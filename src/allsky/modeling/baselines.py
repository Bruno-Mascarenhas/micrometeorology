"""Baseline models: climatology (V0), sensor-only (V1), image-only (V2).

- :class:`ClimatologyModel` — predicts constant per-target values (the
  train-split means in **normalized** space) irrespective of the input, plus
  sky logits equal to the train class-frequency log-probabilities.  It carries
  a single dummy parameter so an optimizer has something to step, and every
  output is tied to that parameter (multiplied by zero) so ``loss.backward()``
  always has a graph.
- :class:`SensorOnlyModel` — sensor encoder -> trunk -> heads (no visual
  branch).
- :class:`ImageOnlyModel` — visual encoder -> trunk -> heads (no sensor
  branch).

All three honour the :class:`allsky.modeling.contracts.MultimodalModel`
contract (``forward(batch) -> ModelOutputs``).
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from allsky.config import TargetsConfig
from allsky.features.normalization import TargetNormalizer
from allsky.modeling.contracts import ModelOutputs
from allsky.modeling.heads import Heads, Trunk
from allsky.modeling.sensor_encoder import SensorEncoder
from allsky.modeling.visual_encoder import split_backbone_param_groups
from labmim_core.sky import SKY_CLASS_COUNT

__all__ = [
    "ClimatologyModel",
    "ImageOnlyModel",
    "SensorOnlyModel",
]


class ClimatologyModel(nn.Module):
    """Constant-prediction baseline (per-target train means; class frequencies).

    Regression constants live in **normalized** space (the engine denormalizes
    for reporting); :meth:`fit_from_targets` accepts raw target arrays and, when
    given the train-split :class:`TargetNormalizer` mapping, stores the
    normalized mean (otherwise the raw mean).

    Parameters
    ----------
    targets:
        Which heads are enabled (:class:`allsky.config.TargetsConfig`); the DHI
        head also emits a zero ``dhi_log_var`` when heteroscedastic so its key
        set matches the trained model.
    n_classes:
        Number of sky classes for the frequency logits.
    """

    dhi_const: Tensor
    kindex_const: Tensor
    cloud_fraction_const: Tensor
    sky_logits_const: Tensor

    def __init__(self, targets: TargetsConfig, *, n_classes: int = SKY_CLASS_COUNT) -> None:
        super().__init__()
        self.n_classes = n_classes
        self._enabled = {
            "dhi": targets.dhi.enabled,
            "kindex": targets.kindex.enabled,
            "cloud_fraction": targets.cloud_fraction.enabled,
            "sky": targets.sky.enabled,
        }
        self._heteroscedastic = targets.dhi.enabled and targets.dhi.loss == "heteroscedastic"
        self._dummy = nn.Parameter(torch.zeros(1))
        self.register_buffer("dhi_const", torch.zeros(1))
        self.register_buffer("kindex_const", torch.zeros(1))
        self.register_buffer("cloud_fraction_const", torch.zeros(1))
        self.register_buffer("sky_logits_const", torch.zeros(n_classes))

    def fit_from_targets(
        self,
        *,
        dhi: ArrayLike | None = None,
        kindex: ArrayLike | None = None,
        cloud_fraction: ArrayLike | None = None,
        sky_class: ArrayLike | None = None,
        target_normalizers: Mapping[str, TargetNormalizer] | None = None,
    ) -> None:
        """Set the constant buffers from raw train-split target arrays.

        Only the arrays provided are used; missing ones keep their zero default.
        Regression means are computed over finite values and normalized with
        *target_normalizers* (keyed by ``"dhi"`` / ``"kindex"`` /
        ``"cloud_fraction"``) when available.  Sky logits are the
        log-frequencies of the valid (``>= 0``) class labels.

        Parameters
        ----------
        dhi:
            ``(N,)`` train-split diffuse target as the head receives it: W m-2,
            or the ratio to the clear-sky reference under the ``clearsky_index``
            parameterization; NaN marks a missing target.
        kindex:
            ``(N,)`` raw train-split clear-sky index (dimensionless ratio);
            NaN marks a missing target.
        cloud_fraction:
            ``(N,)`` raw train-split cloud fraction in ``[0, 1]``
            (dimensionless); NaN marks a missing target.
        sky_class:
            ``(N,)`` integer sky-class labels; ``-1`` marks a missing label and
            is excluded from the frequency count.
        target_normalizers:
            Train-split normalizers used to map each regression mean into the
            normalized space the trained models predict in.

        Raises
        ------
        ValueError
            If a provided regression array holds no finite value: a constant
            ``0.0`` would be reported as a plausible baseline, and 0 W m-2 is a
            physically valid irradiance rather than a missing one.
        """
        for name, values in (
            ("dhi", dhi),
            ("kindex", kindex),
            ("cloud_fraction", cloud_fraction),
        ):
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if not finite.size:
                raise ValueError(
                    f"climatology target {name!r} has no finite value to average; a constant "
                    "0.0 would be reported as a plausible baseline, and for an irradiance "
                    "0 W m-2 is a physically valid reading rather than a missing one"
                )
            mean = float(finite.mean())
            if target_normalizers is not None and name in target_normalizers:
                mean = float(target_normalizers[name].normalize(mean))
            getattr(self, f"{name}_const").fill_(mean)

        if sky_class is not None:
            labels = np.asarray(sky_class)
            valid = labels[labels >= 0].astype(np.int64)
            counts = np.bincount(valid, minlength=self.n_classes)[: self.n_classes].astype(
                np.float64
            )
            total = counts.sum()
            freq = counts / total if total > 0 else np.full(self.n_classes, 1.0 / self.n_classes)
            logits = np.log(np.clip(freq, 1e-8, None))
            self.sky_logits_const.copy_(torch.tensor(logits, dtype=torch.float32))

    def forward(self, batch: dict[str, Tensor]) -> ModelOutputs:
        """Broadcast the fitted constants to the batch, keeping a gradient path.

        Parameters
        ----------
        batch:
            Batch dict; only its leading dimension is read (from ``features``
            when present, else from an arbitrary entry).

        Returns
        -------
        ModelOutputs
            The enabled heads' keys, each ``(B,)`` (or ``(B, n_classes)`` for
            ``sky_logits``) float32.  Regression entries are the train means in
            **normalized** space; every entry is tied to the dummy parameter
            through a multiply-by-zero so ``loss.backward()`` always has a graph.
        """
        reference = batch["features"] if "features" in batch else next(iter(batch.values()))
        batch_size = int(reference.shape[0])
        zero = self._dummy * 0.0
        outputs: dict[str, Tensor] = {}
        if self._enabled["dhi"]:
            outputs["dhi"] = self.dhi_const.expand(batch_size) + zero
            if self._heteroscedastic:
                outputs["dhi_log_var"] = torch.zeros(batch_size, device=zero.device) + zero
        if self._enabled["kindex"]:
            outputs["kindex"] = self.kindex_const.expand(batch_size) + zero
        if self._enabled["cloud_fraction"]:
            outputs["cloud_fraction"] = self.cloud_fraction_const.expand(batch_size) + zero
        if self._enabled["sky"]:
            outputs["sky_logits"] = self.sky_logits_const.expand(
                batch_size, self.n_classes
            ) + zero.unsqueeze(-1)
        return cast("ModelOutputs", outputs)


class SensorOnlyModel(nn.Module):
    """Sensor-only baseline: ``sensor encoder -> trunk -> heads``.

    Parameters
    ----------
    n_features:
        Number of engineered feature columns ``F`` served to the model.
    targets:
        Enabled heads (:class:`allsky.config.TargetsConfig`).
    sensor_hidden:
        Sensor-encoder block widths; the last one is the sensor embedding width
        that feeds the trunk.
    trunk_hidden, trunk_layers:
        Trunk width and depth.
    dropout:
        Dropout shared by the sensor encoder and the trunk.
    """

    def __init__(
        self,
        n_features: int,
        targets: TargetsConfig,
        *,
        sensor_hidden: Sequence[int] = (64, 128),
        trunk_hidden: int = 256,
        trunk_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sensor_encoder = SensorEncoder(n_features, sensor_hidden, dropout=dropout)
        self.trunk = Trunk(
            int(self.sensor_encoder.out_dim), trunk_hidden, trunk_layers, dropout=dropout
        )
        self.heads = Heads(int(self.trunk.out_dim), targets)

    def forward(self, batch: dict[str, Tensor]) -> ModelOutputs:
        """Encode the sensor vector and return the enabled head outputs.

        Parameters
        ----------
        batch:
            Batch dict whose ``features`` entry is ``(B, F)`` float32
            standardized engineered features (dimensionless).

        Returns
        -------
        ModelOutputs
            The enabled heads' outputs (regression entries in normalized space).
        """
        sensor = self.sensor_encoder(batch["features"])
        outputs: ModelOutputs = self.heads(self.trunk(sensor))
        return outputs


class ImageOnlyModel(nn.Module):
    """Image/embedding-only baseline: ``visual encoder -> trunk -> heads``.

    Parameters
    ----------
    visual_encoder:
        A visual source from
        :func:`allsky.modeling.visual_encoder.build_visual_encoder` exposing an
        integer ``out_dim`` and a ``forward(batch)``.
    targets:
        Enabled heads.
    trunk_hidden, trunk_layers:
        Trunk width and depth.
    dropout:
        Dropout inside the trunk.
    backbone_lr:
        Learning rate for the image backbone's own parameter group (see
        :meth:`param_groups`); ``None`` trains everything at the run's ``train.lr``.
    """

    def __init__(
        self,
        visual_encoder: nn.Module,
        targets: TargetsConfig,
        *,
        trunk_hidden: int = 256,
        trunk_layers: int = 2,
        dropout: float = 0.1,
        backbone_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.backbone_lr = backbone_lr
        visual_dim = cast("int", visual_encoder.out_dim)
        self.trunk = Trunk(visual_dim, trunk_hidden, trunk_layers, dropout=dropout)
        self.heads = Heads(int(self.trunk.out_dim), targets)

    def forward(self, batch: dict[str, Tensor]) -> ModelOutputs:
        """Encode the visual input and return the enabled head outputs.

        Parameters
        ----------
        batch:
            Batch dict read by the visual encoder: ``image`` ``(B, 3, H, W)``
            float32 in image mode, or ``embedding`` / ``embedding_seq`` in
            embedding mode.  ``features`` is ignored by this baseline.

        Returns
        -------
        ModelOutputs
            The enabled heads' outputs (regression entries in normalized space).
        """
        visual = self.visual_encoder(batch)
        outputs: ModelOutputs = self.heads(self.trunk(visual))
        return outputs

    def param_groups(self, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        """Optimizer parameter groups; the image backbone gets its own LR.

        Same split (and same group order) as
        :meth:`allsky.modeling.multimodal.MultimodalNet.param_groups`: without it
        the engine would fall back to one group and train the unfrozen ViT blocks
        at the head learning rate.

        Parameters
        ----------
        backbone_lr:
            Overrides the constructor's ``backbone_lr``; ``None`` falls back to
            it, and a ``None`` on both sides collapses to a single group.

        Returns
        -------
        list[dict[str, Any]]
            ``[backbone_group, everything_else]``, or one group of every
            trainable parameter when no backbone rate applies.
        """
        lr = backbone_lr if backbone_lr is not None else self.backbone_lr
        return split_backbone_param_groups(self, self.visual_encoder, lr)
