"""Multi-task training loss for the multimodal all-sky heads.

:class:`MultitaskLoss` is assembled from an
:class:`allsky.config.TargetsConfig` and the train-split
:class:`allsky.features.normalization.TargetNormalizer` mapping.  Each enabled
head contributes a masked, unweighted component; the reported ``loss`` is the
weighted sum.

Target/prediction spaces
------------------------
Model regression outputs live in **normalized** space (the heads predict the
standardized quantity).  Targets arrive in the batch as **raw physical units**,
so this module normalizes ``dhi`` and ``kindex`` internally with the supplied
:class:`TargetNormalizer` before comparing them to the model outputs.
``cloud_fraction`` is the one exception: it is already a bounded fraction in
``[0, 1]`` (the head is sigmoid-bounded), so it is compared raw with no
normalization.

Masking
-------
A regression head only counts rows whose target is finite
(:func:`torch.isfinite`); the sky head only counts rows with a valid class
(``sky_class >= 0``).  A head with **zero** valid targets in a batch contributes
an exact, grad-safe zero (never a NaN) so the total stays finite and
differentiable.

``torch`` is a hard runtime dependency here (the loss is an ``nn.Module``), so it
is imported eagerly — this module is only ever imported lazily from the
training engine / CLIs, keeping ``import allsky`` torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional alias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from allsky.config import TargetsConfig
    from allsky.features.normalization import TargetNormalizer
    from allsky.modeling.contracts import ModelOutputs

__all__ = ["MultitaskLoss"]

#: Regression loss kinds usable by the DHI/kindex/cloud-fraction heads.
_REGRESSION_KINDS = frozenset({"mse", "mae", "huber"})


class MultitaskLoss(nn.Module):
    """Weighted multi-task loss over the enabled prediction heads.

    Parameters
    ----------
    targets:
        Which heads are enabled and their per-head ``weight`` / ``loss`` kind
        (:class:`allsky.config.TargetsConfig`).  The DHI head is a Gaussian NLL
        (heteroscedastic) when ``targets.dhi.loss == "heteroscedastic"``.
    target_normalizers:
        Train-split normalizers keyed by ``"dhi"`` / ``"kindex"``.  Their
        mean/std map the raw physical targets into the normalized space the
        model predicts in.  ``cloud_fraction`` is never normalized.
    huber_delta:
        Transition point of the Huber loss for the ``"huber"`` kind.
    learned_uncertainty:
        **Off by default.** Interface stub for learned (homoscedastic) task
        weighting in the sense of Kendall & Gal (2018): replace the fixed
        per-head ``weight`` with learned ``log_sigma`` parameters so the total
        becomes ``sum_i exp(-s_i) * L_i + s_i``.  Not yet implemented; enabling
        it raises :class:`NotImplementedError` so callers cannot silently rely
        on unweighted behaviour.

    Notes
    -----
    :meth:`forward` returns ``{"loss": total, "loss_<head>": component, ...}``
    where the total is the weighted sum and every ``loss_<head>`` component is
    the **unweighted** per-head loss (only enabled heads appear).
    """

    def __init__(
        self,
        targets: TargetsConfig,
        target_normalizers: Mapping[str, TargetNormalizer],
        *,
        huber_delta: float = 1.0,
        learned_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if learned_uncertainty:
            raise NotImplementedError(
                "learned uncertainty weighting is not implemented yet; leave "
                "learned_uncertainty=False to use the configured fixed weights"
            )
        self.learned_uncertainty = learned_uncertainty
        self._huber_delta = float(huber_delta)

        self._dhi_enabled = bool(targets.dhi.enabled)
        self._dhi_weight = float(targets.dhi.weight)
        self._dhi_kind = str(targets.dhi.loss)
        self._kindex_enabled = bool(targets.kindex.enabled)
        self._kindex_weight = float(targets.kindex.weight)
        self._kindex_kind = str(targets.kindex.loss)
        self._sky_enabled = bool(targets.sky.enabled)
        self._sky_weight = float(targets.sky.weight)
        self._cloud_enabled = bool(targets.cloud_fraction.enabled)
        self._cloud_weight = float(targets.cloud_fraction.weight)
        # cloud_fraction has no configurable loss kind in the current config;
        # default to MSE but honour a `loss` attribute if a future config adds it.
        self._cloud_kind = str(getattr(targets.cloud_fraction, "loss", "mse"))

        self._dhi_mean, self._dhi_std = _norm_stats(target_normalizers, "dhi")
        self._kindex_mean, self._kindex_std = _norm_stats(target_normalizers, "kindex")

    def forward(self, outputs: ModelOutputs, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the per-head components and the weighted total for *batch*.

        Parameters
        ----------
        outputs:
            The model's :class:`allsky.modeling.contracts.ModelOutputs` (only
            the enabled heads' keys are read).
        batch:
            Batch dict with raw physical targets ``dhi`` / ``kindex`` /
            ``cloud_fraction`` (``float``, NaN = missing) and ``sky_class``
            (``int64``, ``-1`` = missing).
        """
        components: dict[str, Tensor] = {}
        total: Tensor | None = None

        if self._dhi_enabled:
            component = self._dhi_loss(outputs, batch["dhi"])
            components["loss_dhi"] = component
            total = _accumulate(total, self._dhi_weight, component)
        if self._kindex_enabled:
            component = self._regression_loss(
                outputs["kindex"],
                batch["kindex"],
                self._kindex_kind,
                self._kindex_mean,
                self._kindex_std,
            )
            components["loss_kindex"] = component
            total = _accumulate(total, self._kindex_weight, component)
        if self._sky_enabled:
            component = self._sky_loss(outputs["sky_logits"], batch["sky_class"])
            components["loss_sky"] = component
            total = _accumulate(total, self._sky_weight, component)
        if self._cloud_enabled:
            component = self._regression_loss(
                outputs["cloud_fraction"],
                batch["cloud_fraction"],
                self._cloud_kind,
                mean=0.0,
                std=1.0,
            )
            components["loss_cloud_fraction"] = component
            total = _accumulate(total, self._cloud_weight, component)

        if total is None:
            # No head enabled: return a finite, grad-free zero.
            total = torch.zeros((), dtype=torch.float32)
        return {"loss": total, **components}

    # -- per-head helpers ---------------------------------------------------

    def _dhi_loss(self, outputs: ModelOutputs, target: Tensor) -> Tensor:
        """DHI component: heteroscedastic Gaussian NLL or a plain regression loss."""
        pred = outputs["dhi"]
        if self._dhi_kind != "heteroscedastic":
            return self._regression_loss(
                pred, target, self._dhi_kind, self._dhi_mean, self._dhi_std
            )
        log_var = outputs["dhi_log_var"]
        mask = torch.isfinite(target)
        if not bool(mask.any()):
            return (pred * 0.0).sum() + (log_var * 0.0).sum()
        normalized = (target[mask] - self._dhi_mean) / self._dhi_std
        residual = pred[mask] - normalized
        lv = log_var[mask]
        # Gaussian NLL (dropping the 0.5*log(2*pi) constant): larger log-variance
        # trades a linear penalty for a shrunk squared-error term, so it lowers
        # the loss for large residuals and raises it for small ones.
        nll = 0.5 * (torch.exp(-lv) * residual.pow(2) + lv)
        return nll.mean()

    def _regression_loss(
        self, pred: Tensor, target: Tensor, kind: str, mean: float, std: float
    ) -> Tensor:
        """Masked regression loss; normalizes the (finite) targets before comparing."""
        if kind not in _REGRESSION_KINDS:
            raise ValueError(f"unknown regression loss kind {kind!r}; expected {_REGRESSION_KINDS}")
        mask = torch.isfinite(target)
        if not bool(mask.any()):
            return (pred * 0.0).sum()
        normalized = (target[mask] - mean) / std
        selected = pred[mask]
        if kind == "mse":
            return F.mse_loss(selected, normalized)
        if kind == "mae":
            return F.l1_loss(selected, normalized)
        return F.huber_loss(selected, normalized, delta=self._huber_delta)

    @staticmethod
    def _sky_loss(logits: Tensor, sky_class: Tensor) -> Tensor:
        """Masked cross-entropy over rows with a valid (``>= 0``) class label."""
        mask = sky_class >= 0
        if not bool(mask.any()):
            return (logits * 0.0).sum()
        return F.cross_entropy(logits[mask], sky_class[mask])


def _accumulate(total: Tensor | None, weight: float, component: Tensor) -> Tensor:
    """Add ``weight * component`` to the running (possibly unset) total."""
    weighted = weight * component
    return weighted if total is None else total + weighted


def _norm_stats(normalizers: Mapping[str, TargetNormalizer], key: str) -> tuple[float, float]:
    """Return ``(mean, std)`` for *key*, or the identity ``(0.0, 1.0)`` if absent."""
    normalizer = normalizers.get(key)
    if normalizer is None:
        return 0.0, 1.0
    return float(normalizer.mean), float(normalizer.std)
