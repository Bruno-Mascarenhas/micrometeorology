"""Multi-task training loss for the multimodal all-sky heads.

:class:`MultitaskLoss` is assembled from an
:class:`allsky.config.TargetsConfig` and the train-split
:class:`allsky.features.normalization.TargetNormalizer` mapping.  Each enabled
head contributes a masked, unweighted component; the reported ``loss`` is the
weighted sum.

Target/prediction spaces
------------------------
Model regression outputs live in **normalized** space (the heads predict the
standardized quantity).  Targets arrive in the batch in the unit the dataset
serves them in — ``dhi`` in W m-2 under ``targets.dhi.parameterization="raw"``,
and a dimensionless ratio to the clear-sky reference under
``"clearsky_index"`` — so this module normalizes ``dhi`` and ``kindex``
internally with the supplied :class:`TargetNormalizer`, which was fitted on that
same served quantity, before comparing them to the model outputs.
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

``torch`` is imported eagerly here (the loss is an ``nn.Module``), so this module
must only ever be imported lazily from the training engine / CLIs, which is what
keeps ``import allsky`` torch-free.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

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

    Raises
    ------
    NotImplementedError
        If *learned_uncertainty* is enabled.

    Notes
    -----
    :meth:`forward` returns ``{"loss": total, "loss_<head>": component, ...}``
    where the total is the weighted sum and every ``loss_<head>`` component is
    the **unweighted** per-head loss (only enabled heads appear).

    ``TargetsConfig.cloud_fraction`` declares no loss kind and forbids extra keys,
    so the cloud head is MSE by construction.
    """

    _sky_class_weights: Tensor | None

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
        self._huber_delta = float(huber_delta)

        self._dhi_enabled = bool(targets.dhi.enabled)
        self._dhi_weight = float(targets.dhi.weight)
        self._dhi_kind = str(targets.dhi.loss)
        self._kindex_enabled = bool(targets.kindex.enabled)
        self._kindex_weight = float(targets.kindex.weight)
        self._kindex_kind = str(targets.kindex.loss)
        self._sky_enabled = bool(targets.sky.enabled)
        self._sky_weight = float(targets.sky.weight)
        self._sky_label_smoothing = float(targets.sky.label_smoothing)
        self._sky_ordinal_tau = targets.sky.ordinal_tau
        self.register_buffer(
            "_sky_class_weights",
            None
            if targets.sky.class_weights is None
            else torch.tensor(targets.sky.class_weights, dtype=torch.float32),
        )
        self._cloud_enabled = bool(targets.cloud_fraction.enabled)
        self._cloud_weight = float(targets.cloud_fraction.weight)
        self._cloud_kind = "mse"

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
            Batch dict with the served targets: ``dhi`` ``(B,)`` in W m-2 under
            ``targets.dhi.parameterization="raw"`` and dimensionless under
            ``"clearsky_index"``,
            ``kindex`` ``(B,)`` (dimensionless ratio) and ``cloud_fraction``
            ``(B,)`` in ``[0, 1]`` — all ``float``, NaN = missing — plus
            ``sky_class`` ``(B,)`` ``int64`` with ``-1`` = missing.

        Returns
        -------
        dict[str, Tensor]
            ``{"loss": total, "loss_<head>": component, ...}``, every value a
            scalar tensor.  With no head enabled the total is an exact
            grad-free zero rather than an empty or NaN value.
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
            total = torch.zeros((), dtype=torch.float32)
        return {"loss": total, **components}

    def _dhi_loss(self, outputs: ModelOutputs, target: Tensor) -> Tensor:
        """DHI component: heteroscedastic Gaussian NLL or a plain regression loss."""
        pred: Tensor = outputs["dhi"]
        if self._dhi_kind != "heteroscedastic":
            return self._regression_loss(
                pred, target, self._dhi_kind, self._dhi_mean, self._dhi_std
            )
        log_var: Tensor = outputs["dhi_log_var"]
        mask = torch.isfinite(target)
        normalized = (_sanitised(target, mask) - self._dhi_mean) / self._dhi_std
        residual = pred - normalized
        # Gaussian NLL dropping the constant 0.5*log(2*pi).
        nll = 0.5 * (torch.exp(-log_var) * residual.pow(2) + log_var)
        return _masked_mean(nll, mask)

    def _regression_loss(
        self, pred: Tensor, target: Tensor, kind: str, mean: float, std: float
    ) -> Tensor:
        """Masked regression loss; normalizes the (finite) targets before comparing."""
        if kind not in _REGRESSION_KINDS:
            raise ValueError(f"unknown regression loss kind {kind!r}; expected {_REGRESSION_KINDS}")
        mask = torch.isfinite(target)
        normalized = (_sanitised(target, mask) - mean) / std
        if kind == "mse":
            per_row = functional.mse_loss(pred, normalized, reduction="none")
        elif kind == "mae":
            per_row = functional.l1_loss(pred, normalized, reduction="none")
        else:
            per_row = functional.huber_loss(
                pred, normalized, delta=self._huber_delta, reduction="none"
            )
        return _masked_mean(per_row, mask)

    def _sky_loss(self, logits: Tensor, sky_class: Tensor) -> Tensor:
        """Masked cross-entropy over rows with a valid (``>= 0``) class label.

        Hard targets by default; ``label_smoothing`` mixes them with the uniform
        distribution, and ``ordinal_tau`` replaces them by the soft targets of
        :func:`ordinal_soft_targets`. ``class_weights`` scales each row by its
        class under either target; the masked mean is over rows, not over
        weights, so a re-weighted batch is not re-normalised.
        """
        mask = sky_class >= 0
        # `clamp(min=0)` only feeds the masked-out rows a valid index; their loss
        # is zeroed below. `ignore_index=-1` with the default reduction would do
        # the masking too, but returns NaN for a batch where every row is masked.
        safe = sky_class.clamp(min=0)
        weights = self._sky_class_weights
        if self._sky_ordinal_tau is None:
            per_row = functional.cross_entropy(
                logits,
                safe,
                weight=weights,
                reduction="none",
                label_smoothing=self._sky_label_smoothing,
            )
        else:
            soft = ordinal_soft_targets(safe, logits.shape[-1], self._sky_ordinal_tau)
            per_row = -(soft * functional.log_softmax(logits, dim=-1)).sum(dim=-1)
            if weights is not None:
                per_row = per_row * weights[safe]
        return _masked_mean(per_row, mask)


def ordinal_soft_targets(labels: Tensor, n_classes: int, tau: float) -> Tensor:
    """Soft target distribution that decays with the rank distance to the label.

    The encoding of Díaz & Marathe (2019, "Soft Labels for Ordinal Regression",
    CVPR): ``softmax_k(-|k - y| / tau)``, with the absolute rank distance as the
    metric so the penalty grows one step per class — the same distance the
    evaluation's ordinal MAE counts. As ``tau`` shrinks the distribution
    converges to the one-hot target and the loss to plain cross-entropy.

    Parameters
    ----------
    labels:
        ``(B,)`` int64 class indices in ``[0, n_classes)``.
    n_classes:
        Number of ordered classes.
    tau:
        Temperature of the decay, in class-rank units; positive.

    Returns
    -------
    Tensor
        ``(B, n_classes)`` float32 distributions, each row summing to one.
    """
    ranks = torch.arange(n_classes, device=labels.device, dtype=torch.float32)
    distance = (ranks[None, :] - labels[:, None].to(torch.float32)).abs()
    return torch.softmax(-distance / float(tau), dim=-1)


def _sanitised(target: Tensor, mask: Tensor) -> Tensor:
    """*target* with its masked-out entries replaced by a finite placeholder.

    Boolean-mask INDEXING (``target[mask]``) copies to a host-shaped result, so
    it has to know how many rows survive — a device synchronisation, once per
    head per batch, which the engine's own loop is written to avoid. Every loss
    below therefore computes over the whole batch and zeroes the rows it does
    not want, which needs the masked entries to be finite: a NaN times zero is
    still NaN, and it would reach the gradient.
    """
    return torch.where(mask, target, torch.zeros_like(target))


def _masked_mean(per_row: Tensor, mask: Tensor) -> Tensor:
    """Mean of *per_row* over the rows *mask* keeps, zero when it keeps none.

    The denominator is clamped rather than branched on, for the same reason: a
    Python ``if`` over a device tensor is a synchronisation. A batch with no
    valid row yields exactly ``0.0`` with the gradient path intact, which is
    what the branch it replaces returned.
    """
    kept = torch.where(mask, per_row, torch.zeros_like(per_row))
    return kept.sum() / mask.sum().clamp(min=1)


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
