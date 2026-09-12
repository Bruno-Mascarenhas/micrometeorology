"""C-Mixup over one training batch (Yao, Wang, Zhang, Zou & Finn 2022, NeurIPS).

Mixup for regression: each row is blended with a partner whose label is close,
so the interpolated label stays meaningful — vanilla mixup between an overcast
and a clear frame would label a physically impossible sky with the mean k*.
The pairing kernel is the paper's eq. 6 on the primary regression target and
the blend its eq. 2, with ``lambda ~ Beta(alpha, alpha)``.

Two decisions depart from the paper's Algorithm 1 and are deliberate:

- the partner is drawn from the current batch rather than from the whole
  training set, so the kernel is a ``(B, B)`` matrix per batch instead of an
  ``(N, N)`` one held in memory;
- a row never draws itself. The paper's kernel gives ``d(i, i) = 0`` the
  largest weight, and at the bandwidth a k* target calls for a batch of a few
  dozen rows would then mostly pair rows with themselves and mix nothing.

The random draws happen on the CPU batch, before it is moved to the device,
from a generator the engine seeds per epoch; the blend itself runs on the
device with the batch. ``torch`` is imported eagerly, so this module is only
ever imported from the training engine.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from labmim_core.sky import SKY_CLASS_COUNT

__all__ = ["CMixup", "MixPlan"]

#: Which batch entries carry the inputs the blend applies to. Both are mixed
#: with the same ``lambda`` as the targets, so the interpolated sample stays
#: one point on the segment between its two sources.
_MIXED_INPUTS = ("image", "features")


@dataclass(frozen=True, slots=True)
class MixPlan:
    """The pairing one batch was dealt: partner, weight and whether to mix at all.

    Attributes
    ----------
    partner:
        ``(B,)`` int64, the row each row is blended with; a row's own index
        where ``mixed`` is False.
    lam:
        ``(B,)`` float32, the weight on the row's own sample; exactly ``1.0``
        where ``mixed`` is False.
    mixed:
        ``(B,)`` bool, False for a row kept as it is — its own or its partner's
        target missing, or no partner within the kernel's reach.
    """

    partner: Tensor
    lam: Tensor
    mixed: Tensor

    def to(self, device: str | torch.device) -> MixPlan:
        """The same plan with its tensors on *device*."""
        return MixPlan(self.partner.to(device), self.lam.to(device), self.mixed.to(device))


@dataclass(frozen=True, slots=True)
class CMixup:
    """C-Mixup of a batch: draw the pairing on the CPU, blend on the device.

    Parameters
    ----------
    alpha:
        Shape of the ``Beta(alpha, alpha)`` the blend weight is drawn from.
    bandwidth:
        Width of the pairing kernel, in the unit of ``kindex`` (k*).
    p:
        Fraction of batches mixed at all; the others pass through untouched.
    regression_targets:
        The enabled regression heads, among ``dhi``, ``kindex`` and
        ``cloud_fraction``; each is blended linearly and each must be present
        on both rows for the pair to mix. ``kindex`` is the target the kernel
        pairs on and must be among them.
    sky:
        Whether the sky head trains: its class then becomes the soft label
        ``lambda * onehot_i + (1 - lambda) * onehot_j`` under ``sky_distribution``,
        and a row without a class does not mix.

    Raises
    ------
    ValueError
        If ``kindex`` is not among *regression_targets*.
    """

    alpha: float
    bandwidth: float
    p: float
    regression_targets: tuple[str, ...]
    sky: bool

    def __post_init__(self) -> None:
        if "kindex" not in self.regression_targets:
            raise ValueError(
                "C-Mixup pairs rows on their kindex target, which is not among the regression "
                f"targets to mix {self.regression_targets}"
            )

    def plan(self, batch: Mapping[str, Tensor], rng: np.random.Generator) -> MixPlan | None:
        """Draw the pairing for one CPU batch, or ``None`` when it is not mixed.

        Parameters
        ----------
        batch:
            The collated batch as the loader emits it — every target tensor on
            the CPU — with ``kindex`` ``(B,)`` float, the other regression
            targets ``(B,)`` float (NaN = missing) and ``sky_class`` ``(B,)``
            int64 (``-1`` = missing).
        rng:
            Seeded generator; three draws per call — the batch gate, the
            partner of every row and the weight of every row — so the stream
            does not depend on which rows turn out mixable.

        Returns
        -------
        MixPlan or None
            ``None`` when the batch gate fails or fewer than two rows carry
            every target the blend needs.
        """
        gate = rng.random()
        kindex = batch["kindex"].numpy().astype(np.float64)
        present = np.isfinite(kindex)
        for name in self.regression_targets:
            present &= np.isfinite(batch[name].numpy())
        if self.sky:
            present &= batch["sky_class"].numpy() >= 0
        n_rows = int(kindex.shape[0])
        uniform = rng.random(n_rows)
        lam = rng.beta(self.alpha, self.alpha, size=n_rows).astype(np.float32)
        if gate >= self.p or int(present.sum()) < 2:
            return None

        labels = np.where(present, kindex, 0.0)
        # Yao et al. 2022, eq. 6: a Gaussian kernel on the label distance.
        weights = np.exp(-((labels[:, None] - labels[None, :]) ** 2) / (2.0 * self.bandwidth**2))
        weights[:, ~present] = 0.0
        weights[~present, :] = 0.0
        np.fill_diagonal(weights, 0.0)
        totals = weights.sum(axis=1)
        mixed = present & (totals > 0.0)
        cdf = np.cumsum(weights / np.where(mixed, totals, 1.0)[:, None], axis=1)
        cdf[mixed] /= cdf[mixed, -1:]
        partner = np.minimum((cdf < uniform[:, None]).sum(axis=1), n_rows - 1)
        own = np.arange(n_rows)
        return MixPlan(
            partner=torch.from_numpy(np.where(mixed, partner, own).astype(np.int64)),
            lam=torch.from_numpy(np.where(mixed, lam, np.float32(1.0)).astype(np.float32)),
            mixed=torch.from_numpy(mixed),
        )

    def apply(self, batch: dict[str, Tensor], plan: MixPlan) -> dict[str, Tensor]:
        """Blend the batch by *plan*; the rows *plan* leaves unmixed come back as they are.

        Parameters
        ----------
        batch:
            The batch on the training device, with ``image`` ``(B, C, H, W)``,
            ``features`` ``(B, F)``, the regression targets ``(B,)`` and
            ``sky_class`` ``(B,)``.
        plan:
            A :class:`MixPlan` on the same device.

        Returns
        -------
        dict[str, Tensor]
            A new dict: the inputs and the regression targets blended, and —
            when the sky head trains — ``sky_distribution`` ``(B, K)`` float32,
            one-hot on the unmixed rows. ``sky_class`` and ``dhi_scale`` stay
            the row's own: the first still marks which rows are labelled, and
            the second is the reference the row's own diffuse was divided by.

        Raises
        ------
        ValueError
            If the batch carries no single ``image``; a window (``image_seq``)
            or an embedding has no one partner frame to blend.
        """
        if "image" not in batch:
            raise ValueError(
                "C-Mixup blends one partner frame per row and needs batch['image']; a windowed "
                "or embedding batch has none"
            )
        out = dict(batch)
        for key in _MIXED_INPUTS:
            out[key] = _blend(batch[key], plan)
        for name in self.regression_targets:
            out[name] = _blend(batch[name], plan)
        if self.sky:
            onehot = functional.one_hot(batch["sky_class"].clamp(min=0), SKY_CLASS_COUNT).to(
                plan.lam.dtype
            )
            out["sky_distribution"] = _blend(onehot, plan)
        return out


def _blend(value: Tensor, plan: MixPlan) -> Tensor:
    """``lam * value + (1 - lam) * value[partner]`` on the mixed rows, *value* elsewhere.

    Selected with ``where`` rather than by arithmetic: at ``lam = 1`` the
    partner's term is ``0 * NaN`` for a missing target, which is NaN.
    """
    shape = (-1,) + (1,) * (value.ndim - 1)
    lam = plan.lam.view(shape).to(value.dtype)
    blended = lam * value + (1.0 - lam) * value[plan.partner]
    return torch.where(plan.mixed.view(shape), blended, value)
