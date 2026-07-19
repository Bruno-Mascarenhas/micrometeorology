"""Shared trunk and per-task prediction heads.

The :class:`Trunk` maps a fused vector through two ``Linear -> LayerNorm ->
GELU -> Dropout`` blocks (``fusion_dim -> 256 -> 256``) with a residual
connection wherever input and output widths match.  Independent heads read the
trunk output:

- :class:`DHIHead` — a single normalized DHI value.
- :class:`DHIHeteroscedasticHead` — mean and log-variance (clamped to
  ``[-10, 10]``) for a Gaussian NLL loss.
- :class:`KIndexHead` — a single normalized k-index value.
- :class:`SkyHead` — three class logits.
- :class:`CloudFractionHead` — a sigmoid-bounded fraction in ``[0, 1]``.

:class:`Heads` assembles exactly the heads enabled by a
:class:`allsky.config.TargetsConfig`; its ``forward`` returns the corresponding
:class:`allsky.modeling.contracts.ModelOutputs` subset.  Regression heads emit
**normalized-space** values (the engine denormalizes).
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from allsky.config import TargetsConfig
from allsky.modeling.contracts import ModelOutputs

__all__ = [
    "CloudFractionHead",
    "DHIHead",
    "DHIHeteroscedasticHead",
    "Heads",
    "KIndexHead",
    "SkyHead",
    "Trunk",
]

#: Clamp range for the predicted DHI log-variance (numerical stability).
_LOG_VAR_MIN = -10.0
_LOG_VAR_MAX = 10.0


class _TrunkBlock(nn.Module):
    """One ``Linear -> LayerNorm -> GELU -> Dropout`` block, residual when square."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = in_dim == out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block, adding a residual when in/out widths match."""
        y = self.drop(self.act(self.norm(self.linear(x))))
        out: Tensor = x + y if self.residual else y
        return out


class Trunk(nn.Module):
    """Shared MLP trunk mapping the fused vector to a task-agnostic embedding.

    Parameters
    ----------
    in_dim:
        Fused-vector width (the fusion block's ``out_dim``).
    hidden_dim:
        Width of each trunk block (``256`` per the spec).
    n_layers:
        Number of trunk blocks.
    dropout:
        Dropout inside every block.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        dims = [in_dim] + [hidden_dim] * n_layers
        self.blocks = nn.ModuleList(
            _TrunkBlock(dims[i], dims[i + 1], dropout) for i in range(n_layers)
        )
        self._out_dim = hidden_dim

    @property
    def out_dim(self) -> int:
        """Width of the trunk embedding."""
        return self._out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Map a fused vector ``(B, in_dim)`` to ``(B, out_dim)``."""
        for block in self.blocks:
            x = block(x)
        out: Tensor = x
        return out


class DHIHead(nn.Module):
    """Single-value DHI regression head (normalized space)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{"dhi": (B,)}``."""
        return {"dhi": self.linear(x).squeeze(-1)}


class DHIHeteroscedasticHead(nn.Module):
    """DHI head predicting a mean and a clamped log-variance (Gaussian NLL)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{"dhi": (B,), "dhi_log_var": (B,)}`` with log-var clamped."""
        out = self.linear(x)
        mean = out[..., 0]
        log_var = out[..., 1].clamp(_LOG_VAR_MIN, _LOG_VAR_MAX)
        return {"dhi": mean, "dhi_log_var": log_var}


class KIndexHead(nn.Module):
    """Single-value k-index regression head (normalized space)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{"kindex": (B,)}``."""
        return {"kindex": self.linear(x).squeeze(-1)}


class SkyHead(nn.Module):
    """Sky-condition classification head (three logits)."""

    def __init__(self, in_dim: int, n_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{"sky_logits": (B, n_classes)}``."""
        return {"sky_logits": self.linear(x)}


class CloudFractionHead(nn.Module):
    """Cloud-fraction regression head bounded to ``[0, 1]`` via sigmoid."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return ``{"cloud_fraction": (B,)}`` in ``[0, 1]``."""
        return {"cloud_fraction": self.act(self.linear(x)).squeeze(-1)}


class Heads(nn.Module):
    """Bundle of the enabled prediction heads.

    Builds exactly the heads that *targets* enables; the DHI head is the
    heteroscedastic variant when ``targets.dhi.loss == "heteroscedastic"``.
    ``forward`` merges each head's output into a single
    :class:`~allsky.modeling.contracts.ModelOutputs`.
    """

    def __init__(self, in_dim: int, targets: TargetsConfig, *, n_classes: int = 3) -> None:
        super().__init__()
        heads: list[nn.Module] = []
        if targets.dhi.enabled:
            if targets.dhi.loss == "heteroscedastic":
                heads.append(DHIHeteroscedasticHead(in_dim))
            else:
                heads.append(DHIHead(in_dim))
        if targets.kindex.enabled:
            heads.append(KIndexHead(in_dim))
        if targets.sky.enabled:
            heads.append(SkyHead(in_dim, n_classes))
        if targets.cloud_fraction.enabled:
            heads.append(CloudFractionHead(in_dim))
        self.heads = nn.ModuleList(heads)

    def forward(self, x: Tensor) -> ModelOutputs:
        """Return the merged outputs of every enabled head."""
        outputs: dict[str, Tensor] = {}
        for head in self.heads:
            outputs.update(head(x))
        return cast("ModelOutputs", outputs)
