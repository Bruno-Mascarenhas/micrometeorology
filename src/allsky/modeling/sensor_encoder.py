"""Sensor-feature encoder: a small standardized-vector MLP.

Maps the standardized engineered feature vector ``(B, F)`` to a dense sensor
embedding ``(B, out_dim)`` used by every fusion and by the sensor-only
baseline.  Each block is ``Linear -> LayerNorm -> GELU -> Dropout``; the default
widths ``F -> 64 -> 128`` follow the executor spec.

``torch`` is a hard runtime dependency of this module (it defines an
``nn.Module``), so it is imported eagerly here — the package
:mod:`allsky.modeling` keeps ``import allsky`` torch-free by importing its
submodules lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["SensorEncoder"]


class SensorEncoder(nn.Module):
    """MLP encoder for the standardized sensor-feature vector.

    Parameters
    ----------
    in_dim:
        Number of engineered feature columns ``F`` (must be positive).
    hidden_dims:
        Output width of each block; the last entry is :attr:`out_dim`.  Default
        ``(64, 128)`` gives the spec's ``F -> 64 -> 128``.
    dropout:
        Dropout probability applied at the end of every block.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (64, 128),
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        widths = list(hidden_dims)
        if not widths:
            raise ValueError("hidden_dims must contain at least one width")
        layers: list[nn.Module] = []
        prev = in_dim
        for width in widths:
            layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)]
            prev = width
        self.net = nn.Sequential(*layers)
        self._out_dim = prev

    @property
    def out_dim(self) -> int:
        """Width of the sensor embedding this encoder produces."""
        return self._out_dim

    def forward(self, features: Tensor) -> Tensor:
        """Encode a ``(B, F)`` standardized feature vector to ``(B, out_dim)``."""
        out: Tensor = self.net(features)
        return out
