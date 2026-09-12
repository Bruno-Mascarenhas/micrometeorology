"""Exponential moving average of a model's weights, kept as a shadow state dict.

A checkpoint chosen by a noisy validation metric lands anywhere between the
third and the fiftieth epoch and moves by as much as the seed does; the
running average of the iterates is the classical remedy (Polyak & Juditsky
1992), and its exponential form is what Izmailov et al. 2018 compare their
weight averaging against. The shadow lives on the model's own device, outside
the autograd graph, and exposes ``state_dict`` / ``load_state_dict`` with the
model's own keys so :func:`allsky.training.checkpointing.save_checkpoint`
writes it in exactly the format of ``last.ckpt``.
"""

import contextlib
from collections.abc import Iterator, Mapping

import torch
from torch import Tensor, nn

__all__ = ["ExponentialMovingAverage"]

KEYS_SHOWN_IN_MISMATCH_ERROR = 5


class ExponentialMovingAverage:
    """Shadow copy of *model* moved toward the live weights after every step.

    Parameters
    ----------
    model:
        The module whose weights are averaged. The shadow is seeded from its
        current ``state_dict`` — after any transfer or resume has been applied.
    decay:
        Weight of the shadow in each update, strictly inside ``(0, 1)``:
        ``shadow = decay * shadow + (1 - decay) * weight``.

    Notes
    -----
    Every ``state_dict`` key is shadowed, so the checkpoint carries a complete
    model: parameters that require a gradient are averaged, buffers are copied
    verbatim on every update (a transformer holds none that move, but a
    BatchNorm's running statistics would), and a frozen parameter keeps the
    value it had when the average started, which is also its value at every
    later step. Keys are used rather than modules so a tensor registered under
    two names — the geometry adapter's projection is — is shadowed under both.
    """

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must lie strictly inside (0, 1), got {decay}")
        self.decay = float(decay)
        sources = model.state_dict(keep_vars=True)
        self._shadow: dict[str, Tensor] = {
            key: tensor.detach().clone() for key, tensor in sources.items()
        }
        self._averaged_keys = [
            key
            for key, tensor in sources.items()
            if isinstance(tensor, nn.Parameter) and tensor.requires_grad
        ]
        self._copied_keys = [
            key for key, tensor in sources.items() if not isinstance(tensor, nn.Parameter)
        ]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move the shadow one step toward *model*'s current weights.

        Parameters
        ----------
        model:
            The module the average was built from, after an optimizer step.
        """
        live = model.state_dict(keep_vars=True)
        # Polyak & Juditsky 1992, exponential form: decay * shadow + (1 - decay) * w
        torch._foreach_lerp_(
            [self._shadow[key] for key in self._averaged_keys],
            [live[key].detach() for key in self._averaged_keys],
            1.0 - self.decay,
        )
        for key in self._copied_keys:
            self._shadow[key].copy_(live[key])

    def state_dict(self) -> dict[str, Tensor]:
        """The averaged weights, keyed exactly like the model's own ``state_dict``."""
        return dict(self._shadow)

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Replace the shadow with *state*, which must carry the model's keys.

        Raises
        ------
        KeyError
            If *state* is missing a key the model has or carries one it lacks.
        """
        missing = sorted(set(self._shadow) - set(state))
        unexpected = sorted(set(state) - set(self._shadow))
        if missing or unexpected:
            raise KeyError(
                "averaged weights do not match the model: missing "
                f"{missing[:KEYS_SHOWN_IN_MISMATCH_ERROR]}, "
                f"unexpected {unexpected[:KEYS_SHOWN_IN_MISMATCH_ERROR]}"
            )
        for key, shadow in self._shadow.items():
            shadow.copy_(state[key])

    @contextlib.contextmanager
    def applied_to(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily load the averaged weights into *model*.

        The live weights are cloned before the swap and restored on exit,
        whatever happens inside the block, so an evaluation of the average
        leaves the training state exactly as it found it.
        """
        live = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        model.load_state_dict(self._shadow)
        try:
            yield model
        finally:
            model.load_state_dict(live)
