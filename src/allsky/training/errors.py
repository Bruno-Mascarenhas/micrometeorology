"""The training engine's own error type.

Kept in its own torch-free module so ``allsky train`` can catch it without
importing the engine — and therefore torch — before it needs to.
"""

__all__ = ["TrainingError"]


class TrainingError(RuntimeError):
    """A run cannot proceed for a reason the operator can act on.

    Raised for a requested device the host does not have, a backbone that cannot
    be constructed from the config, a resume against a different dataset, an AMP
    dtype the device cannot autocast, and a run whose monitored metric was
    non-finite in every epoch.  All of them carry a message naming the knob to
    change, so the CLI reports the message and exits non-zero rather than
    printing a traceback of the engine's internals.

    Subclasses :class:`RuntimeError`, which is what the engine raised before this
    type existed, so callers that catch that keep working.
    """
