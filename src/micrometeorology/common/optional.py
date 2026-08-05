"""Actionable errors for the optional dependency extras.

The CLIs import their heavy dependencies lazily so ``--help`` stays fast in a
minimal install.  The cost of that design is that a missing extra surfaces as a
bare ``ModuleNotFoundError: No module named 'torch'`` deep inside a command,
naming neither the extra nor the install command.  :func:`require` is the
precondition check that turns it into the message
``micrometeorology.wrf.animation`` already emits for the ``video`` extra.

It is a *precondition* check (``importlib.util.find_spec``), not an import
wrapper: it never executes the module, so it cannot mask an ``ImportError``
raised from inside first-party code that the guarded import pulls in, and it
keeps the lazy-import guarantee intact (``find_spec`` does not import torch).
"""

import importlib.util

__all__ = ["require"]


def require(module: str, extra: str) -> None:
    """Raise an actionable :class:`ImportError` when *module* is not installed.

    Parameters
    ----------
    module:
        Top-level importable name to check (e.g. ``"torch"``).
    extra:
        The ``labmim-micrometeorology`` extra that provides it (e.g. ``"allsky"``).
    """
    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"{module} is required for this command.  "
            f"Install with: pip install labmim-micrometeorology[{extra}]"
        )
