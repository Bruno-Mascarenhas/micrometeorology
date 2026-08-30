"""Which device a run executes on — one answer for the whole package.

An explicit ``"cuda"`` request fails rather than falling back: a run that asks
for the GPU and quietly gets the CPU reports timings and, through AMP, numerics
that belong to a different machine. It lives here rather than inside the
dataloader module because the models resolve their device too.

torch is imported inside the function, so importing this module never pulls it.
"""

__all__ = ["resolve_device"]


def resolve_device(requested: str = "auto") -> str:
    """Resolve a user-requested device into an available torch device string.

    An explicit ``"cuda"`` request is never downgraded: a run that asks for the
    GPU and silently gets the CPU would report timings and, through AMP,
    numerics that belong to a different machine.

    Parameters
    ----------
    requested:
        ``"auto"``, ``"cpu"`` or ``"cuda"``, as written in the runtime config.

    Returns
    -------
    str
        ``"cpu"`` or ``"cuda"``. An install without torch resolves ``"auto"`` to
        ``"cpu"`` rather than raising, since there is nothing to ask.

    Raises
    ------
    ValueError
        If ``"cuda"`` is requested where CUDA is unavailable, or if the request
        is not one of the three accepted values.
    """
    if requested == "cpu":
        return "cpu"
    if requested not in ("auto", "cuda"):
        raise ValueError(f"Unknown device request: {requested}")
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise ValueError(
                "runtime.device='cuda' was requested, but torch is not installed"
            ) from None
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("runtime.device='cuda' was requested, but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
