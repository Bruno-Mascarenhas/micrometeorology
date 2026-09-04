"""Reproducibility: global seed control."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for numpy, random, and torch (if available) for reproducibility.

    Seeds the three global streams a run actually draws from — Python's
    ``random``, NumPy's legacy global generator (what scikit-learn and every
    bare ``np.random.*`` call use) and torch's CPU generator, plus all CUDA
    devices when one is present. ``PYTHONHASHSEED`` is exported for any
    subprocess this one spawns; it does not affect the already-running
    interpreter.

    On CUDA this also pins cuDNN to ``deterministic=True`` and
    ``benchmark=False``: the autotuner would otherwise pick a different
    convolution algorithm per run, which changes results in the last bits even
    with every seed fixed. Torch is optional here, so an install without it is
    seeded for Python and NumPy only rather than failing.

    Parameters
    ----------
    seed:
        Value applied to every stream. Comes from the experiment config, so the
        run records what it was seeded with.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
