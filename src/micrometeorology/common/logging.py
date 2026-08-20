"""One logging setup shared by every entry point.

Call ``setup_logging()`` once at application startup (e.g. in a CLI script).
Individual modules obtain their loggers via::

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a uniform format.

    Parameters
    ----------
    level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, …), case-insensitive.

    Raises
    ------
    ValueError
        When *level* names no logging level. Falling back to ``INFO`` would run
        the whole pipeline at a verbosity the operator did not ask for and never
        say so.

    Notes
    -----
    ``matplotlib``, ``PIL``, ``fiona`` and ``rasterio`` are pinned to
    ``WARNING``: at ``INFO`` they bury the pipeline's own messages under font
    cache and driver chatter.
    """
    known_levels = logging.getLevelNamesMapping()
    requested = level.upper()
    if requested not in known_levels:
        raise ValueError(
            f"unknown logging level {level!r}; expected one of {', '.join(sorted(known_levels))}"
        )

    fmt = "%(asctime)s | %(name)-40s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=known_levels[requested],
        format=fmt,
        datefmt=datefmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        # An imported library may already have configured the root logger; this
        # entry point's format and level are the ones that must win.
        force=True,
    )

    for name in ("matplotlib", "PIL", "fiona", "rasterio"):
        logging.getLogger(name).setLevel(logging.WARNING)
