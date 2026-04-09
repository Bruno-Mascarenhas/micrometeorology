"""Structured logging configuration.

Provides a consistent logging setup across all modules.  Import and call
``setup_logging()`` once at application startup (e.g. in a CLI script).
Individual modules obtain their loggers via::

    import logging
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", *, log_file: str | None = None) -> None:
    """Configure the root logger with a uniform format.

    Parameters
    ----------
    level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    log_file:
        Optional path to a log file.  If given, a ``FileHandler`` is added
        alongside the stream handler.
    """
    fmt = "%(asctime)s | %(name)-40s | %(levelname)-7s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # override any prior basicConfig calls
    )

    # Quieten noisy third-party loggers
    for name in ("matplotlib", "PIL", "fiona", "rasterio"):
        logging.getLogger(name).setLevel(logging.WARNING)
