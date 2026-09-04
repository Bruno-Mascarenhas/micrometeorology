"""Cross-platform path utilities.

Every helper here returns ``pathlib.Path`` so the same code runs unmodified on
Windows and Linux.
"""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and its parents if absent, and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def find_files(directory: str | Path, pattern: str) -> list[Path]:
    """Glob for files matching *pattern* inside *directory*, sorted by name.

    The glob is not recursive unless *pattern* says so with ``**``, and the sort
    is lexicographic, which for the station's ``LBM_*_<year>.dat`` naming is not
    the same as chronological -- callers that need ingest order state it
    themselves (see :data:`micrometeorology.sensors.archive.LENTA_MANIFEST`).
    """
    return sorted(Path(directory).glob(pattern))
