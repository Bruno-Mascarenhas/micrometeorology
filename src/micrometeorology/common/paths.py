"""Cross-platform path utilities.

All path handling uses ``pathlib.Path`` so that the same code runs on
Windows and Linux without modification.
"""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.  Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_files(directory: str | Path, pattern: str = "*.dat") -> list[Path]:
    """Glob for files matching *pattern* inside *directory*, sorted by name."""
    return sorted(Path(directory).glob(pattern))
