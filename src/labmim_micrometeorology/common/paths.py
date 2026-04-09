"""Cross-platform path utilities.

All path handling uses ``pathlib.Path`` so that the same code runs on
Windows and Linux without modification.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_dir(env_var: str = "LABMIM_DATA_DIR", fallback: str | Path = "data") -> Path:
    """Return the data directory, reading from an env var or falling back."""
    raw = os.environ.get(env_var)
    if raw:
        return Path(raw).resolve()
    return Path(fallback).resolve()


def resolve_output_dir(env_var: str = "LABMIM_OUTPUT_DIR", fallback: str | Path = "output") -> Path:
    """Return the output directory, creating it if necessary."""
    raw = os.environ.get(env_var)
    p = Path(raw).resolve() if raw else Path(fallback).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.  Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_files(directory: str | Path, pattern: str = "*.dat") -> list[Path]:
    """Glob for files matching *pattern* inside *directory*, sorted by name."""
    return sorted(Path(directory).glob(pattern))


def find_netcdf_files(directory: str | Path) -> list[Path]:
    """Find all NetCDF files (wrfout_*) in a directory, sorted by name."""
    d = Path(directory)
    files: list[Path] = []
    for ext in ("wrfout_*", "*.nc", "*.nc4"):
        files.extend(d.glob(ext))
    return sorted(set(files))
