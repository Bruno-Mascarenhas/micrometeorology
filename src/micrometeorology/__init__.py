"""LabMiM Micrometeorology — UFBA data-processing toolkit.

Provides modules for:
- WRF model output processing and visualization
- Meteorological sensor data ingestion and aggregation
- Statistical comparison between model and observational data
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("labmim-micrometeorology")
except PackageNotFoundError:  # Support direct source-tree imports before installation.
    __version__ = "0+unknown"
