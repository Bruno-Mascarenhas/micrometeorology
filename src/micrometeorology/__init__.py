"""LabMiM Micrometeorology — UFBA data-processing toolkit.

Provides modules for:
- WRF model output processing and visualization
- Meteorological sensor data ingestion and aggregation
- Statistical comparison between model and observational data

``__version__`` is ``0+unknown`` when the package is imported from the source
tree before installation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("labmim-micrometeorology")
except PackageNotFoundError:
    __version__ = "0+unknown"
