"""Shared type definitions, enums, and data classes.

Centralizes all domain-specific types used across the package so that
modules depend on stable, well-documented interfaces rather than raw strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# WRF variable definitions
# ---------------------------------------------------------------------------


class WRFVariable(StrEnum):
    """Meteorological variables produced by WRF."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    WIND = "wind"
    RAIN = "rain"
    VAPOR = "vapor"
    HFX = "HFX"
    LH = "LH"
    SWDOWN = "SWDOWN"
    WEIBULL = "weibull"

    # Eolic potential is height-dependent; the suffix is handled at runtime.
    WIND_POTENTIAL = "poteolico"


class GridLevel(StrEnum):
    """WRF nested grid levels."""

    D01 = "D01"
    D02 = "D02"
    D03 = "D03"
    D04 = "D04"
    D05 = "D05"


# ---------------------------------------------------------------------------
# Default colormaps per WRF variable
# ---------------------------------------------------------------------------

VARIABLE_COLORMAPS: dict[WRFVariable | str, str] = {
    WRFVariable.TEMPERATURE: "hot_r",
    WRFVariable.WIND: "PuBu",
    WRFVariable.VAPOR: "YlGnBu",
    WRFVariable.PRESSURE: "Blues",
    WRFVariable.RAIN: "afmhot_r",
    WRFVariable.HFX: "jet",
    WRFVariable.LH: "jet",
    WRFVariable.SWDOWN: "hot_r",
    WRFVariable.WEIBULL: "jet",
    WRFVariable.WIND_POTENTIAL: "Blues",
}

# Map from our enum to the NetCDF variable / output file suffix
VARIABLE_NETCDF_MAP: dict[WRFVariable | str, str] = {
    WRFVariable.TEMPERATURE: "TEMP",
    WRFVariable.PRESSURE: "PRES",
    WRFVariable.WIND: "WIND",
    WRFVariable.RAIN: "RAIN",
    WRFVariable.VAPOR: "VAPOR",
    WRFVariable.HFX: "HFX",
    WRFVariable.LH: "LH",
    WRFVariable.SWDOWN: "SWDOWN",
    WRFVariable.WEIBULL: "K_WEIB",
}

WEEKDAY_PT: dict[int, str] = {
    1: "Segunda",
    2: "Terça",
    3: "Quarta",
    4: "Quinta",
    5: "Sexta",
    6: "Sábado",
    7: "Domingo",
}


# ---------------------------------------------------------------------------
# Sensor definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorLimit:
    """Physical limits for quality-control filtering of a sensor column."""

    column: str
    lower: float
    upper: float


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable record of a sensor calibration for a specific date range.

    These are historical facts — they represent the actual calibration state
    of a physical instrument during a specific period and must not be altered.
    New calibration periods should be *appended* to the configuration.
    """

    column: str
    start_date: str | None  # ISO format YYYY-MM-DD; None = beginning of time
    end_date: str | None  # ISO format YYYY-MM-DD; None = end of time
    factor: float  # multiplicative correction factor
    description: str = ""


@dataclass
class SensorConfig:
    """Configuration for a sensor dataset ingestion.

    Headers may vary between files because sensors are added/removed from
    the datalogger.  The parser handles this dynamically.
    """

    name: str
    separator: str = ","
    skip_rows: list[int] = field(default_factory=lambda: [0, 2, 3])
    timestamp_column: str = "TIMESTAMP"
    drop_columns: list[str] = field(default_factory=list)
    sentinel_value: float = -900.0
    limits: list[SensorLimit] = field(default_factory=list)
    # Columns that should NOT be averaged (e.g. precipitation=sum, wind_dir=vector)
    sum_columns: list[str] = field(default_factory=list)
    wind_dir_columns: list[str] = field(default_factory=list)
    min_samples_per_hour: int = 6
    extra: dict[str, Any] = field(default_factory=dict)
