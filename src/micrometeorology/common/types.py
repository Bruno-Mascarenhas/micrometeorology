"""Shared type definitions, enums, and data classes.

Centralizes all domain-specific types used across the package so that
modules depend on stable, well-documented interfaces rather than raw strings.
"""

from __future__ import annotations

from enum import StrEnum

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
    SKIN_TEMPERATURE = "skin_temperature"
    RELATIVE_HUMIDITY = "relative_humidity"
    HFX = "HFX"
    LH = "LH"
    SWDOWN = "SWDOWN"
    GLW = "GLW"
    WEIBULL = "weibull"

    # Derived surface radiation budget. wrfout carries the downwelling fluxes
    # (SWDOWN, GLW) directly but the upwelling ones only when the RRTMG
    # bottom-of-atmosphere diagnostics are enabled, so these are reconstructed
    # from EMISS/TSK/ALBEDO/COSZEN — fields every run carries. See
    # ``micrometeorology.wrf.variables`` for formulas and validation.
    LWUP = "lwup"
    SWUP = "swup"
    LWNET = "lwnet"
    SWNET = "swnet"
    RNET = "rnet"
    SKY_EMISSIVITY = "sky_emissivity"
    CLEARNESS_INDEX = "clearness_index"

    # Eolic potential is height-dependent; the suffix is handled at runtime.
    WIND_POTENTIAL = "poteolico"
    WIND_POWER_DENSITY_10M = "wind_power_density_10m"


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
    WRFVariable.SKIN_TEMPERATURE: "hot_r",
    WRFVariable.RELATIVE_HUMIDITY: "YlGnBu",
    WRFVariable.PRESSURE: "Blues",
    WRFVariable.RAIN: "afmhot_r",
    WRFVariable.HFX: "jet",
    WRFVariable.LH: "jet",
    WRFVariable.SWDOWN: "hot_r",
    WRFVariable.GLW: "magma",
    WRFVariable.WEIBULL: "jet",
    WRFVariable.WIND_POTENTIAL: "Blues",
    WRFVariable.WIND_POWER_DENSITY_10M: "YlOrRd",
    # Upwelling/net fluxes share the family of their waveband — magma for
    # longwave (matching GLW), hot_r for shortwave (matching SWDOWN). The two
    # signed net fields get diverging maps so radiative gain and loss read as
    # opposite directions rather than as two ends of one ramp. Note the neutral
    # tone does NOT land on zero: the shared renderer normalizes linearly
    # between the percentile scale bounds, so the midpoint is wherever
    # (vmin + vmax) / 2 falls.
    WRFVariable.LWUP: "magma",
    WRFVariable.SWUP: "hot_r",
    WRFVariable.LWNET: "RdBu_r",
    WRFVariable.SWNET: "hot_r",
    WRFVariable.RNET: "RdBu_r",
    WRFVariable.SKY_EMISSIVITY: "BuPu",
    WRFVariable.CLEARNESS_INDEX: "cividis",
}

# Map from our enum to the NetCDF variable / output file suffix
VARIABLE_NETCDF_MAP: dict[WRFVariable | str, str] = {
    WRFVariable.TEMPERATURE: "TEMP",
    WRFVariable.PRESSURE: "PRES",
    WRFVariable.WIND: "WIND",
    WRFVariable.RAIN: "RAIN",
    WRFVariable.VAPOR: "VAPOR",
    WRFVariable.SKIN_TEMPERATURE: "TSK",
    WRFVariable.RELATIVE_HUMIDITY: "RH2",
    WRFVariable.HFX: "HFX",
    WRFVariable.LH: "LH",
    WRFVariable.SWDOWN: "SWDOWN",
    WRFVariable.GLW: "GLW",
    WRFVariable.WEIBULL: "K_WEIB",
    WRFVariable.WIND_POWER_DENSITY_10M: "WIND_POWER_DENSITY_10M",
    WRFVariable.LWUP: "LWUP",
    WRFVariable.SWUP: "SWUP",
    WRFVariable.LWNET: "LWNET",
    WRFVariable.SWNET: "SWNET",
    WRFVariable.RNET: "RNET",
    WRFVariable.SKY_EMISSIVITY: "EPS_SKY",
    WRFVariable.CLEARNESS_INDEX: "KT",
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
