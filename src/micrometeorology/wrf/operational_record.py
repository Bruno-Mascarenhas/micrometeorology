"""What ``*_series_operacional.dat`` IS: its schema, its rows, and its repair.

The operational record is not the product of a single run. The server simulates
one day at a time and each run appends 24 more rows, so the record spans years
while every block comes from a different execution of the model. Three
properties follow, and this module holds all three:

**The file's own header is the schema.** Nothing here writes a column by position
from a hardcoded list. :func:`read_header` reads the header the file carries and
:func:`format_row` fills exactly those fields, in that order, writing ``nan``
wherever the run could not produce a value. A wrfout that stops carrying a
variable therefore leaves its column present and empty instead of shifting every
column after it, and a column this code has never heard of survives untouched.

**The schema grows, it never shifts.** A new quantity is appended to the header
by :func:`extend_header`; the rows already in the file stay shorter than the
header and pandas reads their missing tail as ``NaN``, which is what they mean.

**Timestamps are naive station-local (UTC-03).** Hour 21 local is 00 UTC, each
run's initialisation step. The conversion happens once, in
:mod:`micrometeorology.wrf.operational_series`, at the boundary where a wrfout
becomes a block; nothing here re-derives it.

Turning a wrfout INTO a block is
:mod:`micrometeorology.wrf.operational_series`, which is where the netCDF
dependency lives. This module needs only numpy and pandas, so the CLIs that
merely read the record -- ``labmim-climatology``, ``labmim-station-graphs`` --
do not pay for a NetCDF stack they never touch.

The v1 record and what was wrong with it
----------------------------------------
The extraction that wrote the 26,087 rows between 2022-06-15 and 2026-03-18 was
never committed to this repository; its formulas were recovered from the rows
themselves and reproduce them to their fourth decimal. Recovering them surfaced
four defects, all of which :func:`migrate_to_v2` repairs in place and none of
which this module reproduces:

1. **A Kelvin-to-Celsius subtraction applied to two dimensionless quantities.**
   ``ALBD`` was ``ALBEDO - 273.15`` and ``EMISS`` was ``emissivity - 273.15``,
   and both propagated: ``Swup_calc`` reached -294,069 W/m2 and ``Lwup_calc``
   was ``emissivity_broken * sigma * T_celsius^4`` -- a broken emissivity AND
   Celsius where the Stefan-Boltzmann law needs Kelvin.
2. **The specific-humidity conversion applied to a mixing ratio.** WRF's ``Q2``
   is a mixing ratio (``QV at 2 M``), so vapour pressure is
   ``q*p / (epsilon + q)``; v1 used ``q*p / (epsilon + (1-epsilon)*q)``, which
   is the conversion for specific humidity. That made ``e`` 1.57% too high and
   ``ur`` 1.23 percentage points too high on average, and is the sole cause of
   the 322 rows -- 314 distinct hours -- above 100% relative humidity that the
   monitoring page framed its axis to show.
3. **The cold-start step published as measurement.** WRF's first output step
   precedes its first radiation and surface call, so every flux was written
   identically zero -- and zero is a physically valid irradiance, so nothing
   downstream could tell the two apart. ``GLW == 0`` marks the step exactly:
   1087 rows, every one of them hour 21, and no other row in the record.
4. **Two columns written at the end of the row.** Until 2022-10-07 the row was
   47 fields wide with ``Swup_calc`` and ``Lwup_calc`` in unnamed trailing
   fields and ``nan`` in their named columns.

Column semantics
----------------
``swdown_farms_w_m2``, ``swddif_farms_w_m2`` and ``swddir_farms_w_m2`` have no
source in the current model configuration: ``SWDDIR + SWDDIF`` equals ``SWDOWN``
exactly in every wrfout of this era, so there is one direct/diffuse pair and it
already feeds ``swddir_w_m2``/``swddif_w_m2``. They keep their 2022-2026 values,
minus the cold-start step, and are written ``nan`` from here on.

``sst_c`` is WRF's ``SST`` at the serving cell. Over a LAND cell -- which the
tower's is -- WRF fills that field with the skin temperature at initialisation
and never updates it, so at this site the column is a per-run constant that is
not a sea-surface temperature. Pointed at a water cell it is one.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from labmim_core.atomic import atomic_write
from labmim_core.site import STATION_SITE
from micrometeorology.common.physics import (
    KELVIN_AT_ZERO_CELSIUS,
    PASCAL_PER_HECTOPASCAL,
    STEFAN_BOLTZMANN,
    saturation_vapor_pressure,
    vapor_pressure,
)
from micrometeorology.sensors.wind import wind_direction_from_components
from micrometeorology.wrf.columns import (
    ALBEDO,
    E_HPA,
    EMISSIVITY,
    ES_PA,
    GLW_W_M2,
    GRDFLX_W_M2,
    HFX_W_M2,
    LH_W_M2,
    LWDNB_W_M2,
    LWUP_AIR_W_M2,
    LWUP_W_M2,
    PBLH_M,
    PRECIP_MM,
    PSFC_HPA,
    Q2_G_KG,
    RH_PCT,
    SST_C,
    SWDDIF_FARMS_W_M2,
    SWDDIF_W_M2,
    SWDDIR_FARMS_W_M2,
    SWDDIR_W_M2,
    SWDNB_W_M2,
    SWDOWN_FARMS_W_M2,
    SWDOWN_W_M2,
    SWUP_W_M2,
    SWUPB_W_M2,
    T2_C,
    U10_M_S,
    USTAR_M_S,
    V10_M_S,
    WIND_DIR_DEG,
    WIND_SPEED_M_S,
)
from micrometeorology.wrf.variables import (
    compute_upwelling_longwave,
    compute_upwelling_shortwave,
    rotate_components,
)

logger = logging.getLogger(__name__)

#: What the shared physical conversions accept. They are elementwise, and the
#: v1 repair in :func:`migrate_to_v2` applies them one cell at a time while the
#: extraction applies them to a whole window, so both spellings must type.
type Elementwise = NDArray | float

__all__ = [
    "DEFAULT_HEADER",
    "DEFAULT_STATION",
    "OPERATIONAL_CATALOG",
    "STATION_FILE_SUFFIX",
    "TIME_COLUMNS",
    "V1_COLUMNS",
    "V1_TO_V2",
    "V1_UNREPAIRED_COLUMNS",
    "MigrationReport",
    "PointSample",
    "SeriesColumn",
    "Station",
    "append_block",
    "build_columns",
    "extend_header",
    "format_row",
    "legacy_spellings",
    "migrate_to_v2",
    "parse_station",
    "read_header",
    "read_stations",
    "rename_v1_columns",
    "render_rows",
]


#: Leading integer fields; ``pd.to_datetime`` reads them by these exact names,
#: which is why they are the one part of the schema the v2 rename left alone.
TIME_COLUMNS: tuple[str, ...] = ("year", "month", "day", "hour")

_GRAMS_PER_KILOGRAM = 1000.0


@dataclass(frozen=True, slots=True)
class PointSample:
    """Every wrfout series one block needs, already reduced to the target cell.

    Attributes
    ----------
    steps:
        Number of hourly steps in the window, ``H``.
    fields:
        ``(H,)`` float64 series per wrfout variable name, in the file's own
        units, for the window itself. A variable the file does not carry is
        absent from the mapping rather than present as NaN, so a column can tell
        "the model did not write this" from "the model wrote no-value".
    increments:
        ``(H,)`` float64 per-step differences for the variables a column
        declared as accumulated. The first entry is NaN when the window starts
        at the run's own first step, which has no predecessor to difference
        against -- the same rule
        :func:`~micrometeorology.wrf.variables.extract_rain_step` states.
    """

    steps: int
    fields: Mapping[str, NDArray]
    increments: Mapping[str, NDArray] = field(default_factory=dict)

    def missing(self, sources: Sequence[str], increments: Sequence[str] = ()) -> tuple[str, ...]:
        """Which of the requested wrfout variables this file did not carry."""
        absent = [name for name in sources if name not in self.fields]
        absent += [name for name in increments if name not in self.increments]
        return tuple(absent)

    def blank(self) -> NDArray:
        """An all-NaN ``(H,)`` series, for a column with no source this run."""
        return np.full(self.steps, np.nan, dtype=np.float64)

    def uninitialised(self) -> NDArray:
        """``(H,)`` bool: steps that precede WRF's first radiation call.

        ``GLW`` is identically zero at such a step and never zero afterwards --
        downwelling longwave is never zero over land -- which makes it an
        unambiguous marker rather than a threshold. This is the same test
        :func:`~micrometeorology.wrf.variables.blank_uninitialised_radiation`
        applies to the gridded products, and it separates the record's 1087
        cold-start rows from every other row exactly. A continuation run keeps
        its radiation state and is untouched.
        """
        glw = self.fields.get("GLW")
        if glw is None:
            return np.zeros(self.steps, dtype=bool)
        return np.asarray(glw == 0.0)


@dataclass(frozen=True, slots=True)
class SeriesColumn:
    """One column of the operational file and how a run produces it.

    Adding a quantity to the extraction is adding one entry to
    :data:`OPERATIONAL_CATALOG`; nothing else in this module enumerates columns.

    Attributes
    ----------
    name:
        Header field, written verbatim; it carries the unit as a suffix.
    sources:
        wrfout variable names read over the window. When any is absent the
        column is written ``nan`` and *compute* is never called.
    compute:
        Maps the sample to the ``(H,)`` published series, in the unit *name*
        declares.
    increments:
        wrfout variable names whose per-step DIFFERENCE the column consumes,
        because the file writes them accumulated since the run start.
    optional_sources:
        Read when the file carries them, but never a reason to blank the column.
        These are the RRTMG bottom-of-atmosphere diagnostics a run may or may
        not have been configured to write, which *compute* prefers over its own
        reconstruction when they are there.
    cold_start_blank:
        Whether the column is a physics output, and therefore has no value at a
        step preceding WRF's first radiation and surface call. Dynamics -- the
        state the run was initialised FROM -- is valid at that step and is not
        blanked.
    """

    name: str
    sources: tuple[str, ...]
    compute: Callable[[PointSample], NDArray]
    increments: tuple[str, ...] = ()
    optional_sources: tuple[str, ...] = ()
    cold_start_blank: bool = False

    def evaluate(self, sample: PointSample) -> NDArray:
        """The column's ``(H,)`` series, all-NaN when a source is missing."""
        absent = sample.missing(self.sources, self.increments)
        if absent:
            logger.info("%s: no value this run, wrfout carries no %s", self.name, ", ".join(absent))
            return sample.blank()
        values = np.asarray(self.compute(sample), dtype=np.float64)
        if not self.cold_start_blank:
            return values
        blanked: NDArray = np.where(sample.uninitialised(), np.nan, values)
        return blanked


def _temperature_celsius(sample: PointSample) -> NDArray:
    celsius: NDArray = sample.fields["T2"] - KELVIN_AT_ZERO_CELSIUS
    return celsius


def _pressure_hpa(sample: PointSample) -> NDArray:
    hectopascal: NDArray = sample.fields["PSFC"] / PASCAL_PER_HECTOPASCAL
    return hectopascal


def relative_humidity_percent(vapor_hpa: Elementwise, saturation_pa: Elementwise) -> Elementwise:
    """Relative humidity, percent, NOT clipped at saturation.

    Deliberately not
    :func:`~micrometeorology.wrf.variables.compute_relative_humidity`, which
    clips to 0-100% for a colour scale: this file is read as model state, and a
    model that reports supersaturated air should say so rather than have it
    rounded away at the boundary.
    """
    humidity: Elementwise = 100.0 * (vapor_hpa * PASCAL_PER_HECTOPASCAL) / saturation_pa
    return humidity


def _vapor_pressure_hpa(sample: PointSample) -> NDArray:
    return np.asarray(vapor_pressure(sample.fields["Q2"], _pressure_hpa(sample)))


def _saturation_vapor_pressure_pa(sample: PointSample) -> NDArray:
    return np.asarray(saturation_vapor_pressure(_temperature_celsius(sample)))


def _relative_humidity_percent(sample: PointSample) -> NDArray:
    return np.asarray(
        relative_humidity_percent(
            _vapor_pressure_hpa(sample), _saturation_vapor_pressure_pa(sample)
        )
    )


def _mixing_ratio_g_kg(sample: PointSample) -> NDArray:
    grams: NDArray = sample.fields["Q2"] * _GRAMS_PER_KILOGRAM
    return grams


def _earth_relative_wind(sample: PointSample) -> tuple[NDArray, NDArray]:
    return rotate_components(
        sample.fields["U10"],
        sample.fields["V10"],
        sample.fields["COSALPHA"],
        sample.fields["SINALPHA"],
    )


def _wind_east(sample: PointSample) -> NDArray:
    return _earth_relative_wind(sample)[0]


def _wind_north(sample: PointSample) -> NDArray:
    return _earth_relative_wind(sample)[1]


def _wind_speed(sample: PointSample) -> NDArray:
    east, north = _earth_relative_wind(sample)
    speed: NDArray = np.hypot(east, north)
    return speed


def _wind_direction_deg(sample: PointSample) -> NDArray:
    """Bearing the wind blows FROM, degrees clockwise from true north.

    So 270 is a westerly and 90 an easterly, and the value is 180 from the
    direction the air actually moves toward. The arithmetic is
    :func:`~micrometeorology.sensors.wind.wind_direction_from_components`,
    which the station's own hourly means already use, so the model layer and
    the observed layer of the direction chart cannot disagree by a convention.
    """
    return np.asarray(wind_direction_from_components(*_earth_relative_wind(sample)))


def _downwelling_longwave_bottom(sample: PointSample) -> NDArray:
    if "LWDNB" in sample.fields:
        return sample.fields["LWDNB"]
    return sample.fields["GLW"]


def _upwelling_longwave(sample: PointSample) -> NDArray:
    if "LWUPB" in sample.fields:
        return sample.fields["LWUPB"]
    return compute_upwelling_longwave(
        sample.fields["EMISS"], sample.fields["TSK"], sample.fields["GLW"]
    )


def upwelling_longwave_from_air(emissivity: Elementwise, temperature_k: Elementwise) -> Elementwise:
    """Graybody emission at SCREEN-LEVEL air temperature, W/m2.

    ``eps * sigma * T2^4``, kept beside the surface's own upwelling flux so the
    two can be differenced. Shared with :func:`migrate_to_v2`, which recomputes
    the v1 rows that evaluated it in Celsius and with a broken emissivity.
    """
    emission: Elementwise = emissivity * STEFAN_BOLTZMANN * temperature_k**4
    return emission


def _upwelling_longwave_from_air(sample: PointSample) -> NDArray:
    return np.asarray(upwelling_longwave_from_air(sample.fields["EMISS"], sample.fields["T2"]))


def _upwelling_shortwave_from_albedo(sample: PointSample) -> NDArray:
    return compute_upwelling_shortwave(sample.fields["ALBEDO"], sample.fields["SWDOWN"])


def _sea_surface_temperature_celsius(sample: PointSample) -> NDArray:
    celsius: NDArray = sample.fields["SST"] - KELVIN_AT_ZERO_CELSIUS
    return celsius


def _precipitation_mm(sample: PointSample) -> NDArray:
    total: NDArray = sample.increments["RAINC"] + sample.increments["RAINNC"]
    return total


def _no_source(sample: PointSample) -> NDArray:
    return sample.blank()


def _passthrough(name: str) -> Callable[[PointSample], NDArray]:
    def read(sample: PointSample) -> NDArray:
        return sample.fields[name]

    return read


def _raw(name: str) -> SeriesColumn:
    # Blanked at the cold start: nothing here knows whether an out-of-catalogue
    # variable is dynamics, which is valid at that step, or a physics output,
    # which is not. A blanked hour is recoverable from the run; a zero flux
    # published as a measurement is not.
    return SeriesColumn(
        name=name, sources=(name,), compute=_passthrough(name), cold_start_blank=True
    )


_WIND_SOURCES = ("U10", "V10", "COSALPHA", "SINALPHA")

#: Every column this extraction knows how to produce, in the order a new file
#: receives them. The first four fields of a row are :data:`TIME_COLUMNS` and are
#: not listed here: they are the stamp, not a measurement.
OPERATIONAL_CATALOG: tuple[SeriesColumn, ...] = (
    SeriesColumn(T2_C, ("T2",), _temperature_celsius),
    SeriesColumn(RH_PCT, ("T2", "PSFC", "Q2"), _relative_humidity_percent),
    SeriesColumn(PSFC_HPA, ("PSFC",), _pressure_hpa),
    SeriesColumn(E_HPA, ("PSFC", "Q2"), _vapor_pressure_hpa),
    SeriesColumn(ES_PA, ("T2",), _saturation_vapor_pressure_pa),
    SeriesColumn(Q2_G_KG, ("Q2",), _mixing_ratio_g_kg),
    SeriesColumn(WIND_SPEED_M_S, _WIND_SOURCES, _wind_speed),
    SeriesColumn(WIND_DIR_DEG, _WIND_SOURCES, _wind_direction_deg),
    SeriesColumn(U10_M_S, _WIND_SOURCES, _wind_east),
    SeriesColumn(V10_M_S, _WIND_SOURCES, _wind_north),
    SeriesColumn(SWDOWN_W_M2, ("SWDOWN",), _passthrough("SWDOWN"), cold_start_blank=True),
    SeriesColumn(SWDNB_W_M2, ("SWDNB",), _passthrough("SWDNB"), cold_start_blank=True),
    SeriesColumn(SWDOWN_FARMS_W_M2, (), _no_source, cold_start_blank=True),
    SeriesColumn(SWUPB_W_M2, ("SWUPB",), _passthrough("SWUPB"), cold_start_blank=True),
    SeriesColumn(
        SWUP_W_M2,
        ("ALBEDO", "SWDOWN"),
        _upwelling_shortwave_from_albedo,
        cold_start_blank=True,
    ),
    SeriesColumn(SWDDIF_W_M2, ("SWDDIF",), _passthrough("SWDDIF"), cold_start_blank=True),
    SeriesColumn(SWDDIF_FARMS_W_M2, (), _no_source, cold_start_blank=True),
    SeriesColumn(SWDDIR_W_M2, ("SWDDIR",), _passthrough("SWDDIR"), cold_start_blank=True),
    SeriesColumn(SWDDIR_FARMS_W_M2, (), _no_source, cold_start_blank=True),
    SeriesColumn(GLW_W_M2, ("GLW",), _passthrough("GLW"), cold_start_blank=True),
    SeriesColumn(
        LWDNB_W_M2,
        ("GLW",),
        _downwelling_longwave_bottom,
        optional_sources=("LWDNB",),
        cold_start_blank=True,
    ),
    SeriesColumn(
        LWUP_W_M2,
        ("EMISS", "TSK", "GLW"),
        _upwelling_longwave,
        optional_sources=("LWUPB",),
        cold_start_blank=True,
    ),
    SeriesColumn(
        LWUP_AIR_W_M2,
        ("EMISS", "T2"),
        _upwelling_longwave_from_air,
        cold_start_blank=True,
    ),
    SeriesColumn(ALBEDO, ("ALBEDO",), _passthrough("ALBEDO"), cold_start_blank=True),
    SeriesColumn(EMISSIVITY, ("EMISS",), _passthrough("EMISS"), cold_start_blank=True),
    SeriesColumn(HFX_W_M2, ("HFX",), _passthrough("HFX"), cold_start_blank=True),
    SeriesColumn(LH_W_M2, ("LH",), _passthrough("LH"), cold_start_blank=True),
    SeriesColumn(GRDFLX_W_M2, ("GRDFLX",), _passthrough("GRDFLX"), cold_start_blank=True),
    SeriesColumn(USTAR_M_S, ("UST",), _passthrough("UST"), cold_start_blank=True),
    SeriesColumn(PBLH_M, ("PBLH",), _passthrough("PBLH"), cold_start_blank=True),
    SeriesColumn(SST_C, ("SST",), _sea_surface_temperature_celsius),
    SeriesColumn(PRECIP_MM, (), _precipitation_mm, increments=("RAINC", "RAINNC")),
)

#: Header a file created by this module receives.
DEFAULT_HEADER: tuple[str, ...] = TIME_COLUMNS + tuple(
    column.name for column in OPERATIONAL_CATALOG
)

_CATALOG_BY_NAME = {column.name: column for column in OPERATIONAL_CATALOG}

#: v1 column name -> v2 column name, in the v1 file's own field order. The four
#: time fields keep their names: ``export_climatology.read_wrf_series`` builds
#: its index with ``pd.to_datetime(frame[["year", "month", "day", "hour"]])``,
#: which resolves them by name.
V1_TO_V2: tuple[tuple[str, str], ...] = (
    ("year", "year"),
    ("month", "month"),
    ("day", "day"),
    ("hour", "hour"),
    ("T", T2_C),
    ("ur", RH_PCT),
    ("pressure", PSFC_HPA),
    ("e", E_HPA),
    ("es", ES_PA),
    ("q", Q2_G_KG),
    ("WS", WIND_SPEED_M_S),
    ("WD", WIND_DIR_DEG),
    ("u", U10_M_S),
    ("v", V10_M_S),
    ("Swdw", SWDOWN_W_M2),
    ("Swdw_b", SWDNB_W_M2),
    ("Swdw_farms", SWDOWN_FARMS_W_M2),
    ("Swup_b", SWUPB_W_M2),
    ("Swup_calc", SWUP_W_M2),
    ("Swdf", SWDDIF_W_M2),
    ("Swdf_farms", SWDDIF_FARMS_W_M2),
    ("Swdr", SWDDIR_W_M2),
    ("Swdr_farms", SWDDIR_FARMS_W_M2),
    ("Lwdw_glw", GLW_W_M2),
    ("Lwdw_b", LWDNB_W_M2),
    ("Lwup_b", LWUP_W_M2),
    ("Lwup_calc", LWUP_AIR_W_M2),
    ("ALBD", ALBEDO),
    ("EMISS", EMISSIVITY),
    ("H", HFX_W_M2),
    ("LE", LH_W_M2),
    ("G", GRDFLX_W_M2),
    ("ustar", USTAR_M_S),
    ("PBLH", PBLH_M),
    ("TSM", SST_C),
)

#: The v1 header, in its own order.
V1_COLUMNS: tuple[str, ...] = tuple(old for old, _ in V1_TO_V2)

_V1_NAME_OF = dict(V1_TO_V2)


def build_columns(
    requested: Sequence[str] | None,
    available: Callable[[str], bool],
) -> tuple[SeriesColumn, ...]:
    """Resolve requested names to columns, admitting new wrfout variables.

    Parameters
    ----------
    requested:
        Column names, or ``None`` for the whole catalogue. A name the catalogue
        does not define is admitted as a raw passthrough when the wrfout carries
        a variable of that name, which is how a quantity the model starts
        writing becomes a column with no change here.
    available:
        Predicate telling whether the wrfout carries a variable of that name,
        normally :meth:`~micrometeorology.wrf.reader.WRFDataset.has_variable`.

    Returns
    -------
    tuple of SeriesColumn
        In the order requested; catalogue order when *requested* is ``None``.

    Raises
    ------
    KeyError
        For a name that is neither a catalogue column nor a variable of this
        wrfout. Silently dropping it would publish a short row under a header
        the caller believes it filled.
    """
    if requested is None:
        return OPERATIONAL_CATALOG

    resolved: list[SeriesColumn] = []
    unknown: list[str] = []
    for name in requested:
        if name in _CATALOG_BY_NAME:
            resolved.append(_CATALOG_BY_NAME[name])
        elif name in _V1_NAME_OF and _V1_NAME_OF[name] != name:
            raise KeyError(
                f"{name} is the v1 spelling of {_V1_NAME_OF[name]}; extracting it "
                "under that name would collide once a reader renames the record"
            )
        elif available(name):
            logger.info("%s: not in the catalogue, extracted as a raw wrfout variable", name)
            resolved.append(_raw(name))
        else:
            unknown.append(name)
    if unknown:
        raise KeyError(
            f"unknown column(s) {', '.join(unknown)}: not in the catalogue "
            f"({', '.join(sorted(_CATALOG_BY_NAME))}) and not a variable of this wrfout"
        )
    return tuple(resolved)


def format_row(stamp: pd.Timestamp, values: Mapping[str, float], header: Sequence[str]) -> str:
    """Render one row against *header*, in that order, ``nan`` where absent.

    The time fields are written as integers and everything else to four
    decimals, matching the block the v1 extraction wrote from 2022-10-07 on --
    including its ``-0.0000`` for a negative zero, which is what the flux fields
    carry at night.
    """
    stamps = {
        "year": stamp.year,
        "month": stamp.month,
        "day": stamp.day,
        "hour": stamp.hour,
    }
    fields: list[str] = []
    for name in header:
        if name in stamps:
            fields.append(str(stamps[name]))
        else:
            fields.append(f"{float(values.get(name, np.nan)):.4f}")
    return ",".join(fields)


def read_header(path: Path) -> tuple[str, ...]:
    """Column names the file declares, trailing empty fields dropped.

    The v1 header names 35 columns and then carries 12 empty fields, the remnant
    of a layout in which ``Swup_calc`` and ``Lwup_calc`` were written at the end
    of the row. Those blanks are not columns and are dropped here, which is also
    how every reader of this file already treats them.

    Raises
    ------
    ValueError
        When the file is empty, or when a name repeats -- a duplicate makes the
        header ambiguous as a positional schema.
    """
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    if not first.strip():
        raise ValueError(f"{path} has no header line")

    fields = [name.strip() for name in first.rstrip("\n").split(",")]
    while fields and not fields[-1]:
        fields.pop()
    if "" in fields:
        raise ValueError(f"{path}: header has an empty field between named columns: {fields}")
    duplicates = duplicated(fields)
    if duplicates:
        raise ValueError(f"{path}: header names {', '.join(duplicates)} more than once")
    return tuple(fields)


def extend_header(path: Path, extra: Sequence[str]) -> tuple[str, ...]:
    """Append *extra* names to the file's header and return the new header.

    Only the header line changes: the rows already written stay shorter than it,
    and a reader fills their missing tail with no-value, which is what they mean.
    Nothing is reordered, so every existing column keeps its position.
    """
    header = read_header(path)
    new_header = header + tuple(extra)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = ",".join(new_header) + "\n"
    atomic_write(path, lambda scratch: scratch.write_text("".join(lines), encoding="utf-8"))
    logger.info("%s: header extended with %s", path.name, ", ".join(extra))
    return new_header


def render_rows(frame: pd.DataFrame, header: Sequence[str]) -> list[str]:
    """Render *frame* as the exact lines an append would write.

    Shared with the CLI's dry run, so what the operator is shown is the bytes
    the file would receive rather than a second rendering of them.
    """
    return [
        format_row(stamp, dict(zip(frame.columns, values, strict=True)), header)
        for stamp, values in zip(pd.DatetimeIndex(frame.index), frame.to_numpy(), strict=True)
    ]


def _ends_with_newline(path: Path, size: int) -> bool:
    with path.open("rb") as handle:
        handle.seek(size - 1)
        return handle.read(1) == b"\n"


def _repair_tail(path: Path) -> None:
    """Finish a complete last row lacking its newline; cut a partial one so its
    hour is appended again in full."""
    size = path.stat().st_size
    if not size or _ends_with_newline(path, size):
        return
    data = path.read_bytes()
    cut = data.rfind(b"\n") + 1
    width = data.split(b"\n", 1)[0].count(b",") + 1
    if cut and data[cut:].count(b",") + 1 < width:
        with path.open("r+b") as handle:
            handle.truncate(cut)
        logger.warning("%s: dropped a partial last row left by an interrupted append", path.name)
        return
    with path.open("ab") as handle:
        handle.write(b"\n")


def _existing_stamps(path: Path) -> set[tuple[int, int, int, int]]:
    """The (year, month, day, hour) of every row already in the file."""
    stamps: set[tuple[int, int, int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            fields = line.split(",", 4)
            if len(fields) < 4:
                continue
            try:
                stamps.add((int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])))
            except ValueError:
                continue
    return stamps


def append_block(
    path: Path,
    frame: pd.DataFrame,
    header: Sequence[str],
    force: bool = False,
) -> int:
    """Append *frame* to *path* under *header*, creating the file if needed.

    Parameters
    ----------
    path:
        The operational file. Created with *header* when absent.
    frame:
        Indexed by naive station-local hour, one column per named field.
    header:
        The positional schema every row is rendered against, normally the one
        the file already declares.
    force:
        Append even when every hour of the block is already in the file. The
        default skips instead, so a cron line that fires twice for the same run
        does not double the block; the record does tolerate duplicates, and its
        readers keep the last row for an hour, but a re-run should be a decision.

    Returns
    -------
    int
        Rows written; ``0`` when the block was skipped as already present.
    """
    legacy = legacy_spellings(header)
    if legacy:
        # Extending a v1 header would weld both schemas into one file: the rows
        # already written would keep filling the v1 half while every new row
        # filled the v2 half, `rename_v1_columns` would then produce duplicate
        # labels, and `migrate_to_v2` would refuse the result as neither schema.
        raise ValueError(
            f"{path.name} is still on the v1 schema ({', '.join(legacy)}); "
            "migrate it before appending"
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(header) + "\n", encoding="utf-8")
        logger.info("%s: created with %d columns", path.name, len(header))
    else:
        _repair_tail(path)
        if not force:
            stamps = _existing_stamps(path)
            block = {(t.year, t.month, t.day, t.hour) for t in frame.index}
            if block and block <= stamps:
                logger.warning(
                    "%s: all %d hours of this block are already in the file; "
                    "skipping (use --force to append anyway)",
                    path.name,
                    len(block),
                )
                return 0

    rows = render_rows(frame, header)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    logger.info("%s: appended %d rows", path.name, len(rows))
    return len(rows)


#: Position of ``Swup_calc`` and ``Lwup_calc`` in the 47-field rows the v1
#: extraction wrote before 2022-10-03: the two values were carried at the END of
#: the row, in the fields the header leaves unnamed, while their named columns
#: held ``nan``. Verified against all 2424 such rows; fields 35..44 are empty in
#: every one of them.
LEGACY_TRAILING_FIELDS: tuple[tuple[int, str], ...] = ((45, "Swup_calc"), (46, "Lwup_calc"))

#: Field count of the rows that layout produced.
LEGACY_ROW_WIDTH = 47

#: Surface emissivity the v1 extraction used, recovered from the record itself:
#: its ``EMISS`` column is the constant -272.27 over all 23,662 rows that carry
#: one, and dividing ``Lwup_calc`` by ``sigma * T_celsius^4`` on the 2022 rows,
#: which carry no ``EMISS``, returns the same value to +/-0.009 -- the rounding
#: of a column written to four decimals with magnitudes as small as 1.57. Only
#: those 2022 rows need it; every later row is repaired from its own column.
V1_SURFACE_EMISSIVITY = 0.88

#: How far a row's implied emissivity may sit from the one it is repaired with
#: before the migration refuses the row. Wide enough for the four-decimal
#: rounding above, far too narrow to admit a different surface.
V1_EMISSIVITY_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """What :func:`migrate_to_v2` changed, per column, for the operator to read.

    Attributes
    ----------
    rows:
        Data rows rewritten.
    recovered:
        Cells moved out of the v1 trailing fields into their named column.
    repaired:
        Cells recomputed, per v2 column name.
    blanked:
        Cells set to no-value because their step precedes WRF's first radiation
        call, per v2 column name.
    """

    rows: int
    recovered: int
    repaired: Mapping[str, int]
    blanked: Mapping[str, int]


def duplicated(names: Iterable[str]) -> list[str]:
    """The names that appear more than once, sorted."""
    return sorted(name for name, count in Counter(names).items() if count > 1)


def _v1_number(row: Mapping[str, str], name: str) -> float:
    raw = row.get(name, "").strip()
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _repair_v1_row(row: dict[str, str], line_number: int, repaired: Counter[str]) -> None:
    """Recompute in place the v1 cells whose formula was wrong.

    Every repair is exactly invertible from the row's own fields, which is why
    the record can be corrected rather than merely annotated:

    - ``ALBD`` and ``EMISS`` had 273.15 subtracted from a dimensionless value;
    - ``Swup_calc`` was ``(albedo - 273.15) * Swdw``, so adding ``273.15 * Swdw``
      restores ``albedo * Swdw`` without needing the albedo at all -- which
      matters, because the 2022 rows carry no albedo;
    - ``Lwup_calc`` was ``(eps - 273.15) * sigma * T_celsius^4``; the emissivity
      comes from the repaired ``EMISS`` where the row has one and from
      :data:`V1_SURFACE_EMISSIVITY` otherwise, and the temperature becomes Kelvin;
    - ``e`` used the specific-humidity conversion on a mixing ratio, and ``ur``
      follows from it.

    Raises
    ------
    ValueError
        When a row's own emissivity cannot be reconciled with the one it would
        be repaired with. Repairing from a value that does not check out would
        put a fabricated flux in the record.
    """
    for name in ("ALBD", "EMISS"):
        value = _v1_number(row, name)
        if np.isfinite(value):
            row[name] = f"{value + KELVIN_AT_ZERO_CELSIUS:.4f}"
            repaired[name] += 1

    shortwave = _v1_number(row, "Swdw")
    reflected = _v1_number(row, "Swup_calc")
    if np.isfinite(reflected) and np.isfinite(shortwave):
        row["Swup_calc"] = f"{reflected + KELVIN_AT_ZERO_CELSIUS * shortwave:.4f}"
        repaired["Swup_calc"] += 1

    temperature_c = _v1_number(row, "T")
    emission = _v1_number(row, "Lwup_calc")
    if np.isfinite(emission) and np.isfinite(temperature_c):
        emissivity = _v1_number(row, "EMISS")
        if not np.isfinite(emissivity):
            emissivity = V1_SURFACE_EMISSIVITY
        implied = emission / (STEFAN_BOLTZMANN * temperature_c**4) + KELVIN_AT_ZERO_CELSIUS
        if abs(implied - emissivity) > V1_EMISSIVITY_TOLERANCE:
            raise ValueError(
                f"line {line_number}: Lwup_calc implies an emissivity of {implied:.4f}, "
                f"but the row would be repaired with {emissivity:.4f}; the v1 formula "
                "does not explain this cell and repairing it would fabricate a flux"
            )
        kelvin = temperature_c + KELVIN_AT_ZERO_CELSIUS
        row["Lwup_calc"] = f"{upwelling_longwave_from_air(emissivity, kelvin):.4f}"
        repaired["Lwup_calc"] += 1

    mixing_ratio = _v1_number(row, "q") / _GRAMS_PER_KILOGRAM
    pressure = _v1_number(row, "pressure")
    if np.isfinite(mixing_ratio) and np.isfinite(pressure):
        vapor = vapor_pressure(mixing_ratio, pressure)
        row["e"] = f"{vapor:.4f}"
        repaired["e"] += 1
        saturation = _v1_number(row, "es")
        if np.isfinite(saturation):
            row["ur"] = f"{relative_humidity_percent(vapor, saturation):.4f}"
            repaired["ur"] += 1


def _blank_v1_cold_start(row: dict[str, str], blanked: Counter[str]) -> None:
    """No-value the physics columns of a step preceding the first radiation call."""
    if _v1_number(row, "Lwdw_glw") != 0.0:
        return
    for old, new in V1_TO_V2:
        column = _CATALOG_BY_NAME.get(new)
        if column is not None and column.cold_start_blank and row.get(old, "").strip():
            row[old] = "nan"
            blanked[new] += 1


def migrate_to_v2(path: Path) -> MigrationReport:
    """Rewrite a v1 operational file onto the v2 schema, keeping a ``.bak``.

    Four things happen to every row, in this order, and the module docstring
    says why each is a defect rather than a preference:

    1. a 47-field row's trailing ``Swup_calc``/``Lwup_calc`` move into their
       named columns -- truncating the row would drop them, and the repairs
       below would then miss them;
    2. :func:`_repair_v1_row` recomputes the cells whose formula was wrong;
    3. :func:`_blank_v1_cold_start` no-values the physics of a step that
       precedes WRF's first radiation call;
    4. the row is emitted under the v2 header, with ``precip_mm`` as no-value
       because the v1 extraction never sampled precipitation.

    Cells nothing touched are passed through as the exact strings the file
    carried, so a diff against the ``.bak`` shows only what the migration
    changed.

    Parameters
    ----------
    path:
        The operational file, rewritten in place after ``<path>.bak`` is written.

    Returns
    -------
    MigrationReport
        Counts per column, for the operator to check before the file is used.

    Raises
    ------
    ValueError
        When the file is already v2 -- so a cron line that fires twice cannot
        apply the repairs twice -- when its header is not v1's, or when a row is
        neither of the two known widths.
    """
    current = read_header(path)
    if current == DEFAULT_HEADER:
        raise ValueError(f"{path.name} is already on the v2 schema; nothing to migrate")
    if current != V1_COLUMNS:
        raise ValueError(
            f"{path.name}: header is neither v1 nor v2. Expected the v1 columns "
            f"{list(V1_COLUMNS)}, got {list(current)}"
        )

    repaired: Counter[str] = Counter()
    blanked: Counter[str] = Counter()
    recovered = 0
    out: list[str] = [",".join(DEFAULT_HEADER) + "\n"]

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()[1:], start=2):
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) == LEGACY_ROW_WIDTH:
            named = fields[: len(V1_COLUMNS)]
            for position, column in LEGACY_TRAILING_FIELDS:
                value = fields[position].strip()
                if value:
                    named[V1_COLUMNS.index(column)] = value
                    recovered += 1
        elif len(fields) == len(V1_COLUMNS):
            named = list(fields)
        else:
            raise ValueError(
                f"{path.name}:{number} has {len(fields)} fields; expected exactly "
                f"{len(V1_COLUMNS)} or {LEGACY_ROW_WIDTH}"
            )

        row = dict(zip(V1_COLUMNS, named, strict=True))
        _repair_v1_row(row, number, repaired)
        _blank_v1_cold_start(row, blanked)
        out.append(",".join([*(row[old] for old in V1_COLUMNS), "nan"]) + "\n")

    path.with_suffix(f"{path.suffix}.bak").write_bytes(path.read_bytes())
    atomic_write(path, lambda scratch: scratch.write_text("".join(out), encoding="utf-8"))

    report = MigrationReport(
        rows=len(out) - 1,
        recovered=recovered,
        repaired={_V1_NAME_OF[old]: count for old, count in repaired.items()},
        blanked=dict(blanked),
    )
    logger.info(
        "%s: migrated %d rows to v2, recovering %d trailing values, "
        "repairing %d cells and blanking %d cold-start cells",
        path.name,
        report.rows,
        report.recovered,
        sum(report.repaired.values()),
        sum(report.blanked.values()),
    )
    return report


#: Suffix every station's file carries, after its own name: the record for the
#: tower is ``labmim_series_operacional.dat``.
STATION_FILE_SUFFIX = "_series_operacional.dat"

#: What a station may be called. The name becomes a file name, so it is
#: restricted rather than sanitised: a station silently renamed to fit the
#: filesystem would append its rows to a file nobody is looking at.
_STATION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class Station:
    """A point the extraction publishes a series for.

    Attributes
    ----------
    name:
        Identifier and file-name stem, ``[A-Za-z0-9][A-Za-z0-9_-]*``.
    latitude:
        Degrees north.
    longitude:
        Degrees east.
    """

    name: str
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not _STATION_NAME.match(self.name):
            raise ValueError(
                f"station name {self.name!r} must match {_STATION_NAME.pattern} — "
                "it becomes a file name"
            )
        if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0:
            raise ValueError(
                f"{self.name}: ({self.latitude}, {self.longitude}) is not a "
                "latitude/longitude pair in degrees"
            )

    @property
    def filename(self) -> str:
        """File this station's rows are appended to."""
        return f"{self.name}{STATION_FILE_SUFFIX}"


#: The tower this repository was built around, and the station every run covers
#: when the caller names none.
DEFAULT_STATION = Station(
    name="labmim",
    latitude=STATION_SITE.latitude,
    longitude=STATION_SITE.longitude,
)


def parse_station(token: str) -> Station:
    """Parse a ``name:lat:lon`` option value.

    Raises
    ------
    ValueError
        When the token has the wrong number of parts, or a part that is not a
        number. A mistyped coordinate would otherwise publish a series for a
        point nobody asked about.
    """
    parts = token.split(":")
    if len(parts) != 3:
        raise ValueError(f"expected name:lat:lon, got {token!r}")
    name, latitude, longitude = parts
    try:
        return Station(name=name.strip(), latitude=float(latitude), longitude=float(longitude))
    except ValueError as error:
        raise ValueError(f"{token!r}: {error}") from None


def read_stations(path: Path) -> tuple[Station, ...]:
    """Read a station list from a CSV of ``name,lat,lon``.

    A header row naming those three columns is optional; anything else in the
    file is a parse error rather than a skipped line, because a station dropped
    without a word is a series that silently stops being published.

    Raises
    ------
    ValueError
        When the file is empty, a row is malformed, or a name repeats — two
        stations of one name would append to the same file.
    """
    stations: list[Station] = []
    seen_content = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        fields = [field.strip() for field in row.split(",")]
        # The header is the first CONTENT line, not physical line 1: a leading
        # comment or blank line would otherwise shift it and turn a header into
        # a station named "name".
        if not seen_content and fields[0].lower() in {"name", "nome", "estacao", "station"}:
            seen_content = True
            continue
        seen_content = True
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected name,lat,lon, got {row!r}")
        try:
            stations.append(
                Station(name=fields[0], latitude=float(fields[1]), longitude=float(fields[2]))
            )
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from None

    if not stations:
        raise ValueError(f"{path} lists no station")
    duplicates = duplicated(station.name for station in stations)
    if duplicates:
        raise ValueError(f"{path}: {', '.join(duplicates)} listed more than once")
    return tuple(stations)


def legacy_spellings(header: Sequence[str]) -> tuple[str, ...]:
    """v1 column names *header* still carries, in v1 order.

    The four time fields are excluded because both schemas share them; only a
    column whose v2 spelling DIFFERS proves the header was never migrated.
    """
    return tuple(old for old, new in V1_TO_V2 if old != new and old in header)


#: The v2 columns whose v1 values are wrong until :func:`migrate_to_v2` runs.
#: These are exactly what :func:`_repair_v1_row` recomputes; naming them is what
#: lets a reader of a v1 file be told which of its numbers not to trust.
V1_UNREPAIRED_COLUMNS: frozenset[str] = frozenset(
    {"albedo", "emissivity", "swup_w_m2", "lwup_air_w_m2", "e_hpa", "rh_pct"}
)


def rename_v1_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Give a frame read from a v1 file its v2 column names.

    Applied by every reader of the operational record, so a consumer names its
    columns once -- in the v2 spelling -- whether the file on disk has been
    through :func:`migrate_to_v2` or not. A frame already on v2, or one carrying
    columns from neither schema, comes back untouched.

    Renaming is ALL this does. The six columns :func:`_repair_v1_row` recomputes
    are still the values the v1 extraction wrote, and those are wrong by a known
    formula: ``albedo`` and ``emissivity`` carry a dimensionless value with
    273.15 subtracted, and ``swup_w_m2``, ``lwup_air_w_m2``, ``e_hpa`` and
    ``rh_pct`` are derived from them. Reading a v1 file therefore serves an
    albedo near -273 and a reflected shortwave in the -10^5 W/m2, which is why
    the caller is warned by name rather than told the file was handled.

    Parameters
    ----------
    frame:
        Any frame read from an operational file; the time fields may already
        have been consumed into the index, which is why nothing here requires
        them.

    Returns
    -------
    pandas.DataFrame
        The same object when no v1 name is present, so the common path copies
        nothing; otherwise a renamed view.
    """
    present = {old: new for old, new in V1_TO_V2 if old != new and old in frame.columns}
    if not present:
        return frame
    renamed = frame.rename(columns=present)
    unrepaired = sorted(V1_UNREPAIRED_COLUMNS.intersection(renamed.columns))
    if unrepaired:
        logger.warning(
            "reading a v1 operational file: %s carry the values the v1 extraction wrote, "
            "which are wrong by a known formula. Run `labmim-wrf-series migrate` to repair "
            "the record before publishing from it.",
            ", ".join(unrepaired),
        )
    else:
        logger.info("v1 column names found, reading them as v2: %s", ", ".join(sorted(present)))
    return renamed


def read_wrf_series(path: str | Path, *, consumes: Collection[str] = ()) -> pd.DataFrame:
    """Read ``series_operacional.dat`` defensively.

    The file is an append-only log of successive operational runs: **not**
    chronologically sorted, with duplicated timestamps and twelve trailing
    anonymous fields that only the oldest rows fill. Reading with the full header
    and sorting afterwards is the only order that does not misalign columns —
    ``names=`` turns the surplus fields into a twelve-level MultiIndex.

    Hour 21 local is 00 UTC, each run's initialisation hour, where surface fluxes
    and boundary-layer height are identically zero; those rows are dropped
    wholesale so one uniform rule can be stated on the page.

    Parameters
    ----------
    path:
        The ``series_operacional.dat`` the operational extraction appends to.
    consumes:
        Column names the caller will actually publish from. An unmigrated v1
        file is REFUSED when they include one of
        :data:`V1_UNREPAIRED_COLUMNS`, whose v1 values are wrong by a known
        formula — ``rh_pct`` is the humidity histogram and the humidity
        overlay, ``swup_w_m2`` sits in the -10^5 W/m2. Left empty the file is
        read as before, warning only: a caller that publishes none of the six
        is unaffected by the schema.

    Returns
    -------
    pandas.DataFrame
        Hourly model variables on a sorted, de-duplicated
        :class:`~pandas.DatetimeIndex` of naive station-local hours (UTC-03),
        with the spin-up hour removed. A repeated timestamp keeps the LAST row,
        which is the most recent run's value for that hour.

    Raises
    ------
    ValueError
        When the file is still on v1 and *consumes* names a column
        :func:`migrate_to_v2` has not repaired.
    """
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")])
    unmigrated = bool(legacy_spellings(list(frame.columns)))
    frame = rename_v1_columns(frame)
    if unmigrated:
        # Present AND consumed: a v1 file that never carried the column cannot
        # serve a wrong value from it, and a caller that publishes none of the
        # six is unaffected by the schema.
        unrepaired = sorted(V1_UNREPAIRED_COLUMNS & set(consumes) & set(frame.columns))
        if unrepaired:
            raise ValueError(
                f"{path} is still a v1 operational file and this run publishes "
                f"{', '.join(unrepaired)}, which carry the values the v1 extraction "
                "wrote — wrong by a known formula. Run `labmim-wrf-series migrate` "
                "on the record first."
            )
    stamps = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    frame.index = pd.DatetimeIndex(stamps)
    frame = frame.drop(columns=["year", "month", "day", "hour"])
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"the hourly database must be indexed by time, got {type(index).__name__}")
    spin_up = index.hour == 21
    logger.info("WRF: %d rows, dropping %d spin-up rows at hour 21", len(frame), int(spin_up.sum()))
    trimmed: pd.DataFrame = frame.loc[~spin_up]
    return trimmed
