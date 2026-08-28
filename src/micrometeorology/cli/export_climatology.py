"""CLI: publish the observed-distribution artifacts the climatology page reads.

Consumes the hourly database built by ``labmim-archive`` plus the WRF point
extraction, and writes ``labmim-climatology-v1`` JSON into the directory the site
declares as ``dataset.paths.climatology``.

The artifacts are **not** committed to the site repository: they derive from the
laboratory's non-public sensor archive, so like the WRF map data they are
gitignored there and attached at deploy time.

The hourly database and the WRF series both enter as naive station-local stamps,
each carrying its instrument's own clock. This module is the manifest boundary:
the ``UTC-03`` it stamps into the published artifacts comes from the pinned
:data:`~micrometeorology.common.site.STATION_UTC_OFFSET_HOURS`, never from the
host's zone.

Examples
--------
Publish straight into a checkout of the site::

    labmim-climatology -i output/archive/station_hourly.parquet \\
        -w data/series/labmim_series_operacional.dat \\
        -o ../site-labmim/site/Climatologia

Restrict to the observed record (no model subsets)::

    labmim-climatology -i output/archive/station_hourly.parquet -o out/ --no-wrf
"""

import functools
import logging
import time
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import typer

# allsky.solar is pure numpy/pandas (no torch) and ships in the same wheel, so
# this CLI reuses NOAA's formulas without pulling a training dependency in.
from allsky.solar import extraterrestrial_ghi
from micrometeorology.common.git import run_git, source_root
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.stats import distributions as dist
from micrometeorology.stats.climatology import seasonal_groups
from micrometeorology.stats.climatology_export import (
    CLIMATOLOGY_VARIABLES,
    MANIFEST_FILENAME,
    RAIN_BUCKET_MM,
    Atom,
    build_manifest,
    build_variable_payload,
    write_json,
)
from micrometeorology.stats.daylight import (
    MIN_SOLAR_ELEVATION_DEG,
    elevation_bounds,
)
from micrometeorology.wrf.operational_record import rename_v1_columns

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

#: Re-exported under this module's shorter local names; the numbers themselves
#: live in :mod:`micrometeorology.common.site`, shared with the sensor archive's
#: quality checks so the two cannot describe different towers.
SITE = STATION_SITE
UTC_OFFSET_HOURS = STATION_UTC_OFFSET_HOURS

STATION = {
    "name": "Estação Micrometeorológica LabMiM",
    "institution": "Instituto de Física — UFBA",
    "latitude": SITE.latitude,
    "longitude": SITE.longitude,
    "elevation_m": 46.0,
    "timezone": "America/Bahia",
}

# The four options the page shows. Every source-by-season pair is computed
# anyway, so the selector can widen later without regenerating anything.
SELECTOR = ("observed_all", "observed_djf", "observed_jja", "wrf_all")

# Unified column in the hourly database for each published variable. The unified
# names are what unify_sensor_columns builds from the sensor_switches era map.
OBSERVED_COLUMN = {
    "air_temperature": "T",
    "relative_humidity": "ur",
    "relative_humidity_wxt": "RH_WXT_Avg",
    "pressure": "pressure",
    "wind_speed": "WS",
    "wind_direction": "WD",
    "precipitation": "precip",
    "shortwave_down": "Sw_dw",
    "shortwave_up": "Sw_up",
    "longwave_down": "Lw_dw",
    "longwave_up": "Lw_up",
    "net_radiation_day": "Net_CNR1",
    "net_radiation_night": "Net_CNR1",
    "par_early": "Sw_par",
    "par_late": "Sw_par",
    "clearness_index": "Sw_dw",
}

# Same, for the WRF point extraction. A variable absent here simply has no model
# subset — the page renders the observed ones and says so.
WRF_COLUMN = {
    "air_temperature": "t2_c",
    "relative_humidity": "rh_pct",
    "pressure": "psfc_hpa",
    "wind_speed": "wind_speed_m_s",
    "wind_direction": "wind_dir_deg",
    "clearness_index": "swdown_w_m2",
    "shortwave_down": "swdown_w_m2",
    "longwave_down": "glw_w_m2",
}

# One date, two instruments. The PAR sensor changed here and the later era's
# PAR/global ratio is inconsistent with the literature, so the two PAR eras are
# published side by side rather than pooled. The albedo breaks on the same date —
# monthly median daytime Sw_up/Sw_dw of 0.345 / 0.353 / 0.370 in the three months
# before, 0.155 in the month after, near 0.13 for the rest of the record — inside
# a 17-day gap ending here, so reflected shortwave is restricted to the later era
# too rather than given a constant of its own.
ERA_SPLIT = pd.Timestamp("2019-03-15")

# Wind direction was recorded as an ARITHMETIC mean of an angle over this window,
# a scalar average across the 0/360 wrap: not one 5-minute sample fell within
# 15 deg of north in the whole of 2021. Kept in the database, excluded here.
INVALID_DIRECTION = (pd.Timestamp("2019-05-31"), pd.Timestamp("2023-02-20 13:30"))

# Where inside its own stamp each source's row lives, for the VALUES each source
# computes — the clearness index and the extraterrestrial mixture, never the
# daylight selection, which is shared (see _daytime_selection).
# The observed rows are MEANS over [T, T+1h) — sensors.aggregation.aggregate_to_hourly
# resamples on pandas' defaults, which label a window by its start — and
# docs/arqueologia/qc/lit-radiation-qc.md prescribes the correction: "If
# interval-averaged data is later ingested, evaluate mu0 at the interval midpoint"
# (Long & Dutton, BSRN Global Network recommended QC tests V2.0). The WRF field is
# instantaneous at the stamped hour (wrf.variables.compute_clearness_index), so it
# shifts by nothing.
GeometrySource = Literal["observed", "wrf"]

GEOMETRY_OFFSET: dict[GeometrySource, pd.Timedelta] = {
    "observed": pd.Timedelta(minutes=30),
    "wrf": pd.Timedelta(0),
}


# Variables restricted to daylight. Ungated, night fills half the record with
# zeros and the histogram collapses to one unreadable bar. PAR is a shortwave band
# too: 38% of the early era is exactly zero for being night, and no density
# survives that (KS distance 0.55).
DAYTIME_ONLY = (
    "shortwave_down",
    "shortwave_up",
    "net_radiation_day",
    "par_early",
    "par_late",
)

# Denominator floor for every ratio taken here (albedo, PAR fraction). A
# STATISTICAL necessity, nobody's published threshold: a ratio whose denominator
# approaches zero has unbounded variance, so a handful of twilight hours would
# dominate a bulk mean. Printed wherever it is applied.
RATIO_DENOMINATOR_FLOOR = 50.0

# Above this, PAR would exceed 60% of the global flux it is a sub-band of, which
# no instrument state explains. Removed as a point mass rather than clipped.
MAX_PAR_FRACTION = 0.6

# Equal-count bins of the extraterrestrial irradiance used to marginalise every
# induced density. Sixty is enough: against the exact mixture over all 35,436
# observed values the CDF differs by 1.2e-4, three orders below the reported KS.
INDUCED_BINS = 60

# Net radiation is a two-regime mixture, so its night half is published on its
# own: with no shortwave term it is the net longwave loss, narrow enough for a
# Gaussian to be defensible where it is indefensible for the pooled quantity.
NIGHTTIME_ONLY = ("net_radiation_night",)

# Anemometer starting threshold. Below it the instrument reports its floor rather
# than a wind, so those hours are the calm atom instead of Weibull samples.
CALM_THRESHOLD_MS = 0.281

# Relative humidity at or above this is the sensor's saturation clip, an atom the
# beta family cannot represent.
SATURATION_RH = 99.5


def read_wrf_series(path: str | Path) -> pd.DataFrame:
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

    Returns
    -------
    pandas.DataFrame
        Hourly model variables on a sorted, de-duplicated
        :class:`~pandas.DatetimeIndex` of naive station-local hours (UTC-03),
        with the spin-up hour removed. A repeated timestamp keeps the LAST row,
        which is the most recent run's value for that hour.
    """
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")])
    frame = rename_v1_columns(frame)
    stamps = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    frame.index = pd.DatetimeIndex(stamps)
    frame = frame.drop(columns=["year", "month", "day", "hour"])
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    spin_up = _times(frame).hour == 21
    logger.info("WRF: %d rows, dropping %d spin-up rows at hour 21", len(frame), int(spin_up.sum()))
    trimmed: pd.DataFrame = frame.loc[~spin_up]
    return trimmed


def _times(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """Narrow the index at the boundary, the way stats.climatology already does."""
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"the hourly database must be indexed by time, got {type(index).__name__}")
    return index


def _season_slices(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = seasonal_groups(frame)
    return {"all": frame, "DJF": groups["DJF"], "JJA": groups["JJA"]}


def _geometry_times(frame: pd.DataFrame, source: GeometrySource) -> pd.DatetimeIndex:
    """The instants the sun's position is evaluated at, for one source's rows.

    Parameters
    ----------
    frame:
        Block indexed by naive station-local (UTC-03) stamps, ``(N,)``.
    source:
        ``"observed"`` or ``"wrf"``; anything else raises, since a source with no
        declared averaging convention has no defensible geometry.

    Returns
    -------
    pandas.DatetimeIndex
        Naive station-local instants, ``(N,)``, in the same order as the index.
        The frame's own labels are untouched: only the geometry moves.
    """
    return _times(frame) + GEOMETRY_OFFSET[source]


def _selection_elevation_bounds(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest solar elevation over the hour each row averages.

    The bracket spans :data:`SELECTION_BRACKET_OFFSETS` — the averaging window
    ``[T, T+1h]`` — rather than the two per-source geometry offsets.  Solar
    elevation has at most one turning point inside an hour, and the only one that
    can fall in a window either gate is deciding is solar midnight (a minimum), so
    sampling the endpoints and the midpoint brackets the window.

    Parameters
    ----------
    frame:
        Block indexed by naive station-local (UTC-03) stamps, ``(N,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Minimum and maximum solar elevation in degrees, ``(N,)`` each, in the
        frame's own row order.  Both are read-only: they are shared between the
        callers that ask about the same block.
    """
    stamps = _times(frame).to_numpy()
    return _elevation_bounds_of(stamps.tobytes(), stamps.dtype.str)


@functools.lru_cache(maxsize=16)
def _elevation_bounds_of(stamps: bytes, stamps_dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """The bracket for one block of stamps, memoized on the stamps themselves.

    Every published variable asks the same gates about the same handful of blocks
    — the whole record and its seasonal slices — so an export recomputed this
    around 200 times over 61k rows, some 10 s of arithmetic for a dozen distinct
    answers.  The key is the block's own stamps, so a cache hit is the same
    question and never a near-miss; the results are frozen before being handed
    out, since every caller of a shared array gets the same object.

    *stamps_dtype* travels with the buffer and is not assumed: a ``datetime64``
    buffer carries no unit of its own, and reading microsecond stamps as
    nanoseconds moves every sun position by decades without failing.
    """
    times = pd.DatetimeIndex(np.frombuffer(stamps, dtype=stamps_dtype))
    lowest_deg, highest_deg = elevation_bounds(times, SITE, UTC_OFFSET_HOURS)
    lowest_deg.setflags(write=False)
    highest_deg.setflags(write=False)
    return lowest_deg, highest_deg


def _daytime_selection(frame: pd.DataFrame) -> np.ndarray:
    """Which hours count as daytime, on one rule both sources share.

    Deliberate deviation from evaluating each source's gate at its own instant:
    the offsets differ by half an hour, so near the terminators the two sides
    would admit different hours — around one row per day per terminator — and the
    paired histograms this artifact exists for would be conditional on different
    events. The hour is daytime only when the sun clears the floor throughout the
    hour the row averages, which also keeps each side's own extraterrestrial
    denominator away from zero.

    Parameters
    ----------
    frame:
        Block indexed by naive station-local (UTC-03) stamps, ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``(N,)``, in the frame's own row order.
    """
    lowest, _highest = _selection_elevation_bounds(frame)
    daytime: np.ndarray = lowest > MIN_SOLAR_ELEVATION_DEG
    return daytime


def _nighttime_selection(frame: pd.DataFrame) -> np.ndarray:
    """Which hours carry no sunlight at any instant inside the hour they average.

    The mirror of :func:`_daytime_selection`, and for an hourly MEAN the stricter
    reading of "night": an hour the sun rises or sets inside averages a sunlit
    half into a quantity published as the nocturnal regime, so the bracket has to
    span ``[T, T+1h]`` and not merely the instants the two sources' own values
    are evaluated at.

    Parameters
    ----------
    frame:
        Block indexed by naive station-local (UTC-03) stamps, ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``(N,)``, in the frame's own row order.
    """
    _lowest, highest = _selection_elevation_bounds(frame)
    nighttime: np.ndarray = highest < 0.0
    return nighttime


def _clearness(frame: pd.DataFrame, column: str, source: GeometrySource) -> pd.Series:
    """Clearness index from measured global irradiance, gated on solar elevation."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    extraterrestrial = extraterrestrial_ghi(_geometry_times(frame, source), SITE, UTC_OFFSET_HOURS)
    daylight = _daytime_selection(frame) & (extraterrestrial > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = frame[column].to_numpy() / extraterrestrial
    return pd.Series(np.where(daylight, kt, np.nan), index=frame.index).dropna()


def _strip_atoms(
    spec_id: str, series: pd.Series, speed: pd.Series | None
) -> tuple[np.ndarray, list[Atom]]:
    """Remove the point masses and report them, for EITHER source.

    Shared between the observed and the model branch on purpose: comparing the two
    histograms is only meaningful when both are conditional on the same event.
    That includes the calm cut, though 0,281 m/s is what a stalled cup anemometer
    reports and a model has no stall — comparability over instrumental literalism,
    stated in the variable's caveats rather than left implied.
    """
    if spec_id in ("wind_speed", "wind_direction") and speed is not None:
        # The gate is `<=`, not `<`: 0.281 m/s is the value the cup anemometer
        # REPORTS while stalled. 2,794 hourly means sit on it exactly (4.0% of the
        # record) and 51.8% of them bear north against 5.3% for the record as a
        # whole — a parked vane, which a strict `<` would keep.
        removed = int((speed <= CALM_THRESHOLD_MS).sum())
        calm = removed / len(series) if len(series) else float("nan")
        kept = series.loc[speed > CALM_THRESHOLD_MS]
        label = (
            f"Calmarias (até {CALM_THRESHOLD_MS:.3f} m/s, o valor que o anemômetro reporta parado)"
        ).replace(".", ",")
        return kept.to_numpy(), [Atom("calm", label, calm, removed)]

    if spec_id in ("relative_humidity", "relative_humidity_wxt"):
        removed = int((series >= SATURATION_RH).sum())
        clipped = removed / len(series) if len(series) else float("nan")
        kept = series.loc[series < SATURATION_RH]
        return kept.to_numpy(), [Atom("saturation", "Saturação (UR ≥ 99,5%)", clipped, removed)]

    if spec_id == "precipitation":
        wet = series.loc[series >= RAIN_BUCKET_MM]
        removed = len(series) - len(wet)
        dry = removed / len(series) if len(series) else float("nan")
        return wet.to_numpy(), [Atom("dry", "Horas sem chuva", dry, removed)]

    if spec_id == "shortwave_down":
        # A pyranometer's zero offset makes a few daytime hours slightly negative.
        # They fall outside the induced density's support, so they leave as a mass;
        # clipping to zero would invent a spike the instrument never measured.
        positive = series.loc[series > 0.0]
        removed = len(series) - len(positive)
        share = removed / len(series) if len(series) else float("nan")
        return positive.to_numpy(), [
            Atom(
                "nonpositive",
                "Horas com fluxo não positivo (deslocamento de zero do sensor)",
                share,
                removed,
            )
        ]

    return series.to_numpy(), []


def _paired_speed(
    spec_id: str, frame: pd.DataFrame, series: pd.Series, column: str | None = None
) -> pd.Series | None:
    """The wind speed a calm cut is judged by, or ``None`` when there is none.

    Direction is cut by the speed measured beside it, not by itself: a bearing
    carries no information about whether the cup was turning.
    """
    if spec_id == "wind_speed":
        return series
    if spec_id != "wind_direction":
        return None
    name = column or OBSERVED_COLUMN["wind_speed"]
    return frame[name].reindex(series.index) if name in frame.columns else None


def _observed_sample(spec_id: str, frame: pd.DataFrame) -> tuple[np.ndarray, list[Atom]]:
    """Select, gate and de-atomise one variable from the observed hourly frame."""
    if spec_id == "clearness_index":
        values = _clearness(frame, OBSERVED_COLUMN[spec_id], "observed")
        return values.to_numpy(), []

    column = OBSERVED_COLUMN[spec_id]
    if column not in frame.columns:
        return np.array([]), []
    series = frame[column]

    if spec_id in DAYTIME_ONLY:
        series = series.loc[_daytime_selection(frame)]
    elif spec_id in NIGHTTIME_ONLY:
        series = series.loc[_nighttime_selection(frame)]
    series = series.dropna()

    if spec_id == "par_early":
        series = series.loc[series.index < ERA_SPLIT]
    elif spec_id in ("par_late", "shortwave_up"):
        series = series.loc[series.index >= ERA_SPLIT]
    era_atoms: list[Atom] = []
    if spec_id == "wind_direction":
        first, last = INVALID_DIRECTION
        retained = series.loc[(series.index < first) | (series.index > last)]
        # Published rather than applied in silence: the cut removes 46.5% of the
        # record and every number on the rose is a share of what is left, so the
        # denominator has to be stated. Published even when it removes nothing —
        # a window entirely after the bad era reports 0 h rather than dropping the
        # entry, so the caveat citing this atom stays resolvable and "none
        # excluded" stays distinguishable from "not checked".
        removed = len(series) - len(retained)
        era_atoms = [
            Atom(
                "arithmetic_mean_era",
                f"Direção registrada como média aritmética do ângulo "
                f"({first.date()} a {last.date()}), inutilizável",
                removed / len(series) if len(series) else float("nan"),
                removed,
            )
        ]
        series = retained

    if spec_id in ("par_early", "par_late"):
        return _par_sample(series, frame)

    # Only the wind variables need the paired speed, and only they may require the
    # column to be present: a frame with no anemometer must still publish its rain
    # gauge.
    sample, atoms = _strip_atoms(spec_id, series, _paired_speed(spec_id, frame, series))
    return sample, [*era_atoms, *atoms]


def _par_sample_mask(series: pd.Series, global_flux: pd.Series) -> pd.Series:
    """Which PAR hours are measurements of the atmosphere rather than of the logger.

    One definition for both the fitted sample and the estimator of the curve's one
    free parameter, so the two can never disagree on the population.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = series / global_flux.where(global_flux > RATIO_DENOMINATOR_FLOOR)
    zeros = series <= 0.0
    impossible = (fraction > MAX_PAR_FRACTION).fillna(False)
    return ~(zeros | impossible)


def _par_sample(series: pd.Series, frame: pd.DataFrame) -> tuple[np.ndarray, list[Atom]]:
    """Strip the two point masses the PAR record carries after the daylight gate.

    Both are instrument states, not weather: a logger zero period in late 2018
    writing exact zeros in broad daylight, and hours where PAR exceeds 60% of the
    global flux it is a sub-band of. Fitting over either drags the curve toward a
    spike no atmosphere produced.
    """
    total = len(series)
    if not total:
        return np.array([]), []
    global_flux = frame[OBSERVED_COLUMN["shortwave_down"]].reindex(series.index)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = series / global_flux.where(global_flux > RATIO_DENOMINATOR_FLOOR)
    zeros = series <= 0.0
    impossible = fraction > MAX_PAR_FRACTION
    kept = series.loc[_par_sample_mask(series, global_flux)]
    return kept.to_numpy(), [
        Atom("daytime_zero", "Zeros diurnos do registrador", float(zeros.mean()), int(zeros.sum())),
        Atom(
            "par_over_global",
            f"Razão PAR/global acima de {MAX_PAR_FRACTION:.1f}, fisicamente impossível".replace(
                ".", ","
            ),
            float(impossible.fillna(False).mean()),
            int(impossible.fillna(False).sum()),
        ),
    ]


def _wrf_sample(spec_id: str, frame: pd.DataFrame) -> tuple[np.ndarray, list[Atom]]:
    """Same selection for the model series, where the variable exists there."""
    if spec_id not in WRF_COLUMN:
        return np.array([]), []
    if spec_id == "clearness_index":
        return _clearness(frame, WRF_COLUMN[spec_id], "wrf").to_numpy(), []
    column = WRF_COLUMN[spec_id]
    if column not in frame.columns:
        return np.array([]), []
    series = frame[column]
    # The SAME solar gate and de-atomisation as the observed side — the gate is
    # source-independent by construction — since conditioning the two histograms on
    # different events makes the comparison meaningless.
    if spec_id in DAYTIME_ONLY:
        series = series.loc[_daytime_selection(frame)]
    elif spec_id in NIGHTTIME_ONLY:
        series = series.loc[_nighttime_selection(frame)]
    series = series.dropna()
    return _strip_atoms(
        spec_id, series, _paired_speed(spec_id, frame, series, WRF_COLUMN["wind_speed"])
    )


def _scale_mixture(extraterrestrial: np.ndarray) -> tuple[list[float], list[float]]:
    """Equal-count bins of the extraterrestrial irradiance, as scales and weights.

    Marginalising over the covariate exactly would mean one mixture component per
    observed hour. Binning by QUANTILE rather than by value gives every component
    the same weight, so a tail bin holding three hours cannot speak as loudly as
    the bulk.
    """
    finite = np.sort(extraterrestrial[np.isfinite(extraterrestrial) & (extraterrestrial > 0.0)])
    if finite.size == 0:
        return [], []
    groups = np.array_split(finite, min(INDUCED_BINS, finite.size))
    groups = [group for group in groups if group.size]
    total = float(sum(group.size for group in groups))
    return (
        [float(group.mean()) for group in groups],
        [group.size / total for group in groups],
    )


def _bulk_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    """Energy-weighted ratio of two fluxes: sum over sum, not the mean of ratios.

    The mean of hourly ratios weights a 20 W/m2 twilight hour like a 900 W/m2 noon
    hour. Summing first makes the scalar mean what its name says — the share of
    incoming energy that came back or arrived in the band — and is far less
    sensitive to the denominator floor.
    """
    paired = pd.concat([numerator, denominator], axis=1, join="inner").dropna()
    paired = paired.loc[paired.iloc[:, 1] > RATIO_DENOMINATOR_FLOOR]
    if paired.empty or paired.iloc[:, 1].sum() <= 0.0:
        return float("nan")
    return float(paired.iloc[:, 0].sum() / paired.iloc[:, 1].sum())


def _check_caveats_quote_the_published_scalar(spec: object, payload: dict) -> None:
    """Warn when a caveat's printed number no longer matches the fitted one.

    The induced curves carry one estimated scalar — the era's PAR fraction, the
    albedo — that the caveats print as a prose literal three lines from the
    computed parameter, so the two drift apart whenever the archive changes.

    A warning, not a failure: which of the two is wrong is the laboratory's
    judgement, and it is printed at generation time, when someone is watching.
    """
    # The PUBLISHED sentences, not the specs' templates: a ``{{param:...}}`` marker
    # resolves to the fitted value, and checking the unresolved template would
    # report every interpolated sentence as missing the number it must carry.
    caveats = payload.get("caveats", ())
    fit = (payload.get("subsets", {}).get("observed_all") or {}).get("fit")
    if not caveats or not fit:
        return
    gain = fit.get("params", {}).get("gain")
    # 1.0 is the identity the families carry when no scalar is estimated at all
    # (the incoming shortwave rides its own law), so there is no prose to check.
    if gain is None or gain == 1.0:
        return
    quoted = f"{gain:.4f}".replace(".", ",")
    if not any(quoted in caveat for caveat in caveats):
        logger.warning(
            "%s: the fitted scalar is %s but no caveat quotes it; the printed prose and "
            "the published parameter disagree on the same screen",
            getattr(spec, "id", "?"),
            quoted,
        )


def _induced_options(
    spec_id: str, frame: pd.DataFrame, source: GeometrySource
) -> dict[str, object] | None:
    """The covariate-derived options one induced curve needs for one subset.

    SUBSET-MATCHED INHERITANCE: lambda and kt_max come from fitting the clearness
    index on THIS block, not on the whole record — the same rule every other
    variable follows, and what keeps the induced curve consistent with the
    clearness-index panel beside it.
    """
    spec = {item.id: item for item in CLIMATOLOGY_VARIABLES}[spec_id]
    if not spec.fit_options:
        return None

    global_column = (
        WRF_COLUMN["shortwave_down"] if source == "wrf" else OBSERVED_COLUMN["shortwave_down"]
    )
    if global_column not in frame.columns:
        return None
    parent = dist.fit_distribution(
        "hollands_huget", _clearness(frame, global_column, source).to_numpy()
    )
    if not np.isfinite(parent.params.get("lambda", np.nan)):
        return None

    daylight = frame.loc[_daytime_selection(frame)]
    if spec_id in ("par_late", "shortwave_up"):
        daylight = daylight.loc[daylight.index >= ERA_SPLIT]
    elif spec_id == "par_early":
        daylight = daylight.loc[daylight.index < ERA_SPLIT]
    if daylight.empty:
        return None

    extraterrestrial = extraterrestrial_ghi(
        _geometry_times(daylight, source), SITE, UTC_OFFSET_HOURS
    )
    scales, weights = _scale_mixture(np.asarray(extraterrestrial, dtype=float))
    if not scales:
        return None

    gain = 1.0
    incoming = daylight[global_column]
    if spec_id == "shortwave_up":
        gain = _bulk_ratio(daylight[OBSERVED_COLUMN["shortwave_up"]], incoming)
    elif spec_id in ("par_early", "par_late"):
        # Estimated over the SAME population the curve is scored against: over the
        # unmasked block the logger's daytime zeros re-enter through the
        # denominator and depress the ratio.
        band = _par_sample_mask(daylight[OBSERVED_COLUMN[spec_id]], incoming)
        gain = _bulk_ratio(daylight[OBSERVED_COLUMN[spec_id]].loc[band], incoming.loc[band])
    if not np.isfinite(gain) or gain <= 0.0:
        return None

    options: dict[str, object] = {
        "lam": float(parent.params["lambda"]),
        "kt_max": float(parent.params["kt_max"]),
        "scales": scales,
        "weights": weights,
        # Named, not folded into the scales: it is the one number estimated for
        # this variable, and for PAR the whole point of publishing the two eras.
        "gain": gain,
    }
    if spec_id == "net_radiation_day":
        line = _net_radiation_line(daylight)
        if line is None:
            return None
        options.update(line)
    return options


def _net_radiation_line(daylight: pd.DataFrame) -> dict[str, object] | None:
    """Local slope, intercept and residual spread of net radiation on shortwave.

    Hu, Wang and Liu (2012) establish that the relation is linear; the coefficients
    are this station's own because theirs are in MJ/m2 per day and these are hourly
    W/m2. The residual spread feeds the Gaussian convolution, and measuring its
    shape here is what licenses calling it Gaussian.
    """
    columns = [OBSERVED_COLUMN["shortwave_down"], OBSERVED_COLUMN["net_radiation_day"]]
    if any(column not in daylight.columns for column in columns):
        return None
    paired = daylight[columns].dropna()
    if len(paired) < 100:
        return None
    incoming = paired.iloc[:, 0].to_numpy()
    net = paired.iloc[:, 1].to_numpy()
    slope, intercept = np.polyfit(incoming, net, 1)
    residual = net - (slope * incoming + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_sd": float(residual.std(ddof=1)),
    }


def _available_hours(spec_id: str, block: pd.DataFrame) -> int:
    """Hours the sensor delivered, which is NOT the hours that survive the fit.

    The panel is labelled "horas válidas" and read as sensor availability, so it
    counts before the point masses leave: 91,3% of the rain gauge's hours are dry,
    which is weather and not an outage. Published as ``n + sum(atom counts)``, the
    identity the variable payload makes checkable from the bytes alone.
    """
    sample, atoms = _observed_sample(spec_id, block)
    return int(np.isfinite(sample).sum()) + sum(atom.count for atom in atoms)


def _coverage(frame: pd.DataFrame) -> dict[str, object]:
    """Valid hours per year and per season, per variable — the honesty panel."""
    years = []
    for year, block in frame.groupby(_times(frame).year):
        hours = {}
        for spec in CLIMATOLOGY_VARIABLES:
            hours[spec.id] = _available_hours(spec.id, block)
        years.append({"year": int(str(year)), "hours": hours})

    seasons = []
    for name, block in _season_slices(frame).items():
        if name == "all":
            continue
        present = sorted({int(year) for year in _times(block).year})
        hours = {}
        for spec in CLIMATOLOGY_VARIABLES:
            hours[spec.id] = _available_hours(spec.id, block)
        seasons.append({"season": name, "years": present, "hours": hours})

    return {
        "variables": [spec.id for spec in CLIMATOLOGY_VARIABLES],
        "years": years,
        "seasons": seasons,
    }


@app.command()
def run(
    input_path: Annotated[
        Path,
        typer.Option("-i", "--input", help="Hourly database from labmim-archive.", exists=True),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output", help="Site's dataset.paths.climatology directory."),
    ],
    wrf_path: Annotated[
        Path | None,
        typer.Option("-w", "--wrf", help="WRF series_operacional.dat for the model subsets."),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Publish the climatology JSON the site reads.

    Every source-by-season subset is precomputed; the manifest's `selector`
    names the four the page currently offers.
    """
    setup_logging(log_level)
    version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    observed = pd.read_parquet(input_path)
    typer.echo(
        f"Observado: {len(observed):,} horas, {observed.index.min()} .. {observed.index.max()}"
    )

    sources = [
        {
            "id": "observed",
            "label": "Estação LabMiM (registro observado)",
            "period": {"start": str(observed.index.min()), "end": str(observed.index.max())},
        }
    ]
    blocks: dict[str, tuple[GeometrySource, pd.DataFrame]] = {
        f"observed_{season.lower()}": ("observed", block)
        for season, block in _season_slices(observed).items()
    }

    if wrf_path is not None:
        model = read_wrf_series(wrf_path)
        typer.echo(f"WRF: {len(model):,} horas, {model.index.min()} .. {model.index.max()}")
        sources.append(
            {
                "id": "wrf",
                "label": "Modelo WRF (extração no ponto da estação)",
                "period": {"start": str(model.index.min()), "end": str(model.index.max())},
            }
        )
        blocks |= {
            f"wrf_{season.lower()}": ("wrf", block)
            for season, block in _season_slices(model).items()
        }

    subsets = [
        {
            "id": subset_id,
            "source": source,
            "season": subset_id.split("_", 1)[1],
            "label": _subset_label(source, subset_id.split("_", 1)[1]),
        }
        for subset_id, (source, _block) in blocks.items()
    ]
    selector = [subset_id for subset_id in SELECTOR if subset_id in blocks]

    manifest = build_manifest(
        version=version,
        generated_utc=version,
        station=STATION,
        period={"start": str(observed.index.min()), "end": str(observed.index.max())},
        sources=sources,
        subsets=subsets,
        selector=selector,
        coverage=_coverage(observed),
        variables=list(CLIMATOLOGY_VARIABLES),
        caveats=[
            "Registro observado da estação do LabMiM, não uma normal climatológica de 30 anos.",
            "Horário local de Salvador (UTC-03), sem horário de verão.",
        ],
        package_version=_package_version(),
        commit=_commit(),
    )
    write_json(output_dir / MANIFEST_FILENAME, manifest)

    for spec in CLIMATOLOGY_VARIABLES:
        samples: dict[str, np.ndarray] = {}
        atoms: dict[str, list[Atom]] = {}
        options: dict[str, dict[str, object]] = {}
        curveless: set[str] = set()
        for subset_id, (source, block) in blocks.items():
            if source == "observed":
                sample, subset_atoms = _observed_sample(spec.id, block)
            else:
                sample, subset_atoms = _wrf_sample(spec.id, block)
            samples[subset_id] = sample
            atoms[subset_id] = subset_atoms
            if spec.fit_options and len(sample):
                subset_options = _induced_options(spec.id, block, source)
                if subset_options is None:
                    # No covariate means no curve, but the bars still publish:
                    # dropping the sample would delete measurements over a missing
                    # model input.
                    logger.warning(
                        "%s/%s: no covariate available, publishing bars without a curve",
                        spec.id,
                        subset_id,
                    )
                    curveless.add(subset_id)
                else:
                    options[subset_id] = subset_options
        payload = build_variable_payload(
            spec, samples, version=version, atoms=atoms, options=options, curveless=curveless
        )
        _check_caveats_quote_the_published_scalar(spec, payload)
        path = write_json(output_dir / f"{spec.id}.json", payload)
        counts = " ".join(f"{key}={len(value):,}" for key, value in samples.items() if len(value))
        typer.echo(f"  [ok] {path.name:28s} {counts}")

    typer.echo(f"\n>> {len(CLIMATOLOGY_VARIABLES) + 1} arquivos em {output_dir}")


def _subset_label(source: str, season: str) -> str:
    seasons = {"all": "Ano inteiro", "djf": "Verão (DJF)", "jja": "Inverno (JJA)"}
    name = seasons.get(season, season)
    return name if source == "observed" else f"WRF — {name.lower()}"


def _commit() -> str | None:
    """Short commit of the checkout that produced these bytes, for provenance."""
    return run_git(["rev-parse", "--short", "HEAD"], cwd=source_root())


def _package_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("labmim-micrometeorology")
    except PackageNotFoundError:
        return None


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-climatology``)."""
    app()


if __name__ == "__main__":
    main()
