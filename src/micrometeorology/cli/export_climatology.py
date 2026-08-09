"""CLI: publish the observed-distribution artifacts the climatology page reads.

Consumes the hourly database built by ``labmim-archive`` plus the WRF point
extraction, and writes ``labmim-climatology-v1`` JSON into the directory the site
declares as ``dataset.paths.climatology``.

The artifacts are **not** committed to the site repository — they are derived
from the laboratory's own sensor archive, which is not public, so like the WRF
map data they are gitignored there and attached at deploy time.

Examples
--------
Publish straight into a checkout of the site::

    labmim-climatology -i output/archive/station_hourly.parquet \\
        -w data/series_operacional.dat \\
        -o ../site-labmim/site/Climatologia

Restrict to the observed record (no model subsets)::

    labmim-climatology -i output/archive/station_hourly.parquet -o out/ --no-wrf
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

# allsky.solar is pure numpy/pandas (no torch) and ships in the same wheel. It is
# also what stats/radiation.py's docstring explicitly points at for the case here
# — a GHI series plus timestamps, needing the extraterrestrial term derived from
# solar geometry — so reusing it beats a second implementation of NOAA's formulas.
from allsky.config import SiteConfig
from allsky.solar import extraterrestrial_ghi, solar_elevation
from micrometeorology.common.git import run_git
from micrometeorology.common.logging import setup_logging
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

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

# LabMiM tower, Instituto de Física, UFBA — Ondina, Salvador.
SITE = SiteConfig(latitude=-13.0055, longitude=-38.5089)
# Salvador keeps UTC-3 all year; Brazil abolished daylight saving in 2019 and it
# never applied to Bahia, so a fixed offset is exact rather than an approximation.
UTC_OFFSET_HOURS = -3.0

STATION = {
    "name": "Estação Micrometeorológica LabMiM",
    "institution": "Instituto de Física — UFBA",
    "latitude": -13.0055,
    "longitude": -38.5089,
    "elevation_m": 46.0,
    "timezone": "America/Bahia",
}

# Seasons the page offers, on the Southern-Hemisphere convention already used by
# stats.climatology.seasonal_groups.
SEASONS = ("all", "DJF", "JJA")

# The literal four options the page shows. Every source-by-season pair is
# computed anyway (it is one more pass over data already in memory), so the
# selector can widen later without regenerating anything.
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
    "air_temperature": "T",
    "relative_humidity": "ur",
    "pressure": "pressure",
    "wind_speed": "WS",
    "wind_direction": "WD",
    "clearness_index": "Swdw",
    "shortwave_down": "Swdw",
    # Only the two DOWNWELLING streams are trustworthy in the point extraction.
    # Swup_calc and Lwup_calc are derived from ALBD and EMISS, which the writer
    # emits as the broken constants -273.01 and -272.27 (a Kelvin-to-Celsius
    # conversion applied to a fill value), so the upwelling columns and any net
    # radiation built from them are physically meaningless. Publishing them
    # beside real measurements would invite a comparison that means nothing.
    "longwave_down": "Lwdw_glw",
}

# The PAR instrument changed on 2019-03-15 and the later era's PAR/global ratio
# is inconsistent with the literature, so the two are published side by side
# rather than pooled. See the caveats on those two variable specs.
PAR_ERA_SPLIT = pd.Timestamp("2019-03-15")

# Wind direction was recorded as an ARITHMETIC mean of an angle over this window
# — a scalar average across the 0/360 wrap. Measured symptom: not one 5-minute
# sample within 15 deg of north in the whole of 2021. The values are kept in the
# database and excluded here, because a rose drawn from them is wrong.
INVALID_DIRECTION = (pd.Timestamp("2019-05-31"), pd.Timestamp("2023-02-20 13:30"))

# Solar elevation above which a shortwave sample counts as daytime. Below it the
# airmass is extreme and the relative error swamps the signal, so the clearness
# index and every shortwave distribution are gated on it.
MIN_SOLAR_ELEVATION_DEG = 10.0

# Variables restricted to daylight. Without the gate, night fills half the record
# with zeros and the histogram collapses to a single bar nobody can read.
DAYTIME_ONLY = ("shortwave_down", "shortwave_up", "net_radiation_day")

# Net radiation is a two-regime mixture, so its night half is published on its
# own: with no shortwave term it is the net longwave loss, narrow enough that a
# Gaussian is defensible where it is indefensible for the pooled quantity.
NIGHTTIME_ONLY = ("net_radiation_night",)

# Anemometer starting threshold. Below it the instrument reports its floor rather
# than a wind, so those hours are the calm atom instead of Weibull samples.
CALM_THRESHOLD_MS = 0.281

# Relative humidity at or above this is the sensor's saturation clip, an atom the
# beta family cannot represent.
SATURATION_RH = 99.5


def read_wrf_series(path: str | Path) -> pd.DataFrame:
    """Read ``series_operacional.dat`` defensively.

    The file is an append-only log of successive operational runs, and it shows:
    it is **not** chronologically sorted, carries duplicated timestamps, and its
    header declares twelve trailing anonymous fields that only the oldest rows
    fill. Reading it with the full header and then sorting is the only way that
    does not silently misalign columns — passing ``names=`` turns the surplus
    fields into a twelve-level MultiIndex and shifts every value.

    Hour 21 local is 00 UTC, the initialisation hour of each run: surface fluxes
    and boundary-layer height are identically zero at t=0. Those rows are dropped
    wholesale rather than per-variable, so one uniform rule can be stated on the
    page.
    """
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")])
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


def _elevation(frame: pd.DataFrame) -> np.ndarray:
    """Solar elevation in degrees for every row, for the daylight gates."""
    elevation: np.ndarray = solar_elevation(_times(frame), SITE, UTC_OFFSET_HOURS)
    return elevation


def _clearness(frame: pd.DataFrame, column: str) -> pd.Series:
    """Clearness index from measured global irradiance, gated on solar elevation."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    elevation = _elevation(frame)
    top = extraterrestrial_ghi(_times(frame), SITE, UTC_OFFSET_HOURS)
    daylight = (elevation > MIN_SOLAR_ELEVATION_DEG) & (top > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = frame[column].to_numpy() / top
    return pd.Series(np.where(daylight, kt, np.nan), index=frame.index).dropna()


def _observed_sample(spec_id: str, frame: pd.DataFrame) -> tuple[np.ndarray, list[Atom]]:
    """Select, gate and de-atomise one variable from the observed hourly frame."""
    if spec_id == "clearness_index":
        values = _clearness(frame, OBSERVED_COLUMN[spec_id])
        return values.to_numpy(), []

    column = OBSERVED_COLUMN[spec_id]
    if column not in frame.columns:
        return np.array([]), []
    series = frame[column]

    if spec_id in DAYTIME_ONLY:
        series = series.loc[_elevation(frame) > MIN_SOLAR_ELEVATION_DEG]
    elif spec_id in NIGHTTIME_ONLY:
        series = series.loc[_elevation(frame) < 0.0]
    series = series.dropna()

    if spec_id == "par_early":
        series = series.loc[series.index < PAR_ERA_SPLIT]
    elif spec_id == "par_late":
        series = series.loc[series.index >= PAR_ERA_SPLIT]
    elif spec_id == "wind_direction":
        first, last = INVALID_DIRECTION
        series = series.loc[(series.index < first) | (series.index > last)]

    if spec_id == "wind_speed":
        calm = float((series < CALM_THRESHOLD_MS).mean()) if len(series) else float("nan")
        kept = series.loc[series >= CALM_THRESHOLD_MS]
        label = f"Calmarias (abaixo de {CALM_THRESHOLD_MS:.3f} m/s, limiar de partida)".replace(
            ".", ","
        )
        return kept.to_numpy(), [Atom("calm", label, calm)]

    if spec_id in ("relative_humidity", "relative_humidity_wxt"):
        clipped = float((series >= SATURATION_RH).mean()) if len(series) else float("nan")
        kept = series.loc[series < SATURATION_RH]
        return kept.to_numpy(), [Atom("saturation", "Saturação (UR ≥ 99,5%)", clipped)]

    if spec_id == "precipitation":
        wet = series.loc[series >= RAIN_BUCKET_MM]
        dry = 1.0 - (len(wet) / len(series)) if len(series) else float("nan")
        return wet.to_numpy(), [Atom("dry", "Horas sem chuva", dry)]

    return series.to_numpy(), []


def _wrf_sample(spec_id: str, frame: pd.DataFrame) -> tuple[np.ndarray, list[Atom]]:
    """Same selection for the model series, where the variable exists there."""
    if spec_id not in WRF_COLUMN:
        return np.array([]), []
    if spec_id == "clearness_index":
        return _clearness(frame, WRF_COLUMN[spec_id]).to_numpy(), []
    column = WRF_COLUMN[spec_id]
    if column not in frame.columns:
        return np.array([]), []
    series = frame[column]
    # The SAME solar gate as the observed side. Without it the model subset would
    # carry its night zeros while the observed one does not, and the two
    # histograms — whose whole purpose is to be compared — would sit on
    # different populations.
    if spec_id in DAYTIME_ONLY:
        series = series.loc[_elevation(frame) > MIN_SOLAR_ELEVATION_DEG]
    elif spec_id in NIGHTTIME_ONLY:
        series = series.loc[_elevation(frame) < 0.0]
    series = series.dropna()
    if spec_id == "wind_speed":
        # The model never outputs exactly zero, so its calm atom is empty by
        # construction. Reporting it as 0 rather than omitting it keeps the two
        # sources structurally comparable on the page.
        return series.to_numpy(), [Atom("calm", "Calmarias (modelo não produz zero exato)", 0.0)]
    return series.to_numpy(), []


def _coverage(frame: pd.DataFrame) -> dict[str, object]:
    """Valid hours per year and per season, per variable — the honesty panel."""
    years = []
    for year, block in frame.groupby(_times(frame).year):
        hours = {}
        for spec in CLIMATOLOGY_VARIABLES:
            sample, _atoms = _observed_sample(spec.id, block)
            hours[spec.id] = int(np.isfinite(sample).sum())
        years.append({"year": int(str(year)), "hours": hours})

    seasons = []
    for name, block in _season_slices(frame).items():
        if name == "all":
            continue
        present = sorted({int(year) for year in _times(block).year})
        hours = {}
        for spec in CLIMATOLOGY_VARIABLES:
            sample, _atoms = _observed_sample(spec.id, block)
            hours[spec.id] = int(np.isfinite(sample).sum())
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
    blocks: dict[str, tuple[str, pd.DataFrame]] = {
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
        for subset_id, (source, block) in blocks.items():
            if source == "observed":
                sample, subset_atoms = _observed_sample(spec.id, block)
            else:
                sample, subset_atoms = _wrf_sample(spec.id, block)
            samples[subset_id] = sample
            atoms[subset_id] = subset_atoms
        payload = build_variable_payload(spec, samples, version=version, atoms=atoms)
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
    return run_git(["rev-parse", "--short", "HEAD"])


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
