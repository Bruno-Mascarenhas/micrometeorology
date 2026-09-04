# `micrometeorology` — Documentation

Environmental data processing toolkit for the Micrometeorology Laboratory (LabMiM) at UFBA.

---

## Overview

The `micrometeorology` package provides a complete infrastructure for:

1. **Sensor data ingestion** — flexible reading of Campbell Scientific `.dat` files with dynamic headers
2. **Calibration** — immutable historical calibration records with date-range application
3. **Temporal aggregation** — high-frequency to hourly resolution with vector-mean wind direction
4. **WRF processing** — NetCDF reading, Cartopy map rendering, GeoJSON export, vertical interpolation
5. **Parallel batch rendering** — `ProcessPoolExecutor`-based parallel figure and JSON generation (30–60× speed-up)
6. **Statistics** — RMSE, MAE, MBE, R², correlation, Willmott d-index, IOA, NRMSE

---

## Package Structure

```
src/micrometeorology/
├── __init__.py              # Package version and docstring
├── cli/                     # Console entry points (registered in pyproject.toml)
│   ├── export_wrf_geojson.py    # labmim-wrf-geojson (WebGIS producer)
│   ├── render_wrf_maps.py       # labmim-wrf-figures
│   ├── run_wrf_pipeline.py      # Shared WRF batch driver
│   ├── ingest_sensor_data.py    # labmim-sensor-process
│   ├── build_archive.py         # labmim-archive (merges the .dat archive, verified)
│   ├── export_climatology.py    # labmim-climatology (climatology-page producer)
│   ├── export_monitoring.py     # labmim-monitoring (interactive monitoring payload)
│   ├── export_sky.py            # labmim-sky (the site's Ceu/ sky-condition artifacts)
│   ├── compare_wrf_observations.py  # labmim-comparison
│   ├── compute_metrics.py       # labmim-metrics
│   ├── generate_station_graphs.py   # labmim-station-graphs
│   └── plot_station_graphs.py   # labmim-site-graphs (three-layer monitoring PNGs)
├── common/
│   ├── config.py            # Centralised config (pydantic-settings + YAML, 4 layers)
│   ├── logging.py           # Structured logging setup
│   ├── paths.py             # Cross-platform path utilities (pathlib)
│   ├── cli_options.py       # Shared typer option definitions
│   ├── optional.py          # Optional-dependency guard (`require`)
│   ├── git.py               # Best-effort `run_git`; the one subprocess call to git,
│   │                        # shared by the allsky and solrad provenance stamps
│   └── types.py             # Enums (WRFVariable, GridLevel D01–D05), dataclasses, constants
├── sensors/
│   ├── archive.py           # Explicit archive manifest, clock repairs, sentinel table
│   ├── ingestion.py         # .dat reading with dynamic headers
│   ├── calibration.py       # Date-precise calibration (immutable historical records)
│   ├── aggregation.py       # Hourly aggregation with vector-mean wind direction
│   ├── monitoring.py        # labmim-monitoring-v1 chart catalogue (3 layers, WRF candidates)
│   ├── wind.py              # U/V decomposition and vector-mean direction
│   └── export.py            # Formatted CSV export
├── stats/
│   ├── distributions.py     # Histograms, MLE fits, goodness-of-fit distances
│   ├── climatology_export.py # labmim-climatology-v1 site artifacts
│   ├── metrics.py           # Model vs. observation metrics (RMSE, MAE, etc.)
│   ├── comparison.py        # Alignment + pairing + metric tables (pure pandas)
│   ├── comparison_plots.py  # Time-series/scatter figure for a paired frame
│   ├── climatology.py       # Diurnal / monthly / seasonal groupings of station series
│   └── radiation.py         # Station-Series clearness index (Kt) and diffuse fraction (Kd)
└── wrf/
    ├── reader.py            # NetCDF dataset wrapper (WRFDataset context manager)
    ├── variables.py         # Variable extraction and unit conversion
    ├── value_source.py      # Per-variable frames + scale bounds (figures and JSON)
    ├── plotting.py          # Colormap saturation helper (rendering lives in batch.py)
    ├── batch.py             # Parallel rendering engine (ProcessPoolExecutor)
    ├── animation.py         # PNG → WebM / GIF creation (parallel batch support)
    ├── interpolation.py     # Vectorised vertical interpolation (replaces wrf-python)
    ├── series.py            # Point time-series extraction from gridded data
    ├── operational_record.py # series_operacional.dat: schema, rows, v1->v2 repair
    ├── operational_series.py # wrfout -> one station's block (the netCDF half)
    └── geojson.py           # GeoJSON + value JSON export
```

---

## Installation

```bash
# Micrometeorology only in the active Conda environment:
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact

# With development dependencies:
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact --extra dev

# With video generation (moviepy):
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact --extra video

```

For local development, activate the `micrometeorology` Conda environment first.
The commands above install the audited lock while preserving Conda's interpreter
and bootstrap packages. Run `uv sync --locked` without
`UV_PROJECT_ENVIRONMENT` when you want a separate project `.venv`. On Windows,
set `UV_PYTHON` to the active Conda interpreter first:

```powershell
$env:UV_PYTHON = (python -c "import sys; print(sys.executable)")
```

### Cartopy Shapefiles

Cartopy requires Natural Earth data for coastlines and borders:

```bash
python -c "
import cartopy.io.shapereader as shpreader
shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
shpreader.natural_earth(resolution='10m', category='physical', name='coastline')
"
```

> **Note:** Shapefiles are NOT bundled in the repository. Each developer must download them locally.

---

## Usage

### 1. Configuration

Configuration is loaded from YAML with 4 priority layers:

```
configs/micromet/default.yaml  →  configs/micromet/<LABMIM_ENV>.yaml  →  LABMIM_CONFIG_PATH  →  Environment variables
```

```python
from micrometeorology.common.config import get_settings

settings = get_settings()
print(settings.data_dir)  # Path to data
print(settings.output_dir)  # Path to output
```

Environment variables use the `LABMIM_` prefix:

```bash
export LABMIM_DATA_DIR=/mnt/data/labmim
export LABMIM_ENV=server
```

`configs_dir` is where the sensor pipeline looks for `default.yaml` (QC limits,
summed columns, wind-direction columns) and `calibrations.yaml`, so it must name
the directory those two files actually live in: `configs/micromet/`. The shipped
`default.yaml` deliberately does **not** redeclare `configs_dir` — the field
default in `common/config.py` owns that path, and a YAML copy of it is what let
the two drift apart historically. Pointing `configs_dir` at a directory without
those files does not fail the run: it silently disables quality control,
calibration, precipitation summing and wind-direction vector averaging. Measured
on one hour of 5-minute data, that costs `precip` 0.5 mm instead of 6.0 mm,
`WD_WXT` 180° instead of 0°, `Temp1` +2.1 °C from an unfiltered 50 °C spike,
`PSP1_Wm2` +117 W/m² from an unfiltered 2000 W/m² spike, and `CM3Up_Wm2_Avg`
+24.5 W/m² from the missing 0.9694 sensitivity factor.

### 2. Sensor Data Ingestion

```python
from micrometeorology.sensors.ingestion import read_campbell_dat, merge_dat_files

# Single file
df = read_campbell_dat("data_2023.dat")

# Multiple files (headers may differ between them)
df = merge_dat_files(
    [
        "data_2023_jan.dat",
        "data_2023_feb.dat",
        "data_2023_mar.dat",
    ]
)
```

#### Why do headers vary?

The Campbell Scientific datalogger allows sensors to be added or removed at any time. When a sensor is added, a new column appears in the `.dat`; when removed, the column disappears. `read_campbell_dat()` handles this automatically:

- Missing columns are ignored (no error)
- Extra columns are included automatically
- `merge_dat_files()` performs an ordered merge across all columns

### 3. Calibration

Calibrations are **immutable historical facts**. Each record specifies:

```yaml
# configs/micromet/calibrations.yaml — the two shipped CMP21 records
calibrations:
  # Column names carry the logger's exact suffixes, as merge_dat_files()
  # produces them: `CMP21` without `_Wm2_Avg` matches nothing, and
  # unify_sensor_columns() then builds no unified column at all.
  - column: CMP21_Wm2_Avg
    start_date: null              # null = from the start of the data
    end_date: "2019-10-12"
    factor: null                  # null = invalid data for this period → NaN
    description: "CMP21 not installed before 2019-10-12; data is invalid"

  - column: CMP21_Wm2_Avg
    start_date: "2019-10-13"
    end_date: null                # null = until end of data
    factor: 0.9852941176          # 9.38 / 9.52
    description: "CMP21 sensitivity correction: 9.38 / 9.52"
```

```python
from micrometeorology.sensors.calibration import load_calibrations, apply_calibrations

cals = load_calibrations("configs/micromet/calibrations.yaml")
df = apply_calibrations(df, cals)
```

> ⚠️ **Never edit** existing calibration records. Always **append new** records for new periods.

### 4. Temporal Aggregation

<!-- fmt: off -->
```python
from micrometeorology.sensors.aggregation import aggregate_to_hourly

df_hourly = aggregate_to_hourly(
    df,
    min_samples=6,                  # minimum valid samples per hour
    sum_columns=["Rain_mm_Tot"],    # precipitation is summed
    wind_dir_columns=["WindDir"],   # direction uses vector-mean
    wind_speed_column_map={"WindDir": "WS_ms_Avg"},
)
```
<!-- fmt: on -->

#### Why vector-mean?

Wind direction cannot be averaged arithmetically. Example: the arithmetic mean of 350° and 10° gives 180°, but the correct result is 0° (north). The `wind.py` module decomposes into U/V, averages, and recomposes.

### 5. Metrics

```python
from micrometeorology.stats.metrics import compute_all, rmse, mae

# Single metric
error = rmse(observed, predicted)

# All metrics at once
results = compute_all(observed, predicted)
# {'RMSE': 2.3, 'MAE': 1.8, 'MBE': -0.2, 'R²': 0.95, 'r': 0.97, 'd': 0.98, 'IOA': 0.94, 'NRMSE': 0.08}
```

All metrics:
- Automatically strip NaN pairs before computation
- Return NaN if fewer than 2 valid pairs remain
- Follow the signature `metric(observed, predicted) → float`

### 6. WRF Figure Generation (Parallel)

The parallel rendering engine (`wrf/batch.py`) dispatches frames across all available CPU cores.

```python
from micrometeorology.wrf.batch import (
    FigureTask,
    build_map_config,
    default_workers,
    run_figure_tasks,
)

# Build tasks (one per frame)
tasks: list[FigureTask] = [...]
# Execute in parallel (cpu_count - 4 workers by default)
png_paths = run_figure_tasks(tasks, workers=44)
```

#### Architecture

1. Load each NetCDF **once** → extract all variable data into memory
2. Build a flat list of `FigureTask` NamedTuples (lightweight, picklable)
3. Dispatch to `ProcessPoolExecutor` with Agg backend (no GUI)
4. Each worker renders one frame → saves PNG → returns path
5. Group PNGs by variable+domain → create WebM in parallel

#### Performance

| Machine | Workers | ~2300 frames | Speed-up |
|---|---|---|---|
| Legacy (serial, Basemap) | 1 | ~45 min | 1× |
| 48-core workstation | 44 | ~1.5 min | 30× |
| 96-core workstation | 92 | ~45 sec | 60× |

### 7. Comparison (Model vs. Observation)

```python
from micrometeorology.stats.comparison import (
    read_dataset,
    pair_dataframes,
    compare_all_variables,
)

obs = read_dataset("salvador.dat")
model = read_dataset("wrf_output.csv")

paired = pair_dataframes(obs, model, tolerance="30min")
metrics = compare_all_variables(paired)
print(metrics)
```

---

## CLI (Command Line)

### GeoJSON/JSON Export (Primary)

```bash
labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o output/JSON -g output/GeoJSON \
    -v temperature -v wind --workers 44
```

The static-site artifact contract (all paths relative to `--geojson-dir` `-g`
and `--output-dir` `-o`) is:

```text
GeoJSON/{domain}.geojson              # legacy full FeatureCollection (fallback)
GeoJSON/{domain}.grid.json            # compact grid companion (edges/bounds)
JSON/{domain}_{variableId}_{NNN}.json # per-time-step value payload
JSON/{domain}_WIND_VECTORS_{NNN}.json # wind-arrow overlay for any variable
JSON/{domain}_ISOBARS_{NNN}.json      # sea-level isobar overlay for any variable
JSON/{domain}_{variableId}.series.bin # per-cell time-series (int32 matrix)
JSON/{domain}_{variableId}.summary.json  # per-step domain mean/min/max
JSON/manifest.json                    # run manifest (v2)
```

`{domain}` is `D01`–`D05`, `{NNN}` is the zero-padded three-digit time-step
index (the front-end reads it as the forecast hour), and `{variableId}` is the
output file suffix from `VARIABLE_NETCDF_MAP` (`micrometeorology.common.types`).

The `.series.bin` and `.summary.json` pair are the **consolidated site
artifacts** written by default; pass `--no-site-artifacts` to skip just those
two. Everything else is unchanged: the per-step value JSONs, the wind-vector
overlays (whenever `wind_vectors` is among the requested variables), the grid
files, and `JSON/manifest.json` are always written. The
manifest still declares `"format": "labmim-data-manifest-v2"` with `timezone`,
`index_min`, `index_max` and — when every unit agrees on the anchor —
`start_local`; only the `features` block is omitted, which is the documented
signal for a consumer to fall back to the per-step value JSONs. The v2 fields
are dropped entirely only when the run wrote no per-step value JSON at all, and
`availability` appears only for variables that do not span the full step range.
`--skip-first N` drops the first `N` spin-up time steps (their indices become
gaps in the timeline).

- **`{domain}.grid.json`** — compact cell geometry the front-end prefers over
  the multi-MB `.geojson`. `grid-edges-v1` stores only the shared 1-D
  `lon_edges`/`lat_edges` for separable (regular lat/lon) grids;
  `grid-bounds-v1` stores per-cell `[west, south, east, north]` for
  curvilinear grids. Cell `k` (row-major) equals the legacy GeoJSON
  `linear_index`, so both grid encodings and the value/series payloads share
  one cell order.
- **`{domain}_{variableId}_{NNN}.json`** — `{"metadata":{...},"values":[...]}`.
  `values` is the row-major flattened grid (one entry per cell, `null` where
  masked); `metadata.scale_values` holds six linspace legend stops and
  `metadata.date_time` the local timestamp. Poteolico/wind payloads carry an
  extra `metadata.wind` block.
- **`{domain}_ISOBARS_{NNN}.json`** — `{"metadata":{...},"isobars":[...]}`. One
  entry per traced level, each `{"level": hPa, "paths": [[[lon, lat], ...], ...]}`;
  a level whose contour comes back empty is omitted, so every published entry
  draws something, and a step too flat to name a round level publishes
  `"isobars": []` rather than a line at the field's midpoint.
  `metadata.interval` is the spacing the legend states.

  The spacing is one value per **domain**, not per step — a spacing that changed
  between steps would make the lines jump under the animation. It is the coarsest
  rung of `ISOBAR_INTERVALS_HPA` that still yields `TARGET_LEVELS_PER_STEP` lines
  on the domain's *median* step, so a single flat step cannot drive a whole domain
  to a fine spacing.

  **The ladder holds only 4/2/1 hPa.** Rungs of 0.5/0.2/0.1 hPa were published
  once and withdrawn. On the 2026-05-03 run the innermost domain spans a median
  1.14 hPa across its 84 km, which earned 0.2 hPa and drew a median 6 labelled
  lines over 14 separate polylines (18.4 KB a step); the same run's domain-mean
  pressure moves 2.9 hPa in a typical hour, fourteen of those spacings, so
  consecutive steps shared no level at all and the overlay reshuffled instead of
  drifting. At the 1 hPa floor the same domain draws a median 1 line (4.3 KB) and
  publishes none in 5 of 76 steps, which is the honest reading of a field flatter
  than the coarsest chart anyone draws. Measured effect of the trim, per domain:
  d01 and d02 unchanged (2 and 1 hPa), d03 0.5 → 1 hPa (7 → 3 lines), d04
  0.2 → 1 hPa (6 → 1 line). 3 hPa is deliberately absent: it is not an interval
  charts are drawn at, and it would blank d04 in 39 of 76 steps.

  Isobars are an overlay over whichever field is shaded beneath them, so they are
  published once per domain and step and drawn over the variables listed in
  `features.isobar_overlay.draw_over`. Sea-level reduction is only defensible
  below roughly 1500 m of terrain; a domain that crosses it earns an operator
  warning rather than a failed unit. **The absolute levels are only as good as the
  run's mass field** — see "Reading isobars from a run whose pressure oscillates"
  below.
- **`{domain}_{variableId}.series.bin`** — `cell-series-int32-le-v1`: a
  row-major `cells × steps` little-endian int32 matrix of
  `round(value, 2) × 100`, with sentinel `-2147483648` for never-written /
  masked / NaN steps. Columns span `0..n_steps-1` regardless of skip-first or
  night gaps, so one HTTP Range request returns a single cell's whole series.
- **`{domain}_{variableId}.summary.json`** — `domain-summary-v1`: per-step
  `indices`, `date_times`, `mean`, `min`, `max` over the same rounded values,
  for the lightweight domain-preview panel.
- **`manifest.json`** — see "Run manifest (v2)" below. Only `values_json` and
  `poteolico` work units accumulate `.series.bin`/`.summary.json`;
  `WIND_VECTORS`, `ISOBARS` and the grid GeoJSON do not.

Supported site-oriented variables include the legacy fields `TEMP`, `PRES`,
`VAPOR`, `RAIN`, `WIND`, `SWDOWN`, `HFX`, `LH`, and the wind-potential files
`POT_EOLICO_50M`, `POT_EOLICO_100M`, and `POT_EOLICO_150M` generated by
`poteolico`. Additional 2026 WRF fields include `TSK`, `RH2`, `GLW`, and
`WIND_POWER_DENSITY_10M`. Units, formulas, and limitations are documented in
the extractor docstrings in `src/micrometeorology/wrf/variables.py`.

#### Reading isobars from a run whose pressure oscillates

Isobar levels are absolute pressures, so they inherit every defect in the run's
mass field, and the overlay is the artifact where such a defect is most visible:
a domain-wide shift in the mean moves the whole level family across the map at
once, which reads on the page as flicker rather than as weather.

The 2026-05-03 operational run has such a defect, and it is worth knowing what
it looks like because no contouring choice repairs it:

| | 2026-05-03 | 2013-07-01 (control) | station barometer `BP1` |
|---|---|---|---|
| hourly \|Δ\| of the domain-mean surface pressure | 2.84 hPa/h (max 7.8) | 0.195 hPa/h (max 0.78) | 0.59 hPa/h |

At the station's own cell the run moves 3.18 hPa/h against the barometer's
0.59 — a factor 5.4. The oscillation is spatially **uniform** (spatial standard
deviation of the hourly jump 0.2–0.4 hPa against a mean of −6.7), it is present in
`MU`, the prognostic dry-air mass, and it carries the same phase in all four
domains, so it enters through the outermost domain or the boundary conditions
rather than through nesting. Its largest jumps are phase-locked to the 03→04,
09→10, 15→16 and 21→22 UTC transitions.

None of it comes from the sea-level reduction, which was audited against RIP4
`seaprs` on this data: it contributes at most 0.16 hPa/h of the swing (1.4% at the
station cell), the terrain-locked artifact its own module docstring declares
accounts for 0.2–2.9%, and its reordered hot-surface branch differs from RIP4's
on zero cells across all four domains. 86–94% of the hourly change in the reduced
field is inherited verbatim from the wrfout's own `PSFC`.

Before trusting a new run's isobars, check
`np.abs(np.diff(PSFC.mean(axis=(1, 2))))`: a median above roughly 1 hPa/h is the
input, not the exporter. Coarsening the interval reduces the visual clutter but
cannot damp the sweep, because the sweep is in the field.

JSON export runs coarse (file, variable) work units on ONE persistent process
pool (`micrometeorology.wrf.jobs`). Each worker opens the NetCDF itself with
the eager `netCDF4` reader, derives its variable, computes scale bounds, and
writes every timestep JSON in-process — no arrays cross the process boundary
and no temporary `.npy` payloads are staged. Wind-potential (`poteolico`)
extraction streams U/V/PH/PHB in ~64-step blocks
(`variables.stream_wind_at_heights`), interpolating u/v/speed to all target
heights from one bracket pass per block, so peak worker memory is bounded by
the block size regardless of how many timesteps the file has.

Reliability: every output file is written to a temporary name and atomically
renamed, so consumers never observe truncated JSON. A unit that fails reports
its error without affecting sibling units; if a worker process dies (e.g.
OOM-killed), incomplete units are retried one at a time in isolated pools and
anything still failing makes the CLI exit non-zero with a per-unit report.
On network filesystems where HDF5 file locking fails at open, set
`LABMIM_HDF5_FILE_LOCKING=BEST_EFFORT` (do not disable locking for files that
may still be written by WRF).

Non-finite rejection: every JSON writer serializes with `allow_nan=False`, and
a field whose scale bounds are non-finite (a fully masked/NaN variable) fails
its work unit rather than emitting bare `NaN` tokens — invalid JSON that would
only break later, in every visitor's browser.

Timezone: all exported `date_time` strings and the manifest anchor are
expressed in a single pinned product timezone, `America/Bahia`
(UTC−03:00, no DST), so the daily job produces identical labels regardless of
the host's clock. Override with `LABMIM_TIMEZONE`; prefer fixed-offset zones,
because the front-end labels the timeline with flat one-hour-per-index
arithmetic and a DST transition inside a run would desync those labels from the
per-file `date_time` strings. Timestamps are formatted `DD/MM/YYYY HH:MM:SS`
and truncated to the hour.

#### Run manifest (v2)

Every run writes `JSON/manifest.json`. The front-end fetches it with
`cache: "no-cache"` at startup and re-checks it periodically; its `version`
(a UTC timestamp) is appended as `?v=` to every data URL so the fixed-name
files can be cached aggressively yet cache-bust the moment a new run publishes.

```jsonc
{
  "version": "20260719T013159Z",          // run id → ?v= cache-buster
  "generated_utc": "2026-07-19 01:31:59Z",
  "domains": ["D01", "D02", "D03", "D04"],
  "files": 4844,                            // total files written this run
  "format": "labmim-data-manifest-v2",      // absent → v1 (front-end defaults)
  "timezone": "America/Bahia",
  "index_min": 0,                           // intersection of per-domain ranges
  "index_max": 75,
  "start_local": "02/05/2026 21:00:00",     // local time of file index 0
  "availability": {                         // only variables NOT full-range
    "SWDOWN": [[9, 21], [33, 45], [57, 69]] // inclusive [start, end] step runs
  },
  "features": {                             // consolidated-artifact descriptors
    "domain_summary": {
      "format": "domain-summary-v1",
      "template": "JSON/{domain}_{variable}.summary.json"
    },
    "cell_series": {
      "format": "cell-series-int32-le-v1",
      "template": "JSON/{domain}_{variable}.series.bin",
      "dtype": "int32", "byte_order": "little",
      "scale": 0.01, "missing": -2147483648,
      "index_min": 0, "index_max": 75       // series columns span 0..n_steps-1
    }
  }
}
```

The v2 fields are additive and are derived only from files **actually written
this run** (never re-derived arithmetic that could drift): `availability` lists
only variables missing from the full step range (e.g. `SWDOWN` daylight
windows); the `features` descriptors are advertised only when every unit
succeeded and agreed on the step count, because the consolidated artifacts are
a byte-offset contract and a failed unit could leave a previous run's file in
place. A consumer that sees no `features` block falls back to the per-step
value JSONs — that is also exactly what a `--no-site-artifacts` run emits: still
`labmim-data-manifest-v2`, still with the timeline fields, just without
`features`. `start_local` always pairs with file index `0`, even when
`--skip-first` makes `index_min > 0`.

To run single-process, pass `--workers 1`. There is deliberately no reader or
worker-backend selection anymore: eager block-streamed reads plus one
persistent pool of file-owning workers is the only execution model.

### Figures (Static Maps & Video)

```bash
# Single domain
labmim-wrf-figures -d wrfout_d03_2024-01-01 -o output/figures/ -v temperature -v wind

# Multiple domains with videos
labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain -v SWDOWN \
    -o output/figures/ --workers 44 --also-video
```

Figure frames are spilled to temporary ``.npy`` files and rendered on one
persistent worker pool per run; no reader or backend tuning is exposed.

### Local testing (all-in-one)

```bash
python -m micrometeorology.cli.run_wrf_pipeline \
    --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain \
    -o output/wrf_local/ --workers 8 --also-video
```

### Sensor processing

```bash
labmim-sensor-process --input data/raw/ --output data/hourly/sensor_data.csv
```

### Station archive (`labmim-archive`)

Turns `data/dados-labmim/` into one verified database. It does **not** glob: the
file list is an explicit, ordered manifest in
`micrometeorology.sensors.archive`, because a bare `*.dat` drops the `.backup`
rotation files — three of which are the only source of an austral winter each —
and sweeps in a second station, the calibration campaigns and the 1-minute solar
tables. Three clock defects are repaired into a scratch directory; nothing under
`data/` is ever written.

```bash
labmim-archive -d data -o output/archive --strict
```

`--strict` exits non-zero when the merge loses rows, duplicates a timestamp or
comes out non-monotonic. The floors it checks against are named constants in
`micrometeorology.sensors.archive` — `EXPECTED_LENTA_ROWS`, `EXPECTED_RAIN_ROWS`,
`ARCHIVE_START`, `ARCHIVE_END` — rather than numbers transcribed here, because
the archive grows with every export and a figure written into prose goes stale
the following week. Treat it as the archive's regression test.

Three artifacts plus a report. The shapes below are one run's, on the archive as
of 2026-09-03; `archive_report.json` carries the current ones:

| file | what it is |
|------|------------|
| `station_5min_raw.parquet` | 1,022,917 x 93 — values as the logger wrote them, sentinels included |
| `station_5min_qc.parquet` | 1,022,917 x 111 — after sentinel masking, physical gates, calibrations and era unification |
| `station_hourly.parquet` | 86,866 x 93 — hourly means, sum for the tipping bucket, speed-weighted vector mean for direction, and the fraction of each hour's samples whose logger status read `OK` |
| `archive_report.json` | the verification, plus samples masked per column |

The run prints how many physical limits actually **fired**. That number matters:
a limit naming a column the frame does not carry is skipped in silence, which is
how the shipped config once reached 19 dead entries out of 21.

### Climatology page artifacts (`labmim-climatology`)

Producer for the `labmim-climatology-v1` JSON the site's climatology page fetches
at runtime, the same role `labmim-wrf-geojson` plays for the WebGIS. Consumes the
hourly database above plus the WRF point extraction.

```bash
labmim-climatology -i output/archive/station_hourly.parquet \
    -w data/series/labmim_series_operacional.dat \
    -o ../site-labmim/site/Climatologia
```

Writes `manifest.json` plus one file per variable. Everything the browser draws
is precomputed here — frozen bin edges, the maximum-likelihood fit, the
theoretical density sampled at the bin centres, and goodness-of-fit **distances**
(never p-values: at ~10^5 correlated hourly samples every classical test rejects
every model). Point masses that no continuous family can represent — wind calms,
dry hours, the humidity saturation clip — are removed from the fit and published
beside it as their own probability.

#### Induced densities: how the radiation variables got a citation

Irradiance in W/m2 has no canonical density — half the record is night and the
extraterrestrial forcing swings with the hour and the season, so the shape of a
raw-flux histogram is mostly solar geometry. The quantity the literature models
is the clearness index. Since the flux **is** that index times the
extraterrestrial irradiance, the published kt law *induces* a density on the
flux: an exact change of variable, marginalised over the observed covariate in
60 equal-count bins, inheriting its parameters and introducing no new ones.

Three families implement it (`stats/distributions.py`):

| family | variables | what is estimated |
|---|---|---|
| `compound_hollands_huget` | shortwave down/up, both PAR eras | nothing, or one gain scalar |
| `power_normal_mixture` | longwave up, via brightness temperature | the two-regime mixture |
| `compound_hollands_gaussian` | daytime net radiation | the local Rn-vs-Rs line |

Longwave down is the only one with a published density for the raw flux itself
and uses the existing `normal`.

**Inheritance is subset-matched**: each recorte's induced curve rides on the
clearness index fitted to that same recorte, exactly as every other variable on
the page is fitted per subset. `VariableSpec.fit_options` declares what the
family cannot get from the sample, and a missing option is an error rather than
a curve quietly fitted on the wrong covariate.

The gain is a **named parameter**, not folded into the scales, because it is the
scientific point for PAR: 0.4475 before the 2019 instrument change against
0.2579 after, a ratio of 1.735 that quantifies the suspected scale error.

**PAR gained the daylight gate it was missing.** It was the only shortwave
variable exported over all hours, so 38% of the early era's published values
were exactly zero — night. The published n falls from 14,450 to 6,412 and the
curve becomes fittable (the same curve against the ungated sample scores a KS
distance of 0.55).

#### Bibliography

`REFERENCES` carries sixteen records — authors, title, venue, resolvable link —
and labels and caveats cite them with `[[key]]` markers instead of prose. The
manifest publishes the registry once and the site turns each marker into a link
with the full record in its tooltip. Every record links through `doi.org`, each
identifier resolved and checked to land on the cited work; a search URL is
refused by test, because a search that silently returns nothing still answers
200. Two guards run at import: a marker with no record, and a record
containing a marker of its own — the second because a global rename of the prose
citations produced exactly that during development.

The output is **not** committed to the site repository: it derives from the
laboratory's private sensor archive, so like the WRF data it is gitignored there
and attached at deploy time.

### Derived surface radiation budget

wrfout carries the downwelling fluxes directly (`SWDOWN`, `GLW`) but the
upwelling ones only when the RRTMG bottom-of-atmosphere diagnostics are
enabled — `LWUPB`/`SWUPB` are in the 2013 archive runs and absent from the
2026 operational runs. The budget is therefore reconstructed from fields every
run carries (`EMISS`, `TSK`, `GLW`, `SWDOWN`, `ALBEDO`, `T2`, `COSZEN`), so a
term publishes identically across wrfout generations:

| `-v` name         | Output id | Formula                                      | Units |
| ----------------- | --------- | -------------------------------------------- | ----- |
| `lwup`            | `LWUP`    | `ε·σ·TSK⁴ + (1−ε)·GLW`                       | W/m²  |
| `swup`            | `SWUP`    | `ALBEDO · SWDOWN`                            | W/m²  |
| `lwnet`           | `LWNET`   | `GLW − LWUP` = `ε·(GLW − σ·TSK⁴)`            | W/m²  |
| `swnet`           | `SWNET`   | `SWDOWN · (1 − ALBEDO)`                      | W/m²  |
| `rnet`            | `RNET`    | `SWNET + LWNET`                              | W/m²  |
| `sky_emissivity`  | `EPS_SKY` | `GLW / (σ·T2⁴)`                              | –     |
| `clearness_index` | `KT`      | `SWDOWN / (S₀·E₀·cos z)`                     | –     |

`LWUP` is the one that matters most: the `(1−ε)·GLW` term is the sky radiance a
non-black surface reflects, and dropping it — as the legacy exporter did, on
top of a Celsius-for-Kelvin bug — costs ~15 W/m². Net fluxes are positive
downward, so `RNET` is negative at night and `LWNET` almost always is.

Validated two independent ways:

- **Against WRF's own fluxes**, where the run wrote them. Over d02 2013,
  `LWUP` reproduces `LWUPB` to MAE 0.82 W/m² on a ~420 W/m² signal (0.11%).
- **Against the surface energy budget**, on the 2026 operational nests that
  carry no `LWUPB`. Averaged over three whole days, `Rn + G = H + LE` closes to
  **−0.6…−1.4 W/m² on all four domains**; dropping the reflected term moves the
  residual to **+21.8…+23.5 W/m²**. The budget alone picks the right formula by
  a factor of ~20 on files with no ground truth in them.

One caveat on `SWUP`. `SWUPB`/`SWDNB` are RRTMG diagnostics frozen at the last
radiation call (`RADT=30 min` on every nest), while `SWDOWN` is rescaled by the
zenith angle every model step. On d04 the output times land on the radiation
calls, so `SWDOWN == SWDNB` and `ALBEDO·SWDOWN` reproduces `SWUPB` to 2e-4
W/m²; on d01–d03 the two clocks drift apart and the fields disagree by up to
~180 W/m² near sunrise purely from that lag. Publishing `ALBEDO·SWDOWN` is the
deliberate choice — it keeps `SWUP`, `SWNET` and `RNET` consistent with the
`SWDOWN` published beside them. The 3-day energy budget is indifferent between
the two (they agree to &lt;0.5 W/m²), so nothing physical is lost.

The `(1−ε)·GLW` term is not an embellishment: RRTMG forms the upwelling
radiance as `radlu = rad0 + reflect*radld` with `rad0 = semiss*B(Ts)` and
`reflect = 1−semiss`, and Noah's urban branch writes the same expression
longhand. `LWUP` is WRF's own formula, evaluated on WRF's own output.

`SWUP`, `SWNET` and `KT` follow the same 06–18 local daylight gate as
`SWDOWN`; the longwave and net terms publish all 24 hours, which is when the
surface energy loss is most visible. `KT` additionally publishes no value
(`null`, and `MISSING` in the series matrix) below 10° solar elevation
(SZA > 80°, the ARM DOE/SC-ARM/TR-008 daytime threshold), since the
extraterrestrial denominator there is a horizon singularity rather than a
sky-condition signal. Formulas, validation and limitations are documented on
each extractor in `src/micrometeorology/wrf/variables.py`; the physics tests
live in `tests/micromet/test_wrf_radiation.py`, whose `_REFERENCES` block
marks each citation with how far it was actually verified.

### Operational point series

Producer for the hourly model series at a point -- the file
`labmim-climatology`, `labmim-monitoring` and `labmim-site-graphs` read as their
WRF layer. The command is `labmim-wrf-series` (`cli/export_operational_series.py`) over two
modules: `wrf/operational_record.py` holds what the record IS -- its schema, its
rows, its repair -- and needs only pandas, so the CLIs that merely read the
record do not pay for a NetCDF stack; `wrf/operational_series.py` is the half
that turns a wrfout into a block.

```bash
# One run's block for every station, appended to each station's own record
labmim-wrf-series run --wrf-dir /data/wrf/20260815/wrf01 --date 20260815 \
    -o data/series -s labmim:-13.0055:-38.5089 -s ilheus:-14.7889:-39.0339

# Once, before the first append: the v2 record is a new artifact at a new path
cp data/series_operacional.dat data/series/labmim_series_operacional.dat
labmim-wrf-series migrate -i data/series/labmim_series_operacional.dat
```

The file is **not** the product of a single run. The server simulates one day at
a time and each run contributes 24 rows, so the record spans years while every
block comes from a different execution of the model. Step 0 is 00 UTC, written
as hour 21 of the previous **local** day (UTC-03) -- the run's initialisation,
where the physics has not been called yet -- and step 23 is 23 UTC, hour 20
local. 1087 such blocks were appended between 2022-06-15 and 2026-03-18.

#### Several stations, one domain each

A nested run covers the same region at four resolutions, so which wrfout answers
for a station depends on where the station is. Pass the whole run and every
station: each is served by the **finest domain whose grid contains it**, and
writes to its own `{name}_series_operacional.dat` in `-o`. A station no domain
reaches is named in the output and the command exits non-zero -- the others are
still written, because a series that silently stops being published is the
failure worth catching.

| Station | Domain that serves it | dx |
|---|---|---|
| LabMiM tower (-13.0055, -38.5089) | `d04` | 1 km |
| Feira de Santana (-12.2664, -38.9663) | `d03` | 3 km |
| Ilhéus (-14.7889, -39.0339) | `d02` | 9 km |
| Brasília (-15.79, -47.88) | `d01` | 27 km |

Stations come from `-s name:lat:lon` (repeatable) or `--stations file.csv`
(`name,lat,lon`, header optional). With neither, the run covers the LabMiM
tower alone and writes `labmim_series_operacional.dat`. A station name becomes a
file name, so it is restricted to `[A-Za-z0-9][A-Za-z0-9_-]*` rather than
sanitised.

#### The header is the schema

Rows are rendered against the header the FILE declares, field by field, with
`nan` wherever the run produced no value. Three consequences, all deliberate:

- a variable the model stops writing **empties one column** instead of shifting
  every column after it;
- a column this extraction has never heard of is **preserved**, not dropped;
- a new quantity is **appended** to the header (`--extend-header`, on by
  default) and the rows already written stay shorter than it, which pandas reads
  as no-value -- which is what they mean.

`-v/--variables` selects the columns to compute. A name the catalogue does not
define but the wrfout carries becomes a raw passthrough column, so a field WRF
starts writing reaches the record with no code change; a name that is neither is
a usage error, not a silently short row.

#### Columns (v2)

Every name carries its unit, because the record does mix them -- `e_hpa` beside
`es_pa` is the file's own historical convention, kept rather than converted.

| Column | Source | Unit |
|---|---|---|
| `year,month,day,hour` | `Times` + `STATION_UTC_OFFSET_HOURS` | local hour |
| `t2_c` | `T2 - 273.15` | °C |
| `rh_pct` | `100·e/es`, **unclipped** | % |
| `psfc_hpa` | `PSFC/100` | hPa |
| `e_hpa` | `w·p/(0.622 + w)`, `w` = `Q2` | hPa |
| `es_pa` | Bolton (1980) eq. 10 | Pa |
| `q2_g_kg` | `Q2·1000` | g/kg |
| `wind_speed_m_s`, `wind_dir_deg`, `u10_m_s`, `v10_m_s` | `U10`,`V10` rotated onto true north | m/s, ° |
| `swdown_w_m2`, `swdnb_w_m2`, `swupb_w_m2` | `SWDOWN`, `SWDNB`, `SWUPB` | W/m² |
| `swddif_w_m2`, `swddir_w_m2` | `SWDDIF`, `SWDDIR` | W/m² |
| `swup_w_m2` | `ALBEDO · SWDOWN` | W/m² |
| `glw_w_m2` | `GLW` | W/m² |
| `lwdnb_w_m2` | `LWDNB`, else `GLW` | W/m² |
| `lwup_w_m2` | `LWUPB`, else `ε·σ·TSK⁴ + (1−ε)·GLW` | W/m² |
| `lwup_air_w_m2` | `ε·σ·T2⁴` | W/m² |
| `albedo`, `emissivity` | `ALBEDO`, `EMISS` | – |
| `hfx_w_m2`, `lh_w_m2`, `grdflx_w_m2` | `HFX`, `LH`, `GRDFLX` | W/m² |
| `ustar_m_s`, `pblh_m` | `UST`, `PBLH` | m/s, m |
| `sst_c` | `SST - 273.15` | °C |
| `precip_mm` | hourly increment of `RAINC+RAINNC` | mm |
| `swdown_farms_w_m2`, `swddif_farms_w_m2`, `swddir_farms_w_m2` | none — `nan` | W/m² |

`rh_pct` is deliberately **not** `variables.compute_relative_humidity`, which
clips to 0-100% for a colour scale: this file is read as model state, and a
model reporting supersaturated air should say so.

`sst_c` is WRF's `SST` at the serving cell. Over a **land** cell -- which the
tower's is -- WRF fills that field with the skin temperature at initialisation
and never updates it, so at this site it is a per-run constant that is not a sea
temperature. Pointed at a water cell it is one.

The `_farms` trio has no source in the current configuration: `SWDDIR + SWDDIF`
equals `SWDOWN` exactly in every wrfout of this era, so there is one
direct/diffuse pair and it already feeds `swddir_w_m2`/`swddif_w_m2`. The
2022-2026 values stay; nothing new is written there.

#### v1 → v2: what `migrate` repairs

The extraction that wrote the record was never committed here; its formulas were
recovered from the 26,087 rows themselves, which surfaced four defects. All four
are exactly invertible from the rows' own fields, so `migrate` repairs the
record rather than annotating it, keeping a `.bak` and passing every untouched
cell through verbatim so a diff shows only what changed.

| # | Defect | Repair | Cells |
|---|---|---|---|
| 1 | `ALBD`/`EMISS` written as `value − 273.15` — a Kelvin-to-Celsius conversion applied to two dimensionless quantities | `+ 273.15` | 23 759 each |
| 1a | `Swup_calc` = `(albedo − 273.15)·Swdw`, down to −294 069 W/m² | `+ 273.15·Swdw`, which needs no albedo — the 2022 rows have none | 26 087 |
| 1b | `Lwup_calc` = `(ε − 273.15)·σ·T_celsius⁴` — broken emissivity **and** Celsius under Stefan-Boltzmann | `ε·σ·T_K⁴`, with ε from the repaired column or the constant 0.88 the record itself implies | 26 087 |
| 2 | `e` used `w·p/(0.622 + 0.378·w)`, the **specific-humidity** conversion, on WRF's `Q2`, which is a **mixing ratio** | `w·p/(0.622 + w)`, and `ur` recomputed from it | 26 087 each |
| 3 | WRF's cold-start step published as measurement: every flux written identically zero, and zero is a physically valid irradiance | no-value, keyed on `GLW == 0` | 1087 rows × 20 columns |
| 4 | Until 2022-10-07 the row was 47 fields with `Swup_calc`/`Lwup_calc` at its **end** and `nan` in their named columns | moved into place before the repairs | 4656 |

Defect 2 is why the record carried 322 rows -- 314 distinct hours -- above 100%
relative humidity: `e`
was 1.57% too high and `ur` 1.23 percentage points too high throughout. After the
repair the record's maximum is 99.80% and no hour exceeds saturation.

Shortening the header alone was never an option: pandas refuses a row wider than
its header, so the 47-field rows and a 35-name header cannot coexist. That is
why this is a file migration and not a header edit. Every reader of the record
goes through `wrf.operational_record.read_wrf_series`, which calls
`rename_v1_columns`, so a file still on v1 is read under the v2 names and no
consumer needs to know which schema is on disk.

### Monitoring window artifacts (`labmim-monitoring`)

Producer for the `labmim-monitoring-v1` document the site's **interactive**
monitoring page fetches, the counterpart of `labmim-climatology` for the rolling
window rather than the whole record.

```bash
labmim-monitoring -i output/archive -o ../site-labmim/site/Monitoramento \
    -w data/series/labmim_series_operacional.dat
```

**It reads the archive; it does not build it.** `-i` points at the directory
`labmim-archive` wrote, and the command loads `station_5min_qc.parquet` and
`station_hourly.parquet` from there — never `station_5min_raw.parquet`, so no
ungated sample reaches the page. The window is anchored on the newest sample IN
THAT ARCHIVE, not on the server clock, which makes the order mandatory for any
scheduled run:

For an hourly job, re-merging ten years to publish seven days is the wrong unit
of work. `labmim-archive --source` builds the window straight from the tables the
datalogger is writing now, skipping the manifest and the audit comparison, and
running the SAME quality-control chain — there is no second copy of it:

```bash
labmim-archive --source data/LBM_lenta_2025.dat --source data/LBM_rain_2025.dat \
    -d data -o /var/tmp/labmim-janela
labmim-monitoring -i /var/tmp/labmim-janela -o ../site-labmim/site/Monitoramento
```

Both `--source` files matter: the rain table is where `precip` comes from, and
the precipitation card is one of the nine. `-d` still points at `data/` because
the shade-ring factors are read from the solar-geometry table there.

That table ships as a 203 MB CSV whose five leading integer columns spell out a
timestamp, and the pipeline reads six of its twenty columns. Rewritten once in
the shape the rest of the archive uses — one `DatetimeIndex` in naive local time,
`float64`, zstd — it is 29 MB and answers in 0.037 s against 1.42 s:

```bash
uv run python scripts/converter_teorica.py --data data
```

`load_shade_ring_factors` prefers `teorica_2016-2030.parquet` when it sits beside
the CSV and falls back to the CSV otherwise, so the step is an optimisation and
never a prerequisite. The CSV stays the source of truth and is not modified. Run
it once per machine, and again whenever the lab replaces the CSV — on the server
it takes the window build from 3.6 s to **2.2 s** and its peak from 563 MB to
473 MB, and leaves the published archive byte for byte identical.

`float64` rather than `float32` is deliberate. Single precision would shrink the
file by a further 3 MB and introduce a 5.4e-08 relative error in `fc`, which
multiplies the measured diffuse — every corrected value in the archive would move.

Measured on this archive, the window pair costs **3.6 s and 0.9 s with a 563 MB
peak**, against **18.9 s and 0.9 s with a 4.7 GB peak** for the full rebuild —
and the payload is identical field for field. The live tables carry 458 days,
far more lead-in than the three hours `mask_persistent_runs` needs to judge a
rail that began before the window.

The full rebuild stays the right command for the record itself — it is what runs
the manifest, the audit comparison and the whole-record detectors — but it
belongs on a daily or on-demand schedule, not hourly:

```bash
labmim-archive -d data -o output/archive --strict
```

The window mode skips what a seven-day slice cannot answer: `verify_frame`
compares against constants measured over the whole record, and the absent-column
warning is expected there, since the live table carries 35 columns against the
88 the historical eras used together. Whatever the mode, running
`labmim-monitoring` without rebuilding first regenerates the payload from the
same seven days for ever: the page keeps rendering, the timestamps never
advance, and nothing errors.

The QC is not re-applied here; it is inherited, and it is genuinely in the window
rather than only in the history. Measured over one seven-day window, the raw
barometer carried 2,017 samples and 436 survived to the payload.

One JSON carries all nine charts in the three layers the researcher asked for —
the five-minute samples, the hourly means over them, and the WRF series where
the model has that variable. The five-minute layer is labelled "raw" for its
RESOLUTION, not its provenance: it is `station_5min_qc`, after every gate. About 133 kB for a fully instrumented week, against
the ~380 kB of PNGs it replaces, and it arrives as numbers the reader can hover,
toggle and download. The time axis is published as `start` + `step_minutes` +
`count` instead of one stamp per sample; that alone is worth ~50 kB.

The WRF column is resolved per series against an **ordered tuple of candidate
names** (`sensors/monitoring.py`). `series_operacional.dat` gains variables over
time, so the payload records which names were looked for and the page says the
layer is missing instead of showing a legend that is silently one entry short.
Precipitation is the case that exercised the mechanism: `labmim-wrf-series` now
writes a `precip_mm` column and the chart picked it up with no change here, while
every hour before that column existed still reads as absent.

The `window` object carries four fields, and two of them are easy to confuse:

| field | meaning |
|-------|---------|
| `start` | first instant the document covers. **Authoritative** — do not re-derive it |
| `end` | last instant the document CARRIES, which is not necessarily the newest observation |
| `station_end` | newest station sample inside the window; the anchor for a "last N days" view |
| `days` | the rolling length that was requested, i.e. the `--days` option |

`end` reaches past `station_end` whenever the model runs ahead of the station,
which is the normal state once the operational extraction accumulates forward: an
hour with a WRF value and no observation is expected, not a fault, and clipping
the window at the newest sample would cut exactly the part of the forecast worth
looking at. A consumer that anchors a "last 7 days" view on `end` therefore
pushes real observations out of view — anchor the **start** on `station_end` and
the **axis maximum** on `end`. `days` is the requested length and reproduces
`start` only on the default path; under an explicit `--end` the two are
independent, so `start` is the field to trust.

A series whose column is present but holds no value over the window is **omitted**
from its layer rather than published as an array of `null`, and a layer left with
no series at all is published as `null`. The page reads an absent key as "nothing
to draw"; an all-null array instead builds a dataset and puts an entry in the
legend for a line the reader cannot see. This is not hypothetical — the Gill
thermohygrometer railed in December 2025, so air temperature and humidity are
empty in every current window.

Like the climatology artifacts, the output is **not** committed to the site
repository: same private archive, same deploy-time attachment.

### Monitoring-page graphs (site)

These are the **static** PNGs. They are not superseded by the interactive page:
they stay because they are what goes into papers, and they draw the same three
layers so the two products can be read the same way.

```bash
# Nine fixed-name PNGs for the site's monitoring page, straight into a checkout
labmim-site-graphs site -i data/hourly/sensor_data.csv \
    -o ../site-labmim/site/assets/graphs --last-days 7

# The same nine in three layers: raw under the hourly mean, model on top
labmim-site-graphs site -i data/hourly/sensor_data.csv -o out/ \
    --raw output/archive/station_5min_qc.parquet --wrf data/series/labmim_series_operacional.dat

# Retarget a renamed logger column without editing code
labmim-site-graphs site -i data/hourly/sensor_data.csv -o out/ \
    --col temperatura=AirT2_C_Avg

# Ad-hoc per-variable graphs (generic secondary command, legacy filenames)
labmim-site-graphs columns -i data/hourly/sensor_data.csv -o out/ -v AirT1_C_Avg -v RH1
```

See [Monitoring page (site-labmim)](#monitoring-page-site-labmim) for the full
nine-image contract and the operational command sequence.

### Comparison & metrics

```bash
# Full comparison with plots
labmim-comparison --obs observed.csv --model modeled.csv --output comparison/

# Metrics between any two datasets
labmim-metrics -a salvador.dat -b rio.dat -o metrics.csv
```

---

## Front-end integration (site-labmim)

`labmim-wrf-geojson` is the **producer** for the LabMiM public WebGIS
(`site-labmim`). The two repositories share a byte-level file contract: the
exporter writes fixed-name artifacts, the static site fetches them by those
exact names, and the daily job overwrites them in place. This section is the
authoritative description of that contract.

### Producer → consumer map

| Producer writes (this repo)            | Site reads (`site-labmim`)                 | Consumed by | Status |
|---|---|---|---|
| `JSON/manifest.json`                   | `JSON/manifest.json` (`cache: "no-cache"`) | `map-init.js` → `applyManifest` | live |
| `GeoJSON/{domain}.grid.json`           | `GeoJSON/{domain}.grid.json`               | `map-manager.loadGridLayer` (primary) | live |
| `GeoJSON/{domain}.geojson`             | `GeoJSON/{domain}.geojson`                 | grid loader fallback + `charts` cell lookup | live (fallback) |
| `JSON/{domain}_{variableId}_{NNN}.json`| same                                        | `map-manager.loadValueData` (the map raster) | live |
| `JSON/{domain}_WIND_VECTORS_{NNN}.json`| same                                        | `map-manager.renderWindVectors` (arrow overlay) | live |
| `JSON/{domain}_ISOBARS_{NNN}.json`     | same                                        | `map-manager.renderIsobars` (line overlay); the variables to draw them over are in `features.isobar_overlay.draw_over`, the legend text comes from `metadata.interval` | live |
| `JSON/{domain}_{variableId}.summary.json` | via `features.domain_summary.template`   | `charts-manager._loadSummaryArtifactSeries` (domain preview) | live |
| `JSON/{domain}_{variableId}.series.bin`   | via `features.cell_series.template`      | `charts-manager._loadCellSeriesFromBinary` (cell modal, HTTP Range) | live |

The site expects the value payloads under `site/JSON/` and the grid files under
`site/GeoJSON/`; deploying a run is copying the exporter's `-o`/`-g` outputs
into those two directories. Every artifact the exporter emits is consumed by
the current front-end. (`site/assets/json/` is an unrelated empty placeholder,
not the manifest location.)

### Variable-id source of truth

The `{variableId}` tokens are the string values of `VARIABLE_NETCDF_MAP` in
`src/micrometeorology/common/types.py` (`TEMP`, `PRES`, `WIND`, `RAIN`,
`VAPOR`, `TSK`, `RH2`, `HFX`, `LH`, `SWDOWN`, `GLW`,
`WIND_POWER_DENSITY_10M`), plus the poteolico expansion
(`POT_EOLICO_50M/100M/150M`) and the two standalone overlays, `WIND_VECTORS`
and `ISOBARS`. On the consumer side the shaded ids are the
`id`/`id_100m`/`id_150m` fields of `VARIABLES_CONFIG` in
`site/assets/js/variables-config.js` — the front-end's registry and the single
source of truth for which ids the map can request as a base layer. The default
exporter variable set (`DEFAULT_VARS`) covers that registry exactly, and adds
the two overlays, which are drawn OVER a base layer and are therefore not
members of it.

### Guarantees the site relies on

- **Shape** — `values` is the row-major flattened grid; cell order is defined by
  the grid file's `linear_index` (`k = row·n_cols + col`) and is shared by the
  value JSON, the `.series.bin` rows, and both grid encodings.
- **Units** — one physical unit per variable, matching the exporter docstrings
  in `wrf/variables.py` and the `unit` field of the site registry.
- **Non-finite** — never emitted; masked cells are `null` in value JSON and the
  `-2147483648` sentinel in `.series.bin`, and an all-NaN field fails its unit
  instead of shipping invalid JSON (see "Non-finite rejection" above).
- **Rounding** — values are rounded to two decimals everywhere; `.series.bin`
  encodes `value × 100` as int32 (`scale: 0.01`), so the binary and per-step
  views always agree.
- **Timezone** — `America/Bahia` (UTC−03:00, no DST); `manifest.start_local`
  anchors file index `0`. The front-end currently hardcodes the UTC−03:00 label
  and does not read `manifest.timezone`, so changing `LABMIM_TIMEZONE` away from
  a −03:00 zone would desync its time labels.

### Cache semantics (why the JSON is a byte contract)

The pipeline reuses the **same filenames every run** and overwrites them in
place. The site never renames on deploy, so cache invalidation rides entirely
on the manifest:

- `manifest.json` is fetched `no-cache` (always revalidated) and re-checked
  every ~15 min and on tab refocus.
- `manifest.version` is appended as `?v=` to every data URL, letting the browser
  cache the fixed-name files long-term while a new run (new `version`)
  cache-busts them all at once.
- When a session detects a changed `version` it drops every cached payload,
  chart series, and grid layer keyed on the old bytes and re-anchors the
  timeline — so a page left open across the daily regeneration never mixes two
  runs. This is why the output must be a stable byte contract: identical names,
  identical shapes, one version stamp per round.

### Refreshing the site's data from a wrfout file

From this repo, export straight into a checkout of the site (read-only sibling
`../site-labmim` shown here; adjust the path):

```bash
labmim-wrf-geojson \
    --wrf-dir /path/to/wrfout/ --date 20260503 \
    -D 1 -D 2 -D 3 -D 4 \
    -o ../site-labmim/site/JSON \
    -g ../site-labmim/site/GeoJSON \
    --workers 44
```

This writes the per-step value JSONs, `WIND_VECTORS`, the `.geojson` +
`.grid.json` grids, the `.series.bin`/`.summary.json` consolidated artifacts,
and the v2 `manifest.json` — the complete set the WebGIS consumes. A single
`wrfout` file is fine too: `-d /path/to/wrfout_d03_2026-05-03_00_00_00`
(the domain is read from the filename). Omit `--date` to batch every `wrfout*`
file in `--wrf-dir`, restricted to `--domains` when given (subdirectories named
`wrfout*` are skipped). A `--date` that is not at least an 8-digit day is
refused rather than reported as a day WRF produced nothing for; separators are
tolerated, so `--date 2026-05-03` and `--date 20260503` are the same request.

An empty or partial selection warns and still exits `0`, so a cron chain
survives a day whose run is late. Pass `--strict` to turn both "no wrfout
selected" and "a requested `--domains` is missing" into a non-zero exit before
anything is written — the right choice when the JSON feeding the WebGIS must
never silently go stale.

### Monitoring page (site-labmim)

The WebGIS above is not the only consumer. The site's **monitoring page**
(`site/monitoring.html`, `https://labmim.if.ufba.br/monitoring.html`) embeds
nine station graphs by **fixed image name** under `site/assets/graphs/`.
`labmim-site-graphs site` is the producer for those PNGs, reading the hourly CSV
that `labmim-sensor-process` exports and writing exactly the names the page
requests, overwriting them in place each run.

> **This consumer is external.** Nothing in this repository imports
> `plot_station_graphs`, and the site repo is a separate, read-only checkout;
> the coupling is a **cron/manual copy** of PNG files, so it is invisible to any
> reverse-import ("who calls this?") dead-code analysis. Treat the nine
> filenames as a byte-name contract exactly like the WebGIS JSON names above.

**Maintenance rule — sensor swaps rename columns.** The column names below are
defaults matching the current CR5000 program. When a sensor is replaced (or the
logger program renames a channel — e.g. a new anemometer reporting
`WS1_ms_GMX` instead of `WS_ms`), the operator MUST update the mapping via
`--col image=NewColumn` or the `--config` YAML, otherwise the affected graph is
skipped with a warning and the site keeps serving the last generated PNG
indefinitely. Run with `--strict` in cron so a broken mapping fails loudly
instead.

The nine-image contract (`site` command; column names are the defaults and are
overridable via `--config` / `--col`):

| Site image (`assets/graphs/`) | Default CSV column | Plot type | Y-axis label |
|---|---|---|---|
| `temperatura.png`      | `AirT1_C_Avg`  | line                     | Temperatura do Ar (°C) |
| `umidade.png`          | `RH1`          | line                     | Umidade Relativa do Ar (%) |
| `pressao.png`          | `BP1_mbar_Avg` | line                     | Pressão Atmosférica (hPa) |
| `precipitacao.png`     | `PL01_mm_Tot`  | bar                      | Precipitação (mm) |
| `velocidade.png`       | `WS_ms`        | line                     | Velocidade do Vento (m/s) |
| `direcao.png`          | `WindDir`      | scatter (0–360, wraps)   | Direção do Vento (°) |
| `balanco.png`          | `Net_Wm2_Avg`  | line + optional CM3/CG3 components | Balanço de Radiação (W/m²) |
| `radiacao_difusa.png`  | `PSP_Wm2_Avg`  | line                     | Radiação Difusa (W/m²) |
| `radiacao_par.png`     | `PAR_Wm2_Avg`  | line                     | Radiação PAR (W/m²) |

A missing source column logs a warning and skips only that image (exit code
stays `0`); pass `--strict` to fail the run instead. Wind direction is scattered
(not lined) because the series wraps at 360°; when the direct column is absent
but U/V components are present it is reconstructed with
`micrometeorology.sensors.wind`. The optional balance components
(`CM3Up_Wm2_Avg`, `CM3Dn_Wm2_Avg`, `CG3Up_Wm2Cr_Avg`, `CG3Dn_Wm2Cr_Avg`) are drawn
only when present, with the upward channels negated per the legacy convention.

Operational command sequence (ingest → hourly CSV → graphs → copy):

```bash
# 1. Raw .dat -> processed hourly CSV
labmim-sensor-process --input data/raw/ --output data/hourly/sensor_data.csv

# 2. Hourly CSV -> the nine fixed-name PNGs (straight into the site checkout)
labmim-site-graphs site \
    -i data/hourly/sensor_data.csv \
    -o ../site-labmim/site/assets/graphs \
    --last-days 7

# 3. (If step 2 wrote to a staging dir instead) copy into the page's asset dir
#    cp output/site_graphs/*.png ../site-labmim/site/assets/graphs/
```

`-o` may point anywhere; the operational target is the site checkout's
`site/assets/graphs/` (the default staging dir is `output/site_graphs`).

---

## FAQ

### What is the sentinel value (-900)?

It is a default that matches **nothing** in this archive, and leaving it on only
creates the impression that missing data has been handled. `read_campbell_dat`
turns values ≤ `sensor_sentinel_value` into NaN, but this station's loggers rail
at 1000 °C, 999 %RH, −46.8, −273.1, −7999, −6673 and a windowed 0 — all of them
above −900, all of them finite, and every one of them a plausible-looking number
to a filter that only checks `np.isfinite`.

The rails that are actually caught are per column and per era, in
`SENTINEL_RANGES` / `mask_sentinels`
(`micrometeorology.sensors.archive`), which `labmim-archive` and
`allsky prepare-local` both apply after reading. Setting
`sensor_sentinel_value: null` turns off a guard that only looks like one; it
changes nothing about what reaches the archive.

### Why does configuration have 4 layers?

To support different environments without code changes:

| Layer | Purpose |
|---|---|
| `default.yaml` | Default values for local development |
| `<env>.yaml` | Production server config (`LABMIM_ENV=server`) |
| `LABMIM_CONFIG_PATH` | Full override (e.g. for tests) |
| Environment variables | Specific value overrides in CI/CD |

### Can I use WRF processing on Windows?

Yes. All NetCDF processing works on both Windows and Linux. Dependencies (`netCDF4`, `cartopy`) are cross-platform. WRF itself typically runs on Linux, but its output files (NetCDF) can be processed on any OS.

For test runs on Windows, prefer a pytest temporary directory outside OneDrive
and outside a corrupted `AppData\Local\Temp\pytest-of-<user>` tree:

```powershell
$env:LABMIM_PYTEST_TMP = "$env:LOCALAPPDATA\labmim-pytest"
New-Item -ItemType Directory -Force $env:LABMIM_PYTEST_TMP
pytest -n auto -v tests --basetemp $env:LABMIM_PYTEST_TMP
```

The xdist-safe tests use per-test temporary files, so parallel workers should
not share mutable YAML/config fixtures.

### How do I add a new sensor?

1. The datalogger already generates the new column in the `.dat` file
2. Ingestion recognises the new column automatically (no code change)
3. If the sensor needs calibration, add a new record in `calibrations.yaml`
4. If it needs physical limits, add them in `default.yaml` under the limits section

### What happened to Basemap?

Basemap is deprecated and no longer maintained. All map generation now uses **Cartopy**, which is actively maintained and does not require a separate conda environment. The visual output matches the legacy maps.

### What is `batch.py`?

The parallel figure-rendering engine. It builds `FigureTask` frames, spills their arrays to temporary `.npy` files, and renders them on a persistent process pool. JSON generation lives in `jobs.py`, where each worker opens the NetCDF itself and writes its files directly — no array payloads cross process boundaries at all.

### Safe WRF execution guardrails

WRF operations fail early when a planned array allocation exceeds the configured memory guardrail. The default single-operation limit is `16 GiB` and can be adjusted with `LABMIM_MAX_ARRAY_GB`. Worker processes are recycled every `64` tasks by default; set `LABMIM_MAX_TASKS_PER_CHILD=0` to disable or raise it if worker startup dominates. Wind-potential extraction streams the 4D fields in ~64-step blocks, so peak worker memory stays bounded regardless of how many timesteps a file has.

Staggered WRF dimensions are destaggered positionally before derived calculations (`U/V` wind speed, `PH+PHB` heights above terrain), so no label alignment ever occurs.

Recommended server commands:

```bash
labmim-wrf-geojson --wrf-dir /data/wrf --date 20240101 --domains 1,4 \
  --variables temperature,wind,rain,wind_vectors --workers 8 \
  -o output/JSON -g output/GeoJSON

labmim-wrf-figures --wrf-dir /data/wrf --date 20240101 --domains 3 \
  --variables temperature,wind,SWDOWN --workers 8 -o output/figures
```

Architecture remains modular:

- `reader.py` owns eager NetCDF access (whole variables and time blocks) and path resolution.
- `safety.py` owns shape, dtype, memory, staggered-grid, and worker-payload guardrails.
- `variables.py` owns physical WRF diagnostics and derived variables.
- `interpolation.py` owns vertical interpolation (`VerticalInterpolator` bracket fast path with an argsort fallback).
- `geojson.py` owns grid/value serialization and writes large outputs incrementally.
- `batch.py` owns worker execution and payload transport.
- CLI modules compose those layers and now flush bounded task batches instead of retaining the full run in memory.

Large JSON/GeoJSON outputs are streamed:

- Grid GeoJSON is written feature-by-feature; `save_geojson()` no longer builds a full `FeatureCollection` feature list in the file-output path.
- Per-timestep value JSON is written in chunks of `65,536` flattened cells; the file format remains `{"metadata":...,"values":[...]}` but the Python process no longer holds the entire values list.
- The legacy in-memory helpers `create_grid_geojson()` and `create_values_json()` remain useful for tests and small arrays, but server workflows should use the file writers through the CLIs.
