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
├── cli.py                   # Console entry points (registered in pyproject.toml)
├── common/
│   ├── config.py            # Centralised config (pydantic-settings + YAML, 4 layers)
│   ├── logging.py           # Structured logging setup
│   ├── paths.py             # Cross-platform path utilities (pathlib)
│   └── types.py             # Enums (WRFVariable, GridLevel D01–D05), dataclasses, constants
├── sensors/
│   ├── ingestion.py         # .dat reading with dynamic headers
│   ├── calibration.py       # Date-precise calibration (immutable historical records)
│   ├── aggregation.py       # Hourly aggregation with vector-mean wind direction
│   ├── wind.py              # U/V decomposition and vector-mean direction
│   └── export.py            # Formatted CSV export
├── stats/
│   ├── metrics.py           # Model vs. observation metrics (RMSE, MAE, etc.)
│   ├── comparison.py        # Full comparison pipeline: alignment + metrics + plots
│   ├── climatology.py       # Diurnal, monthly, and seasonal groupings
│   └── radiation.py         # Clearness index (Kt) and diffuse fraction (Kd)
└── wrf/
    ├── reader.py            # NetCDF dataset wrapper (WRFDataset context manager)
    ├── variables.py         # Variable extraction and unit conversion
    ├── plotting.py          # Cartopy-based map rendering (replaces Basemap)
    ├── batch.py             # Parallel rendering engine (ProcessPoolExecutor)
    ├── animation.py         # PNG → WebM / GIF creation (parallel batch support)
    ├── interpolation.py     # Vectorised vertical interpolation (replaces wrf-python)
    ├── series.py            # Point time-series extraction from gridded data
    └── geojson.py           # GeoJSON + value JSON export
```

---

## Installation

```bash
# Micrometeorology only:
uv pip install -e "."

# With development dependencies:
uv pip install -e ".[dev]"

# With video generation (moviepy):
uv pip install -e ".[video]"

```

For local development, prefer activating the existing `labmim` Conda
environment first and then running the `uv pip` commands inside it. Conda keeps
native scientific binaries stable; `uv` speeds up dependency resolution and
editable installs. On Windows, set `UV_PYTHON` to the active Conda interpreter
first:

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
print(settings.data_dir)        # Path to data
print(settings.output_dir)      # Path to output
```

Environment variables use the `LABMIM_` prefix:

```bash
export LABMIM_DATA_DIR=/mnt/data/labmim
export LABMIM_ENV=server
```

### 2. Sensor Data Ingestion

```python
from micrometeorology.sensors.ingestion import read_campbell_dat, merge_dat_files

# Single file
df = read_campbell_dat("data_2023.dat")

# Multiple files (headers may differ between them)
df = merge_dat_files([
    "data_2023_jan.dat",
    "data_2023_feb.dat",
    "data_2023_mar.dat",
])
```

#### Why do headers vary?

The Campbell Scientific datalogger allows sensors to be added or removed at any time. When a sensor is added, a new column appears in the `.dat`; when removed, the column disappears. `read_campbell_dat()` handles this automatically:

- Missing columns are ignored (no error)
- Extra columns are included automatically
- `merge_dat_files()` performs an ordered merge across all columns

### 3. Calibration

Calibrations are **immutable historical facts**. Each record specifies:

```yaml
# configs/micromet/calibrations.yaml
calibrations:
  - column: CM3Up
    start_date: "2018-11-01"
    end_date: "2019-06-30"
    factor: 1.0526
    description: "Post-maintenance calibration Nov/2018"

  - column: CM3Up
    start_date: "2019-07-01"
    end_date: null      # null = until end of data
    factor: null         # null = invalid data for this period → NaN
    description: "Sensor malfunction"
```

```python
from micrometeorology.sensors.calibration import load_calibrations, apply_calibrations

cals = load_calibrations("configs/micromet/calibrations.yaml")
df = apply_calibrations(df, cals)
```

> ⚠️ **Never edit** existing calibration records. Always **append new** records for new periods.

### 4. Temporal Aggregation

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
    FigureTask, build_map_config, default_workers, run_figure_tasks,
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
    read_dataset, pair_dataframes, compare_all_variables,
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
JSON/{domain}_{variableId}.series.bin # per-cell time-series (int32 matrix)
JSON/{domain}_{variableId}.summary.json  # per-step domain mean/min/max
JSON/manifest.json                    # run manifest (v2)
```

`{domain}` is `D01`–`D05`, `{NNN}` is the zero-padded three-digit time-step
index (the front-end reads it as the forecast hour), and `{variableId}` is the
output file suffix from `VARIABLE_NETCDF_MAP` (`micrometeorology.common.types`).

The last four artifacts are the **consolidated site artifacts** written by
default; pass `--no-site-artifacts` to emit only the per-step value JSONs, the
grid files, and a v1 manifest. `--skip-first N` drops the first `N` spin-up
time steps (their indices become gaps in the timeline).

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
  `WIND_VECTORS` and the grid GeoJSON do not.

Supported site-oriented variables include the legacy fields `TEMP`, `PRES`,
`VAPOR`, `RAIN`, `WIND`, `SWDOWN`, `HFX`, `LH`, and the wind-potential files
`POT_EOLICO_50M`, `POT_EOLICO_100M`, and `POT_EOLICO_150M` generated by
`poteolico`. Additional 2026 WRF fields include `TSK`, `RH2`, `GLW`, and
`WIND_POWER_DENSITY_10M`. Units, formulas, and limitations are documented in
the extractor docstrings in `src/micrometeorology/wrf/variables.py`.

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
value JSONs. `start_local` always pairs with file index `0`, even when
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
labmim-sensor-process --input data/raw/ --output data/hourly/
```

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
(`POT_EOLICO_50M/100M/150M`) and the standalone `WIND_VECTORS`. On the consumer
side the same ids are the `id`/`id_100m`/`id_150m` fields of `VARIABLES_CONFIG`
in `site/assets/js/variables-config.js` — the front-end's registry and the
single source of truth for which ids the map can request. The default exporter
variable set (`DEFAULT_VARS`) matches this registry exactly.
`data/variables-config.js` in this repo is a **reference copy only** — nothing
in the pipeline reads it, and it can lag behind the site's live registry.

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
in `--wrf-dir`.

---

## FAQ

### What is the sentinel value (-900)?

The Campbell Scientific datalogger uses -900 (or similar) to indicate missing or invalid data. The ingestion module automatically converts all values ≤ sentinel to NaN.

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
  --variables temperature wind rain wind_vectors --workers 8 \
  -o output/JSON -g output/GeoJSON

labmim-wrf-figures --wrf-dir /data/wrf --date 20240101 --domains 3 \
  --variables temperature wind SWDOWN --workers 8 -o output/figures
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
