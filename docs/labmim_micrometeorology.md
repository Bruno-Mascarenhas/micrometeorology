# `labmim_micrometeorology` — Documentation

Environmental data processing toolkit for the Micrometeorology Laboratory (LabMiM) at UFBA.

---

## Overview

The `labmim_micrometeorology` package provides a complete infrastructure for:

1. **Sensor data ingestion** — flexible reading of Campbell Scientific `.dat` files with dynamic headers
2. **Calibration** — immutable historical calibration records with date-range application
3. **Temporal aggregation** — high-frequency to hourly resolution with vector-mean wind direction
4. **WRF processing** — NetCDF reading, Cartopy map rendering, GeoJSON export, vertical interpolation
5. **Parallel batch rendering** — `ProcessPoolExecutor`-based parallel figure and JSON generation (30–60× speed-up)
6. **Statistics** — RMSE, MAE, MBE, R², correlation, Willmott d-index, IOA, NRMSE

---

## Package Structure

```
src/labmim_micrometeorology/
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
    └── geojson.py           # GeoJSON + value JSON export for the site
```

---

## Installation

```bash
# Micrometeorology only:
pip install -e "."

# With development dependencies:
pip install -e ".[dev]"

# With video generation (moviepy):
pip install -e ".[video]"
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
from labmim_micrometeorology.common.config import get_settings

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
from labmim_micrometeorology.sensors.ingestion import read_campbell_dat, merge_dat_files

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
from labmim_micrometeorology.sensors.calibration import load_calibrations, apply_calibrations

cals = load_calibrations("configs/micromet/calibrations.yaml")
df = apply_calibrations(df, cals)
```

> ⚠️ **Never edit** existing calibration records. Always **append new** records for new periods.

### 4. Temporal Aggregation

```python
from labmim_micrometeorology.sensors.aggregation import aggregate_to_hourly

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
from labmim_micrometeorology.stats.metrics import compute_all, rmse, mae

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
from labmim_micrometeorology.wrf.batch import (
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
from labmim_micrometeorology.stats.comparison import (
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

### GeoJSON/JSON for Interactive WebGIS (Primary)

```bash
labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o site/JSON -g site/GeoJSON \
    -v temperature -v wind --workers 44
```

### Figures (Static Maps & Video - Legacy)

> Note: The interactive site is now the primary map output. Use this command only if you specifically need `.png` images or `.webm` videos.

```bash
# Single domain
labmim-wrf-figures -d wrfout_d03_2024-01-01 -o output/figures/ -v temperature -v wind

# Multiple domains with videos
labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain -v SWDOWN \
    -o output/figures/ --workers 44 --also-video
```

### Local testing (all-in-one)

```bash
python scripts/micromet/run_wrf_local.py \
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

### How do I add a new sensor?

1. The datalogger already generates the new column in the `.dat` file
2. Ingestion recognises the new column automatically (no code change)
3. If the sensor needs calibration, add a new record in `calibrations.yaml`
4. If it needs physical limits, add them in `default.yaml` under the limits section

### What happened to Basemap?

Basemap is deprecated and no longer maintained. All map generation now uses **Cartopy**, which is actively maintained and does not require a separate conda environment. The visual output matches the legacy maps.

### What is `batch.py`?

The core parallel rendering engine. Instead of rendering frames serially in a loop, it pre-extracts all data, builds lightweight `FigureTask` tuples, and dispatches them to a `ProcessPoolExecutor`. Each worker uses the `Agg` matplotlib backend (no GUI) and a single `pcolormesh` call (instead of the legacy double `contourf` + `pcolor`).
