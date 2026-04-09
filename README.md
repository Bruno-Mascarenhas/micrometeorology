# Micrometeorology Monorepo

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Monorepo containing two Python packages for atmospheric science research:

| Package | Purpose |
|---|---|
| **`labmim_micrometeorology`** | Sensor data ingestion, WRF output analysis, parallel figure generation, GeoJSON export, and statistical comparison |
| **`solrad_correction`** | ML-based bias correction of WRF solar radiation (SVM, LSTM, Transformer) |

> 📘 Detailed per-package documentation lives in [`docs/`](docs/).

---

## Repository Layout

```
src/
├── labmim_micrometeorology/      # Sensor ingestion, calibration, WRF maps, metrics
│   ├── common/                   # Cross-platform config, logging, types
│   ├── sensors/                  # Campbell Scientific .dat processing
│   ├── stats/                    # Metrics, comparison, climatology, radiation indices
│   └── wrf/                      # NetCDF reader, Cartopy plotting, parallel batch, GeoJSON, animation
└── solrad_correction/            # ML correction pipeline (config → train → evaluate)
    ├── models/                   # SVM, LSTM, Transformer (BaseRegressorModel interface)
    ├── training/                 # Trainer, progress tracking, early stopping
    └── evaluation/               # Metrics, reports, experiment comparison

configs/
├── micromet/                     # Sensor & WRF settings, calibrations
└── tcc/experiments/              # Experiment YAML configs

scripts/
├── micromet/                     # CLIs: process_wrf_figures, process_wrf_geojson, run_wrf_local, etc.
└── wrf/                          # Bash automation for workstations

tests/                            # 53 tests (micromet + tcc)
docs/                             # Per-package documentation
legacy/                           # Archived original scripts (Basemap-era)
```

---

## Installation

Requires **Python ≥ 3.11**.

### Micrometeorology only

```bash
pip install -e "."
```

### Micrometeorology + video generation (WebM)

```bash
pip install -e ".[video]"
```

### Micrometeorology + ML correction (CPU)

```bash
pip install -e ".[tcc]"
```

### Micrometeorology + ML correction (CUDA)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[tcc-cuda]"
```

### Development (adds pytest, ruff, mypy, pre-commit)

```bash
pip install -e ".[dev]"
# or together:
pip install -e ".[dev,tcc,video]"
```

### Cartopy shapefiles

Map generation requires Natural Earth data (downloaded on first use):

```bash
python -c "import cartopy; cartopy.io.shapereader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')"
```

---

## Quick Start

### WRF parallel figure generation

```bash
# Single domain
labmim-wrf-figures -d wrfout_d03_2024-01-01 -o output/figures/ -v temperature -v wind

# Multiple domains (4 files → ~2300 figures in parallel)
labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain -v SWDOWN \
    -o output/figures/ --workers 44 --also-video
```

### GeoJSON/JSON for the site

```bash
labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o site/JSON -g site/GeoJSON --workers 44
```

### Local testing (all-in-one)

```bash
python scripts/micromet/run_wrf_local.py \
    --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o output/wrf_local/ --also-video
```

### Sensor data processing

```bash
labmim-sensor-process --input data/raw/ --output data/hourly/
```

### Metrics between datasets

```bash
labmim-metrics -a salvador.dat -b rio.dat -o metrics.csv
```

### Run an ML experiment

```bash
solrad-run --config configs/tcc/experiments/svm_hourly.yaml
```

---

## Testing

```bash
pytest tests/ -v          # all 53 tests
pytest tests/micromet/    # micrometeorology only
pytest tests/tcc/         # ML correction only
ruff check src/ tests/    # lint
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/labmim_micrometeorology.md`](docs/labmim_micrometeorology.md) | Sensor ingestion, calibration, aggregation, WRF parallel pipeline, batch rendering, statistics, CLI reference, FAQ |
| [`docs/solrad_correction.md`](docs/solrad_correction.md) | Model types (SVM/LSTM/Transformer), experiment configs, transfer learning, data leakage prevention, feature engineering, FAQ |

---

## License

MIT
