# LabMiM Micrometeorology & Solar Radiation Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced scientific computing suite for atmospheric science research, maintained by the Micrometeorology Laboratory (LabMiM) at UFBA. This repository unifies data processing, geospatial visualization, and machine learning pipelines into two core packages:

| Package | Purpose |
|---|---|
| **`labmim_micrometeorology`** | High-performance WRF output analysis, interactive WebGIS grid generation, sensor data ingestion, and statistical climatology |
| **`solrad_correction`** | Machine learning pipeline for bias correction of WRF diffuse solar radiation (SVM, LSTM, Transformer architectures) |

> 📘 Detailed technical documentation for each package is located in the [`docs/`](docs/) directory.

---

## Repository Architecture

```
src/
├── labmim_micrometeorology/      # Core data pipelines, WRF spatial processing, and APIs
│   ├── common/                   # Cross-platform config, logging, types
│   ├── sensors/                  # Datalogger ingestion and calibration algorithms
│   ├── stats/                    # Statistical comparison and climatological metrics
│   └── wrf/                      # High-speed NetCDF parsing, GeoJSON export, interpolation
└── solrad_correction/            # Deep learning bias correction pipeline
    ├── models/                   # Neural architectures (SVM, LSTM, Transformer)
    ├── training/                 # PyTorch training loops and early stopping
    └── evaluation/               # Experiment reporting and validation metrics

configs/                          # YAML environments for pipelines and ML experiments
scripts/                          # CLI automation tools and Bash workflows
site/                             # High-performance Vanilla JS WebGIS Interactive Maps
tests/                            # Comprehensive Pytest suite
docs/                             # In-depth package documentation
legacy/                           # Archived Cartopy/Basemap scripts
```

---

## Installation

Requires **Python ≥ 3.11**.

### Base Scientific Suite

```bash
pip install -e "."
```

### Machine Learning Environment (CPU)

```bash
pip install -e ".[tcc]"
```

### Machine Learning Environment (CUDA)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[tcc-cuda]"
```

### Development Environment

```bash
pip install -e ".[dev,tcc,video]"
```

---

## Quick Start

### 1. Generate Interactive WebGIS Grids (Primary Workflow)

The main visualization output for LabMiM is the interactive site frontend. This command extracts spatial data from WRF NetCDF outputs into highly optimized JSON payloads for the web interface.

```bash
labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o site/JSON -g site/GeoJSON --workers 44
```

### 2. Sensor Data Processing & Calibration

```bash
labmim-sensor-process --input data/raw/ --output data/hourly/
```

### 3. Statistical Comparison (WRF vs Observations)

```bash
labmim-metrics -a salvador.dat -b rio.dat -o metrics.csv
```

### 4. Machine Learning Bias Correction

```bash
solrad-run --config configs/tcc/experiments/svm_hourly.yaml
```

### 5. Static Cartopy Map Generation [Legacy Workflow]

If you need static `.png` maps or `.webm` animations for publications instead of the interactive website, you can use the parallel batch renderer:

```bash
labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain -v SWDOWN \
    -o output/figures/ --workers 44 --also-video
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
