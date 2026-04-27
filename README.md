# LabMiM Micrometeorology & Solar Radiation Intelligence

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An advanced scientific computing suite for atmospheric science research, maintained by the Micrometeorology Laboratory (LabMiM) at UFBA. This repository unifies data processing, geospatial visualization, and machine learning pipelines into two core packages:

| Package | Purpose |
|---|---|
| **`micrometeorology`** | High-performance WRF output analysis, data export, sensor data ingestion, and statistical climatology |
| **`solrad_correction`** | Machine learning pipeline for bias correction of WRF diffuse solar radiation (SVM, LSTM, Transformer architectures) |

> 📘 Detailed technical documentation for each package is located in the [`docs/`](docs/) directory.

---

## Repository Architecture

```
src/
├── micrometeorology/      # Core data pipelines, WRF spatial processing, and APIs
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

Dask-backed xarray chunking is included in the base dependencies for large WRF
workloads.

---

## Quick Start

### 1. Export WRF Grid Data (Primary Workflow)

This command extracts spatial data from WRF NetCDF outputs into highly optimized JSON and GeoJSON payloads for external visualization tools.

```bash
labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -o output/JSON -g output/GeoJSON --workers 44
```

WRF CLIs now default to adaptive `auto` performance mode. Small or single-worker
jobs stay on the eager reader and serial JSON writer; larger multi-worker
exports can resolve to the lazy reader and memmap payload references.
When dask-backed xarray chunking is unavailable, auto mode keeps lazy reads
unchunked instead of failing.

To force the previous fixed behavior:

```bash
labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \
    -o output/JSON -g output/GeoJSON \
    --reader eager --chunks none --worker-backend pickle
```

For large exports, you can still force lazy + memmap explicitly:

```bash
labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \
    -o output/JSON -g output/GeoJSON \
    --reader lazy --worker-backend memmap --tmp-dir scratch/wrf-json
```

Static map rendering also supports memmap-backed figure payloads:

```bash
labmim-wrf-figures --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \
    -o output/figures --reader lazy --chunks auto \
    --worker-backend memmap --tmp-dir scratch/wrf-figures
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

Set `model.evaluation_policy: common_sequence_horizon` when comparing SVM
against LSTM/Transformer metrics on the same sequence target horizon. The
default `model_native` policy preserves previous metric semantics.

### 5. Static Cartopy Map Generation

If you need static `.png` maps or `.webm` animations for publications, you can use the parallel batch renderer:

```bash
labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \
    -D 1 -D 4 -v temperature -v wind -v rain -v SWDOWN \
    -o output/figures/ --workers 44 --also-video
```

---

## Testing

```bash
pytest tests/ -v          # all the tests
pytest tests/micromet/    # micrometeorology only
pytest tests/tcc/         # ML correction only
ruff check src/ tests/    # lint
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/micrometeorology.md`](docs/micrometeorology.md) | Sensor ingestion, calibration, aggregation, WRF parallel pipeline, batch rendering, statistics, CLI reference, FAQ |
| [`docs/solrad_correction.md`](docs/solrad_correction.md) | Model types (SVM/LSTM/Transformer), experiment configs, transfer learning, data leakage prevention, feature engineering, FAQ |

---

## License

MIT
