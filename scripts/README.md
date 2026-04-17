# LabMiM Automation Scripts

This directory contains various convenience scripts and automation bash wrappers that orchestrate the core Python CLI tools. 

These are typically used on high-performance workstations to trigger large-scale batch processing without needing to manually type the long CLI arguments.

## Directory Structure

```
scripts/
└── wrf/           # Bash automation for processing WRF model output
```

## Workstation Automation

The `scripts/wrf/` directory contains standard pipelines for transforming raw WRF `netCDF` outputs into the optimized JSON payloads required by the `site/` interactive maps.

### `processa_wrf_04_python.sh`

A robust Bash wrapper around `labmim-wrf-geojson`. It automatically processes multiple domains for a given date across several critical meteorological variables.

**Usage:**
```bash
bash scripts/wrf/processa_wrf_04_python.sh <YYYYMMDD>
```

**Example:**
```bash
bash scripts/wrf/processa_wrf_04_python.sh 20240101
```

**What it does:**
1. Validates the input date.
2. Checks if the corresponding WRF files exist in the network attached storage.
3. Automatically fires `labmim-wrf-geojson` for variables like:
    - `temperature`
    - `wind`
    - `humidity`
    - `solar`
    - `eolico`
    - `rain`
4. Uses 44 CPU cores (configured for the lab's primary workstation) to accelerate GeoJSON grid extraction and serialization.

> **Note**: This script assumes the repository is running on a LabMiM server with the specific `wrfout` directory paths configured in the environment. If you are developing locally, use the CLI directly (see the main README).
