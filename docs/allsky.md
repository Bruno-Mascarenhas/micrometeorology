# `allsky` — Documentation

A package that pairs one-day all-sky camera timelapses with LabMiM radiation-sensor records and trains **SkyFusionNet**, a multi-task DNN that classifies cloud condition (clear / partial / overcast) and predicts diffuse horizontal irradiance from the sky image plus sensor features.

---

## Overview

The pipeline has three stages, each with its own CLI command:

| Stage | Command | Input | Output |
|---|---|---|---|
| **1. Frame extraction** | `allsky extract-frames` | `allsky-YYYYMMDD.mp4` timelapse | Timestamped JPEGs + `manifest.parquet` |
| **2. Frame–sensor pairing** | `allsky build-index` | Frame manifest + Campbell `.dat` files | `index.parquet` (one row per matched frame) |
| **3. Training** | `allsky train` | Pairing index | Checkpoints, metrics, TensorBoard logs |

The model learns two targets jointly:

1. **Cloud condition class** — *weak* labels derived from clearness-index (kt) bins, not human annotation.
2. **Diffuse irradiance (W/m²)** — a measured pyranometer column by default (`PSP_Wm2_Avg`), or an Erbs-decomposition **pseudo-target** derived from GHI when no measured column is configured. Every dataset row carries a `target_source` column (`"measured"` or `"erbs_pseudo"`) so pseudo rows can be identified and replaced later.

All timestamps in the pipeline are **naive local standard time** (the Campbell datalogger clock — no timezone conversion in v0). The UTC offset for solar geometry is inferred from the site longitude as `round(longitude / 15)`; for Salvador-BA (longitude −38.51) this yields UTC−3, correct year-round since Brazil abolished DST in 2019.

---

## Package Structure

```
src/allsky/
├── __init__.py       # Package version and docstring
├── config.py         # Pydantic config models (video/sensor/site/labels/model/train) + YAML loader
├── cli.py            # CLI: allsky info / extract-frames / build-index / train
├── solar.py          # NOAA/Spencer solar position, extraterrestrial GHI, clearness index kt
├── erbs.py           # Erbs (1982) diffuse-fraction decomposition -> pseudo diffuse targets
├── sensors.py        # TOA5 ingestion (via micrometeorology), target derivation, weak labels, QC
├── video.py          # Streamed frame decoding, frame -> wall-clock mapping, JPEG extraction
├── dataset.py        # build_index (merge_asof pairing), FeatureStats, AllSkyDataset
├── models.py         # SkyFusionNet (image CNN + sensor MLP fusion) + multitask loss
└── training.py       # Day-based split, training loop, AMP, TensorBoard, checkpoints
```

Configuration defaults live in `configs/allsky/default.yaml`; tests are under `tests/allsky/`.

---

## Installation

The `allsky` extra pulls the heavy dependencies (`torch`, `imageio-ffmpeg`, `tqdm`, `tensorboard`):

```bash
# CPU PyTorch (uv.lock pins torch to the CPU wheel index on Linux/macOS):
uv pip install -e ".[allsky]"

# CUDA PyTorch (install torch from the CUDA index first):
uv pip install --torch-backend cu121 torch
uv pip install -e ".[allsky]"

# Plain pip also works:
pip install -e ".[allsky]"
```

For local development, activate the `labmim` Conda environment before running these commands, as with the other packages in this repository.

**torch is optional for most of the package.** `torch`, `tqdm`, `tensorboard`, and `imageio-ffmpeg` are imported lazily: `allsky info`, frame extraction, sensor ingestion, `build_index`, and `split_days` all work in a torch-free environment. Only `AllSkyDataset.__getitem__` and the training loop itself require torch.

---

## Quick Start

### 1. Inspect the configuration and video mapping

```bash
allsky info --config configs/allsky/default.yaml
```

Prints the frame-to-time mapping, the videos matched by `video.pattern`, whether the diffuse target is measured or an Erbs pseudo-target, and the fully resolved config.

### 2. Extract frames

```bash
allsky extract-frames data/all-sky/allsky-20260625.mp4 -o output/allsky/frames
```

Writes JPEGs named `allsky-YYYYMMDD-HHMM.jpg` (quality 92) plus a `manifest.parquet` with columns `frame_path`, `timestamp`, `video`, `index`. Use `--step 60` to keep one frame per hour and `--resize 224` to downscale at extraction time. The manifest is overwritten on every call — use one directory per video or per extraction run.

### 3. Build the pairing index

```bash
allsky build-index --manifest output/allsky/frames/manifest.parquet \
    --out output/allsky/index.parquet
```

Loads the Campbell files from `sensor.paths`, derives targets, and pairs each frame with the nearest sensor record within `sensor.tolerance_minutes`. The command reports how many rows survived and the `target_source` breakdown.

### 4. Train

```bash
allsky train --index output/allsky/index.parquet
# Resume an interrupted run:
allsky train --index output/allsky/index.parquet --resume output/allsky/last.pt
```

Device resolves automatically (CUDA → MPS → CPU); useful overrides: `--epochs`, `--batch-size`, `--device`, `--out-dir`, `--val-fraction` (fraction of **days** held out, default 0.2).

### Configuration file

All keys and their defaults (see `configs/allsky/default.yaml`):

```yaml
video:
  pattern: "data/all-sky/allsky-*.mp4"
  filename_date_format: "allsky-%Y%m%d"
  start_time: "06:00"          # local time of frame 0 — match the camera schedule
  minutes_per_frame: 1.0

sensor:
  paths: ["data/LBM_lenta_2025.dat"]
  ghi_column: "CM3Up_Wm2_Avg"
  diffuse_column: "PSP_Wm2_Avg"   # null -> Erbs pseudo-targets (see below)
  feature_columns: ["CM3Up_Wm2_Avg", "CG3Up_Wm2_Avg", "CM3Dn_Wm2_Avg",
                    "Net_Wm2_Avg", "CUV5_Wm2_Avg", "PAR_Wm2_Avg"]
  tolerance_minutes: 5.0

site:
  latitude: -13.00              # LabMiM/UFBA, Salvador-BA
  longitude: -38.51

labels:
  kt_clear: 0.65
  kt_overcast: 0.35
  min_solar_elevation_deg: 10.0
  max_kt: 1.2                   # QC guard: higher kt = sensor artifact, row dropped

model:
  image_size: 224
  backbone: "small"             # "small" (built-in conv net) or "resnet18"
  embed_dim: 128
  hidden_dim: 256
  n_classes: 3

train:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0003
  weight_decay: 0.0001
  num_workers: 2
  device: "auto"                # auto -> cuda | mps | cpu
  amp: true                     # mixed precision on CUDA
  out_dir: "output/allsky"
  seed: 42
  cls_loss_weight: 1.0
  reg_loss_weight: 1.0
```

### Output artifacts

Training writes into `train.out_dir`:

```
output/allsky/
├── index.parquet     # default pairing-index location used by `allsky train`
├── last.pt           # checkpoint every epoch (model + optimizer + epoch + config) — resumable
├── best.pt           # checkpoint at the lowest validation loss
├── config.json       # fully resolved config used for the run
├── metadata.json     # git commit, package versions, device, timing
└── runs/             # TensorBoard event files (tensorboard --logdir output/allsky/runs)
```

---

## Video → Time Mapping

Videos are **one-day timelapse files** named by date (`data/all-sky/allsky-YYYYMMDD.mp4`) where **one frame covers one minute** of real time by default. Frame 0 is captured at `video.start_time` local time (default `06:00`), and frame *i* maps to:

```
timestamp(i) = date + start_time + i * minutes_per_frame
```

Both `start_time` and `minutes_per_frame` are configurable — adjust them to the camera schedule. Videos are always decoded as a stream (`imageio.v3.imiter`); a full one-day 1080p file is never loaded into memory at once.

Limitations to keep in mind:

- The mapping assumes the camera never skips frames — a dropped frame in the timelapse shifts every subsequent timestamp.
- Extracted JPEG filenames carry minute resolution, so with `minutes_per_frame < 1` two frames can map to the same name and overwrite each other.

---

## Sensor Pairing and Targets

### Ingestion

Campbell TOA5 `.dat` files listed in `sensor.paths` are read through `micrometeorology.sensors.ingestion.read_campbell_dat` (sentinel values ≤ −900 become NaN), concatenated, sorted by timestamp, deduplicated (first occurrence wins), and reduced to the GHI, diffuse, and feature columns. A missing configured column raises `KeyError` immediately.

### Target derivation (`allsky.sensors.derive_targets`)

Adds four columns and applies QC:

- `kt` — clearness index `GHI / E0h` (extraterrestrial horizontal irradiance from the NOAA/Spencer solar-position chain in `allsky.solar`).
- `diffuse` — the training target (see below).
- `cloud_class` — weak label from kt bins.
- `target_source` — `"measured"` or `"erbs_pseudo"`.

Rows are **dropped** when: solar elevation is below `labels.min_solar_elevation_deg` (default 10° — night and near-horizon rows where kt is noise-dominated), kt or the diffuse target is NaN (missing GHI), or `kt > labels.max_kt` (default 1.2 — GHI spiking far beyond the physically plausible clear-sky envelope is a sensor artifact, not weather).

### The diffuse target — read this before changing `diffuse_column`

> **Important: the station's primary diffuse pyranometer (CMP21) currently produces no usable data.** The CMP21 W/m² logger channel is zero-filled — only the raw `CMP21_Avg` millivolt channel is live, because the CR5000 program's unit conversion is broken. Until that is fixed, **`PSP_Wm2_Avg` is the working diffuse measurement and the default target**. Switch `sensor.diffuse_column` back to `CMP21_Wm2_Avg` only after the logger program conversion is repaired.

A **dead-channel guard** enforces this: if the configured measured diffuse column is effectively all zeros in the selected daytime rows (more than 99% of finite values exactly zero), `derive_targets` raises `ValueError` instead of silently teaching the model to predict zero.

Setting `diffuse_column: null` switches to **Erbs pseudo-targets**: the Erbs et al. (1982) three-piece `kd(kt)` correlation converts GHI into a pseudo diffuse value (`DHI = kd(kt) × GHI`, bounded by `0 ≤ DHI ≤ GHI`). Such rows carry `target_source: "erbs_pseudo"` — regression metrics on them measure agreement with the Erbs decomposition, not with a real measurement.

### Weak cloud-condition labels

| Condition | kt bin (defaults) | Class |
|---|---|---|
| Clear | `kt >= labels.kt_clear` (0.65) | 0 |
| Partial | `labels.kt_overcast <= kt < kt_clear` | 1 |
| Overcast | `kt < labels.kt_overcast` (0.35) | 2 |

NaN kt yields −1 (unlabelable) and the row is dropped. These are *weak* labels: kt conflates cloudiness with turbidity and calibration drift, and thin cirrus can still reach "clear" kt values.

### Pairing (`allsky.dataset.build_index`)

Frames and sensor rows are joined with `pandas.merge_asof(direction="nearest")` within `sensor.tolerance_minutes` (default 5 minutes). Frames with no sensor record inside the tolerance are dropped — this also removes night frames, because `derive_targets` already removed low-sun sensor rows. Rows with any missing target or feature value are dropped too. The result (optionally written to parquet) has one row per matched frame: manifest columns, `sensor_timestamp`, sensor features, and targets.

---

## The Model: SkyFusionNet

Two branches fused into a shared trunk with two heads (`allsky.models.SkyFusionNet`):

- **Image branch** — `backbone: "small"` (default): a built-in 4-block conv net (Conv3×3 stride 2 → BatchNorm → ReLU, channels 32→64→128→256) with global average pooling; or `backbone: "resnet18"`: torchvision resnet18 with random weights (requires `torchvision`). Either is projected to `embed_dim`.
- **Sensor branch** — MLP `F → 64 → embed_dim` over the standardized feature vector.
- **Fusion trunk** — concatenation → MLP (`hidden_dim`).
- **Heads** — `cls_head` (`n_classes` logits) and `reg_head` (1 output through a final ReLU, so predicted irradiance is **non-negative by construction**).

The multi-task loss is:

```
loss = cls_loss_weight * CrossEntropy(logits, cloud_class)
     + reg_loss_weight * SmoothL1(diffuse_pred / 100, diffuse_true / 100)
```

The division by 100 W/m² scale-normalizes the regression term so it is commensurate with cross-entropy (SmoothL1's quadratic region then covers errors up to ~100 W/m²). Predictions and the reported MAE/RMSE metrics stay in raw W/m² — only the loss is scaled.

---

## Training

`allsky train` (or `allsky.training.train`) reads the index parquet and:

1. **Splits by calendar day, never by row.** Consecutive frames of the same day are near-duplicates, so a row-level split would leak validation information into training. `split_days` assigns whole days to one side (validation gets `round(n_days * val_fraction)` days, at least 1 and at most `n_days - 1`) and re-checks that no day appears on both sides. If the index spans a **single day**, training falls back to reusing that day for validation and logs a loud warning — metrics are then not leakage-free (smoke/debug runs only).
2. **Standardizes features from the training split only.** `FeatureStats` (per-feature mean/std) are computed on the train split and handed to the validation dataset; a validation dataset refuses to compute its own stats.
3. **Resolves the device**: `train.device: auto` picks CUDA → MPS → CPU. On CUDA, automatic mixed precision is enabled when `train.amp: true` (roughly 2× throughput on Colab T4/L4 GPUs) and cuDNN benchmarking is turned on.
4. **Keeps DataLoader workers persistent** across epochs (`num_workers > 0`) — worker re-forking otherwise dominates epoch startup for image-heavy datasets.
5. **Logs and checkpoints**: TensorBoard scalars (loss, accuracy, MAE/RMSE in W/m²) under `out_dir/runs`; `last.pt` every epoch and `best.pt` at the lowest validation loss, both containing model, optimizer, epoch, validation metrics, and the config; `config.json` and `metadata.json` (git commit, versions, timing) for reproducibility.

Pass `--resume <out-dir>/last.pt` to restore the model, optimizer, and epoch counter and continue an interrupted run.

### Google Colab

[`notebooks/allsky_colab.ipynb`](../notebooks/allsky_colab.ipynb) is the GPU quickstart: it installs the `allsky` extra from GitHub, mounts Google Drive for videos and `.dat` files, writes a run config, runs the three CLI stages, and shows live metrics via `%tensorboard`. Training is resumable across Colab disconnects through `--resume`; copy the run directory back to Drive to persist artifacts.

---

## Known Data Gap

The sensor archive currently ends on **2026-04-24**, while the first all-sky video is from **2026-06-25** — there is no temporal overlap yet. `build-index` runs fine on this data but matches zero frames. Pairing yields real training rows once logger files covering the camera dates are added to `sensor.paths`.

---

## CLI Reference

```bash
allsky info [--config FILE]

allsky extract-frames VIDEO --out DIR [--step N] [--resize N] [--config FILE]

allsky build-index --manifest MANIFEST.parquet [--out INDEX.parquet] [--config FILE]

allsky train [--config FILE] [--index INDEX.parquet] [--resume CHECKPOINT.pt]
             [--epochs N] [--batch-size N] [--device auto|cpu|cuda|mps]
             [--out-dir DIR] [--val-fraction F]
```

- `--config` / `-c` accepts a pipeline YAML; without it, built-in defaults are used.
- `extract-frames`: `--out` / `-o` is required; `--step N` keeps every Nth frame; `--resize N` writes N×N JPEGs.
- `build-index`: `--out` defaults to `output/allsky/index.parquet`.
- `train`: `--index` defaults to `<out-dir>/index.parquet`; `--epochs`, `--batch-size`, `--device`, and `--out-dir` override the corresponding `train.*` config keys.

---

## Frequently Asked Questions

### Why is the diffuse target PSP and not CMP21?

CMP21 is the station's primary diffuse pyranometer, but its W/m² logger channel currently writes zeros — only the raw `CMP21_Avg` mV channel is live because the CR5000 program's unit conversion is broken. Training on it would teach the model to predict zero, which is why the dead-channel guard raises on effectively all-zero measured columns. `PSP_Wm2_Avg` is the working diffuse measurement today; switch `sensor.diffuse_column` to `CMP21_Wm2_Avg` once the logger program is fixed.

### What does `target_source` mean?

Every index row records where its diffuse target came from: `"measured"` (a real pyranometer column) or `"erbs_pseudo"` (derived from GHI via the Erbs decomposition when `diffuse_column: null`). Pseudo rows bootstrap the pipeline when no measured diffuse exists — treat regression metrics on them as consistency checks against Erbs, not accuracy, and replace them once real measurements are available.

### Why split train/validation by day instead of by row?

Sky frames one minute apart are near-duplicates. A shuffled row-level split would place near-identical images on both sides and inflate validation metrics. `split_days` assigns whole calendar days to one split, so frames of the same day never cross splits; `train()` re-checks the invariant at runtime.

### How do I add a real shaded-pyranometer column later?

Point `sensor.diffuse_column` at the new column name (and add it to the logger file read by `sensor.paths`), then rebuild the index with `allsky build-index`. New rows will carry `target_source: "measured"`; the dead-channel guard verifies the column actually contains data. No code changes are needed.

### Can I use the package without PyTorch installed?

Yes, for everything except training and dataset tensor access. Config loading, `allsky info`, frame extraction, sensor ingestion, target derivation, `build_index`, and `split_days` import torch lazily (or not at all). Only `AllSkyDataset.__getitem__` and `allsky train` require torch — install the `allsky` extra for those.

### Why do some daytime rows disappear from the index?

Three QC filters remove them: solar elevation below `labels.min_solar_elevation_deg` (kt is noise-dominated near the horizon), `kt > labels.max_kt` (GHI spikes beyond the clear-sky envelope are sensor artifacts), and missing GHI/feature values. Unmatched frames (no sensor record within `sensor.tolerance_minutes`) are dropped at pairing time.
