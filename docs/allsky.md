# `allsky` — Documentation

A package that pairs one-day all-sky camera timelapses with LabMiM radiation-sensor records into a portable **v2 multimodal dataset** (manifest + frames + precomputed visual embeddings), then trains a ladder of multimodal models (V0–V7) that predict **diffuse horizontal irradiance** — and optionally a clear-sky index and a clear / partially-cloudy / overcast **sky class** — from the sky image plus engineered sensor features.

> **Getting the videos.** The Planetário da UFBA camera archive is mirrored by `allsky sync-archive`, which also extracts frames and can push them to Google Drive — see [`allsky-archive.md`](allsky-archive.md). Read it before extracting frames from that camera: its capture interval changes between day and night and its videos do not all start at the same hour, so the `VideoConfig` `start_time`/`minutes_per_frame` model mislabels frames by up to 2 h 33 min. `sync-archive --extract` reads the timestamp the camera burns into each frame instead.

The legacy v0 SkyFusionNet pipeline (`allsky info` / `build-index` / `allsky train --index`) has been **retired**. This document describes the current multimodal stack only. The full internal design — module map, artifact contracts, anti-leakage policy, alignment strategies, and the V0–V7 ladder — is in [`allsky-architecture.md`](allsky-architecture.md); this file is the CLI + config quickstart.

---

## Overview

The pipeline has four stages, each with its own CLI command:

| Stage | Command | Input | Output |
|---|---|---|---|
| **1. Prepare** | `allsky prepare-local` | `allsky-YYYYMMDD.mp4` timelapses + Campbell `.dat` files | Frames, `manifest.parquet` (+ `.meta.json`), `splits.json` |
| **2. Validate** | `allsky validate-dataset` | Prepared manifest + splits | Pass/fail report (leakage, schema, QC) |
| **3. Embeddings** | `allsky precompute-embeddings` | Prepared manifest + frames | DINOv2 fp16 safetensors shards + index/meta |
| **4. Train / Evaluate** | `allsky train` / `allsky evaluate` | Manifest + splits + embeddings | Checkpoints, metrics, stratified reports |

The models learn diffuse irradiance and, per the experiment's `targets` block, optionally a clear-sky index (k\*) and a weak sky-condition class. Diffuse targets are a measured pyranometer column by default (`PSP_Wm2_Avg`), or an Erbs-decomposition **pseudo-target** derived from GHI when no measured column is configured. Every manifest row carries a `target_source` column (`"measured"` or `"erbs_pseudo"`) so pseudo rows can be identified and replaced later.

All timestamps in the pipeline are **naive local standard time** (the Campbell datalogger clock — no timezone conversion). The UTC offset for solar geometry is inferred from the site longitude as `round(longitude / 15)`; for Salvador-BA (longitude −38.51) this yields UTC−3, correct year-round since Brazil abolished DST in 2019.

---

## Package Structure

```
src/allsky/
├── __init__.py        # Package version and docstring
├── config.py          # VideoConfig/SiteConfig + Experiment/Prepare config trees + YAML loaders
├── video.py           # Streamed frame decoding, frame -> wall-clock mapping, JPEG extraction
├── archive.py         # Mirrors the Planetário camera archive (HTTPS client, ledger, TLS repair)
├── overlay.py         # Reads the timestamp the camera burns into each frame + timestamped extraction
├── drive.py           # rclone uploads to Google Drive
├── snapshot.py        # Live-frame capture + single-image prediction
├── preprocessing.py   # Static mask / crop / resize + per-frame visual QC
├── solar.py           # NOAA/Spencer solar position, extraterrestrial GHI, clearness index kt
├── clearsky.py        # Haurwitz clear-sky GHI + clear-sky index k*
├── erbs.py            # Erbs (1982) diffuse-fraction decomposition -> pseudo diffuse targets
├── atomic.py          # Atomic file / JSON writes         provenance.py # code + content hashes
├── bundle.py          # Colab bundle export
├── cli/               # frames, prepare, embeddings, train, evaluate command groups
├── data/              # manifest, contracts, alignment, splits, datasets, loading, validation
├── features/          # engineering, normalization, anti-leakage policy
├── embeddings/        # visual backbones, extraction loop, safetensors storage
├── modeling/          # sensor/visual encoders, fusions, heads, model registry (V0–V7)
├── training/          # experiment engine, losses, checkpointing, device resolution
└── evaluation/        # evaluator, metrics, stratified reports
```

Configs live under `configs/allsky/`; tests are under `tests/allsky/`.

---

## Installation

The `allsky` extra pulls the heavy dependencies (`torch`, `imageio-ffmpeg`, `tensorboard`, `safetensors`):

```bash
# CPU PyTorch in the active Conda environment (locked):
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact --extra allsky

# CUDA PyTorch plus the locked TCC and all-sky dependencies:
make install-cuda

# A separate project .venv instead of Conda:
uv sync --locked --extra allsky

# Plain pip remains available for downstream installs outside this checkout:
pip install -e ".[allsky]"
```

For the Conda workflows, activate the `micrometeorology` environment first. The Make target installs the locked extras without the CPU torch wheel, then force-installs Torch 2.13 from the CUDA 13.0 index. `TORCH_BACKEND` and `TORCH_VERSION` can be overridden together when another matching index/version pair is required.

**torch is optional for most of the package.** Heavy modules are imported lazily, and `imageio-ffmpeg` is loaded as a video backend only when needed: `allsky --help`, frame extraction (`extract-frames`), dataset preparation, manifest building, day splits, and validation all work in a torch-free environment. Only embedding extraction, training, and evaluation require torch — install the `allsky` extra for those.

---

## Quick Start (local)

```bash
allsky prepare-local          --config configs/allsky/data/local_prepare.yaml
allsky validate-dataset       --config configs/allsky/data/local_prepare.yaml
allsky precompute-embeddings  --config configs/allsky/data/local_prepare.yaml

allsky train    --config configs/allsky/experiments/v4_film.yaml \
                --data-root output/allsky-mm/dataset \
                --out-dir   output/allsky-mm/experiments/v4_film/run --device cuda --amp
allsky evaluate --checkpoint output/allsky-mm/experiments/v4_film/run/best.ckpt \
                --split test --data-root output/allsky-mm/dataset

allsky export-colab-bundle -o bundle.tar.gz --config configs/allsky/data/local_prepare.yaml
```

Swap `--device cuda --amp` for `--device cpu --no-amp` for a CPU smoke run (the `fp16` AMP in the shipped configs requires CUDA).

### Single-video frame extraction

`prepare-local` runs frame extraction as its first step, but the low-level, single-video entry point is still available:

```bash
allsky extract-frames data/all-sky/allsky-20260625.mp4 -o output/allsky/frames --step 60
```

It writes JPEGs named `allsky-YYYYMMDD-HHMM.jpg` (quality 92) plus a `manifest.parquet` with columns `frame_path`, `timestamp`, `video`, `index`, using the `video` section of the PrepareConfig (`--config`, or built-in defaults) for the frame→time mapping. Use `--resize 224` to downscale at extraction time. The manifest is overwritten on every call — use one directory per video or per extraction run.

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

## Sensor Ingestion and Targets

### Ingestion

Campbell TOA5 `.dat` files listed in `sensor.paths` are read through `micrometeorology.sensors.ingestion.read_campbell_dat` (sentinel values ≤ −900 become NaN), concatenated, sorted by timestamp, and deduplicated (first occurrence wins). Raw logger columns are kept as-is; the manifest builder selects and validates the policy columns it needs, and `sensor.column_map` optionally renames logger columns to the policy source names before building.

### The manifest and its targets

`allsky prepare-local` aligns each frame to the nearest sensor record (see the alignment strategy in the config), derives targets, and writes the v2 manifest plus a `.meta.json` sidecar carrying a content `manifest_sha256`, dataset version, and provenance. Target columns include:

- `kt` — clearness index `GHI / E0h` (extraterrestrial horizontal irradiance from the NOAA/Spencer solar-position chain in `labmim_core.solar`).
- `dhi` — the diffuse target (see below).
- `kindex` — the clear-sky index k\* (GHI over Haurwitz clear-sky GHI) or the clearness index k\_t, per `targets.kindex_kind`.
- `sky_class` — weak sky-condition label from k-index bins (`-1` marks missing/unlabelable).
- `target_source` — `"measured"` or `"erbs_pseudo"`.

Rows are flagged in a `qc_flags` bitmask (low sun, sensor gap, far alignment, k-index artifact) and night frames below the configured solar-elevation floor are dropped. A `prepare-local` run ORs two more bits in from the frame-QC pass: `FRAME_DARK` when the frame's mean BT.601 luminance is below `DARK_LUMINANCE_THRESHOLD = 10.0`, and `FRAME_SATURATED` when more than `SATURATED_FRACTION_THRESHOLD = 0.2` of the pixels have every channel at `SATURATED_LEVEL = 255`. QC is computed on the raw extracted frame *before* mask/crop/resize, so the black surround outside the fisheye disc drags the dark-luminance mean down. Manifests built without that pass (a standalone `extract-frames` run, or `build_manifest` called directly) leave both bits unset, so filter on them explicitly rather than treating `qc_flags == 0` as "clean".

### The diffuse target — read this before changing `targets.diffuse_column`

> **Important: the station's primary diffuse pyranometer (CMP21) currently produces no usable data.** The CMP21 W/m² logger channel is zero-filled — only the raw `CMP21_Avg` millivolt channel is live, because the CR5000 program's unit conversion is broken. Until that is fixed, **`PSP_Wm2_Avg` is the working diffuse measurement and the default target**. Switch `targets.diffuse_column` back to `CMP21_Wm2_Avg` only after the logger program conversion is repaired.

Setting `targets.diffuse_column: null` switches to **Erbs pseudo-targets**: the Erbs et al. (1982) three-piece `kd(kt)` correlation converts GHI into a pseudo diffuse value (`DHI = kd(kt) × GHI`, bounded by `0 ≤ DHI ≤ GHI`). Such rows carry `target_source: "erbs_pseudo"` — regression metrics on them measure agreement with the Erbs decomposition, not with a real measurement.

### Weak sky-condition labels

The four conditions of Escobedo, Gomes, Oliveira & Soares (2009), *Applied Energy* 86(3):299–309, sec. 3.1, binned on the clearness index **Kt** — the index those bounds are published on. The bounds are not configurable: they are the reference's.

| Condition | Kt bin | Class | Portuguese |
|---|---|---|---|
| Cloudy | `Kt <= 0.35` | 0 | nebuloso |
| Partly cloudy, diffuse-dominated | `0.35 < Kt <= 0.55` | 1 | parcialmente nebuloso com dominância para o difuso |
| Partly cloudy, clear-dominated | `0.55 < Kt <= 0.65` | 2 | parcialmente nebuloso com dominância para o claro |
| Clear | `Kt > 0.65` | 3 | claro |

Rows whose solar elevation is below `min_elevation_deg` (10°) are labelled `-1` (`SKY_CLASS_MISSING`) rather than binned, because Kt's denominator collapses near the horizon.

These are *weak* labels: Kt conflates cloudiness with turbidity and calibration drift, and thin cirrus can still reach "clear" values. The sky-class head is off by default and enabled per experiment via `targets.sky.enabled`.

---

## Anti-leakage feature policy

The default `safe` feature set is **solar geometry + standard meteorology only — no radiometry**. GHI, the diffuse pyranometer, and every derived target are *forbidden* as features and raise `ForbiddenFeatureError` if requested. This is the central anti-leakage guarantee of the v2 stack: a model must learn diffuse irradiance from the *sky image* and non-radiometric context, not from a radiometric shortcut. The `extended` set adds ablation-only radiometric auxiliaries and is never selected silently. `validate-dataset` fails if a forbidden feature reaches the manifest.

> **`safe` needs a working thermohygrometer, and the station's has not had one since 2025-12-19.** The Gill MetSENS1 railed at 1000 °C air, 1000 °C dew point and 999 %RH; `mask_sentinels` turns each rail into NaN and `build_manifest` drops any row with a non-finite feature, so `safe` retains 0.02 % of the rows paired against the all-sky camera archive. `features.set: minimal` drops those three channels (10 features: solar geometry, barometer, anemometer).

> **`minimal` needs a working barometer, and on 2026-08-10 13:05 the same instrument took that too.** The MetSENS1 stopped railing and moved to a single `Unknown Fault` fill state on every subsequent row: `AirT1_C_Avg` writes the literal text `NAN`, `DP1_C_Avg` 214, `RH1` 2.62, and **`BP1_mbar_Avg` a constant 2.62 hPa**. That last one falls outside `SENTINEL_RANGES["BP1_mbar_Avg"] = (950, 1100)`, so it masks to NaN and the row-wise finite filter drops every sample from that timestamp on — silently, since rows before it still pass. Use **`features.set: bare`** (9 features: solar geometry + the *mechanical* anemometer `WS_ms`/`WindDir`, a different instrument that survived both failures) until the MetSENS1 is repaired. `configs/allsky/data/local_prepare.yaml` and `configs/allsky/experiments/_base.yaml` both ship that way, and they must always agree — the engine resolves feature columns from the experiment config while the manifest carries whatever prepare built.

---

## CLI Reference

```bash
allsky extract-frames         VIDEO --out DIR [--step N] [--resize N] [--config FILE]

allsky prepare-local          [--config FILE] [--steps a,b,c] [--dry-run] [--force]
allsky validate-dataset       [--config FILE] [--manifest FILE] [--strict]
allsky precompute-embeddings  --config FILE [--manifest FILE] [--out DIR]
                              [--device auto|cpu|cuda|mps] [--resume/--no-resume] [--dry-run]
allsky export-colab-bundle    --out BUNDLE.tar.gz [--config FILE]
                              [--include-embeddings/--no-include-embeddings]

allsky train    --config EXPERIMENT.yaml [--data-root DIR] [--out-dir DIR]
                [--epochs N] [--batch-size N] [--device auto|cpu|cuda|mps]
                [--amp/--no-amp] [--resume auto|CHECKPOINT.ckpt]
allsky evaluate --checkpoint CHECKPOINT.ckpt [--split val|test|train]
                [--config FILE] [--data-root DIR] [--report-dir DIR]
                [--device ...] [--batch-size N] [--predictions/--no-predictions] [--strict]
```

- `prepare-local` runs `extract-frames → build-manifest → splits`; steps are resumable and skip up-to-date outputs unless `--force`. A `--steps build-manifest` run without `extract-frames` cannot re-extract anything, so it aborts (exit 1) on a video whose frames carry no recorded provenance, or one written under a different video/mask/crop/resize config, instead of stamping the manifest with a config that did not produce those JPEGs — include the `extract-frames` step, or pass `--force` to build from the frames as they are. `--dry-run` logs the full plan and writes nothing.
- `precompute-embeddings` reads the `embeddings` section of the PrepareConfig (backbone / pooling / batch / shard-size / dtype); backbone `"fake"` is the offline dev/test hook; the DINOv2 and DINOv3 ViTs download via `torch.hub` on first use, and `resnet50`/`efficientnet_v2_s` come from torchvision. See the [architecture reference](allsky-architecture.md#backbone-families) for the full list. `--resume` (default) skips `sample_id`s already in `index.parquet`, but refuses to resume into an embeddings dir built with a different backbone/pooling/dim/config — rerun with `--no-resume` (or a fresh `--out` dir) to overwrite.
- `train` **requires** an experiment config (a YAML declaring `experiment: true`); any other config (or none) is rejected with a pointer to `configs/allsky/experiments/`. `--resume auto` finds `last.ckpt` in the run dir; `--epochs` is the **total** budget (resuming trains only the remainder and never clobbers a better `best.ckpt`).
- `evaluate` rebuilds the model from the checkpoint, restores the train-split normalizers (no refit — leakage-safe), denormalizes to physical units, verifies `manifest_sha256`/`split_id` (warn, or error under `--strict`), and writes `metrics.json`, `stratified.csv`, `report.md` and (optionally) `predictions.parquet`.

---

## Config tree (`configs/allsky/`)

```
configs/allsky/
├── data/local_prepare.yaml      # PrepareConfig: prepare/validate/embeddings/bundle
├── models/                      # model-section fragments (name + arch params)
│   ├── sensor_only.yaml  image_only.yaml  concat.yaml  film.yaml  cross_attention.yaml
└── experiments/
    ├── _base.yaml               # shared data / targets / train blocks
    └── v0_climatology.yaml … v7_cross_attention.yaml
```

Experiment files stay tiny by composing with `extends:` (a path or list, resolved relative to the including file, deep-merged, later wins; cycles raise). A typical experiment `extends: [_base.yaml, ../models/film.yaml]` then overrides only `name`, `output_dir`, and any target/data/train key it changes. All experiment / prepare configs are strict (`extra="forbid"`) so a typo fails loudly. The V0–V7 ladder is summarised in [`allsky-architecture.md`](allsky-architecture.md).

---

## Training

`allsky train` routes an `experiment: true` config to the multimodal engine, which:

1. **Uses persisted day splits.** The split artifact (`splits.json`) assigns whole calendar days to train/val/test, so near-duplicate frames of the same day never cross splits, and carries a `split_id` (no silent regeneration). Consecutive frames one minute apart are near-duplicates — a row-level split would leak validation information into training.
2. **Standardizes features and targets from the training split only.** The `FeatureNormalizer` / `TargetNormalizer` are fit on the train split and stored in the checkpoint; validation/test reuse them verbatim (computing one locally is refused).
3. **Resolves the device**: `device: auto` picks CUDA → MPS → CPU; on CUDA, automatic mixed precision is available (`--amp`, `fp16`/`bf16`).
4. **Runs the engine**: optimizer/param groups (optional separate backbone LR), scheduler (`none`/`cosine`/`plateau`), gradient accumulation and clipping, early stopping, and full resume. It writes `last.ckpt` every epoch, `best.ckpt` at the best monitored metric, `metrics.json` and `metrics.csv` every epoch, and TensorBoard events under `runs/`. There is no separate run-manifest file: run provenance (`split_id`, `manifest_sha256`, `feature_columns`, `feature_groups`, `dataset_version`, the resolved config dump and `code_version_info`) is embedded in **both** checkpoints, and that is what `allsky evaluate` re-verifies.

`--resume auto` restores from `last.ckpt` in the run dir and continues; `--epochs` is the total budget, so resuming trains only the remainder.

---

## Frequently Asked Questions

### Why is the diffuse target PSP and not CMP21?

CMP21 is the station's primary diffuse pyranometer, but its W/m² logger channel currently writes zeros — only the raw `CMP21_Avg` mV channel is live because the CR5000 program's unit conversion is broken. Training on it would teach the model to predict zero. `PSP_Wm2_Avg` is the working diffuse measurement today; switch `targets.diffuse_column` to `CMP21_Wm2_Avg` once the logger program is fixed.

### What does `target_source` mean?

Every manifest row records where its diffuse target came from: `"measured"` (a real pyranometer column) or `"erbs_pseudo"` (derived from GHI via the Erbs decomposition when `diffuse_column: null`). Pseudo rows bootstrap the pipeline when no measured diffuse exists — treat regression metrics on them as consistency checks against Erbs, not accuracy, and replace them once real measurements are available.

### Why split train/validation by day instead of by row?

Sky frames one minute apart are near-duplicates. A shuffled row-level split would place near-identical images on both sides and inflate validation metrics. The persisted split artifact assigns whole calendar days to one split, so frames of the same day never cross splits.

### Why can't features include GHI or the pyranometers?

That would be leakage: the diffuse target is derived from radiometry, so a model given GHI as a feature learns a radiometric shortcut instead of reading the sky. The `safe` feature policy forbids all radiometry; `validate-dataset` enforces it.

### Can I use the package without PyTorch installed?

Yes, for everything except embedding extraction, training, and evaluation. Config loading, `allsky --help`, frame extraction, dataset preparation, manifest building, day splits, and validation import torch lazily (or not at all). Install the `allsky` extra for the torch-backed stages.

---

## Size of the paired archive

Pairing yields **46,014 rows over 81 days**, split by day into 55 train / 12 val
/ 12 test. That is a small archive for a visual model, which is why arms are
compared against persistence and clear-sky baselines rather than in isolation,
and why the stack carries transfer learning from a larger external source (see
[datasets-de-ceu-abertos.md](datasets-de-ceu-abertos.md)).

---

## Quickstart (Colab)

[`notebooks/allsky_multimodal_colab.ipynb`](../notebooks/allsky_multimodal_colab.ipynb) is the thin GPU notebook: it provisions a CPython 3.14 venv with `uv` (the package requires Python ≥ 3.14, which the Colab base runtime is not assumed to provide), unpacks the exported bundle, then runs `validate-dataset → train (--resume auto, AMP) → evaluate → copy to Drive`. All knobs live in one CONFIG cell; the default experiment is `configs/allsky/experiments/v4_film.yaml`. It has not been executed on a real Colab runtime from the dev environment (documented in the notebook).
```
