# `allsky` multimodal architecture (v2 pipeline)

Reference for the **multimodal DHI-estimation stack** that lives alongside the
legacy SkyFusionNet pipeline (see [`allsky.md`](allsky.md) for the legacy v0
path and shared physics). The v2 stack estimates **diffuse horizontal irradiance
(DHI)**, a **clear-sky / clearness index**, and a **sky-condition class** from an
all-sky image (as a precomputed DINOv2 embedding *or* end-to-end) plus
non-radiometric sensor context.

All timestamps are naive local **America/Bahia** (fixed UTC-3, no DST). The
manifest stores tz-aware `timestamp_utc`; `day_id` is the local calendar day; a
`sample_id` is `allsky-YYYYMMDD-HHMM` (local, matching the frame filename stem).

---

## Local → bundle → Colab flow

```
                          LOCAL WORKSTATION (CPU/GPU)
  ┌───────────────────────────────────────────────────────────────────────┐
  │  allsky-*.mp4          LBM_lenta_*.dat (Campbell TOA5)                  │
  │        │                        │                                       │
  │        ▼                        ▼                                       │
  │  extract frames  ─────►  build v2 manifest  ◄─── feature policy (safe)  │
  │  (+ QC, mask/crop)       (solar geometry, k-index, sky_class, targets)  │
  │        │                        │                                       │
  │        │                        ▼                                       │
  │        │                  day-level splits (splits.json, split_id)      │
  │        │                        │                                       │
  │        ▼                        ▼                                       │
  │   precompute-embeddings ──► embeddings/ (fp16 shards + index + meta)    │
  │        │                                                                │
  │        ▼                                                                │
  │   export-colab-bundle ─────────────► bundle.tar.gz ───────────────┐    │
  └───────────────────────────────────────────────────────────────────┼────┘
                                                                        │
        allsky prepare-local / validate-dataset / train / evaluate     │  (Drive
        run identically here for a fully-local workflow.               │   or upload)
                                                                        ▼
                                    GOOGLE COLAB (GPU)   notebooks/allsky_multimodal_colab.ipynb
  ┌───────────────────────────────────────────────────────────────────────┐
  │  uv venv (CPython 3.14)  ─►  unpack bundle  ─►  validate-dataset        │
  │        └─────────────────────────────────►  train (--resume auto, AMP)  │
  │                                              └──►  evaluate (test split) │
  │                                                    └──►  copy → Drive     │
  └───────────────────────────────────────────────────────────────────────┘
```

The bundle is the single portable artifact: it carries the manifest + sidecar,
the split, the embedding shards, and the configs used, all with **relative POSIX
paths**, so training on Colab is byte-identical to training locally.

---

## Module map

New code lives in packages under `src/allsky/`; the legacy modules
(`models.py`, `dataset.py`, `training.py` re-export, `video.py`, `sensors.py`,
`erbs.py`, `solar.py`) are untouched and stay import-torch-free where they were.

| Package | Responsibility |
|---|---|
| `features/` | Anti-leakage **policy** (`safe`/`extended`/`forbidden`, feature groups), cyclic **engineering** (solar geometry, wind, day-of-year), train-only **normalization** (`FeatureNormalizer`, `TargetNormalizer`). |
| `data/` | `contracts` (v2 column registry, `QCFlag`, sky classes, relative-path helpers), `manifest` builder + atomic parquet + `.meta.json` sidecar, `validation` (every failure mode), `splits` (day-level artifact with `split_id`), `alignment` strategies, lazy-torch `datasets` (the batch contract). |
| `embeddings/` | `backbone` (`VisualBackbone` protocol, pinned DINOv2 via `torch.hub`, `FakeBackbone` for tests), `storage` (safetensors shards + parquet index + `embeddings.meta.json`, `SafetensorsEmbeddingReader`), `extract` (resumable, batched, atomic). |
| `modeling/` | `contracts` (`ModelOutputs`, `group_slices`), `sensor_encoder`, `visual_encoder` (embedding passthrough / image backbone), `fusion` (concat / FiLM / cross-attention), `heads` (trunk + DHI / heteroscedastic / k-index / sky / cloud-fraction), `baselines` (climatology / sensor-only / image-only), `multimodal` assembly, `registry`. |
| `training/` | `losses` (`MultitaskLoss`, maskable per-head), `engine` (`run_experiment`: AMP, grad accum/clip, scheduler, early stop, TensorBoard + `metrics.csv`/`metrics.json`, full resume), `checkpointing` (atomic, full payload, RNG). Legacy `train` re-exported. |
| `evaluation/` | `metrics` (regression + classification), `evaluator` (rebuild from checkpoint, denormalize, stratify), `reports` (`report.md` / `metrics.json` / `stratified.csv` / `predictions.parquet`, `compare_experiments`). |
| `cli/` | `legacy` (`info`/`extract-frames`/`build-index`/`train` with experiment dispatch), `prepare` (`validate-dataset`/`prepare-local`/`export-colab-bundle`), `embeddings` (`precompute-embeddings`), `evaluate`. Every command imports torch lazily so `allsky --help` stays light. |

---

## Artifact contracts

### Manifest v2 (`manifest.parquet`)

One row per paired sample. Columns (see `data/contracts.py`):

- **Identity/metadata**: `sample_id`, `timestamp_utc` (tz-aware UTC), `day_id`
  (local `YYYY-MM-DD`), `image_path` (relative POSIX against `data_root`),
  `frame_index`, `video`.
- **Solar geometry** (degrees): `solar_elevation`, `solar_azimuth`,
  `solar_zenith`. Elevation/zenith double as features; azimuth is fed as the
  `azimuth_sin`/`azimuth_cos` cyclic pair.
- **Feature columns** per the active policy set (13 for `safe`, +4 for
  `extended`).
- **Targets/labels**: `target_dhi`, `target_source` (`measured`/`erbs_pseudo`),
  `target_kindex`, `kindex_kind` (`kstar`/`kt`), `sky_class`
  (`0/1/2`, `-1` = missing), `cloud_fraction` (nullable, all-NaN today),
  `qc_flags` (`int64` `QCFlag` bitmask: `LOW_SUN`, `SENSOR_GAP`,
  `ALIGNMENT_FAR`, `KT_ARTIFACT`, `FRAME_DARK`, `FRAME_SATURATED` — the manifest
  builder sets only the first four; `FRAME_DARK`/`FRAME_SATURATED` are reserved
  for the image-preprocessing wave and stay unset here).
- **Provenance (constant per row)**: `dataset_version`, `alignment_id` (mirror
  the sidecar meta so a manifest is self-describing), and `split` — a **nullable**
  label, empty at build and filled in place by
  `attach_split_column(manifest, split)` after the day split exists.

Night frames below `night_min_elevation_deg` (default 5°) are **dropped** at
build (not just flagged); `LOW_SUN` marks the surviving band between the night
threshold and the k-index floor (`min_elevation_deg`, default 10°). The
`KT_ARTIFACT` ceiling defaults per k-index kind — `1.5` for `kstar` (cloud
enhancement over the Haurwitz clear-sky reference legitimately exceeds `1.2`) and
`1.2` for `kt`. `sample_id` is minute-resolution (`allsky-YYYYMMDD-HHMM`), so a
build with two frames in the same minute raises rather than silently colliding.

### Meta sidecar (`manifest.parquet.meta.json`)

`dataset_version`, `alignment_id`, `feature_set`, the ordered `feature_columns`,
`kindex_kind`, `target_source`, `config_sha256`, `code_version`/git commit,
`created_at`, `row_count`, the `timezone` and `site`, thresholds (incl.
`night_min_elevation_deg` and the resolved `max_kindex`), an optional `split_id`
(recorded by `attach_split_column`), and a content **`manifest_sha256`**
(order-sensitive, parquet-container-independent) that ties a trained checkpoint to
the exact data it saw. **Attaching the split column changes the manifest content,
so it re-hashes `manifest_sha256`** (by design — the split label is part of the
bytes a checkpoint trains on). Both files are written atomically
(temp + `os.replace`).

### Split artifact (`splits.json`)

`{split_id (sha256 of the day→split map + params), seed, fractions, assignment
day_id→split, created_at, dataset_version, per-split day counts}`. Splits are
**day-level** (never row-level) so
near-duplicate consecutive frames cannot cross splits. Overwriting an existing
artifact with different content requires `force=True`, else `SplitExistsError`.

### Embeddings (`embeddings/`)

- `embeddings-{i:05d}.safetensors` — one fp16 tensor per shard.
- `index.parquet` — `sample_id → (shard, row)`.
- `embeddings.meta.json` — backbone, revision, pooling, dim, transform,
  `config_sha256`, count, storage `dtype` (`fp16`).

`SafetensorsEmbeddingReader` resolves a `sample_id` to its vector lazily.
Extraction is resumable — the index is the source of truth, so every `sample_id`
already recorded in `index.parquet` is skipped (a shard and its index rows land
together atomically, so a crash leaves a consistent, possibly shorter index). It
**refuses to resume** into a store whose `embeddings.meta.json` records a
different backbone / revision / pooling / dim / config, rather than silently
mixing two encoders' vectors — rerun with `--no-resume` (or a fresh out dir) to
overwrite.

### Checkpoint payload (`last.ckpt` / `best.ckpt`)

`torch.save` (atomic; `weights_only=False` — a trusted local file). Contains:
`model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`, `epoch`,
`global_step`, `epochs_no_improve` (the early-stopping patience counter, so a
resumed run continues from the exact point on the patience curve; optional —
`None` on pre-field checkpoints, from which the engine reconstructs a lower bound
`max(0, epoch - best_epoch)`), `best_metric{name,value,epoch}`, full `config`
dump, `normalizers` (`feature_normalizer` + per-target `target_normalizers` as
JSON-able dicts), ordered `feature_columns`, `feature_groups`, `dataset_version`,
`split_id`, `manifest_sha256`, `backbone` info (image mode only), `code_version`
(package + git commit), and `rng_state` (python / numpy / torch / cuda) for
deterministic resume. Loading strips a compiled `_orig_mod.` prefix.

Resume is deterministic and crash-safe: the train batch order is drawn from a
dedicated `RandomSampler` generator re-seeded to `seed * 100003 + epoch` each
epoch (a pure function of `(seed, epoch)`, independent of the resume point,
`persistent_workers`, or global-RNG draw count), and a per-loader generator
isolates the worker `base_seed` draw from the global RNG that drives dropout. On
resume the engine also truncates any `metrics.csv`/`metrics.json` rows past the
resumed epoch (metrics are flushed before the checkpoint each epoch, so a crash
in that gap can leave a row the checkpoint never completed) before appending
again, so re-running the interrupted epoch never duplicates it.

### Bundle (`bundle.tar.gz`)

`manifest.parquet` + `.meta.json`, `splits.json`, `embeddings/` (shards + index +
meta), the configs used, and a generated `BUNDLE_README.md`. All members are
relative; `validate_bundle` re-checks the manifest content hash against the
sidecar.

---

## Batch contract (new stack)

Keys emitted by the datasets (all `float32` unless noted):

| Key | Shape | Notes |
|---|---|---|
| `features` | `(B, F)` | standardized sensor vector (train-only stats) |
| `embedding` | `(B, D)` | embedding mode (`center_frame`, or `mean_embedding` pooled in the dataset) |
| `image` | `(B, 3, H, W)` | image mode, `[0, 1]` |
| `embedding_seq` + `frame_mask` | `(B, T, D)` + `(B, T)` bool | `attention_pooling` window: zero-padded to fixed `T = ceil(window_minutes)+1`, mask True = real frame |
| `dhi`, `kindex` | `(B,)` | **raw physical units**, NaN = missing |
| `sky_class` | `(B,)` int64 | `-1` = missing |
| `cloud_fraction` | `(B,)` | NaN = missing |

Losses mask absent targets (`isfinite` / `sky_class >= 0`). The engine normalizes
targets for the loss; metrics and evaluation are always in physical units.
Regression **model outputs are in normalized space** — the engine/evaluator
denormalize with the stored `TargetNormalizer`.

---

## Anti-leakage policy

DHI is derived from the station radiometers (GHI drives `kt`/`k*`; the diffuse
pyranometer *is* the label), so those channels must never be model inputs.
`features/policy.py` pins three tiers:

- **`SAFE_FEATURES`** (default, 13): solar geometry (`solar_elevation`,
  `solar_zenith`, `azimuth_sin`, `azimuth_cos`, `doy_sin`, `doy_cos`) + standard
  met (`air_temp_c`, `dew_point_c`, `rel_humidity`, `pressure_mbar`,
  `wind_speed_ms`, `wind_dir_sin`, `wind_dir_cos`). **No radiometry.**
- **`EXTENDED_FEATURES`** (ablation only, never silent): `uv_wm2`, `par_wm2`,
  `longwave_up_wm2`, `longwave_down_wm2`. Selected only by `features.set:
  extended`.
- **`FORBIDDEN_FEATURES`** (always fail): `CM3Up_Wm2_Avg` (GHI), `CM3Dn_Wm2_Avg`,
  `Net_Wm2_Avg`, `PSP_Wm2_Avg`/`CMP21_*`/`PSP_Avg` (diffuse), `kt`, `kstar`,
  `dhi`, `diffuse`, any `target_*` column, plus any configured target column.

**How it fails loudly:** `validate_features(names, target_columns=...)` raises
`ForbiddenFeatureError` (a `ValueError` subclass) naming the first offender;
`validate_manifest` reports forbidden feature columns as errors. There is no
silent drop — a leakage-prone request stops the run.

`FEATURE_GROUPS` (used for cross-attention tokens): `solar`, `temperature`,
`humidity`, `pressure`, `wind`, and `radiometry_aux` (extended set only).

---

## Alignment strategies

Registered in `data/alignment.py` (`get_strategy(name)`, extend via
`register_strategy`):

| Strategy | Stage | Status |
|---|---|---|
| `center_frame` | build-time pairing | **Default, fully wired.** Picks the frame nearest the window centre within `tolerance_minutes`; flags `ALIGNMENT_FAR`. |
| `mean_embedding` | dataset-level window | **Wired end-to-end.** `MultimodalEmbeddingDataset(window="mean_embedding")` averages the *available* embeddings of the co-frames in each row's window (same `day_id`, within `window_minutes/2`; missing co-frames skipped, all-missing falls back to the own frame) and emits a plain `embedding`. |
| `attention_pooling` | dataset-level window | **Wired end-to-end.** The dataset emits a zero-padded `embedding_seq` + `frame_mask`; `PrecomputedEmbedding` pools it either with a mask-aware mean (`temporal_pooling=mean`, default) or a small **learned** pooler (`temporal_pooling=attention`: one learnable **single query** over one `nn.MultiheadAttention` with `key_padding_mask=~frame_mask`; all-masked rows fall back to zeros). Both the **engine** (build/train) and the **evaluator** (rebuild/score) derive `temporal_pooling` from `data.alignment.strategy` via `temporal_pooling_for_strategy` — so an attention-pooled checkpoint, which carries the extra query/attention weights, is reloaded with the matching pooler instead of failing `load_state_dict`. (Cross-attention *fusion* is between modalities — distinct from this temporal pooler.) |

---

## Model ladder (V0–V7)

| # | Config | Model | Input | What it adds | Use when |
|---|---|---|---|---|---|
| V0 | `v0_climatology` | `climatology` | — | Constant train-mean per target (+ class freqs). The floor. | Always — the baseline every model must beat. |
| V1 | `v1_sensor_only` | `sensor_only` | sensor | MLP over geometry + met, no image. | Quantify the sensor-only ceiling. |
| V2 | `v2_image_only` | `image_only` | embedding | Visual signal alone (frozen DINOv2 embedding). | Quantify the image-only ceiling. |
| V3 | `v3_concat` | `concat` | embedding + sensor | First multimodal model: concatenated fusion. | Default multimodal baseline. |
| V4 | `v4_film` | `film` | embedding + sensor | Sensor modulates the visual embedding (FiLM, zero-init = concat at start). | Default experiment; usually ≥ concat. |
| V5 | `v5_multitask` | `film` | embedding + sensor | **Heteroscedastic DHI** (predicts its own variance) + k-index + sky (+ disabled cloud-fraction head). | Want calibrated DHI uncertainty. |
| V6 | `v6_film_finetune` | `film` | **image** | End-to-end: unfreezes the last 2 ViT blocks (`backbone_lr 1e-5`). | Have a GPU and want to adapt DINOv2 to sky images. |
| V7 | `v7_cross_attention` | `cross_attention` | embedding + sensor | Visual query attends to per-group sensor tokens. | Want the image to select relevant sensor context. |

**Embeddings vs. patch tokens vs. finetune.** V0–V5 and V7 train on **precomputed
embeddings** (cheap, CPU-friendly, but the backbone is frozen). V6 switches
`data.input_mode: image` to decode JPEGs and **finetune** the backbone — heavier,
GPU-only, and impossible on the embedding path. There is no separate patch-token
path today; cross-attention (V7) attends over *sensor* group tokens, not visual
patch tokens.

---

## How to add a sensor feature

1. **Policy** (`features/policy.py`): add the engineered name → source logger
   column to `SAFE_FEATURES` (or `EXTENDED_FEATURES` for radiometric auxiliaries).
   The insertion position is the canonical feature order.
2. **Engineering** (`features/engineering.py`): compute the column in
   `build_feature_frame` (read the source column, or derive from geometry/time;
   use `wind_components` for direction sin/cos). Cyclic quantities become a
   sin/cos pair.
3. **Groups** (`FEATURE_GROUPS`): add the name to the right group so
   cross-attention builds a token for it. `active_feature_groups` intersects with
   the resolved set, so a superset name is harmless.

Rebuild the manifest (`prepare-local`) so the new column is baked in; `n_features`
and the sensor-encoder width follow automatically.

## How to add a fusion strategy

1. **`modeling/fusion.py`**: implement a `nn.Module` exposing `out_dim` and a
   uniform `forward(visual, sensor, ...)`; set `needs_features = True` if it needs
   the raw standardized vector (like cross-attention). Register it in
   `build_fusion`.
2. **`modeling/registry.py`**: add a `_multimodal_builder("<name>")` entry to
   `MODEL_BUILDERS` (or a bespoke builder).
3. **Config**: add a `configs/allsky/models/<name>.yaml` fragment and an
   experiment that `extends` it. The arch params ride the permissive
   `ExperimentModelConfig` (`extra="allow"`).

---

## Reproduce an experiment (exact commands)

```bash
# 0) Prepare the dataset locally (frames → v2 manifest → day splits).
allsky prepare-local --config configs/allsky/data/local_prepare.yaml
allsky validate-dataset --config configs/allsky/data/local_prepare.yaml

# 1) Precompute DINOv2 embeddings (resumable; use backbone "fake" offline).
allsky precompute-embeddings --config configs/allsky/data/local_prepare.yaml

# 2) Train an experiment (V4 shown). --resume auto continues from last.ckpt.
allsky train --config configs/allsky/experiments/v4_film.yaml \
    --data-root output/allsky-mm/dataset \
    --out-dir  output/allsky-mm/experiments/v4_film/run \
    --device cuda --amp
allsky train --config configs/allsky/experiments/v4_film.yaml \
    --data-root output/allsky-mm/dataset \
    --out-dir  output/allsky-mm/experiments/v4_film/run \
    --device cuda --amp --resume auto        # after an interruption

# 3) Evaluate on the held-out test split (writes report.md + metrics.json + …).
allsky evaluate --checkpoint output/allsky-mm/experiments/v4_film/run/best.ckpt \
    --split test --data-root output/allsky-mm/dataset

# 4) Export a Colab bundle (manifest + splits + embeddings + configs).
allsky export-colab-bundle -o bundle.tar.gz \
    --config configs/allsky/data/local_prepare.yaml
```

Cross-model comparison table (in Python, from several eval report dirs):

```python
from allsky.evaluation.reports import compare_experiments
compare_experiments(
    ["output/allsky-mm/experiments/v3_concat/run/eval-test",
     "output/allsky-mm/experiments/v4_film/run/eval-test"],
    out_dir="output/allsky-mm/compare",
)  # writes comparison.csv + comparison.md
```

For CPU or a smoke run, swap `--device cuda --amp` for `--device cpu --no-amp`.

---

## Current limitations (honest)

- **No cloud-fraction ground truth.** The `cloud_fraction` head exists (V5) but is
  disabled everywhere: the manifest column is all-NaN. Enable it only once a label
  source exists.
- **Temporal windowing vs. cross-attention fusion are distinct.** `mean_embedding`
  and `attention_pooling` are now wired end to end — the engine builds the dataset
  with `window=data.alignment.strategy` and builds/evaluates the model with the
  matching temporal pooler. The dataset resolves each row's co-frame window and
  emits a pooled `embedding` or a padded `embedding_seq` + `frame_mask`; the
  encoder pools with a mask-aware mean or a learned attention pooler. Two honest
  bounds on that attention pooler: it is a **single-query** MHA (one learnable
  query attends over the window — a compact summary, not a full temporal
  transformer), and the co-frame window is built from **embeddings that must
  already be present for the same `day_id`** — missing co-frame reads are skipped,
  and a row whose whole window is missing falls back to its own frame (attention:
  an all-`False` mask yields a zero-pooled embedding with a logged warning). This
  is a **temporal** pooler over a frame window and is separate from
  cross-attention *fusion* (V7), which attends between the visual embedding and
  sensor group tokens. The V0–V7 shipped configs still default to `center_frame`;
  the windowed modes are opt-in via `data.alignment.strategy` + `window_minutes`.
- **DINOv2 / image mode (V6) and Colab are not executed here.** The real DINOv2
  backbone downloads via `torch.hub` on first local use only (never in CI —
  `FakeBackbone` there). The Colab notebook provisions CPython 3.14 with `uv` but
  has not been run on a real Colab runtime; the `[allsky]` extra installs a **CPU**
  torch wheel, so GPU runs need a CUDA torch build installed into the venv.
- **Single site, and a data gap.** Everything is for LabMiM/UFBA (Salvador-BA,
  −13.00/−38.51). The sensor archive currently ends **2026-04** while the first
  all-sky video is **2026-06** — there is no temporal overlap yet, so a real
  paired dataset needs logger files covering the camera dates.
```
