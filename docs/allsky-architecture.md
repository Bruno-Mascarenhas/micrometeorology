# `allsky` multimodal architecture (v2 pipeline)

Reference for the **multimodal DHI-estimation stack** — the only all-sky
pipeline (see [`allsky.md`](allsky.md) for the CLI/config quickstart and shared
physics). It estimates **diffuse horizontal irradiance (DHI)**, a **clear-sky /
clearness index**, and a **sky-condition class** from an all-sky image — as a
precomputed embedding *or* end-to-end — plus non-radiometric sensor context.

All timestamps are naive local **America/Bahia** (fixed UTC-3, no DST): the
instrument-clock exception to the lab's tz-aware-UTC rule, because the Campbell
datalogger and the camera overlay both stamp local time. **The manifest layer is
the UTC boundary** — it writes tz-aware `timestamp_utc` from the site's pinned
offset, never the host timezone. `day_id` is the local calendar day.

---

## Local → bundle → Colab flow

```
  LOCAL WORKSTATION                                    GOOGLE COLAB (GPU)
  ─────────────────                                    ─────────────────
  allsky-*.mp4   LBM_lenta_*.dat
        │              │
        ▼              ▼
   extract frames → build v2 manifest ← feature policy
   (+ QC, mask/crop)   (geometry, k-index, targets)
        │              │
        │              ▼
        │        day-level splits (splits.json)
        ▼              │
   precompute-embeddings → embeddings/ (fp16 + index + meta)
        │
        ▼
   export-colab-bundle → bundle.tar.gz ──────────────► unpack → validate
                                                       → train → evaluate
  prepare-local / validate-dataset / train / evaluate    → copy to Drive
  run identically here for a fully-local workflow.
```

The bundle is the single portable artifact — manifest + sidecar, split, embedding
shards and configs, all with **relative POSIX paths** — so training on Colab is
byte-identical to training locally.

---

## Module map

Code lives under `src/allsky/`. The shared physics helpers (`video.py`,
`erbs.py`, `clearsky.py`, `preprocessing.py`, `geometry.py`, `lens.py`) stay
import-torch-free; site and solar-position primitives live in `labmim_core`.

| Package | Responsibility |
|---|---|
| `features/` | Anti-leakage **policy** (`safe`/`minimal`/`bare`/`extended`/`forbidden`), cyclic **engineering**, train-only **normalization**. |
| `data/` | `contracts` (v2 column registry, `QCFlag`, sky classes), `manifest` builder + atomic parquet + `.meta.json` sidecar, `validation`, `splits` (day-level, `split_id`), `alignment`, `folsom` (UCSD-Folsom adapter), lazy-torch `datasets` (the batch contract). |
| `embeddings/` | `backbone` (`VisualBackbone` protocol; DINOv2/DINOv3 via `torch.hub`, torchvision ResNet50/EfficientNetV2-S, `FakeBackbone` for tests), `storage` (safetensors shards + parquet index + meta), `extract` (resumable, atomic). |
| `modeling/` | `contracts`, `sensor_encoder`, `visual_encoder`, `backbone_families` (what fine-tuning needs per family), `geometry_adapter` (extra input channels), `transfer` (initialise from another run's weights), `fusion`, `heads`, `baselines`, `multimodal`, `registry`. |
| `training/` | `losses`, `engine` (`run_experiment`), `checkpointing`, `device`. |
| `evaluation/` | `metrics`, `evaluator`, `reports`. |
| `cli/` | `frames`, `prepare`, `embeddings`, `train`, `evaluate`. Every command imports torch lazily so `allsky --help` stays light. |

Top-level `allsky` modules outside those packages: `archive`/`snapshot`/`drive`
(camera mirror), `augmentation`, `bundle`, `config`, `frame_pixels`, `overlay`,
`provenance`.

---

## Artifact contracts

### Manifest v2 (`manifest.parquet`)

One row per paired sample. Columns (see `data/contracts.py`):

- **Identity**: `sample_id`, `timestamp_utc` (tz-aware UTC), `day_id` (local
  `YYYY-MM-DD`), `image_path` (relative POSIX against `data_root`),
  `frame_index`, `video`.
- **Solar geometry** (degrees): `solar_elevation`, `solar_azimuth`,
  `solar_zenith`. Azimuth is fed as the `azimuth_sin`/`azimuth_cos` pair.
- **Feature columns** per the active policy set (13 `safe`, 10 `minimal`,
  9 `bare`, +4 `extended`).
- **Targets**: `target_dhi`, `target_source` (`measured`/`erbs_pseudo`),
  `target_kindex`, `kindex_kind` (`kstar`/`kt`), `sky_class` (the four `labmim_core.sky.SKY_CLASS_VALUES`, `0/1/2/3`; `-1` =
  missing), `cloud_fraction` (nullable, all-NaN today), `qc_flags` (`int64`
  `QCFlag` bitmask: `LOW_SUN`, `SENSOR_GAP`, `ALIGNMENT_FAR`, `KT_ARTIFACT`,
  `FRAME_DARK`, `FRAME_SATURATED`). `build_manifest` sets the first four; the
  two frame bits are ORed in by `prepare-local`'s frame-QC pass, so they are
  present only when the per-video parquet carries `qc_frame_flags`.
- **Provenance (constant per row)**: `dataset_version`, `alignment_id`, and
  `split` — nullable, empty at build, filled by `attach_split_column`.

Night frames below `night_min_elevation_deg` (default 5°) are **dropped** at
build; `LOW_SUN` marks the surviving band up to the k-index floor
(`min_elevation_deg`, default 10°). The `KT_ARTIFACT` ceiling defaults per
k-index kind — `1.5` for `kstar` (cloud enhancement over the Haurwitz reference
legitimately exceeds `1.2`) and `1.2` for `kt`.

`sample_id` is built from the frame time by `sample_id_format`. This station
uses minute resolution (`allsky-%Y%m%d-%H%M`), one frame a minute; a source
whose frames land anywhere inside the minute needs a finer format and a prefix
of its own (Folsom uses `folsom-%Y%m%d-%H%M%S`). A build that produces two
identical ids raises rather than colliding silently.

### Meta sidecar (`manifest.parquet.meta.json`)

`dataset_version`, `alignment_id`, `feature_set`, ordered `feature_columns`,
`kindex_kind`, `target_source`, `sample_id_format`, `config_sha256`,
`code_version`/git commit, `created_at`, `row_count`, thresholds, an optional
`split_id`, and a content **`manifest_sha256`** tying a checkpoint to the exact
data it saw. Attaching the split column re-hashes it, by design.

The `timezone` block records `utc_offset_hours` and, for this station only, the
zone `name`. Any other site is recorded by offset alone — naming a zone the
pipeline never resolved would be an assertion nobody checked. Consumers read it
with `site_utc_offset_hours`; a manifest whose sidecar lost the block falls back
to this station's offset **loudly**, because guessing silently is how a Folsom
manifest would get Salvador's clock.

Both files are written atomically (temp + `os.replace`).

### Split artifact (`splits.json`)

`{split_id, seed, fractions, day_id→split assignment, created_at,
dataset_version, per-split day counts}`. Splits are **day-level**, never
row-level, so near-duplicate consecutive frames cannot cross splits.
Overwriting with different content requires `force=True`.

### Embeddings (`embeddings/`)

`embeddings-{i:05d}.safetensors` (one fp16 tensor per shard), `index.parquet`
(`sample_id → shard, row`), and `embeddings.meta.json` (backbone, revision,
pooling, dim, transform, `config_sha256`, `pixel_config_sha256`, count, dtype).

Extraction is resumable — the index is the source of truth, and a shard lands
atomically with its index rows. It **refuses to resume** into a store recording
a different backbone / revision / pooling / dim / `config_sha256` rather than
mixing two encoders' vectors, and no mismatch is negotiable — a store stamped by
a superseded digest formula is refused like any other, and re-extracting it means
`--no-resume`. An indexed store carrying no provenance at all is refused too:
there is nothing to check the incoming backbone against. `pixel_config_sha256` is
recorded beside the resume digest as provenance a reader can consult, never as a
migration key: it lets someone tell "the formula widened" from "the pixels
changed", and nothing reads it to carry a store across either.

### Checkpoint payload (`last.ckpt` / `best.ckpt`)

`torch.save`, atomic. Read back under torch's **restricted** unpickler
(`weights_only=True` plus an allowlist of the payload's own types): a checkpoint
travels through Colab and shared Drives, so it is not a trusted local file.
`--trust-checkpoint` (default off, on `train` and `evaluate`) opts back into the
unrestricted reader for a file you produced yourself.

Contains `model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`,
`epoch`, `global_step`, `epochs_no_improve`, `best_metric`, the full `config`
dump, `normalizers`, ordered `feature_columns`, `feature_groups`,
`dataset_version`, `split_id`, `manifest_sha256`, `backbone` info (image mode),
`code_version`, and `rng_state` for deterministic resume.

Resume is crash-safe: the train batch order is drawn from a dedicated sampler
generator re-seeded to `seed * 100003 + epoch` — a pure function of
`(seed, epoch)`, independent of the resume point — and metrics rows past the
resumed epoch are truncated before appending, so re-running the interrupted
epoch never duplicates it.

### Bundle (`bundle.tar.gz`)

Manifest + sidecar, `splits.json`, `embeddings/`, the configs used, and a
generated `BUNDLE_README.md`. All members relative; `validate_bundle` re-checks
the manifest hash against the sidecar.

---

## Batch contract

Keys emitted by the datasets (all `float32` unless noted):

| Key | Shape | Notes |
|---|---|---|
| `features` | `(B, F)` | standardized sensor vector (train-only stats) |
| `embedding` | `(B, D)` | embedding mode, `center_frame` or pooled `mean_embedding` |
| `image` | `(B, C, H, W)` | image mode, `[0, 1]`; `C` = 3 + any geometry channels |
| `embedding_seq` + `frame_mask` | `(B, T, D)` + `(B, T)` bool | windowed embedding mode |
| `image_seq` + `frame_mask` | `(B, T, C, H, W)` + `(B, T)` bool | windowed image mode, zero-padded to `seq_len` |
| `dhi`, `kindex` | `(B,)` | **raw physical units**, NaN = missing |
| `dhi_scale` | `(B,)` | divisor that produced `dhi`; exactly `1.0` under `raw` |
| `sky_class` | `(B,)` int64 | `-1` = missing |
| `cloud_fraction` | `(B,)` | NaN = missing |

Losses mask absent targets. The engine normalizes targets for the loss; metrics
and evaluation are always in physical units, and regression outputs are
denormalized with the stored `TargetNormalizer`.

**Target parameterization.** `raw` fits DHI in W/m². `clearsky_index` fits
`DHI / DHI_clearsky`, the same reference `skill_clearsky` is scored against, and
the engine and evaluator multiply back. The `raw` path keeps a scale of exactly
`1.0`, so existing arms are unchanged bit for bit.

The target normalizer is fitted from what the dataset **serves**, not from the
manifest column, so it describes by construction the quantity the head receives.

---

## Anti-leakage policy

DHI comes from the station radiometers (GHI drives `kt`/`k*`; the diffuse
pyranometer *is* the label), so those channels must never be inputs.
`features/policy.py` pins five tiers:

- **`SAFE_FEATURES`** (default, 13) — solar geometry (`solar_elevation`,
  `solar_zenith`, `azimuth_sin/cos`, `doy_sin/cos`) plus standard met
  (`air_temp_c`, `dew_point_c`, `rel_humidity`, `pressure_mbar`,
  `wind_speed_ms`, `wind_dir_sin/cos`). **No radiometry.**
- **`MINIMAL_FEATURES`** (10) — safe minus the thermohygrometer. The Gill
  MetSENS1 has been railed at 1000 °C / 1000 °C / 999 %RH since 2025-12-19;
  `mask_sentinels` turns each rail into NaN and `build_manifest` drops those rows
  whole, which over the camera archive costs 99.98 % of the dataset.
- **`BARE_FEATURES`** (9) — minimal minus the barometer. From 2026-08-10 13:05
  `BP1_mbar_Avg` is a constant 2.62 hPa, outside `SENTINEL_RANGES`, so every
  later row is dropped whole. What survives is solar geometry plus the
  **mechanical** anemometer, a separate instrument from the dead Gill unit.
- **`EXTENDED_FEATURES`** (ablation only) — `uv_wm2`, `par_wm2`,
  `longwave_up_wm2`, `longwave_down_wm2`.
- **`FORBIDDEN_FEATURES`** (always fail) — `CM3Up_Wm2_Avg` (GHI),
  `CM3Dn_Wm2_Avg`, `Net_Wm2_Avg`, `PSP_Wm2_Avg`/`CMP21_*` (diffuse), `kt`,
  `kstar`, `dhi`, `diffuse`, any `target_*`, plus any configured target column.

`validate_features` raises `ForbiddenFeatureError` naming the first offender;
there is no silent drop. `FEATURE_GROUPS` (cross-attention tokens): `solar`,
`temperature`, `humidity`, `pressure`, `wind`, `radiometry_aux`.
`active_feature_groups` drops groups the resolved set leaves empty.

---

## Alignment and temporal windows

Accepted names are the `AlignmentStrategyName` literal in `config.py`;
`data/alignment.py` implements build-time pairing and `data/datasets.py` the
dataset-level windowing.

| Strategy | Stage | Behaviour |
|---|---|---|
| `center_frame` | build-time pairing | Default. Frame nearest the window centre within `tolerance_minutes`; flags `ALIGNMENT_FAR`. |
| `mean_embedding` | dataset window | Averages the available members of each row's window (same `day_id`, within `window_minutes/2`). In embedding mode it emits a pooled `embedding`; in image mode a padded `image_seq` + `frame_mask` the encoder reduces by masked mean. |
| `attention_pooling` | dataset window | Padded sequence pooled by a **single-query** `nn.MultiheadAttention` with `key_padding_mask`. Embedding mode only — the learned pooler lives on `PrecomputedEmbedding`, and asking for it in image mode is rejected at config load. |

A window never crosses a day boundary: the night gap between the last frame of
one day and the first of the next is not a neighbourhood. `window_max_frames`
caps members, evenly subsampled, always keeping the ends — the image path needs
that cap because each member costs a decode plus a backbone forward.

Engine and evaluator both derive the pooler from `data.alignment.strategy`, so
an attention-pooled checkpoint reloads with the matching pooler instead of
failing `load_state_dict`. This temporal pooler is distinct from cross-attention
*fusion* (V7), which attends between modalities.

---

## Solar geometry as image channels

Geometry reaches the model as scalars beside the image, or as extra image
channels — one value per pixel saying where that pixel points relative to the
sun. A convolutional tokeniser can read the second and cannot read the first: a
scalar is constant across the frame, so it carries nothing about *which patch*
holds the circumsolar region that governs the diffuse split.

`allsky.geometry` builds every map through `LensCalibration`, which owns the
projection, the east-west mirror and the mount rotation. Channels:
`cos_sun_angle`, `cos_pixel_zenith`, `solar_disc`. The zenith channel is fixed
for a fixed camera — a spatial prior, carrying no information *between* samples.

**How the channels attach.** Widening the pretrained convolution would put the
new weights inside the backbone, where the freeze sweep owns them:
`patch_embed` is not part of `blocks`, so `unfreeze_last_n` never reaches it,
and channels initialised at zero and frozen at zero are structurally inert — the
arm returns the control's number and reads as a null finding about the physics
when it is a finding about the wiring. `GeometryPatchProjection` instead leaves
the pretrained convolution untouched and sums a **separate** zero-initialised
one: at initialisation the sum equals the pretrained projection exactly, and the
new convolution is a normal trainable module no freeze sweep owns.

Measured: one channel (`cos_sun_angle`) drops the elevation-band bias amplitude
from 17.47 to 2.65 W/m². The adapter weight moves from exactly 0.0 to
|w| = 445.93 after one epoch, which is how reachability is verified.

---

## Backbone families

A ViT and a convolutional network answer three questions differently, and
`modeling/backbone_families.py` is where each is answered:

1. **pooling** — how a `(B, C, H, W)` frame becomes `(B, dim)`. A DINOv2 ViT
   returns CLS and patch tokens; a ResNet has neither and pools its last map.
2. **stages** — what `unfreeze_last_n` counts. Flattening `Sequential` gives
   ResNet50 16 blocks, the ViT 12, EfficientNetV2-S 8. Unfreezing "the last 2"
   of one thinking it is the other reports a depth the run never used.
3. **the first convolution** — where extra channels attach: `patch_embed.proj`
   on a ViT, `conv1` on a ResNet, the stem of `features` on an EfficientNet.

A family that cannot answer raises `BackboneCapabilityError` rather than
guessing. `AVAILABLE_BACKBONES` today: DINOv2 `vits14/vitb14/vitl14/vitg14`,
DINOv3 `vits16/vits16plus/vitb16/vitl16`, `resnet50`, `efficientnet_v2_s`, and
`fake` for tests.

---

## Transfer learning

Transfer is not `--resume`: only the **weights** move, and the fresh run brings
its own optimizer, schedule and normalizers. Nie et al. (2022,
arXiv:2211.02108) found pre-train-then-transfer superior to local-only and to
joint training on fused datasets, reaching the local baseline with 80 % less
target data — which matters here, where the station has 55 training days.

The risk is silent partial loading: `load_state_dict(strict=False)` accepts a
checkpoint sharing three tensors and drops the rest, and the run then trains a
nearly-random network while its log says it transferred. So nothing is skipped
without being counted and named, and a mismatch *inside* the backbone — a
different architecture, not a different task — is an error.

**Direction is an invariant**: Folsom is always the source, this station always
the target. A station arm pre-training another station arm would report a
transfer gain that is really a longer schedule; a Folsom arm initialising from
anywhere would stop being reproducible from public data alone. Neither failure
announces itself in a metric, so `tests/allsky/test_configs_repo.py` asserts it.

---

## Model ladder (V0–V7)

| # | Config | Model | Input | What it adds |
|---|---|---|---|---|
| V0 | `v0_climatology` | `climatology` | — | Constant train-mean per target. The floor every model must beat. |
| V1 | `v1_sensor_only` | `sensor_only` | sensor | MLP over geometry + met, no image. |
| V2 | `v2_image_only` | `image_only` | embedding | Visual signal alone. |
| V3 | `v3_concat` | `concat` | embedding + sensor | First multimodal model. |
| V4 | `v4_film` | `film` | embedding + sensor | Sensor modulates the visual embedding (zero-init = concat at start). |
| V5 | `v5_multitask` | `film` | embedding + sensor | Heteroscedastic DHI + k-index + sky heads. |
| V6 | `v6_film_finetune` | `film` | **image** | End-to-end; unfreezes the last backbone stages. |
| V7 | `v7_cross_attention` | `cross_attention` | embedding + sensor | Visual query attends to per-group sensor tokens. |

V0–V5 and V7 train on **precomputed embeddings** — cheap, CPU-friendly, backbone
frozen. V6 sets `data.input_mode: image` to decode JPEGs and fine-tune.
Cross-attention (V7) attends over *sensor* group tokens, not visual patches.

### Experiment arms

Beyond the ladder, `configs/allsky/experiments/` carries one directory per arm,
each a seed sweep over a single question: `iso` (isotropic re-extraction, the
control), `sunmap`/`sunangle` (geometry channels), `kdindex`/`kdsun` (clear-sky
index target), `janela` (temporal window), `resnet50`/`effnet`/`dinov3s`
(backbone family), `folsom`/`transfer` (pre-train and transfer), `ceu` (the sky
condition as the primary target: sky + k* + clear-sky-index DHI heads, fine-tuned,
`cls+mean` pooling, the annealed recipe, three seeds for an ensemble), plus the
earlier `control`/`exposure`/`shuffled`/`finetune`/`anneal`/`loss`/`normlr`/`res`
sweeps. Every arm pins its seed and every run records the commit hash.

---

## How to add a sensor feature

1. **Policy** (`features/policy.py`) — add the engineered name → source column to
   the right tier. Insertion position is the canonical feature order.
2. **Engineering** (`features/engineering.py`) — compute it in
   `build_feature_frame`. Cyclic quantities become a sin/cos pair.
3. **Groups** (`FEATURE_GROUPS`) — add the name so cross-attention builds a token.

Rebuild with `prepare-local`; `n_features` and the encoder width follow.

## How to add a fusion strategy

1. **`modeling/fusion.py`** — an `nn.Module` exposing `out_dim` and a uniform
   `forward(visual, sensor, ...)`; set `needs_features = True` if it needs the
   raw standardized vector. Register it in `build_fusion`.
2. **`modeling/registry.py`** — add a `_multimodal_builder("<name>")` entry.
3. **Config** — a `configs/allsky/models/<name>.yaml` fragment plus an
   experiment that `extends` it.

---

## Reproduce an experiment

```bash
# 0) Prepare the dataset locally (frames → v2 manifest → day splits).
allsky prepare-local   --config configs/allsky/data/local_prepare.yaml
allsky validate-dataset --config configs/allsky/data/local_prepare.yaml

# 1) Precompute embeddings (resumable; backbone "fake" works offline).
allsky precompute-embeddings --config configs/allsky/data/local_prepare.yaml

# 2) Train. --resume auto continues from last.ckpt after an interruption.
allsky train --config configs/allsky/experiments/v4_film.yaml \
    --data-root output/allsky-mm/dataset \
    --out-dir   output/allsky-mm/experiments/v4_film/run \
    --device cuda --amp

# 3) Evaluate on the held-out test split.
allsky evaluate --checkpoint output/allsky-mm/experiments/v4_film/run/best.ckpt \
    --split test --data-root output/allsky-mm/dataset

# 4) Export a Colab bundle.
allsky export-colab-bundle -o bundle.tar.gz \
    --config configs/allsky/data/local_prepare.yaml
```

For CPU or a smoke run, swap `--device cuda --amp` for `--device cpu --no-amp`.
`allsky.evaluation.reports.compare_experiments([...], out_dir=...)` writes a
cross-model `comparison.csv` + `comparison.md` from several eval report dirs.

---

## Current limitations (honest)

- **No cloud-fraction ground truth.** The head exists (V5) but is disabled
  everywhere: the manifest column is all-NaN. Enable it only once a label source
  exists.
- **The station's paired archive is small.** 46,014 rows over 81 days, of which
  55 are training days. That is what makes transfer learning worth the machinery
  above, and it is why every metric is reported against persistence and
  clear-sky baselines rather than in isolation.
- **The sensor archive constrains the feature tier.** With the Gill MetSENS1
  railed since 2025-12-19 and the barometer gone since 2026-08-10, the `safe`
  tier drops almost every row; production arms run `minimal` or `bare`.
- **Colab is provisioned but lightly exercised.** The notebook installs CPython
  3.14 with `uv`; the `[allsky]` extra pulls a **CPU** torch wheel, so GPU runs
  need a CUDA build installed into the venv.
- **Single site for the target.** Everything is LabMiM/UFBA (Salvador-BA,
  −13.00/−38.51). Folsom (38.6°N) is a pre-training source only — a different
  climate with the same camera geometry, which is the trade the
  [dataset survey](open-sky-datasets.md) documents.
