# The sky page: artifacts, producer and cadence (`allsky publish-site`)

What the public sky page (`site-labmim`, `ceu.html`) shows about the all-sky
model, where every number on it comes from, and the one command that writes it.
The page has two claims to support — *the network is good* and *the network
reads the camera* — and every artifact below exists to carry evidence for one
of them, with its provenance beside it and its limits written next to it.

The directory is `site/Ceu/` on the site side: deploy-only, gitignored,
`Disallow`-ed in `robots.txt`, `Cache-Control: no-cache` under the site's
`.htaccess`. Two files there are **not** this command's — `ktkd.json` and
`kt_cumulative.json` come from `labmim-sky` (`micrometeorology.cli.export_sky`,
this repository's own console script) and describe the pyranometer archive,
not the model — and this command never touches them.

## What is published

| File | Schema | Rewritten | Carries |
| --- | --- | --- | --- |
| `frame.json` | `labmim-allsky-frame-v2` | every publish | the latest scored live frame: prediction, solar geometry, the two counterfactuals, the attribution map's metadata and grid, the members that scored it, and a `status` block that says whether the frame is fresh and, when it is not, why |
| `allsky.jpg` | — | when the scored frame changes | that frame, the camera's own JPEG re-encoded to 1280 px wide |
| `input.jpg` | — | when the scored frame changes | exactly the array the network received (crop → pad → resize → preprocess), at the trained `image_size` |
| `attribution.png` | — | when the scored frame changes | RGBA occlusion-sensitivity map warped into `allsky.jpg`'s pixel grid; alpha is the normalised sensitivity, transparent outside the network's field of view |
| `timeline.json` | `labmim-allsky-timeline-v1` | every publish | the last *N* days of 5-minute block predictions from the watch on a regular axis, with the clear-sky reference per block, the measured diffuse when a station export is supplied, and every skipped block with its reason |
| `model.json` | `labmim-allsky-model-v1` | every publish (its numbers change only when the pin or a report changes; the stamp changes every run) | the model card: served checkpoints and how they were chosen, the dataset and its day split, the training curve, the held-out test metrics against the controls and the references, the per-class, per-band and per-day tables, the seed spread |

Every document of one publish shares one `version` stamp
(`YYYYMMDDTHHMMSSZ`, UTC), also written as `generated_utc` in the same compact
form the monitoring payload uses. The three images are rewritten only when the
scored frame changes; `frame.json` names each image with a short content hash
(`sha256_12`) the page appends to the image URL, so a fresh `frame.json` never
pairs with a cached image. The probe cache under the watch's `.state/` is keyed
by the capture stamp **and** by everything the probes depend on (the
attribution member's and the controls' digests, the target, the window and
stride, the device, whether a station export was fed) and is honoured only
while the three images it names are in the destination with the recorded
hashes; otherwise the frame is probed again and the images rewritten.

Timestamps of the *data* — `captured_at`, block ends, day ids — are **naive
station-local** on the camera's clock, `YYYY-MM-DDTHH:MM:SS`, as in every
all-sky record (see `allsky-operacional.md`); each document repeats the
`timezone` block (`America/Bahia`, −3 h). Day ids are full midnight stamps.

Numbers are rounded at the writer and every non-finite value becomes `null`
before serialisation (`allow_nan=False`); the page parses with a strict
`response.json()` and one bare `NaN` would cost the whole document. Files are
written atomically (temp sibling + rename) in the compact site encoding; images
through the same writer with an explicit `format=`. Write order within one
publish: images, then `timeline.json` and `model.json`, then `frame.json` last,
so a reader never sees a `frame.json` that names an image not yet on disk.

No document carries a filesystem path, a hostname or a credential: members are
named by arm id and seed, the station export by presence only. A publisher test
asserts no string value in any document starts with `/` or `output/`.

## Evidence map: which artifact carries which claim

**The network reads the camera.**

1. `model.json → evaluation.arms`: the same test rows scored by the served
   checkpoints and by two controls trained on the same manifest, the same
   `split_id` and the same targets — `climatology` (train-mean per target,
   class frequencies) and `sensor_only` (an MLP over the nine non-radiometric
   scalars: solar geometry and the mechanical anemometer). **The served arm
   is `image_only`: it receives the pixels and nothing else** — the scalars
   reach it only through the clear-sky denominator of its diffuse head. The
   comparison is therefore *information content*: pixels alone vs scalars
   alone vs the train mean, over identical rows and targets. It is the
   strongest evidence here, and it is not a same-inputs ablation; the card
   says so (`served.inputs.scalars_consumed: false`).
2. `frame.json → counterfactuals.no_image`: what the two controls say at the
   same timestamp without the picture — `sensor_only` (the scalars-only MLP)
   and `climatology` (the pinned train means) — each beside its own test
   error, so the reader can weigh the reference. Both are pinned checkpoints.
   The scalars-only control beats the train mean on k\* (MAE 0.213 vs 0.266)
   and is **worse** than it on the diffuse irradiance (RMSE 87.6 vs 73.6 W
   m⁻²), and the page says so; a live delta against a bad reference proves
   the reference bad, not the network good, which is why both are printed.
3. `frame.json → counterfactuals.overlay_neutralised`: the same frame scored
   with the camera's burned-in text band (clock, sensor temperature,
   **exposure time**) replaced by the network's mean level, sky disc intact.
   The exposure text is a shortcut strongly correlated with the target
   (`configs/allsky/experiments/shuffled/shuffled_s42.yaml` header, r =
   −0.83 with DHI) and it lies inside the field of view; this number says how
   much the served network leans on it. `delta = counterfactual − prediction`.
4. `attribution.png` and `frame.json → attribution`: occlusion sensitivity —
   the frame is scored again with one square window at a time replaced by the
   network's mean level, and each cell records the signed change of the head
   named in `attribution.target` (default `kindex`, the clear-sky index the
   checkpoints' early stopping monitored). The window and stride are in the
   record; the map is the absolute change, max-normalised, rasterised at the
   input resolution and warped through the recorded geometry onto
   `allsky.jpg`. `mass_by_region` splits the sensitivity between the sky
   disc, the overlay band, the padding and the rest, so the page can say where the
   sensitivity sits instead of leaving it to the eye. It is a causal probe
   of *where* the output depends on the pixels, at window resolution; it is
   not attention and is not named as such.

**The network is good.**

5. `model.json → evaluation`: RMSE, MAE and MBE together for the diffuse
   irradiance (W m⁻²) and the clear-sky index; accuracy, balanced accuracy,
   macro-F1, quadratic κ, the per-class recall and the full confusion matrix
   for the four sky conditions of Escobedo et al. (2009); all on the
   chronological held-out test days, with a one-day gap after validation.
   Skill is reported against four references, every one over the same paired
   rows with its `n` printed: climatology, sensor-only, the clear-sky diffuse
   reference, and 5-minute persistence.
6. `model.json → evaluation.references.persistence`: the previous **paired
   logger row** of the same day — the 5-minute mean the pyranometer wrote
   five minutes earlier — not the previous frame. Consecutive frames one
   minute apart pair to the same 5-minute row in 79 % of the test rows, so
   a frame-shifted persistence is the target itself most of the time and its
   skill (−0.66 in the evaluator's report) an artifact of the pairing; over
   the paired row the served arm's DHI skill is positive (about +0.24, the
   reference RMSE ≈ 24 W m⁻² against the model's 18) and its k\* skill is
   near zero. The horizon (5 min) and the number of distinct rows
   (`n_rows`) are printed beside every persistence number. The evaluator's
   own `rmse_persistence` is **not** published; see *Follow-ups* below.
7. `model.json → evaluation.references.clearsky`: named exactly — Haurwitz
   clear-sky GHI decomposed by the Erbs correlation at the clear-sky
   clearness index — with its own RMSE and MBE on the clear-class rows it
   should fit best, and the statement that the diffuse head is parameterised
   as a ratio to it.
8. `model.json → dataset` and `stratified`: one austral-winter season of 88
   days; `solar_elevation_range_deg` per split (the test days reach 69.5°
   where training stopped at 59.8°, so a quarter of the test rows are
   extrapolation, and from mid-September the noon sun is outside anything
   the network trained on); the class share of each split; DHI RMSE/MAE/MBE
   by elevation band, by sky class and **by test day**, so the headline is
   readable as the mixture it is.
9. `model.json → served.selection`: the pin was decided on the **validation**
   split, but the test split was consulted while the recipe family was
   developed (the arm configs' headers quote test numbers), so the test
   metrics are not a single-shot holdout. The card says so
   (`selection.selection_split`, the `recipe_history` caveat). The only clean
   holdout left is the live timeline against the pyranometer on days after
   2026-09-05, never used for any decision; the card names it as such and
   publishes it as `measured` when the export is supplied.
10. `model.json → seeds`: the members' test metrics side by side with
    `n_seeds`, min and max — two seeds are not a spread estimate and no
    standard deviation is printed. The served ensemble's metrics are
    recomputed from the pinned members' `predictions.parquet` only (mean
    prediction; sky = argmax of the mean probabilities, the rule the watch
    serves).
11. `model.json → domain_check`: the live frame is the camera's JPEG, the
    training frames came through the day's H.264 timelapse. Until the
    first-day check of `allsky-operacional.md` has been run and pinned
    (`reports.domain_check`), the field is `null` and the page prints "não
    medido".
12. `timeline.json`: the model running unattended, block by block, against
    the clear-sky envelope, with blocks above the training elevation flagged
    `extrapolation`. When a station export is supplied the measured diffuse
    of the same blocks is published beside the prediction (screened through
    the archive's sentinel and range gates); when it is not, `measured` is
    `null` and the page says the live comparison is pending — it does not
    substitute the test-split numbers for it.

## The pin: `configs/allsky/serving/ceu.yaml`

The best network is **declared**, never discovered: a versioned YAML names the
checkpoints, their SHA-256, the roles their heads play, the control checkpoint
the live counterfactual is scored with, the dataset and the evaluation reports
the card is built from, and the sentence that says why they were chosen.
Auto-picking the minimum RMSE across `output/` would silently compare arms
trained on different manifests and alignments.

The pinned checkpoints live in an **immutable serving directory** outside the
training tree (`~/labmim/serving/<arm>/<seed>/best.ckpt`, mode 0444): the
watch re-reads the file on every frame, so a `best.ckpt` rewritten by a later
run under `output/` would change the served weights without any refusal. The
file keeps its `best.ckpt` stem because the watch reads a member's role off
the stem. Pin paths may start with `~`, which the loader expands, so the
tracked pin names no machine; the systemd templates use `%h` for the same
reason. Every prose field this command publishes leaves the machine and falls
under the same rules as commit text.

```yaml
serving: true
id: ceuv3res512
label: DINOv3 ViT-S/16+ a 512 px, ajuste fino dos 12 blocos, só imagem
frame_checkpoints:
  - path: ~/labmim/serving/ceuv3res512/s42/best.ckpt
    sha256: 5b45dfc7…
  - path: ~/labmim/serving/ceuv3res512/s43/best.ckpt
    sha256: fc6f9daa…
frame_sky_role: all
frame_dhi_role: all
attribution_checkpoint: 0
attribution_target: kindex
min_elevation_deg: 10.0
controls:
  sensor_only:
    checkpoint: {path: ~/labmim/serving/ceucontrol/v1_sensor_only/best.ckpt, sha256: …}
    report: output/allsky-mm/experiments/ceucontrol/v1_sensor_only/run/eval-test
  climatology:
    checkpoint: {path: ~/labmim/serving/ceucontrol/v0_climatology/best.ckpt, sha256: …}
    report: output/allsky-mm/experiments/ceucontrol/v0_climatology/run/eval-test
reports:
  dataset: output/allsky-mm/dataset-iso-20260906
  members:
    - output/allsky-mm/experiments/ceuv3res512/ceuv3res512_s42/run/eval-test
    - output/allsky-mm/experiments/ceuv3res512/ceuv3res512_s43/run/eval-test
  training_history: output/allsky-mm/experiments/ceuv3res512/ceuv3res512_s42/run/metrics.csv
  domain_check: null
selection:
  criterion: >-
    menor MAE do índice de céu claro na validação entre os braços de quadro único
    treinados sobre o mesmo manifesto e o mesmo split cronológico; as duas sementes
    são servidas em média
  selection_split: val
  decided_on: 2026-09-11
```

Relative paths resolve against the working directory, like every
`output/...` path under `configs/allsky/`; the systemd units set
`WorkingDirectory` to the repository checkout.

**Verification, and what each failure blocks.** Checkpoint digests are
verified (`allsky.serving.verify_pinned_checkpoints`, hash cached by size and
mtime under the watch's `.state/`) before anything is written; a mismatch
blocks the whole publish. A member report whose `meta.split_id` or
`meta.manifest_sha256` differs from the served checkpoints', a control whose
`enabled_targets` differ, a dataset directory whose `splits.json` or manifest
sidecar disagrees with the checkpoints, or a missing report **block
`model.json` only**: `frame.json`, the images and `timeline.json` still
publish from a digest-verified pin, and the page says the card is pending. A
card with stale or mismatched provenance is worse than none; a live frame is
not held hostage to the slowest control's evaluation.

`allsky watch --serving configs/allsky/serving/ceu.yaml` reads the same file
for its frame checkpoints, roles and elevation floor, verifies the digests
and **builds every member once at start-up**, exiting 1 when one cannot be
built (missing DINOv3 weights, a moved file): a watch that starts cleanly and
then swallows a per-frame error all day, scoring nothing with exit 0, is the
failure the service manager cannot see.

## The producer

```
allsky watch  --serving configs/allsky/serving/ceu.yaml --out <watch>
                └── frames/<stem>.jpg + .json + .prediction.json     (per poll)
                └── blocks/<YYYYMMDD-HHMM>.prediction.json | .skipped.json

allsky publish-site --serving configs/allsky/serving/ceu.yaml \
                    --watch-dir <watch> --out <site>/Ceu \
                    [--days 3] [--sensor-csv FILE] [--device cpu|cuda]
                    [--prune-frames-days 14] [--rclone-remote NAME:path]
                └── allsky.jpg input.jpg attribution.png timeline.json model.json frame.json
```

The watch is the only process that talks to the camera; the publisher reads
its directory. `frame.json` reproduces the watch's own prediction for the
latest **scored** frame — single-member and ensemble records are normalised
through one function into the same `members[]` shape — and adds what only the
publisher computes, from the `attribution_checkpoint` member and the
`controls.sensor_only.checkpoint`: the occlusion map and the two
counterfactuals.

Exit codes: 0 published; 1 the pin is malformed or failed verification, the
watch directory holds neither `frames/` nor `blocks/`, or an artifact could
not be written; 2 published, but the watch looks dead (see *Freshness*). A
`timeline.json` or `model.json` that cannot be built (a refused report, an
unreadable export) keeps the previous file and is logged; the frame still
publishes.

### `frame.json` (`labmim-allsky-frame-v2`)

| key | content |
| --- | --- |
| `schema`, `version`, `generated_utc`, `timezone` | as above |
| `status` | `{scored, reason, reason_pt, latest_scored_at, solar_elevation_deg, watch_alive, camera_clock_offset_s, camera_clock_drift_s}`; `reason` ∈ `fresh` / `night` (sun below the floor, the day's last scored frame stays) / `no_scored_frame` (nothing scored yet, whatever the hour) / `watch_stale` (the sun has been above the floor for the whole liveness window and the newest prediction record was **written** more than `stale_after_blocks` blocks ago — liveness is judged on the host clock, the data stamps stay the camera's; the first blocks after sunrise are a grace period); `camera_clock_offset_s` is the scored frame's `server_last_modified_as_local − captured_at` and `camera_clock_drift_s` the same with its whole-hour part removed — on the first served day the raw offset was observed at −3 h (a zone-label error in the header) and later at +9 s, so only the residual is read as drift; it is the cheapest daily health number for the geometry inputs |
| `captured_at`, `captured_at_source` | the scored frame's stamp (naive local) and its source (`overlay` for every filed frame) |
| `image` | `{file, sha256_12, width, height, original_width, original_height, source: "camera JPEG", alt_pt}` |
| `input` | `{file, sha256_12, size, geometry: {crop, pad, resize}, preprocess: {overlay, band_fraction, roi_radius_fraction}, content_box: {left, top, width, height}, alt_pt}` — the geometry and the preprocessing the checkpoint records, and the rectangle of `input.jpg` holding camera pixels (the rest is padding) |
| `field_of_view` | `{left, top, width, height}` in `allsky.jpg` pixels: the rectangle the network saw |
| `solar` | `{elevation_deg, azimuth_deg, clearsky_ghi_w_m2, clearsky_dhi_w_m2, extrapolation}` at `captured_at`; `extrapolation` is true above the training split's maximum elevation |
| `prediction` | `{dhi_w_m2, kindex, sky: {name, condition, id, name_pt, probabilities: {i, ii, iii, iv}}}` — the watch's record; `condition` is 1–4 and `id` is `i`–`iv`, both present because the exporter's class integer is 0-based |
| `counterfactuals` | `no_image: {sensor_only: {…}, climatology: {…}}`, each `{kind, member, dhi_w_m2, kindex, sky, delta, test: {dhi_rmse, kindex_mae}, scalars_imputed, station_export}` with `delta = counterfactual − prediction` (the served ensemble's), the control's own test error beside it, and which scalar columns were imputed at the training mean (all anemometer columns when no `--sensor-csv` was supplied — the controls then run on solar geometry alone); `overlay_neutralised: {kind, member, base, dhi_w_m2, kindex, sky, delta, band: {…input px}}` with `delta = counterfactual − base`, both from the attribution member on the same planes |
| `attribution` | `{file, sha256_12, alt_pt, summary_pt, method: "occlusion_sensitivity", fill: "network_mean_level", window_px, stride_px, target, member, base_value, grid_shape, grid (signed Δtarget, row-major, input frame), mass_by_region: {disc, overlay_band, pad, other}, peak: {row, col}}` (four disjoint regions that sum to one; `other` is the camera pixels around the lens horizon); `summary_pt` is generated from the data ("62 % da sensibilidade no disco do céu, 30 % na faixa de texto") |
| `members` | one per served checkpoint: `{name, seed, role, heads, checkpoint_sha256, code_version}` |
| `targets` | glossary: `{kindex: {kind, symbol, label_pt, definition_pt, reference}, dhi: {symbol, unit, label_pt}}` — the served index is **k\***, GHI over the Haurwitz clear-sky GHI, not the Kt the archive charts use |
| `sky_conditions` | the four conditions as `ktkd.json` publishes them (`conditions[]` with `condition`, `id`, `name`, `name_pt`, `kt_range`) |
| `caveats`, `references` | prose with `[[key]]` markers, and `{key: {short, citation, url}}` |

When nothing was scored yet `status.scored` is false, `captured_at` and the
prediction keys are `null`, the images are left as they were and `frame.json`
is still rewritten with the shared `version`.

### `timeline.json` (`labmim-allsky-timeline-v1`)

The monitoring page's convention: one regular axis and parallel arrays with
`null` where a block has no value, so a night or an outage is a hole, never a
segment drawn across it.

| key | content |
| --- | --- |
| `axis` | `{start, step_minutes, count}`; `start` is the first block end of the window, naive local |
| `series` | parallel arrays of length `count`: `dhi_w_m2`, `kindex`, `condition`, `p_i`, `p_ii`, `p_iii`, `p_iv`, `n_frames`, `solar_elevation_deg`, `clearsky_dhi_w_m2`, `clearsky_ghi_w_m2`, `extrapolation`, `measured_dhi_w_m2` |
| `source` | per block: `frame_aggregate` / `block_model` / `null` |
| `skipped` | sparse: `[{t, reason}]` with `reason` ∈ `insufficient_frames`, `below_elevation_floor`, `no_frame_predictions`, `prediction_failed`; `reason_labels_pt` maps each to its sentence |
| `latest` | `{last_scored_block, last_block_status, reason}` |
| `measured` | `null`, or `{source_column: "PSP_Wm2_Avg", screening: "sentinels + sensor_limits", n}`; a logger row stamped `t` averages `(t − 5 min, t]` and matches the block ending at `t` directly, no offset |
| `measured_status` | `{available, reason: ok / no_export / export_stale / no_valid_rows / interval_mismatch, last_row_at, source_label}` — so the page prints "última leitura da estação: 2026-08-11" instead of "pendente" forever; an export whose rows are not on the block grid is refused as measured, since its rows are not the means the timeline compares |
| `live` | `null`, or the running comparison since `selection.decided_on` over every block in `blocks/` joined with the export: `{since, n_days, n_blocks, dhi: {rmse, mae, mbe}, kindex: {mae}, sky: {balanced_accuracy}}` — the only clean holdout, never used for a decision. `kindex.mae` and `sky.balanced_accuracy` are `null` until the export carries the global channel the k\* and the Kt bands are computed from |
| `days` | per day: `date` (midnight stamp), `blocks_scored`, `blocks_skipped`, `frames`, `condition_share` |
| `targets`, `sky_conditions`, `caveats` | as in `frame.json` |

The clear-sky reference and the elevation are evaluated at the block centroid
(`t − step/2`), the instant `predict_block` uses.

### `model.json` (`labmim-allsky-model-v1`)

| key | content |
| --- | --- |
| `served` | `{id, label, members: [{name, seed, role, checkpoint_sha256, epoch, best_metric}], attribution_member, roles, selection: {criterion, selection_split, decided_on}, architecture: {name, backbone, image_size, pooling, unfreeze_last_n, trunk_hidden, parameters}, inputs: {image_geometry, preprocess, scalars_consumed: false, radiometry_forbidden: true}, targets, training, code_version}` |
| `dataset` | `{rows, days, days_assigned, period, season_note, split: {strategy, gap_days, train/val/test: {days, start, end, rows, solar_elevation_range_deg, class_share}}, manifest_sha256, split_id, dataset_version, target_source, min_elevation_deg, frame_geometry, camera, frames_from}` |
| `training_curve` | `{epochs, train_kindex_mae, val_kindex_mae, train_dhi_mae, val_dhi_mae, train_sky_balanced_acc, val_sky_balanced_acc, best_epoch}` of the seed named in `reports.training_history` |
| `evaluation` | `{split: "test", n, arms, references, skill, per_class, per_day, sky_conditions, targets}`; `arms` has the served ensemble, each member, `sensor_only` and `climatology`, every one with `dhi: {rmse, mae, mbe, r2, n}`, `kindex`, `sky: {accuracy, balanced_accuracy, macro_f1, kappa_quadratic, confusion, per_class}`; `references.clearsky` `{label, dhi_rmse, model_rmse_on_paired, skill, n, on_clear_rows: {rmse, mbe, n}}`, `references.persistence` `{label, horizon_minutes: 5, dhi_rmse, model_rmse_on_paired, skill, n, n_rows}`; `skill` has one entry per reference over the same paired rows; `per_day` rows: `{date, n, n_frames, class_share, rmse_model, rmse_persistence, rmse_clearsky}` — the three RMSEs over the same `n` paired rows (frames with a persistence and a clear-sky reference), `n_frames` the day's frames |
| `stratified` | the attribution member's DHI RMSE/MAE/MBE by `solar_elevation` band and by `sky_class` |
| `seeds` | `{n_seeds, members: [...], range: {metric: {min, max}}}` |
| `attribution_summary` | method, window, stride, target — the same as `frame.json` so the card explains the live map |
| `domain_check` | `null` or the pinned first-day check `{day, n, dhi_mbe, dhi_rmse, class_agreement}` |
| `caveats`, `references` | prose and bibliography the page renders |

## Cadence, freshness and the units that keep it running

Two `systemd --user` units on the serving machine, templates under
`deploy/systemd/` (user units: no `User=`, `WantedBy=default.target`, no
`network-online` dependency — the watch's own capture loop survives outages;
absolute `ExecStart` to the environment's `allsky`, absolute `--out` paths,
`Environment=` for `OMP_NUM_THREADS`, `ALLSKY_DINOV3_REPO` and
`ALLSKY_DINOV3_WEIGHTS`):

- `allsky-watch.service` — `allsky watch --serving … --out ~/labmim/allsky-watch/<pin id>`;
  `KillSignal=SIGINT`, `Restart=on-failure`, `MemoryMax=`, thread cap 6. One
  watch directory **per pin id**: block and frame records carry no pin
  identity, and a timeline read across a re-pin would mix two networks.
- `allsky-publish.timer` (`OnCalendar=*:0/10`) → `allsky-publish.service`,
  `Type=oneshot`, `Nice=10`, thread cap 4, `OnFailure=allsky-publish-failed.service`
  which leaves a marker file the operator can watch. Exit 2 (documents
  written, watch looks dead) counts as a failure on purpose, so the marker
  fires while the timer keeps publishing.

`loginctl enable-linger <user>` or the units stop with the login session.

**Freshness.** The publisher computes it: when the sun is above the pin's
floor and the newest `frames/*.prediction.json` is older than
`stale_after_blocks` (default 3) blocks, `status.watch_alive` is false,
`status.reason` is `watch_stale` and the command exits 2. The page prints the
age of `captured_at` beside the frame and the sentence for the reason.

**Retention.** The camera advances about once a minute in daylight and a
frame is ~0.9 MB, so `frames/` grows 0.6–1.3 GB per day. `publish-site
--prune-frames-days N` deletes frames (JPEG, sidecar, prediction record) older
than *N* days; `blocks/` is never pruned (288 small records a day, and the
timeline reads them). *N* must exceed `--days`; the default is 14 and the
value is the operator's call.

**Upload is a separate, explicit step.** The production host takes FTP only
and its credentials are the operator's; `publish-site --rclone-remote
NAME:path` copies exactly the six artifacts, one `rclone copyto` per file in
the same order as the local writes (rclone walks a file list alphabetically,
so a single batch could land `frame.json` before its images; never `sync`,
which would delete the `labmim-sky` files), and does nothing when the flag is
absent. Production is
several site versions behind `main` and answers 404 for `/Ceu/`: the first
publish only lands after a full site deploy.

## Decisions left to the operator

- The FTP remote for rclone (plain FTP vs explicit TLS given the host's broken
  chain), where `rclone.conf` lives so the user unit can read it, and the
  remote path.
- The retention window `N`.
- Whether to serve the block model (`ceubloco` arms, positive persistence
  skill, one prediction per closed 5-minute block) instead of, or beside, the
  frame ensemble; the pin has room for `block_checkpoints` and the watch
  already scores blocks.
- Whether to add a daily `allsky-domain-check.timer` that scores the
  previous day's timelapse frames with the pinned member and pairs them with
  the live records (the first-day check of `allsky-operacional.md`, made
  routine); until it exists `domain_check` stays `null`.
- Re-pinning when the GCP queue finishes: never automatic, and more than a
  path edit. The candidate's `eval-test` must be on the same manifest and
  split as the controls it names — the 1024 px queue trains on
  `dataset-iso-1024-20260910`, so `v0_climatology` and `v1_sensor_only` are
  retrained there first and `reports.dataset` points there. Then: copy the
  chosen `best.ckpt` into the serving directory, hash it, edit the pin, give
  the watch a new `--out` directory named by the new pin id, restart it.
- The block estimator (`ceubloco` arms): it matches the 5-minute pyranometer
  mean the timeline is compared with and has positive skill against the
  paired-row persistence too, but its numbers are on another manifest (so it
  needs its own controls and card), its target is smoother (RMSE not
  comparable one-to-one with the frame arms), and it can only score a closed
  block, so it cannot drive "Agora". The pin has no `block_checkpoints` field
  yet; adding one is a schema change.

## Documentation and consumers this change touches

- micrometeorology: `README.md` (documentation table row for this file;
  the `allsky-operacional.md` row mentions `--serving`), `docs/allsky.md`
  (module map: `serving.py`, `attribution.py`, `publish/`; CLI reference:
  `watch --serving`, `publish-site`), `docs/allsky-operacional.md`
  (cross-reference to this file), `deploy/systemd/`.
- site-labmim: `src/template/pages/ceu.html`, `site/assets/js/ceu.js`,
  `site/assets/css/components.css`, the `sky` page SEO description in
  `src/sites/<id>/pages.js`, and the comments in `.htaccess`,
  `src/datasets/*.js` and `scripts/site-builder/validate.js` that still
  describe a segmentation mask. `npm run build` after any `src/**` change,
  then `lint:all`, `format:check`, `build:check`.

## Follow-ups this design surfaced

- `allsky.evaluation.evaluator._previous_observation_same_day` shifts by one
  **frame**; on this manifest that is the same 5-minute row for 79 % of the
  rows, so the evaluator's `rmse_persistence` / `skill_persistence` are not a
  persistence forecast. The card computes persistence over the paired logger
  row instead. The evaluator should follow, with a test that fails on a
  frame-shifted reference; not changed here because it moves every report's
  numbers.
- The watch reloads each checkpoint on every frame; loading once per member
  at start-up (the `ServedModel` seam the publisher uses) would cut per-frame
  latency and is where a digest of the weights that scored a frame would be
  recorded.

## What this does not claim

- No segmentation: the model has no segmentation head. `attribution.png` is
  an occlusion-sensitivity map and is labelled as one on the page.
- No cloud fraction: the head is wired but disabled (no ground truth), so no
  document carries a `cloud_fraction`.
- The sky conditions the network predicts are the Kt bands of the
  pyranometer's 5-minute mean it was trained to reproduce, not a visual label.
- The domain gap between the live JPEG and the timelapse frames is unmeasured
  until `domain_check` is pinned.
- Persistence is scored over the paired 5-minute logger row; the page
  reports the number as measured, with its horizon and its `n_rows`.
- Test-time augmentation: the reports the card is built from are single-pass
  (`meta.tta_rotations == 0`), the estimator the watch runs; a report with
  rotations is refused for the card, which states `evaluation.inference`.
