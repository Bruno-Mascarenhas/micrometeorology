# All-sky serving on a workstation (`allsky snapshot` / `allsky watch`)

How to run the trained checkpoints against the live camera on a machine with
no GPU: what to install, which files to carry over, the two serving modes and
the exact commands, what lands on disk, a unit file to keep it running, and
what to check on the first day.

## Install without a GPU

The `allsky` extra resolves `torch` from the CPU wheel index pinned in
`pyproject.toml` (`[tool.uv.sources]`, index `pytorch-cpu`), so a plain sync
never pulls the CUDA stack. The first command is `make install` with the
extra added, flags as the `Makefile` spells them; the second is the `.venv`
variant from [allsky.md](allsky.md#installation):

```sh
# into the active micrometeorology Conda environment:
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked --inexact --extra allsky

# or into a project .venv, with no Conda at all:
uv sync --locked --extra allsky
```

`--locked` refuses a `uv.lock` that disagrees with `pyproject.toml`, which is
what you want on a machine that only serves. `make install-cuda` is the GPU
path and is not needed here.

## What to copy

Per chosen seed, two files from the run directory: `best.ckpt` (the epoch the
validation monitor picked) and `last.ckpt` (the final epoch). Each checkpoint
carries everything the serving path reads — the experiment `config`, the
ordered `feature_columns`, the train-split `normalizers`, the `frame_geometry`
the dataset was prepared with and the `sensor_pairing` rule — so no config
file, manifest or dataset travels with it (the payload is itemised in
[allsky-architecture.md](allsky-architecture.md#checkpoint-payload-lastckpt--bestckpt)).

Bring both files: the epoch the monitor picked and the final epoch need not
serve every head equally well, and the role options below read each head
group from whichever file the run's own evaluation favours. Decide from the
`eval-<split>/eval_metrics.json` that `allsky evaluate` writes beside each
checkpoint (see [allsky.md](allsky.md#cli-reference)), not from a rule of thumb.

Two constraints on what can be served:

- Copy **image-mode** checkpoints. `--checkpoint-block` refuses an
  embedding-mode one outright. `--checkpoint-frame` would serve it only if
  the embedding store path baked in at training resolves on this machine or
  the checkpoint carries its own encoding recipe — the watch has no
  `--embeddings-dir` to point it at a store copied alongside; only
  `allsky snapshot` takes that flag.
- `--checkpoint-frame` takes checkpoints trained under
  `alignment.strategy: center_frame` (one frame per row);
  `--checkpoint-block` takes `sensor_block` checkpoints whose
  `alignment.window_minutes` equals `--block-minutes`. Each side refuses the
  other kind at start-up (exit 1) instead of scoring it through the wrong
  path.

Checkpoints are read under torch's restricted unpickler. A file from your own
run that the restricted reader refuses is loaded with `--trust-checkpoint`;
never pass it for a file you did not produce.

Keep the checkpoints outside the repository and name them by their role: the
watch reads a checkpoint's role off its **file stem** — `best.ckpt` is `best`,
`last.ckpt` is `last`, any other stem is `other`. A renamed copy such as
`seed3-best.ckpt` plays `other` and is picked up only by the `all` selector.

## From a serving pin

When the checkpoints are the ones the public sky page is served from, name
the pin instead of the files: `allsky watch --serving configs/allsky/serving/ceu.yaml --out <dir>`
verifies each pinned SHA-256, builds every member once (a missing backbone
weight stops the start with exit 1 instead of a day of unscored frames) and
takes the roles and the elevation floor from the pin. The publisher that
reads this directory, the documents it writes and the systemd user units
that keep both running are in [allsky-site.md](allsky-site.md).

## The two modes

### One frame, now

```sh
allsky snapshot -k /opt/labmim/checkpoints/best.ckpt --out /opt/labmim/allsky-live
```

Captures `image.jpg`, writes `allsky-YYYYMMDD-HHMMSS.jpg`, its sidecar and
`<stem>.prediction.json` under `--out`, and exits. `-k` is
`--checkpoint`; `--sensor-csv` supplies the station features and is optional
(see [allsky-archive.md](allsky-archive.md#the-snapshot-prediction-caveat) for
what gets imputed without it).

### Continuous, per frame

```sh
allsky watch --out /opt/labmim/allsky-watch \
  --checkpoint-frame /opt/labmim/checkpoints/best.ckpt \
  --checkpoint-frame /opt/labmim/checkpoints/last.ckpt \
  --frame-sky-role best --frame-dhi-role last \
  --min-elevation-deg 10
```

Every checkpoint is loaded once at start-up and stays resident; a member that
cannot be built stops the watch (exit 1) instead of failing every frame.
Every `--poll-seconds` (default 20) the watch fetches the live frame, files it
under `frames/` and, when the sun is at or above the elevation floor,
scores it with every `--checkpoint-frame` given. With more than one the
members are averaged: `--frame-sky-role best` reads the sky probabilities
from the `best` members only, `--frame-dhi-role last` reads `dhi`, `kindex`
and `cloud_fraction` from the `last` members only; `all` (the default for
both) averages every member. A selector that picks no member stops the watch
at start-up (exit 1).

Every 5-minute block is still closed in this mode: its record is the mean of
the frame predictions written inside it (`source: frame_aggregate`).

### Continuous, per block

```sh
allsky watch --out /opt/labmim/allsky-watch \
  --checkpoint-block /opt/labmim/checkpoints/block/best.ckpt \
  --checkpoint-block /opt/labmim/checkpoints/block/last.ckpt \
  --block-sky-role best --block-dhi-role last \
  --min-elevation-deg 10
```

Frames are archived and scored only once their block closes — when a frame
stamped after the block's end arrives, or `--grace-seconds` (default 90)
after it — through `predict_block`, which feeds the block's frames as the
window the checkpoint was trained on. `--block-minutes` (default 5) must be
the checkpoint's own window; a block with fewer than `--min-frames`
(default 3) frames is skipped.

### Both

```sh
allsky watch --out /opt/labmim/allsky-watch \
  --checkpoint-frame /opt/labmim/checkpoints/frame/best.ckpt \
  --checkpoint-frame /opt/labmim/checkpoints/frame/last.ckpt \
  --frame-sky-role best --frame-dhi-role last \
  --checkpoint-block /opt/labmim/checkpoints/block/best.ckpt \
  --min-elevation-deg 10
```

Frames are scored as they arrive and the block record carries both the block
model's prediction (which is what its top-level `predictions` are) and the
frame aggregate, so the two can be compared block by block.

## What lands on disk

```
<out>/
  .state/                          TLS intermediate cache (see allsky-archive.md)
  frames/
    allsky-20260906-120100.jpg     the live frame
    allsky-20260906-120100.json    capture sidecar: captured_at, its source, headers
    allsky-20260906-120100.prediction.json   with --checkpoint-frame, sun above the floor
  blocks/
    20260906-1205.prediction.json  one per closed and scored block
    20260906-1210.skipped.json     one per closed block that was not scored
```

Timestamps are the camera's own naive local clock, read off the overlay
burned into the frame; a frame whose stamp had to come from the server header
stays on disk but is not filed under a block, and one named from the host
clock is deleted again (details in the `run_watch` docstring). The watch
restarts from these files: a block that has either record is never rewritten.

### Frame `prediction.json`

With one frame checkpoint, the `predict_snapshot` record: `predictions`,
`features` (values fed, which were imputed, the pairing gap), `model` and
`image`. With more than one, the ensemble record:

| key | content |
| --- | --- |
| `predictions` | `dhi` (W m⁻²), `kindex`, `cloud_fraction`, `sky_probabilities`, `sky_class` — whichever the selected members carry |
| `members[]` | per checkpoint: `checkpoint`, `role` (`best` / `last` / `other`), `heads` it contributed (`dhi`, `kindex`, `cloud_fraction`, `sky`), its own `predictions` |
| `models[]` | each member's `model` record (checkpoint path, architecture, code and dataset versions) |
| `image` | the frame scored |

### Block `prediction.json`

| key | content |
| --- | --- |
| `block_end`, `closed_by`, `n_frames` | naive local end of the block; `later_frame` or `grace`; frames on disk inside it |
| `source` | `block_model` when a block checkpoint scored it, else `frame_aggregate` |
| `predictions` | the `source`'s predictions, same keys as a frame's |
| `block_model` | present with a block checkpoint: its full `predict_block` record — `predictions`, `block` (frames fed, `representative`, `solar_elevation_deg`, `ignored`), `features`, `model`; or the ensemble record with `members` and `models` |
| `frame_aggregate` | present when any frame of the block has a `prediction.json`: mean `predictions` over those frames (probabilities averaged, class = their argmax), `n_frames` used, the `frames` used. A frame file the mean cannot take — a head that is not a finite number, a probability map missing a class another frame carries — is left out with a warning |

`n_frames` at the top counts the frames on disk; `block_model.block.n_frames`
counts the frames fed after the checkpoint's `max_frames` cap;
`frame_aggregate.n_frames` counts the frames that had a prediction.

### Block `skipped.json`

Every skipped record carries `block_end`, `closed_by`, `n_frames` and a
`reason`:

| `reason` | when | extra fields |
| --- | --- | --- |
| `insufficient_frames` | fewer than `--min-frames` frames, including a capture gap (`n_frames: 0`) | `min_frames` |
| `below_elevation_floor` | the block model's representative frame has the sun below the floor | `solar_elevation_deg`, `min_solar_elevation_deg` |
| `prediction_failed` | the block model raised; recorded once, not retried every poll | `error` |
| `no_frame_predictions` | no block checkpoint and none of the block's frames was scored: night, every frame prediction failed, or the frames were captured before there was a `--checkpoint-frame` (an archive-only history, or a watch restarted between a frame's capture and its prediction) | — |

A block model that raises is skipped even when its frames were scored: the
failure is recorded, not papered over by the aggregate.

A restart re-indexes the frames on disk and scores only what arrives from
then on; a frame that was never scored stays unscored, and its block, once
recorded, is never revisited.

## A systemd unit

`Restart=on-failure` covers a crash and the exit-1 refusals. The watch
handles no signal but Ctrl-C (`KeyboardInterrupt`), so `KillSignal=SIGINT`
makes `systemctl stop` the same clean exit 0; without it the process dies
under the default SIGTERM — nothing is left half-written, every record is
an atomic rename — and an explicit stop is not restarted either way. Replace
`/opt/labmim` with the install root and `labmim` with the service account.

```ini
[Unit]
Description=all-sky live frame scoring
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=labmim
WorkingDirectory=/opt/labmim/micrometeorology
Environment=OMP_NUM_THREADS=8
ExecStart=/opt/labmim/micrometeorology/.venv/bin/allsky watch \
  --out /opt/labmim/allsky-watch \
  --checkpoint-frame /opt/labmim/checkpoints/best.ckpt \
  --checkpoint-frame /opt/labmim/checkpoints/last.ckpt \
  --frame-sky-role best --frame-dhi-role last \
  --min-elevation-deg 10
KillSignal=SIGINT
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

`ExecStart` names the `allsky` of the environment the sync above created
(`$CONDA_PREFIX/bin/allsky` for the Conda path). `configs/micromet/default.yaml`
— the `sensor_limits` the snapshot screens a station row against — is found
from the installed package, not from the working directory: the loader walks
up from `micrometeorology/common/config.py` to the `pyproject.toml` of an
editable checkout, and falls back to the copy force-included in the wheel.
`WorkingDirectory` only anchors relative paths on the command line.

## Why `--min-elevation-deg 10`

The three local prepare manifests (`configs/allsky/data/local_prepare.yaml`,
`local_prepare_raw.yaml`, `local_prepare_iso.yaml`) all set
`night_filter.min_solar_elevation_deg: 10.0`, so no checkpoint trained on
them ever saw a frame with the sun under 10°. A checkpoint written since the
floor joined its provenance records it (`night_filter` in the payload) and
the watch takes it from there: `--min-elevation-deg` is then a cross-check,
refused when it disagrees, and required only for a checkpoint written before
the floor was recorded. Two members recording different floors are refused
too. Below the floor a frame is logged and left unscored and a block is
recorded as `below_elevation_floor` (block model) or `no_frame_predictions`
(frames only). A checkpoint trained under another manifest carries that
manifest's value, not 10.

## First-day check: are the live frames the frames the model saw?

Training frames were **extracted from the day's `.mp4` timelapse** — a
H.264 re-encoding of the camera's JPEGs, written through the prepare
geometry (`resize: 512` in the local manifests). The watch scores the
**original `image.jpg`** the camera publishes. The checkpoint's
`frame_geometry` is applied to the live frame too, so the mask, crop and
resize match; what remains is the codec, and nothing measured yet says how
much it costs. Compare, on the first served day:

1. Let the watch run for the day; its frames and predictions are under
   `<out>/frames/`.
2. On the next day, after the video is published (around 08:40 local, see
   [allsky-archive.md](allsky-archive.md#commands)), fetch it and extract its
   frames the way the training manifest's were: `--config` fixes the clock
   mapping (the overlay stamp) and `--resize` the pixel size, which
   `extract-frames` takes from the flag, not from the config:

   ```sh
   allsky sync-archive --data-dir /opt/labmim/all-sky --since 2026-09-06 --until 2026-09-06
   allsky extract-frames /opt/labmim/all-sky/videos/allsky-20260906.mp4 \
     --out /opt/labmim/domain-check/20260906 \
     --config configs/allsky/data/local_prepare.yaml --resize 512
   ```

   The frames are named `allsky-YYYYMMDD-HHMM.jpg` from the overlay stamp,
   with a `manifest.parquet` beside them.
3. Score the extracted frames with the same checkpoints, from Python — there
   is no command that scores a directory:

   ```python
   from pathlib import Path
   import pandas as pd
   from allsky.snapshot import predict_snapshot

   manifest = pd.read_parquet("/opt/labmim/domain-check/20260906/manifest.parquet")
   rows = []
   for frame_path, stamp in zip(manifest["frame_path"], manifest["timestamp"], strict=True):
       record = predict_snapshot(
           frame_path, "/opt/labmim/checkpoints/last.ckpt", timestamp=pd.Timestamp(stamp)
       )
       rows.append({"timestamp": pd.Timestamp(stamp), **record["predictions"]})
   pd.DataFrame(rows).to_parquet("/opt/labmim/domain-check/20260906/from-mp4.parquet")
   ```

4. Pair each extracted frame with the live `prediction.json` nearest in time
   (the cadence differs; a minute is a reasonable tolerance) and compare
   `dhi` and `sky_class` — MBE and RMSE of the diffuse across the day,
   agreement rate of the class. The station's PSP reading for the same
   blocks is the reference both should be judged against, not each other.

A large, systematic gap is a domain shift the checkpoint does not know about
and a reason to retrain on frames prepared from the original JPEGs; a small
one is a number worth recording in the run's notes before the watch is
trusted. Nothing here promises which of the two it will be.

## Cost

No measurement is versioned yet. Take one on the serving machine before
trusting the poll interval, and record it in the run's notes together with
the command, the checkpoints (architecture and `image_size`), the number of
frames scored and the thread count.

Memory: run the watch at its normal cadence for long enough to score several
distinct frames (a poll that reads the same overlay stamp again scores
nothing, so back-to-back polls measure only the start-up) and read
`Maximum resident set size`:

```sh
OMP_NUM_THREADS=8 /usr/bin/time -v allsky watch --out /tmp/allsky-cost \
  --checkpoint-frame /opt/labmim/checkpoints/best.ckpt \
  --checkpoint-frame /opt/labmim/checkpoints/last.ckpt \
  --min-elevation-deg 10 --max-polls 30
```

Latency: time `predict_snapshot` directly on one archived frame, with the
call the first-day check above already spells out, over a few repetitions
after the first (which pays the checkpoint load); `predict_block` on one
block's frames the same way for the block model.
