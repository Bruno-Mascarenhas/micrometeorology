# Notebooks — Google Colab GPU Training Guide

How to run the LabMiM training notebooks on Google Colab with your models and
logs persisted to Google Drive.

## Opening a notebook in Colab

Every notebook in this repository can be opened directly from GitHub:

```
https://colab.research.google.com/github/Bruno-Mascarenhas/micrometeorology/blob/master/notebooks/<NOTEBOOK>.ipynb
```

| Notebook | Purpose |
|---|---|
| [`allsky_multimodal_colab.ipynb`](allsky_multimodal_colab.ipynb) | Train the **multimodal** all-sky pipeline (V0–V7 DHI / k-index / sky models) from a prepared Colab bundle |
| [`allsky_colab.ipynb`](allsky_colab.ipynb) | Train **SkyFusionNet** (cloud condition + diffuse radiation) on all-sky frames + sensors (legacy v0 pipeline) |
| [`tcc/02_colab_training.ipynb`](tcc/02_colab_training.ipynb) | Train the **solrad_correction** models (SVM / LSTM / Transformer) via `solrad-colab` |
| `exploratory/*.ipynb` | Local data exploration (sensor merging, WRF time series) — no GPU needed |

The multimodal notebook is **thin**: it provisions a CPython 3.14 venv with `uv`
(the package requires Python ≥ 3.14, which the Colab base runtime is not assumed
to provide), unpacks an `allsky export-colab-bundle` archive, and drives the
`allsky train` / `allsky evaluate` CLIs. Its default experiment is
`configs/allsky/experiments/v4_film.yaml`. It has **not** been executed on a real
Colab runtime from the dev environment — versions/timings are best-effort.

Before running anything: **Runtime → Change runtime type → GPU** (a T4 is
enough; both training CLIs enable AMP mixed precision on CUDA automatically /
via `--amp`). The first cell of each training notebook runs `nvidia-smi -L` —
if it fails, the GPU runtime is not enabled, and `solrad-colab` will refuse to
start rather than waste minutes loading data.

## Recommended Google Drive layout

Mount Drive in the notebook (`drive.mount("/content/drive")`) and keep one
folder for everything:

```
MyDrive/labmim/
├── data/            # sensor .dat files (LBM_lenta_*.dat), parquet caches
├── all-sky/         # allsky-YYYYMMDD.mp4 camera videos
├── allsky-mm/       # prepared multimodal bundle (bundle.tar.gz from export-colab-bundle)
└── runs/
    ├── allsky/      # legacy SkyFusionNet runs (best.pt, last.pt, runs/ tensorboard)
    ├── allsky-mm/   # multimodal runs (best.ckpt, last.ckpt, metrics.*, runs/, eval-*/)
    └── solrad/      # solrad_correction experiments
```

The multimodal notebook keeps `OUTPUT_DIR` on the local `/content` disk for fast
checkpoint I/O and copies it to `runs/allsky-mm/` on Drive (final cell). Build the
bundle locally with
`allsky export-colab-bundle -o bundle.tar.gz --config configs/allsky/data/local_prepare.yaml`
and drop it under `allsky-mm/`.

## Saving the best models to your Drive

Both pipelines write checkpoints wherever the output directory points, so the
only thing needed to persist models across Colab sessions is an output path on
Drive — the notebooks are already wired this way:

- **allsky**: set `train.out_dir` to a Drive path in the config cell
  (`out_dir: /content/drive/MyDrive/labmim/runs/allsky`). `best.pt`, `last.pt`,
  `config.json`, `metadata.json` and the TensorBoard events land there directly.
- **solrad_correction**: pass `--output-dir /content/drive/MyDrive/labmim/runs/solrad`
  to `solrad-colab`. Checkpoints, `metrics.json`, and `predictions.csv`
  (always timestamped) persist per experiment name.

Tip: keep *frame extraction* output (`allsky extract-frames`) on the local
`/content` disk — thousands of small JPEG writes are slow on Drive — and put
only the training `out_dir` on Drive. The pairing index is tiny and lives with
the run.

## Resuming after a disconnect

Colab sessions die; the checkpoints don't (they are on Drive):

```bash
# allsky multimodal — --resume auto finds last.ckpt in the run dir and continues,
# never overwriting a better best.ckpt (epochs is the TOTAL budget)
allsky train --config configs/allsky/experiments/v4_film.yaml \
    --data-root .../allsky-mm/data --out-dir .../runs/allsky-mm/out \
    --device cuda --amp --resume auto

# allsky legacy (SkyFusionNet)
allsky train --config config.yaml --index .../index.parquet --resume .../runs/allsky/last.pt

# solrad_correction — max_epochs is the TOTAL budget: resuming trains only the
# remaining epochs and never overwrites a better best.pt from the earlier run
solrad-colab --config experiment.yaml --output-dir .../runs/solrad --resume .../checkpoints/last.pt
```

## Watching metrics

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/labmim/runs   # covers both pipelines
```

## Session-restart checklist

1. Re-run the install cell (`%pip install ...`) — the environment resets.
2. Re-mount Drive.
3. Skip extraction/index steps if their outputs already exist on Drive.
4. Use the resume cell instead of the train cell.

Package documentation: [`docs/allsky.md`](../docs/allsky.md) ·
[`docs/solrad_correction.md`](../docs/solrad_correction.md) ·
[`docs/micrometeorology.md`](../docs/micrometeorology.md)
