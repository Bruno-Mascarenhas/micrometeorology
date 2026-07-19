# `solrad_correction` — Documentation

A package for bias correction of diffuse solar radiation from the WRF model using machine learning. Designed to be generic — works with data from any meteorological station and geographic coordinates.

---

## Overview

WRF (Weather Research and Forecasting) often exhibits systematic bias when estimating diffuse solar radiation (`SW_dif`). This package trains ML models to correct this bias using observational data from meteorological stations as a baseline.

### Available Models

Current public support is intentionally scoped to SVM, LSTM, and Transformer.

| Model | Type | Input | When to Use |
|---|---|---|---|
| **SVM** | Scikit-learn (SVR) | Tabular (1 row = 1 sample) | Fast baseline, small datasets |
| **LSTM** | PyTorch (RNN) | Temporal windows (seq_len × features) | Capturing temporal dependencies |
| **Transformer** | PyTorch (Attention) | Temporal windows (seq_len × features) | Long-range relationships, larger datasets |

---

## Package Structure

```
src/solrad_correction/
├── __init__.py              # Version and docstring
├── config/                  # Modular config package + public facade
├── cli.py                   # CLI: solrad-run (Typer app)
├── cli_colab.py             # CLI: solrad-colab (Typer app, Colab defaults)
├── data/
│   ├── loaders.py           # CSV/Parquet loading, projection, dtype, row limits
│   ├── preprocessing.py     # Train-only fitted state + strict schema validation
│   └── splits.py            # Chronological split, walk-forward, temporal K-fold
├── features/
│   ├── engineering.py       # Lags, rolling means, differences
│   ├── temporal.py          # Hour, day of year, month + cyclic encoding (sin/cos)
│   └── sequence.py          # Sliding windows construction for LSTM/Transformer
├── datasets/
│   ├── tabular.py           # TabularDataset (X, y)
│   ├── sequence.py          # Dense and lazy windowed torch datasets
│   └── serialization.py     # Dataset artifact serialization
├── models/
│   ├── base.py              # BaseRegressorModel (ABC): unified interface
│   ├── sklearn_base.py      # Wrapper for scikit-learn regressors
│   ├── torch_base.py        # Base for PyTorch models (device, transfer learning)
│   ├── svm.py               # SVMRegressor (SVR)
│   ├── lstm.py              # LSTMRegressor + LSTMNet (nn.Module)
│   └── transformer.py       # TransformerRegressor + TimeSeriesTransformer
├── training/
│   ├── trainer.py           # Full training loop
│   ├── loops.py             # train_one_epoch(), evaluate_epoch()
│   ├── callbacks.py         # Early stopping, checkpointing
│   └── progress.py          # Progress with batch %, epoch %, and ETA
├── evaluation/
│   ├── metrics.py           # Metrics (reuses labmim + MAPE)
│   ├── reports.py           # ExperimentReport: saves metrics, config, history
│   └── comparison.py        # Comparative table between experiments
├── experiments/
│   ├── artifacts.py         # Canonical artifact layout and manifest
│   ├── pipeline.py          # Composable pipeline stages
│   └── runner.py            # Public compatibility wrapper
└── utils/
    ├── seeds.py             # Seed control (numpy, torch, random)
    ├── io.py                # JSON I/O, CSV predictions
    └── serialization.py     # Serialization: joblib (sklearn) / torch (checkpoint)
```

---

## Installation

```bash
# With CPU PyTorch:
uv pip install --torch-backend cpu -e ".[tcc]"

# With CUDA PyTorch (install torch first):
uv pip install --torch-backend cu121 torch
uv pip install -e ".[tcc-cuda]"
```

For local development, activate the `labmim` Conda environment before running
these commands. Conda remains the environment boundary for native scientific
packages; `uv pip` is used as the faster package installer inside that env. On
Windows, set `UV_PYTHON` to the active Conda interpreter first:

```powershell
$env:UV_PYTHON = (python -c "import sys; print(sys.executable)")
```

### Check if GPU is available

```python
from solrad_correction.utils.seeds import get_device
print(get_device())  # "cuda" or "cpu"
```

---

## Quick Start

### 1. Create a Configuration File

```yaml
# configs/tcc/experiments/my_experiment.yaml
name: svm_baseline_salvador
description: "SVM with hourly data from Salvador"
seed: 42

data:
  hourly_data_path: data/hourly/sensor_data.csv
  source_format: auto       # "auto", "csv", or "parquet"
  datetime_column: 0        # First CSV column by default; can be a column name
  datetime_index: true
  dtype_map: {}             # Optional pandas dtypes for loaded columns
  target_column: SW_dif
  feature_columns:
    - SWDOWN
    - T2
    - Q2
    - PSFC
  station_lat: -12.95
  station_lon: -38.51

split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  shuffle: false    # NEVER use shuffle for time series

preprocess:
  scaler_type: standard    # "standard", "minmax", "none"
  impute_strategy: drop    # "drop", "ffill", "mean", "interpolate"

features:
  add_temporal: true       # hour, day, month
  cyclic_encoding: true    # sin/cos
  lag_steps: []
  rolling_windows: []

model:
  model_type: svm
  svm_kernel: rbf
  svm_c: 10.0
  svm_epsilon: 0.1
  svm_gamma: scale

output_dir: output/experiments
```

### 2. Run the Experiment

`solrad-run` and `solrad-colab` are installed console scripts (Typer apps), so
they can be invoked directly after `uv pip install -e ".[tcc]"`:

```bash
solrad-run --config configs/tcc/experiments/my_experiment.yaml
```

Fast pre-flight checks that never train: `--validate-config` (validate the YAML
and exit), `--print-config` (print the resolved config as JSON), `--dry-run`
(validate without loading data), and `--smoke-test` (a small synthetic
CPU-safe experiment that needs no `--config`). Override flags — `--device`,
`--amp/--no-amp`, `--compile/--no-compile`, `--num-workers`,
`--pin-memory/--no-pin-memory`, `--limit-rows`, `--profile`, `--resume`,
`-o/--output-dir`, `-n/--name` — take precedence over the config file.

Or via Python:

```python
from solrad_correction.config import ExperimentConfig
from solrad_correction.experiments.runner import run_experiment

config = ExperimentConfig.from_yaml("configs/tcc/experiments/my_experiment.yaml")
report = run_experiment(config)
report.print_summary()
```

### 3. Output Structure

Each experiment generates a directory containing everything needed to reproduce it:

```
output/experiments/svm_baseline_salvador/
├── manifest.json                    # Artifact list, byte sizes, checksums
├── configs/
│   ├── config.yaml                  # Exact YAML config used
│   └── config_resolved.json         # Resolved config dumped to JSON
├── metrics/
│   ├── metrics.json                 # Results (RMSE, MAE, R², etc.)
│   └── training_history.csv         # Loss per epoch (if neural network)
├── predictions/
│   └── predictions.csv              # y_true vs y_pred
├── metadata/
│   ├── metadata.json                # Environment/device/git/model metadata
│   └── preprocessing_state.json     # Human-readable fitted preprocessing state
├── preprocessing/
│   └── preprocessing_pipeline.joblib
├── models/
│   └── model.joblib (or model.pt)
├── checkpoints/                     # best.pt and last.pt for neural runs
├── datasets/
│   ├── train/
│   ├── val/
│   └── test/
├── profiles/
│   └── profile.json                 # Written when --profile/runtime.profile is enabled
├── logs/
└── cache/
```

Migration note: root-level artifacts from the previous layout are no longer
written. Use `configs/config.yaml`, `metrics/metrics.json`,
`predictions/predictions.csv`, `models/model.*`, and
`preprocessing/preprocessing_pipeline.joblib`.

Crash safety: the fitted preprocessing pipeline is written right after fitting
and the trained model right after `fit()` returns — before prediction and
evaluation run — so a crash in a later stage never loses hours of training.

---

## Using Each Model

### SVM

```yaml
model:
  model_type: svm
  svm_kernel: rbf       # "rbf", "linear", "poly"
  svm_c: 10.0           # Regularization (higher = less regularization)
  svm_epsilon: 0.1      # Tolerance margin
  svm_gamma: scale      # "scale", "auto", or float
```

### LSTM

```yaml
model:
  model_type: lstm
  lstm_hidden_size: 64       # Neurons in hidden layer
  lstm_num_layers: 2         # Number of stacked LSTM layers
  lstm_dropout: 0.1          # Dropout between layers
  sequence_length: 24        # Temporal window size (hours)
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  patience: 10               # Early stopping: stops after 10 epochs without improvement
```

### Transformer

```yaml
model:
  model_type: transformer
  tf_d_model: 64             # Embedding dimension
  tf_nhead: 4                # Number of attention heads (d_model must be divisible)
  tf_num_encoder_layers: 2   # Number of encoder blocks
  tf_dim_feedforward: 128    # Internal FFN dimension
  tf_dropout: 0.1
  sequence_length: 24
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
  patience: 10
```

---

## Transfer Learning (Resume Training)

Training can be resumed from a previous checkpoint:

```yaml
model:
  model_type: lstm
  max_epochs: 100      # TOTAL epoch budget for the run (not additional)

runtime:
  resume: output/experiments/lstm_v1/checkpoints/last.pt
```

`model.max_epochs` is the **total** epoch budget: a run resumed at epoch 30
with `max_epochs: 100` trains only the remaining 70 epochs. Resuming also
restores the previous run's best monitor metric, so a resumed epoch that is
worse than the earlier best never overwrites `checkpoints/best.pt`, and early
stopping measures improvement against that best rather than restarting from
scratch. If the checkpoint is already at or past `max_epochs`, the run logs a
warning and exits cleanly without training; to extend a finished run, raise
`model.max_epochs`.

Resume restores the model, optimizer, scheduler, scaler, epoch, and
best-metric state from `last.pt`. Neural runs write:

- `checkpoints/best.pt` (best validation/train monitor)
- `checkpoints/last.pt` (latest epoch)

Each checkpoint saves:

- `model_state_dict` (model weights)
- `optimizer_state_dict` (optimizer state)
- `scheduler_state_dict` (if a scheduler is active)
- `scaler_state_dict` (if AMP scaling is active)
- `epoch` (epoch it stopped at)
- `config` (architecture parameters for reconstruction)
- `metadata` (checkpoint kind, monitor metric, run-wide `best_metric`/`best_epoch`)

### PyTorch Compile

`torch.compile` is opt-in because compile overhead is workload- and platform-
dependent. Keep it disabled for short CPU experiments and enable it explicitly
for longer GPU runs after a smoke test:

```yaml
runtime:
  torch_compile: true
```

Compiled runs persist checkpoints with plain `state_dict` keys, so they load
back into uncompiled modules. Legacy checkpoints written with `_orig_mod.*`
prefixed keys remain loadable — the prefix is normalized at load time.

### Inference Precision

`predict()` never runs under autocast, even when AMP was enabled for training:
inference is always full float32 and returns a float32 array. The same
checkpoint therefore produces identical predictions and metrics on CUDA and
CPU. AMP (`runtime.amp` / `--amp`) is a training-time optimization only.

### Google Colab GPU Training

Use the Typer-based `solrad-colab` entry point for remote GPU runs. It uses the
same override path and experiment runner as `solrad-run`, so local and Colab
artifacts stay aligned.

```bash
python -m pip install uv
uv pip install --torch-backend cu121 torch
uv pip install -e ".[tcc-cuda]"

solrad-colab \
  --config configs/tcc/experiments/lstm_hourly.yaml \
  --output-dir /content/drive/MyDrive/LabMiM/experiments \
  --device cuda \
  --amp \
  --num-workers 2
```

The compatibility script remains available:

```bash
solrad-colab --config configs/tcc/experiments/lstm_hourly.yaml --device cuda --amp
```

Useful Colab management flags:

- `--validate-config` checks YAML and overrides without training.
- `--print-config` prints the resolved config for notebook inspection.
- `--resume /path/to/checkpoints/last.pt` resumes optimizer, scheduler, scaler,
  epoch, model, and best-metric state.
- `--limit-rows N` runs a small GPU smoke pass before a full training campaign.

`solrad-colab` defaults to `--device cuda` and fails fast — before any data is
loaded — when CUDA is unavailable (the Colab default runtime has no GPU).
Enable a GPU runtime (Runtime > Change runtime type) or pass `--device cpu` to
train without a GPU.

---

## Data Leakage Prevention

The package implements multiple protection layers:

### 0. Format-aware Loading and Sensor Ingestion

Preprocessed hourly inputs can be CSV or Parquet:

```yaml
data:
  hourly_data_path: path/to/hourly.parquet
  source_format: parquet   # auto also works for .csv, .parquet, and .pq
  datetime_column: timestamp
  target_column: SW_dif
  feature_columns: [SWDOWN, T2, Q2, PSFC]
  dtype_map:
    SWDOWN: float32
    T2: float32

runtime:
  limit_rows: 10000        # applied during read where possible
```

When `data.load_columns` is empty and `feature_columns` is set, the loader
projects `feature_columns + target_column`. `data.load_columns` can be used for
manual projection when a custom feature stage needs additional columns.

Memory guardrails:

- Dense NumPy materialization for tabular and sequence datasets is checked before allocation.
- The default guard is `SOLRAD_MAX_ARRAY_GB=8`; raise it only when the host has enough RAM for the feature matrix, target vector, and model-specific copies.
- LSTM/Transformer experiments use `WindowedSequenceDataset`, which stores the 2-D base feature matrix and slices windows lazily instead of materializing the full 3-D `(n_windows, sequence_length, n_features)` tensor.
- The legacy `create_sequences()` helper still exists for small in-memory workflows, but now fails early if dense window construction would exceed the guardrail.
- For large experiments, prefer Parquet, `data.load_columns`, `dtype_map` to `float32`, and a realistic `runtime.limit_rows` during development.
- With Parquet inputs (including the Parquet cache written for CSV sources), `runtime.limit_rows` reads only the first batch through PyArrow instead of loading the whole file and slicing afterward. Head reads always include the stored pandas index columns, so the original `DatetimeIndex` survives row limits and column projection; numeric columns are never reinterpreted as epoch timestamps.

Raw LabMiM sensor directories can be loaded through the micrometeorology
ingestion, calibration, and aggregation stack before solrad preprocessing:

```yaml
data:
  sensor_data_path: data/raw/salvador
  sensor_pattern: "*.dat"
  calibrations_path: configs/micromet/calibrations.yaml
  resample_freq: 1h
  sensor_min_samples: 6
  target_column: SW_dif
  feature_columns: [SWDOWN, T2, Q2, PSFC]
```

### 1. Chronological Splitting

```
|←——— train (70%) ———→|←— val (15%) —→|←— test (15%) —→|
        past                 present           future
```

`shuffle=false` is the default. If enabled, a warning is emitted.

### 2. Preprocessing with Fit on Train

```python
pipeline = PreprocessingPipeline(scaler_type="standard")
train_pp = pipeline.fit_transform(train_df)   # ← Fit ONLY here
val_pp   = pipeline.transform(val_df)         # ← Apply train parameters
test_pp  = pipeline.transform(test_df)        # ← Apply train parameters
```

The mean and standard deviation used to normalize are calculated **only** on the training set. Validation and testing use these identical values.

Missing values are handled by `preprocess.impute_strategy` (`drop` is the
default). `interpolate` performs real interpolation of **interior** gaps only —
time-based when the frame has a `DatetimeIndex`, positional otherwise. Leading
and trailing NaNs are never extrapolated or forward-filled: those rows stay NaN
and are dropped, keeping the trailing edge causal-safe.

### 3. Sliding Windows (Sequence)

For LSTM/Transformer, window rows `[i, i + sequence_length)` are paired with
the target at the window's **last row**, `y[i + sequence_length - 1]`:

```
Window 1: [t₀, t₁, t₂, t₃] → target: t₃
Window 2: [t₁, t₂, t₃, t₄] → target: t₄
```

This is **concurrent bias correction** (features at time *t* correct the
biased estimate at time *t*), not one-step-ahead forecasting — the same task
the tabular SVM solves, so models are directly comparable. No row after the
target's timestamp ever enters a window.

Windows also never span temporal gaps: when the data carries a
`DatetimeIndex`, the base sampling frequency is inferred as the median
timestamp delta and any window containing a larger jump between consecutive
rows is dropped.

> **Breaking change:** earlier releases paired each window with the target one
> step *past* the window and allowed windows to cross gaps. Sequence metrics
> from those runs are not comparable with current runs.

### 4. Serialized Pipeline and State

The preprocessing state is saved twice:

- `preprocessing/preprocessing_pipeline.joblib` for executable reuse.
- `metadata/preprocessing_state.json` for audit/debugging.

The state records fitted input/output columns, feature and target columns, row
counts, imputation values, scaling values, and dropped-column reasons. Transform
is strict by default: missing or unexpected columns raise before scaling.

---

## Evaluation Policy

By default, metrics preserve each model's native test row set:

```yaml
model:
  evaluation_policy: model_native
```

This means SVM evaluates the full processed test set, while LSTM and Transformer
evaluate sequence targets starting at position `sequence_length - 1` (the last
row of the first window), minus any windows dropped for temporal gaps. Use this
default when you want model-specific production behavior.

For fair side-by-side SVM/LSTM/Transformer comparison, align SVM metrics to the
same target horizon used by sequence models:

```yaml
model:
  evaluation_policy: common_sequence_horizon
  sequence_length: 24
```

`common_sequence_horizon` intentionally changes the SVM metric row set. The
prediction CSV index follows the selected policy, so metric row counts and
timestamps remain explicit and reproducible.

`predictions/predictions.csv` always carries timestamps, under both policies
(`model_native` included): the index is taken from the built test dataset
itself, so it stays row-aligned with the model's predictions even when rows are
dropped for NaNs or temporal gaps. `save_predictions` raises a `ValueError` on
an index whose length does not match the predictions instead of silently
writing an untimestamped file.

---

## Profiling and Synthetic Benchmarks

Use `--profile` or `runtime.profile: true` to write
`profiles/profile.json` with stable stage timings and total stage time.

Synthetic benchmarks never read `data/`; they generate inputs under `scratch/`.
Run them all with `make bench`, or individually (a smoke test in
`tests/tcc/test_benchmarks_smoke.py` keeps them working against the current APIs):

```bash
python benchmarks/solrad_correction/loading.py --rows 10000 --features 16
python benchmarks/solrad_correction/preprocessing.py --rows 20000 --features 24
python benchmarks/solrad_correction/sequence_dataloader.py --rows 50000 --features 24 --sequence-length 24
python benchmarks/solrad_correction/artifact_checkpoint.py --hidden-size 32 --layers 2
```

---

## Comparing Experiments

```python
from solrad_correction.evaluation.comparison import compare_experiments

df = compare_experiments([
    "output/experiments/svm_baseline",
    "output/experiments/lstm_24h",
    "output/experiments/transformer_48h",
])
print(df)
#                     RMSE     MAE      R²      r      d     MAPE
# svm_baseline      45.23   32.10   0.847  0.921  0.958   18.5
# lstm_24h          38.67   27.45   0.893  0.946  0.972   15.2
# transformer_48h   36.12   25.89   0.908  0.953  0.978   13.8
```

---

## Feature Engineering

### Temporal

```yaml
features:
  add_temporal: true      # Adds: hour, day_of_year, month, weekday
  cyclic_encoding: true   # Converts to sin/cos (avoids 23→0 discontinuity)
```

Engineered columns always reach the models, even when `data.feature_columns`
is set: `feature_columns` selects the base data columns, and every column
added by an enabled feature stage (temporal, cyclic, lag, rolling, diff) is
appended to the model inputs on top of them.

**Why cyclic encoding?** Hour 23 and hour 0 are adjacent in time, but numerically far apart. The sin/cos encoding preserves this proximity:

```
hour=0  → sin=0.00, cos=1.00
hour=6  → sin=1.00, cos=0.00
hour=12 → sin=0.00, cos=-1.00
hour=23 → sin=-0.26, cos=0.97  ← close to hour=0
```

### Lags and Rolling Windows

```yaml
features:
  lag_steps: [1, 3, 6, 12, 24]        # Values from the last 1, 3, 6, 12, 24 hours
  rolling_windows: [3, 6, 12, 24]     # Rolling mean and standard deviation
  rolling_aggs: ["mean", "std"]
```

---

## Training Progress

During neural network training, progress is displayed in real-time:

```
  Epoch 1/100 [100.0%] ETA epoch: 0.0s | Overall:  1.0%
  Epoch 1/100 — train_loss=0.235412  val_loss=0.198765 (2.3s/epoch, ETA: 3m48s)
  Epoch 2/100 — train_loss=0.189234  val_loss=0.167892 (2.1s/epoch, ETA: 3m25s)
  ...
  Epoch 23/100 — train_loss=0.045123  val_loss=0.052345 (2.2s/epoch, ETA: 2m48s) [EARLY STOP]

✓ Training complete in 50.6s
```

### TensorBoard

Set `model.log_dir` to log per-epoch train/validation loss and learning rate
to TensorBoard. The `tensorboard` package is an optional dependency imported
lazily — it is only required when `log_dir` is actually set. Both the `tcc`
and `tcc-cuda` extras ship it.

---

## Adding a New Model

1. Choose the correct base:
   - **Sklearn** → inherit from `SklearnRegressorModel`
   - **PyTorch** → inherit from `TorchRegressorModel`

2. Implement the interface:

```python
from solrad_correction.models.sklearn_base import SklearnRegressorModel

class MyModel(SklearnRegressorModel):
    @property
    def name(self) -> str:
        return "MyModel"

    def __init__(self, param1: float = 1.0) -> None:
        from sklearn.ensemble import RandomForestRegressor
        self._estimator = RandomForestRegressor(n_estimators=100)

    @classmethod
    def from_config(cls, config):
        return cls(param1=config.custom_param)
```

3. Register it in `models/registry.py`:

```python
MODEL_REGISTRY["mymodel"] = ModelSpec(name="mymodel", kind="tabular")
```

Add the matching factory branch in `build_model()` in the same registry module.

For PyTorch, use `TorchRegressorModel` which automatically provides:
- Automatic GPU/CPU detection
- Transfer learning support
- Checkpoint saving
- Trainer integration (progress + early stopping)

---

## Frequently Asked Questions

### Can I use this for variables other than `SW_dif`?

Yes. Change `target_column` and `feature_columns` in the YAML config. The package is generic.

### Can I train with data from another city?

Yes. Change `station_lat` and `station_lon` in the config and provide the corresponding data. The name `solrad_correction` is generic and not tied to any specific location.

### Do I need a GPU?

No. SVM runs exclusively on CPU. LSTM and Transformer work on CPU but are significantly faster with CUDA. The code auto-detects and uses a GPU if available.

### How do I reproduce an experiment exactly?

1. Use the exact same `seed` in the config
2. Use the saved dataset in `experiments/<name>/datasets/`
3. Use the saved config in `experiments/<name>/configs/config.yaml`

```python
config = ExperimentConfig.from_yaml("output/experiments/lstm_v1/configs/config.yaml")
report = run_experiment(config)
```

### How do I see which features were used?

The saved dataset includes `feature_names.csv`:

```python
from solrad_correction.datasets.tabular import TabularDataset
ds = TabularDataset.load("output/experiments/svm_v1/datasets/train")
print(ds.feature_names)
```

### How do I apply an inverse transform to the predictions?

The saved pipeline allows you to undo the normalization:

```python
from solrad_correction.data.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline.load(
    "output/experiments/svm_v1/preprocessing/preprocessing_pipeline.joblib"
)
y_original = pipeline.inverse_transform_column(y_normalized, "SW_dif")
```
