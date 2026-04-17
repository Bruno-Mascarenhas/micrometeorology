# `solrad_correction` — Documentation

A package for bias correction of diffuse solar radiation from the WRF model using machine learning. Designed to be generic — works with data from any meteorological station and geographic coordinates.

---

## Overview

WRF (Weather Research and Forecasting) often exhibits systematic bias when estimating diffuse solar radiation (`SW_dif`). This package trains ML models to correct this bias using observational data from meteorological stations as a baseline.

### Available Models

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
├── config.py                # Experiment configuration (dataclasses + YAML)
├── cli.py                   # CLI: solrad-run
├── data/
│   ├── loaders.py           # Wrappers for loading sensor/WRF data
│   ├── alignment.py         # Temporal alignment sensor ↔ WRF
│   ├── preprocessing.py     # Pipeline with fit on train only (no leakage)
│   └── splits.py            # Chronological split, walk-forward, temporal K-fold
├── features/
│   ├── engineering.py       # Lags, rolling means, differences
│   ├── temporal.py          # Hour, day of year, month + cyclic encoding (sin/cos)
│   └── sequence.py          # Sliding windows construction for LSTM/Transformer
├── datasets/
│   ├── tabular.py           # TabularDataset (X, y) + reproducible save/load
│   └── sequence.py          # SequenceDataset (torch.Dataset) + save/load
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
│   └── runner.py            # Complete pipeline: config → data → train → evaluate
└── utils/
    ├── seeds.py             # Seed control (numpy, torch, random)
    ├── io.py                # JSON I/O, CSV predictions
    └── serialization.py     # Serialization: joblib (sklearn) / torch (checkpoint)
```

---

## Installation

```bash
# With CPU PyTorch:
pip install -e ".[tcc]"

# With CUDA PyTorch (install torch first):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[tcc-cuda]"
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

```bash
solrad-run --config configs/tcc/experiments/my_experiment.yaml
```

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
├── config.yaml                      # Exact config used
├── config_resolved.json             # Config dumped to JSON
├── metrics.json                     # Results (RMSE, MAE, R², etc.)
├── predictions.csv                  # y_true vs y_pred
├── training_history.csv             # Loss per epoch (if neural network)
├── model.joblib (or model.pt)       # Trained model
├── preprocessing_pipeline.joblib    # Preprocessing state
└── datasets/
    ├── train/                       # Saved training dataset
    └── test/                        # Saved testing dataset
```

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
  pretrained_path: output/experiments/lstm_v1/model.pt   # Previous weights
  max_epochs: 50       # ADDITIONAL epochs
```

This loads the weights from `lstm_v1` and trains for an additional 50 epochs. The checkpoint saves:

- `model_state_dict` (model weights)
- `optimizer_state_dict` (optimizer state)
- `epoch` (epoch it stopped at)
- `config` (architecture parameters for reconstruction)

---

## Data Leakage Prevention

The package implements multiple protection layers:

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

### 3. Sliding Windows (Sequence)

For LSTM/Transformer, each window only looks into the **past**:

```
Window 1: [t₀, t₁, t₂, t₃] → target: t₄
Window 2: [t₁, t₂, t₃, t₄] → target: t₅
```

The target is always **after** the end of the window.

### 4. Serialized Pipeline

The preprocessing state is saved with each experiment (`preprocessing_pipeline.joblib`), ensuring that the exact same transform can be applied to future data.

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

3. Register it in `runner.py`:

```python
elif model_type == "mymodel":
    model = MyModel.from_config(config.model)
```

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

### What is `tolerance="30min"` in alignment?

When aligning sensor (observation) data with WRF (model) data, timestamps might not match exactly. The tolerance allows pairing timestamps with up to a 30-minute difference.

### Do I need a GPU?

No. SVM runs exclusively on CPU. LSTM and Transformer work on CPU but are significantly faster with CUDA. The code auto-detects and uses a GPU if available.

### How do I reproduce an experiment exactly?

1. Use the exact same `seed` in the config
2. Use the saved dataset in `experiments/<name>/datasets/`
3. Use the saved config in `experiments/<name>/config.yaml`

```python
config = ExperimentConfig.from_yaml("output/experiments/lstm_v1/config.yaml")
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

pipeline = PreprocessingPipeline.load("output/experiments/svm_v1/preprocessing_pipeline.joblib")
y_original = pipeline.inverse_transform_column(y_normalized, "SW_dif")
```
