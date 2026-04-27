"""Synthetic solrad experiment generators for smoke tests and demos."""

from __future__ import annotations

from pathlib import Path

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    SplitConfig,
)


def build_smoke_config(root: str | Path = "scratch/solrad_smoke") -> ExperimentConfig:
    """Create a tiny CPU-safe LSTM experiment backed by synthetic hourly data."""
    from solrad_correction.utils.torch_runtime import preload_torch

    preload_torch()

    import numpy as np
    import pandas as pd

    scratch = Path(root)
    scratch.mkdir(parents=True, exist_ok=True)
    data_path = scratch / "smoke_hourly.csv"
    if not data_path.exists():
        index = pd.date_range("2024-01-01", periods=96, freq="1h")
        rng = np.random.default_rng(42)
        f1 = rng.normal(size=96).astype("float32")
        f2 = rng.normal(size=96).astype("float32")
        target = (0.4 * f1 - 0.1 * f2 + rng.normal(scale=0.01, size=96)).astype("float32")
        pd.DataFrame({"f1": f1, "f2": f2, "target": target}, index=index).to_csv(data_path)

    return ExperimentConfig(
        name="solrad_smoke",
        data=DataConfig(
            hourly_data_path=str(data_path),
            target_column="target",
            feature_columns=["f1", "f2"],
        ),
        split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        preprocess=PreprocessConfig(scaler_type="standard", impute_strategy="drop"),
        features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
        model=ModelConfig(
            model_type="lstm",
            lstm_hidden_size=4,
            lstm_num_layers=1,
            sequence_length=4,
            batch_size=8,
            max_epochs=1,
            patience=2,
        ),
        output_dir=str(scratch / "output"),
    )
