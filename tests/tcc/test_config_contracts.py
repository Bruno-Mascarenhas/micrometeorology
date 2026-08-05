"""Experiment config validation, YAML parsing, and data-source dispatch contracts."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    RuntimeConfig,
)
from solrad_correction.experiments.pipeline import load_data


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            ExperimentConfig(preprocess=PreprocessConfig(scaler_type="standrd")),
            r"preprocess\.scaler_type",
        ),
        (
            ExperimentConfig(preprocess=PreprocessConfig(impute_strategy="ffil")),
            r"preprocess\.impute_strategy",
        ),
        (
            ExperimentConfig(preprocess=PreprocessConfig(drop_na_threshold=1.5)),
            r"preprocess\.drop_na_threshold",
        ),
        (
            ExperimentConfig(data=DataConfig(use_raw=True, hourly_data_path="data/hourly.csv")),
            r"data\.use_raw=true requires data\.sensor_data_path",
        ),
        (
            ExperimentConfig(
                data=DataConfig(wrf_data_path="data/wrf/", hourly_data_path="data/hourly.csv")
            ),
            r"data\.wrf_data_path",
        ),
        (
            ExperimentConfig(features=FeatureConfig(lag_steps=[1, 24])),
            r"data\.feature_columns, which is empty",
        ),
        (
            ExperimentConfig(features=FeatureConfig(rolling_windows=[3])),
            r"data\.feature_columns, which is empty",
        ),
        (
            ExperimentConfig(features=FeatureConfig(add_diffs=True)),
            r"data\.feature_columns, which is empty",
        ),
        (
            ExperimentConfig(
                data=DataConfig(target_column="SW_dif", feature_columns=["SWDOWN", "SW_dif"]),
                features=FeatureConfig(add_diffs=True),
            ),
            r"'SW_dif_diff_1' are same-row functions of the target",
        ),
        (
            ExperimentConfig(
                data=DataConfig(target_column="SW_dif", feature_columns=["SWDOWN", "SW_dif"]),
                features=FeatureConfig(rolling_windows=[3]),
            ),
            r"'SW_dif_roll_\*'",
        ),
        (
            ExperimentConfig(
                data=DataConfig(feature_columns=["SWDOWN"]),
                features=FeatureConfig(lag_steps=[0]),
            ),
            r"features\.lag_steps entries must be >= 1",
        ),
        (
            ExperimentConfig(
                data=DataConfig(feature_columns=["SWDOWN"]),
                features=FeatureConfig(lag_steps=[-1]),
            ),
            r"features\.lag_steps entries must be >= 1",
        ),
        (
            ExperimentConfig(
                data=DataConfig(feature_columns=["SWDOWN"]),
                features=FeatureConfig(rolling_windows=[0]),
            ),
            r"features\.rolling_windows entries must be >= 1",
        ),
    ],
)
def test_validate_catches_config_errors_before_any_stage_runs(
    cfg: ExperimentConfig, message: str
) -> None:
    """Regression for findings 3-5: --validate-config must not defer these to stage 4."""
    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_shipped_experiment_configs_still_validate() -> None:
    """The feature-stage invariants must not reject a config the project ships."""
    shipped = sorted(Path("configs/tcc/experiments").glob("*.yaml"))

    assert shipped
    for config_path in shipped:
        ExperimentConfig.from_yaml(config_path).validate()


def _write_raw_sensor_tree(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sensor.dat").write_text(
        "TOA5\n"
        "TIMESTAMP,f1,target\n"
        "TS,unit,unit\n"
        "meta,meta,meta\n"
        "2024-01-01 00:00:00,1,10\n"
        "2024-01-01 00:30:00,3,14\n"
        "2024-01-01 01:00:00,5,18\n",
        encoding="utf-8",
    )


def _write_hourly_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2024-01-01", periods=3, freq="1h")
    pd.DataFrame(
        {"f1": np.full(3, 999.0), "target": np.full(3, 999.0)},
        index=index,
    ).to_parquet(path)


def _config_with_both_data_paths(
    raw_dir: Path, hourly_path: Path, *, use_raw: bool
) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            use_raw=use_raw,
            sensor_data_path=str(raw_dir),
            hourly_data_path=str(hourly_path),
            source_format="parquet",
            target_column="target",
            resample_freq="1h",
            sensor_min_samples=1,
        )
    )


def test_use_raw_breaks_the_tie_when_both_data_paths_are_configured(tmp_path: Path) -> None:
    """Regression for finding 4: use_raw: true must not silently load the hourly table."""
    raw_dir = tmp_path / "raw"
    hourly_path = tmp_path / "hourly.parquet"
    _write_raw_sensor_tree(raw_dir)
    _write_hourly_table(hourly_path)

    raw_frame = load_data(_config_with_both_data_paths(raw_dir, hourly_path, use_raw=True)).frame
    hourly_frame = load_data(
        _config_with_both_data_paths(raw_dir, hourly_path, use_raw=False)
    ).frame

    assert raw_frame["f1"].iloc[0] == 2.0
    assert hourly_frame["f1"].iloc[0] == 999.0


def test_station_coordinates_warn_that_they_cannot_retarget_the_site(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression for finding 3: a site retargeted by coordinates must not pass in silence."""
    hourly_path = tmp_path / "hourly.parquet"
    _write_hourly_table(hourly_path)
    cfg = ExperimentConfig(
        data=DataConfig(
            hourly_data_path=str(hourly_path),
            source_format="parquet",
            target_column="target",
            station_lat=-3.7,
            station_lon=-38.5,
        )
    )

    with caplog.at_level(logging.WARNING, logger="solrad_correction.experiments.pipeline"):
        load_data(cfg)

    assert "station_lat" in caplog.text


def test_from_yaml_rejects_unknown_keys_at_every_level(tmp_path: Path) -> None:
    """Regression for finding 6: a mistyped key must never run with the default value."""
    top_level_typo = tmp_path / "top_level.yaml"
    top_level_typo.write_text(
        yaml.safe_dump({"name": "typo", "sed": 7, "outputdir": "/mnt/big/runs"}),
        encoding="utf-8",
    )
    nested_typo = tmp_path / "nested.yaml"
    nested_typo.write_text(
        yaml.safe_dump({"data": {"hourly_data_pathh": "data/hourly.csv"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outputdir, sed"):
        ExperimentConfig.from_yaml(top_level_typo)
    with pytest.raises(ValueError, match="hourly_data_pathh"):
        ExperimentConfig.from_yaml(nested_typo)


def test_from_yaml_accepts_null_sections_and_anchor_keys(tmp_path: Path) -> None:
    """Regression for finding 6: commenting out a section's body is not a config error."""
    config_path = tmp_path / "sparse.yaml"
    config_path.write_text(
        "_defaults: &defaults\n  device: cpu\nname: sparse\ndata:\nruntime:\n  <<: *defaults\n",
        encoding="utf-8",
    )

    cfg = ExperimentConfig.from_yaml(config_path)

    assert cfg.name == "sparse"
    assert cfg.data == DataConfig()
    assert cfg.runtime.device == "cpu"


def test_from_yaml_round_trips_a_written_config(tmp_path: Path) -> None:
    """The strict key check must accept every key ``ExperimentConfig.save`` writes."""
    config_path = tmp_path / "config.yaml"
    cfg = ExperimentConfig(
        name="round_trip",
        data=DataConfig(hourly_data_path="data/hourly.csv", target_column="SW_dif"),
        model=ModelConfig(model_type="lstm"),
        runtime=RuntimeConfig(device="cpu"),
    )
    cfg.save(config_path)

    assert ExperimentConfig.from_yaml(config_path) == cfg
