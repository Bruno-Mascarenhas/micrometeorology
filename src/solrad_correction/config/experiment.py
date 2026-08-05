"""Top-level experiment configuration and validation."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from solrad_correction.config.data import DataConfig
from solrad_correction.config.features import FeatureConfig
from solrad_correction.config.models import ModelConfig
from solrad_correction.config.preprocessing import PreprocessConfig
from solrad_correction.config.runtime import RuntimeConfig
from solrad_correction.config.split import SplitConfig


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "unnamed"
    description: str = ""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_dir: str = "output/experiments"

    @property
    def experiment_dir(self) -> Path:
        """Directory this run's artifacts are written to (``output_dir/name``)."""
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a nested plain-dict (via ``dataclasses.asdict``)."""
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Check cross-field invariants, raising on the first invalid run.

        Every violation is collected and reported together, so a single
        ``ValueError`` lists all problems (unknown model type, split ratios that
        do not sum to 1, non-divisible transformer head count, and so on).

        Raises
        ------
        ValueError
            If any invariant fails; the message enumerates every error.
        """
        errors: list[str] = []

        from solrad_correction.models.registry import supported_model_names

        supported_models = supported_model_names()
        model_type = self.model.model_type.lower()
        if model_type not in supported_models:
            errors.append(f"model.model_type must be one of: {', '.join(supported_models)}")

        if self.data.source_format not in {"auto", "csv", "parquet"}:
            errors.append("data.source_format must be one of: auto, csv, parquet")
        if self.data.sensor_min_samples <= 0:
            errors.append("data.sensor_min_samples must be positive")
        if self.data.use_raw and not self.data.sensor_data_path:
            errors.append("data.use_raw=true requires data.sensor_data_path")
        if self.data.wrf_data_path and (self.data.hourly_data_path or self.data.sensor_data_path):
            errors.append(
                "data.wrf_data_path is read by no loader and would be ignored next to "
                "data.hourly_data_path/data.sensor_data_path; remove it from the config"
            )

        if not self.data.feature_columns and (
            self.features.lag_steps or self.features.rolling_windows or self.features.add_diffs
        ):
            errors.append(
                "features.lag_steps/rolling_windows/add_diffs are applied only to the columns "
                "declared in data.feature_columns, which is empty; declare the base columns "
                "to engineer or disable those stages"
            )
        if self.data.target_column in self.data.feature_columns and (
            self.features.rolling_windows or self.features.add_diffs
        ):
            errors.append(
                f"data.feature_columns must not contain data.target_column "
                f"('{self.data.target_column}') while features.rolling_windows/add_diffs are "
                f"enabled: '{self.data.target_column}_roll_*' (trailing window with "
                f"min_periods=1 includes the current row) and "
                f"'{self.data.target_column}_diff_1' are same-row functions of the target"
            )
        if any(step < 1 for step in self.features.lag_steps):
            errors.append(
                "features.lag_steps entries must be >= 1: 0 copies a column onto itself and a "
                "negative step reads a future value"
            )
        if any(window < 1 for window in self.features.rolling_windows):
            errors.append("features.rolling_windows entries must be >= 1")

        if self.preprocess.scaler_type not in {"standard", "minmax", "none"}:
            errors.append("preprocess.scaler_type must be one of: standard, minmax, none")
        if self.preprocess.impute_strategy not in {"drop", "ffill", "mean", "interpolate"}:
            errors.append(
                "preprocess.impute_strategy must be one of: drop, ffill, mean, interpolate"
            )
        if not 0.0 <= self.preprocess.drop_na_threshold <= 1.0:
            errors.append("preprocess.drop_na_threshold must be in [0, 1]")

        if self.model.evaluation_policy not in {"model_native", "common_sequence_horizon"}:
            errors.append(
                "model.evaluation_policy must be one of: model_native, common_sequence_horizon"
            )

        split_total = self.split.train_ratio + self.split.val_ratio + self.split.test_ratio
        if abs(split_total - 1.0) > 1e-6:
            errors.append(f"split ratios must sum to 1.0, got {split_total:.4f}")
        if min(self.split.train_ratio, self.split.val_ratio, self.split.test_ratio) < 0:
            errors.append("split ratios must be non-negative")

        if self.model.sequence_length <= 0:
            errors.append("model.sequence_length must be positive")
        if self.model.batch_size <= 0:
            errors.append("model.batch_size must be positive")
        if self.model.max_epochs <= 0:
            errors.append("model.max_epochs must be positive")
        if self.model.tf_d_model <= 0:
            errors.append("model.tf_d_model must be positive")
        if self.model.tf_nhead <= 0:
            errors.append("model.tf_nhead must be positive")
        elif self.model.tf_d_model % self.model.tf_nhead != 0:
            errors.append("model.tf_d_model must be divisible by model.tf_nhead")

        if self.runtime.device not in {"auto", "cpu", "cuda"}:
            errors.append("runtime.device must be one of: auto, cpu, cuda")
        if self.runtime.num_workers is not None and self.runtime.num_workers < 0:
            errors.append("runtime.num_workers must be non-negative")
        if self.runtime.prefetch_factor is not None and self.runtime.prefetch_factor <= 0:
            errors.append("runtime.prefetch_factor must be positive when set")
        if self.runtime.num_workers == 0 and self.runtime.prefetch_factor is not None:
            errors.append("runtime.prefetch_factor requires runtime.num_workers > 0")
        if self.runtime.gradient_clip is not None and self.runtime.gradient_clip < 0:
            errors.append("runtime.gradient_clip must be non-negative when set")
        if self.runtime.checkpoint_every is not None and self.runtime.checkpoint_every <= 0:
            errors.append("runtime.checkpoint_every must be positive when set")
        if self.runtime.limit_rows is not None and self.runtime.limit_rows <= 0:
            errors.append("runtime.limit_rows must be positive when set")

        if errors:
            joined = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"Invalid experiment config:\n{joined}")

    def save(self, path: str | Path) -> None:
        """Write the config to ``path`` as YAML, creating parent directories."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Build a config from a YAML file, filling unspecified fields with defaults.

        A key no config field accepts is a typo that would otherwise run with
        the default value, so it is rejected here — at the top level and inside
        every section. Keys starting with ``_`` are left to YAML anchor blocks.

        Raises
        ------
        ValueError
            If the file is not a mapping or carries an unknown key.
        """
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):
            # The documented contract above is a single ValueError for every config
            # fault, so a bad top-level shape must not raise a different type.
            raise ValueError(f"config file {path} must contain a YAML mapping")  # noqa: TRY004

        known_keys = {config_field.name for config_field in dataclasses.fields(cls)}
        unknown_keys = sorted(
            str(key) for key in data if key not in known_keys and not str(key).startswith("_")
        )
        if unknown_keys:
            raise ValueError(
                f"unknown top-level config key(s): {', '.join(unknown_keys)}; "
                f"expected one of: {', '.join(sorted(known_keys))}"
            )

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            data=_build_section(DataConfig, data.get("data"), "data"),
            split=_build_section(SplitConfig, data.get("split"), "split"),
            preprocess=_build_section(PreprocessConfig, data.get("preprocess"), "preprocess"),
            features=_build_section(FeatureConfig, data.get("features"), "features"),
            model=_build_section(ModelConfig, data.get("model"), "model"),
            runtime=_build_section(RuntimeConfig, data.get("runtime"), "runtime"),
            output_dir=data.get("output_dir", "output/experiments"),
        )


def _build_section[SectionT](
    section_type: Callable[..., SectionT], payload: object, section_name: str
) -> SectionT:
    """Build one config section, naming the section and the offending key on failure.

    A missing or null section (``data:`` with every line below it commented
    out) falls back to that section's defaults, and an unknown key surfaces as
    a ``ValueError`` instead of a raw dataclass ``TypeError``.

    Raises
    ------
    ValueError
        If the section is not a mapping or carries a key the section rejects.
    """
    if payload is None:
        return section_type()
    if not isinstance(payload, dict):
        # Same contract as the docstring and the TypeError->ValueError funnel below:
        # every section fault surfaces as a ValueError naming the section.
        raise ValueError(  # noqa: TRY004
            f"config section '{section_name}' must be a mapping, got {type(payload).__name__}"
        )
    try:
        return section_type(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid config section '{section_name}': {exc}") from exc
