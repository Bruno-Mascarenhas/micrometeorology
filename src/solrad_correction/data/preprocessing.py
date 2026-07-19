"""Current-schema preprocessing for leakage-safe solrad experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingState:
    """Serializable preprocessing state learned from the training split."""

    version: int = 3
    scaler_type: str = "standard"
    impute_strategy: str = "drop"
    drop_na_threshold: float = 0.5
    input_columns: list[str] = field(default_factory=list)
    output_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    target_column: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    fill_values: dict[str, float] = field(default_factory=dict)
    last_values: dict[str, float] = field(default_factory=dict)
    scaling: dict[str, dict[str, float]] = field(default_factory=dict)
    dropped_columns: dict[str, str] = field(default_factory=dict)
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-safe dict (inverse of :meth:`from_dict`)."""
        return {
            "version": self.version,
            "scaler_type": self.scaler_type,
            "impute_strategy": self.impute_strategy,
            "drop_na_threshold": self.drop_na_threshold,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "row_counts": self.row_counts,
            "fill_values": self.fill_values,
            "last_values": self.last_values,
            "scaling": self.scaling,
            "dropped_columns": self.dropped_columns,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingState:
        """Rebuild a state from :meth:`to_dict` output; rejects other schema versions.

        Raises
        ------
        ValueError
            If ``data['version']`` is not 3 (only the current schema is supported).
        """
        if int(data.get("version", 0)) != 3:
            raise ValueError("Only preprocessing state version 3 is supported")
        return cls(
            version=3,
            scaler_type=str(data["scaler_type"]),
            impute_strategy=str(data["impute_strategy"]),
            drop_na_threshold=float(data["drop_na_threshold"]),
            input_columns=list(data["input_columns"]),
            output_columns=list(data["output_columns"]),
            feature_columns=list(data.get("feature_columns", [])),
            target_column=data.get("target_column"),
            row_counts={str(key): int(value) for key, value in data.get("row_counts", {}).items()},
            fill_values=_float_dict(data.get("fill_values", {})),
            last_values=_float_dict(data.get("last_values", {})),
            scaling={key: _float_dict(value) for key, value in data.get("scaling", {}).items()},
            dropped_columns={str(k): str(v) for k, v in data.get("dropped_columns", {}).items()},
            fitted=bool(data.get("fitted", False)),
        )


class Preprocessor:
    """Stateful train-only preprocessing with strict schema validation."""

    def __init__(
        self,
        scaler_type: str = "standard",
        impute_strategy: str = "drop",
        drop_na_threshold: float = 0.5,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        strict_schema: bool = True,
    ) -> None:
        if scaler_type not in {"standard", "minmax", "none"}:
            raise ValueError("scaler_type must be one of: standard, minmax, none")
        if impute_strategy not in {"drop", "ffill", "mean", "interpolate"}:
            raise ValueError("impute_strategy must be one of: drop, ffill, mean, interpolate")
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.drop_na_threshold = drop_na_threshold
        self.feature_columns = feature_columns or []
        self.target_column = target_column
        self.strict_schema = strict_schema
        self._state = PreprocessingState(
            scaler_type=scaler_type,
            impute_strategy=impute_strategy,
            drop_na_threshold=drop_na_threshold,
            feature_columns=list(self.feature_columns),
            target_column=target_column,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been run on this instance."""
        return self._state.fitted

    @property
    def state(self) -> PreprocessingState:
        """The learned :class:`PreprocessingState` (statistics and column layout)."""
        return self._state

    @property
    def columns(self) -> list[str]:
        """Output columns kept after fitting (input columns minus the dropped ones)."""
        return list(self._state.output_columns)

    @property
    def dropped_columns(self) -> dict[str, str]:
        """Columns dropped at fit time mapped to the NaN-ratio reason they were dropped."""
        return dict(self._state.dropped_columns)

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn drop list, imputation fills and scaling from ``df`` (train split only).

        All statistics are computed here and frozen into :attr:`state`; call this
        on the training split alone so no validation/test information leaks into
        the fitted parameters. Returns ``self`` for chaining.
        """
        input_columns = list(df.columns)
        na_ratio = df.isna().mean()
        dropped = {
            str(col): f"nan_ratio={ratio:.6f} > threshold={self.drop_na_threshold:.6f}"
            for col, ratio in na_ratio.items()
            if ratio > self.drop_na_threshold
        }
        df_clean = df.drop(columns=list(dropped), errors="ignore")
        output_columns = list(df_clean.columns)
        fill_values = _series_to_float_dict(df_clean.mean(numeric_only=True))
        last_values = _series_to_float_dict(df_clean.ffill().iloc[-1]) if not df_clean.empty else {}
        scaling = self._fit_scaling(df_clean)
        if self.impute_strategy == "drop":
            fit_output_rows = len(df_clean.dropna())
        elif self.impute_strategy == "interpolate":
            fit_output_rows = len(self._interpolate(df_clean).dropna())
        else:
            fit_output_rows = len(df_clean)

        self._state = PreprocessingState(
            scaler_type=self.scaler_type,
            impute_strategy=self.impute_strategy,
            drop_na_threshold=self.drop_na_threshold,
            input_columns=input_columns,
            output_columns=output_columns,
            feature_columns=list(self.feature_columns),
            target_column=self.target_column,
            row_counts={
                "fit_input_rows": len(df),
                "fit_output_rows": int(fit_output_rows),
                "fit_input_columns": len(input_columns),
                "fit_output_columns": len(output_columns),
            },
            fill_values=fill_values,
            last_values=last_values,
            scaling=scaling,
            dropped_columns=dropped,
            fitted=True,
        )
        logger.info(
            "Preprocessor fitted: %d cols, dropped %d, scaler=%s",
            len(output_columns),
            len(dropped),
            self.scaler_type,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted drop/impute/scale steps to ``df``.

        With ``strict_schema`` (the default) the input columns must match those
        seen at fit time exactly.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If the input schema does not match the fitted columns.
        """
        if not self._state.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        self._validate_transform_schema(df)

        out = df[self._state.output_columns].copy()
        out = self._impute(out)
        return self._scale(out)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience for :meth:`fit` followed by :meth:`transform` on the same frame."""
        return self.fit(df).transform(df)

    def inverse_transform_column(self, values: np.ndarray, column: str) -> np.ndarray:
        """Map scaled ``values`` for one column back to their original units.

        Undoes only the scaling step (standard or min-max); a no-op when
        ``scaler_type='none'``.

        Raises
        ------
        ValueError
            If ``column`` was not part of the fitted output columns.
        """
        if column not in self._state.output_columns:
            raise ValueError(f"Column '{column}' is not part of fitted preprocessing output")
        values = np.asarray(values, dtype=np.float64)
        if self.scaler_type == "standard":
            return values * self._state.scaling["std"][column] + self._state.scaling["mean"][column]
        if self.scaler_type == "minmax":
            return (
                values * (self._state.scaling["max"][column] - self._state.scaling["min"][column])
                + self._state.scaling["min"][column]
            )
        return values

    def to_state(self) -> PreprocessingState:
        """Return the fitted :class:`PreprocessingState` for serialization."""
        return self._state

    @classmethod
    def from_state(cls, state: PreprocessingState) -> Preprocessor:
        """Rebuild a ready-to-transform preprocessor from a saved state (no refit)."""
        pipeline = cls(
            scaler_type=state.scaler_type,
            impute_strategy=state.impute_strategy,
            drop_na_threshold=state.drop_na_threshold,
            feature_columns=state.feature_columns,
            target_column=state.target_column,
        )
        pipeline._state = state
        return pipeline

    def save(self, path: str | Path) -> None:
        """Persist the fitted state to ``path`` as a joblib artifact (see :meth:`load`)."""
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._state.to_dict(), path)

    def save_state_json(self, path: str | Path) -> None:
        """Persist the fitted state to ``path`` as human-readable JSON."""
        from solrad_correction.utils.io import save_json

        save_json(self._state.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Preprocessor:
        """Reconstruct a fitted preprocessor from a :meth:`save` joblib artifact."""
        import joblib

        return cls.from_state(PreprocessingState.from_dict(joblib.load(path)))

    def _fit_scaling(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        if self.scaler_type == "standard":
            return {
                "mean": _series_to_float_dict(df.mean(numeric_only=True)),
                "std": _series_to_float_dict(df.std(numeric_only=True).replace(0, 1)),
            }
        if self.scaler_type == "minmax":
            min_values = df.min(numeric_only=True)
            max_values = df.max(numeric_only=True)
            diff = max_values - min_values
            max_values = min_values + diff.mask(diff == 0, 1)
            return {
                "min": _series_to_float_dict(min_values),
                "max": _series_to_float_dict(max_values),
            }
        return {}

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.impute_strategy == "drop":
            return df.dropna()
        if self.impute_strategy == "ffill":
            return df.ffill().fillna(self._state.last_values).fillna(self._state.fill_values)
        if self.impute_strategy == "mean":
            return df.fillna(self._state.fill_values)
        if self.impute_strategy == "interpolate":
            return self._interpolate(df).dropna()
        raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")

    @staticmethod
    def _interpolate(df: pd.DataFrame) -> pd.DataFrame:
        """Linearly interpolate internal gaps only.

        Uses time-based interpolation on a ``DatetimeIndex`` (positional
        otherwise). ``limit_area='inside'`` restricts filling to gaps between
        known points, so leading/trailing NaNs are never extrapolated or
        forward-filled — those rows stay NaN and are dropped by ``_impute``,
        keeping the trailing edge causal-safe (no implied ffill).
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df.interpolate(method="time", limit_area="inside")
        return df.interpolate(method="linear", limit_area="inside")

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_type == "standard":
            return (df - pd.Series(self._state.scaling["mean"])) / pd.Series(
                self._state.scaling["std"]
            )
        if self.scaler_type == "minmax":
            return (df - pd.Series(self._state.scaling["min"])) / pd.Series(
                _dict_subtract(self._state.scaling["max"], self._state.scaling["min"])
            )
        return df

    def _validate_transform_schema(self, df: pd.DataFrame) -> None:
        if not self.strict_schema:
            return
        actual = set(df.columns)
        expected = set(self._state.input_columns)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing columns: {missing}")
            if unexpected:
                parts.append(f"unexpected columns: {unexpected}")
            raise ValueError(
                "Input schema does not match fitted preprocessing state; " + "; ".join(parts)
            )


class PreprocessingPipeline(Preprocessor):
    """Backward-compatible public name for the current preprocessor."""


def _series_to_float_dict(series: Any) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in series.to_dict().items()
        if not np.isnan(float(value))
    }


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _dict_subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left}
