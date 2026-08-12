"""Current-schema preprocessing for leakage-safe solrad experiments."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessingState:
    """Serializable preprocessing state learned from the training split.

    Everything a fitted :class:`Preprocessor` needs to reproduce its transform on
    another split, and nothing that depends on the split it is applied to.
    ``input_columns`` is the schema ``fit`` saw, ``output_columns`` what survived
    the NaN-ratio drop, and ``dropped_columns`` maps each removed column to the
    ratio that removed it. ``fill_values`` (column means), ``last_values`` (the
    final observed row, populated only under the ``ffill`` strategy) and
    ``scaling`` (statistic name -> column -> value, in the columns' original
    units) are all computed on the training split alone. ``row_counts`` records
    the fit's input/output row and column counts as provenance.

    ``version`` is the on-disk schema version; :meth:`from_dict` accepts only
    the current one rather than silently reading an older layout.
    """

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

    def fingerprint(self) -> str:
        """Stable digest of everything that changes what a feature vector means.

        A resumed run refits the scaler from whatever data is on disk now, so a
        changed feature set, target, scaler type or fitted mean/scale silently
        feeds the restored weights a differently-scaled input. Comparing this
        digest against the one baked into the checkpoint is what turns that into
        a refusal. ``row_counts`` and ``dropped_columns`` are deliberately
        excluded: they are provenance, not part of the transform.
        """
        material = json.dumps(
            {
                "version": self.version,
                "scaler_type": self.scaler_type,
                "impute_strategy": self.impute_strategy,
                "drop_na_threshold": self.drop_na_threshold,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column,
                "fill_values": self.fill_values,
                "scaling": self.scaling,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

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
    """Stateful train-only preprocessing with strict schema validation.

    Fit on the training split, then transform every split with the same frozen
    statistics: drop the columns that are too sparse, impute the remaining
    feature gaps, and scale. The target column is deliberately never imputed —
    see :meth:`transform`.

    Parameters
    ----------
    scaler_type:
        ``standard`` (subtract the mean, divide by the standard deviation),
        ``minmax`` (rescale to ``[0, 1]`` over the fitted range) or ``none``.
        A zero spread is replaced by 1 so a constant column maps to 0 instead of
        to infinity.
    impute_strategy:
        ``drop``, ``ffill``, ``mean`` or ``interpolate``.
    drop_na_threshold:
        NaN fraction in ``[0, 1]``; a column whose training-split NaN ratio
        strictly exceeds it is dropped from every split.
    feature_columns, target_column:
        Recorded in the state so a resumed run can be checked against the
        feature set the checkpoint's weights were trained on; ``target_column``
        additionally marks the column imputation must leave untouched.
    strict_schema:
        When true (the default) :meth:`transform` refuses any frame whose column
        set differs from the one seen at fit time.

    Raises
    ------
    ValueError
        If ``scaler_type`` or ``impute_strategy`` is not one of the values above.
    """

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

        Parameters
        ----------
        df:
            Training split, shape ``(n_train_rows, n_columns)``, in the columns'
            original units. Its column set becomes the schema
            :meth:`transform` enforces.

        Returns
        -------
        Preprocessor
            ``self``, now fitted.

        Raises
        ------
        ValueError
            If a column kept by the NaN-ratio filter has no defined scaling
            statistic — a non-numeric dtype, fewer than two observations, or a
            non-finite value. Scaling it would silently return NaN for every row
            of every split.
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
        # Only the ``ffill`` strategy reads this, and it is the one statistic
        # taken over the raw row rather than a numeric reduction — so computing
        # it unconditionally made any non-numeric column kill ``fit`` with a
        # bare "could not convert string to float" naming no column, under
        # strategies that never use the value.
        last_values = (
            _series_to_float_dict(df_clean.ffill().iloc[-1])
            if self.impute_strategy == "ffill" and not df_clean.empty
            else {}
        )
        scaling = self._fit_scaling(df_clean)
        self._reject_unscalable_columns(scaling, output_columns)
        fit_output_rows = self._count_retained_rows(df_clean)

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

        The **target column is never imputed**: it is authoritative ground
        truth. Under any non-``drop`` strategy the feature columns are filled
        but rows whose observed target is missing are dropped, so metrics and
        the validation loss are never computed against fabricated targets.

        Parameters
        ----------
        df:
            Any split, shape ``(n_rows, n_columns)``, in the columns' original
            units.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_kept_rows, n_output_columns)`` holding the fitted output
            columns in scaled (dimensionless) units, or the original units when
            ``scaler_type='none'``. Rows are a subset of the input's, with the
            input index preserved.

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
        ``scaler_type='none'``. The drop and imputation steps are not invertible
        and are not undone, so this maps model outputs back to physical units
        (W m-2 for the diffuse-irradiance target) but does not restore rows.

        Parameters
        ----------
        values:
            Scaled values for a single column, any shape, cast to ``float64``.
        column:
            Name of the fitted output column ``values`` were scaled with.

        Returns
        -------
        np.ndarray
            ``float64``, same shape as ``values``, in ``column``'s original unit.

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
        """Reconstruct a fitted preprocessor from a :meth:`save` joblib artifact.

        The artifact's integrity is checked against a reachable experiment
        ``manifest.json`` before unpickling (raising on a checksum mismatch);
        an unverified load is logged when no manifest covers the file.
        """
        import joblib

        from solrad_correction.utils.serialization import verify_pickle_integrity

        verify_pickle_integrity(path)
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

    def _reject_unscalable_columns(
        self, scaling: dict[str, dict[str, float]], output_columns: list[str]
    ) -> None:
        """Refuse a fitted state that cannot scale a column it decided to keep.

        ``_series_to_float_dict`` drops every entry whose statistic is NaN, and
        the numeric reductions above skip non-numeric columns outright — so a
        column with fewer than two observations, or a single ``inf`` (which no
        NaN threshold and no ``dropna`` removes), silently lost its entry.
        ``_scale`` then aligns on labels and returns that column as NaN for every
        row of train, val AND test, after ``_impute`` has already run, so nothing
        fills it back in: sklearn dies with an unexplained "Input contains NaN"
        and the torch models train on NaN for every epoch, never improving, and
        still exit 0 writing NaN metrics.
        """
        for statistic, values in scaling.items():
            missing = [column for column in output_columns if column not in values]
            if missing:
                raise ValueError(
                    f"cannot fit the {statistic!r} scaling for {missing}: the statistic is "
                    "undefined (non-numeric dtype, fewer than two observations, or a "
                    "non-finite value). Drop the column or fix the source data."
                )

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute feature columns only; keep the target authoritative.

        ``drop`` removes any row containing a NaN (feature *or* target). Every
        other strategy fills feature gaps but never the target column, then
        drops any row whose observed target is missing — so ground truth is
        never fabricated for metrics or the validation loss.
        """
        if self.impute_strategy == "drop":
            return df.dropna()

        target = self.target_column
        has_target = target is not None and target in df.columns
        features = df.drop(columns=[target]) if has_target else df
        features = self._impute_features(features)
        if not has_target:
            return features

        result = features
        result[target] = df.loc[result.index, target]
        result = result[list(df.columns)]
        return result[result[target].notna()]

    def _impute_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured non-``drop`` strategy to feature columns only.

        Fill dictionaries carry the target's statistics too, but pandas'
        ``fillna(dict)`` only touches columns present in ``features`` — the
        target has already been split off — so the target is never filled here.
        """
        if self.impute_strategy == "ffill":
            return features.ffill().fillna(self._state.last_values).fillna(self._state.fill_values)
        if self.impute_strategy == "mean":
            return features.fillna(self._state.fill_values)
        if self.impute_strategy == "interpolate":
            return self._interpolate(features).dropna()
        raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")

    def _count_retained_rows(self, df: pd.DataFrame) -> int:
        """Rows that survive imputation (used only for ``fit`` row-count stats).

        Mirrors :meth:`_impute`'s drop decisions without needing fitted fill
        values: feature imputation only removes leading/trailing rows under
        ``interpolate``, and every non-``drop`` strategy additionally drops
        rows with a missing target.
        """
        if self.impute_strategy == "drop":
            return len(df.dropna())

        target = self.target_column
        has_target = target is not None and target in df.columns
        features = df.drop(columns=[target]) if has_target else df
        if self.impute_strategy == "interpolate":
            features = self._interpolate(features).dropna()
        kept = features.index
        if has_target:
            kept = kept[df.loc[kept, target].notna().to_numpy()]
        return len(kept)

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
