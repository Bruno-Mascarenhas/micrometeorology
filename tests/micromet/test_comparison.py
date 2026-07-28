"""Tests for dataset reading, pairing and comparison plotting.

The round trip that matters is ``sensors.export.export_csv`` ->
``stats.comparison.read_dataset``: the exported file carries its timestamps in
an unnamed leading column, and losing them makes every downstream alignment
positional.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from micrometeorology.sensors.export import export_csv
from micrometeorology.stats.comparison import (
    compare_variables,
    pair_dataframes,
    plot_comparison,
    read_dataset,
)

STRICT_PANDAS_MIGRATION = pytest.mark.filterwarnings("error::pandas.errors.PandasChangeWarning")


def _series_csv(path: Path, start: str, values: list[float]) -> Path:
    """Export a single-column hourly series exactly as ``labmim-sensor-process`` does."""
    index = pd.date_range(start, periods=len(values), freq="1h")
    return export_csv(pd.DataFrame({"Temp1": values}, index=index), path)


class TestReadDataset:
    def test_export_csv_round_trip_keeps_the_datetime_index(self, tmp_path: Path) -> None:
        path = _series_csv(tmp_path / "obs.csv", "2020-01-01", [20.0, 21.0, 22.0])

        df = read_dataset(path)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["Temp1"]
        assert df.index[0] == pd.Timestamp("2020-01-01 00:00:00")
        assert df["Temp1"].to_list() == [20.0, 21.0, 22.0]

    def test_numeric_leading_column_is_not_read_as_epoch_nanoseconds(self, tmp_path: Path) -> None:
        """A ``yr,mo,dy,hr,T`` file has no unnamed index -- nothing may be consumed."""
        path = tmp_path / "counter.csv"
        path.write_text("yr,mo,dy,hr,T\n2020,1,1,0,20.0\n2020,1,1,1,21.0\n", encoding="utf-8")

        df = read_dataset(path)

        assert isinstance(df.index, pd.RangeIndex)
        assert list(df.columns) == ["yr", "mo", "dy", "hr", "T"]
        assert df["yr"].to_list() == [2020, 2020]

    def test_canonical_year_month_day_hour_columns_still_build_the_index(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "wrf.csv"
        path.write_text(
            "year,month,day,hour,T2\n2020,1,1,0,20.0\n2020,1,1,1,21.0\n", encoding="utf-8"
        )

        df = read_dataset(path)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert list(df.columns) == ["T2"]

    @STRICT_PANDAS_MIGRATION
    def test_text_columns_are_coerced_to_numeric(self, tmp_path: Path) -> None:
        path = tmp_path / "text.csv"
        path.write_text("TIMESTAMP,T\n2020-01-01 00:00:00,NAN\n2020-01-01 01:00:00,21.0\n")

        df = read_dataset(path)

        assert df["T"].dtype == np.float64
        assert pd.isna(df["T"].iloc[0])


class TestPairDataframes:
    def test_a_three_hour_lag_is_not_reported_as_a_perfect_model(self, tmp_path: Path) -> None:
        """Positional alignment made a 3 h-lagged copy score RMSE 0 / R2 1."""
        values = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        obs_path = _series_csv(tmp_path / "obs.csv", "2020-01-01 00:00", values)
        model_path = _series_csv(tmp_path / "mod.csv", "2020-01-01 03:00", values)

        paired = pair_dataframes(read_dataset(obs_path), read_dataset(model_path))
        metrics = compare_variables(paired, "Temp1")

        assert metrics["RMSE"] == pytest.approx(3.0)
        assert metrics["MBE"] == pytest.approx(-3.0)

    def test_non_datetime_index_is_rejected_by_name(self) -> None:
        frame = pd.DataFrame({"T": [1.0, 2.0]})

        with pytest.raises(TypeError, match="DatetimeIndex"):
            pair_dataframes(frame, frame)

    def test_a_named_index_survives_the_merge(self) -> None:
        index = pd.DatetimeIndex(
            pd.date_range("2020-01-01", periods=3, freq="1h"), name="TIMESTAMP"
        )
        obs = pd.DataFrame({"T": [1.0, 2.0, 3.0]}, index=index)
        model = pd.DataFrame({"T": [1.5, 2.5, 3.5]}, index=index)

        paired = pair_dataframes(obs, model)

        assert paired.index.name == "time"
        assert list(paired.columns) == ["T_obs", "T_model"]


class TestPlotComparison:
    def test_the_figure_never_reaches_the_pyplot_figure_manager(self, tmp_path: Path) -> None:
        """Every retained pyplot figure is dead weight for the whole CLI run."""
        index = pd.date_range("2020-01-01", periods=48, freq="1h")
        rng = np.random.default_rng(11)
        paired = pd.DataFrame(
            {"T2_obs": rng.normal(25, 3, 48), "T2_model": rng.normal(25, 3, 48)}, index=index
        )
        before = set(plt.get_fignums())

        out = tmp_path / "comparison_T2.png"
        fig = plot_comparison(paired, "T2", output_path=out)

        assert isinstance(fig, Figure)
        assert out.stat().st_size > 0
        assert set(plt.get_fignums()) == before
