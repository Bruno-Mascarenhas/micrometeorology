"""Tests for dataset reading, pairing and comparison plotting.

The round trip that matters is ``sensors.export.export_csv`` ->
``stats.comparison.read_dataset``: the exported file carries its timestamps in
an unnamed leading column, and losing them makes every downstream alignment
positional.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from micrometeorology.sensors.export import export_csv
from micrometeorology.stats.comparison import (
    compare_all_variables,
    compare_variables,
    pair_dataframes,
    read_dataset,
)
from micrometeorology.stats.comparison_plots import plot_comparison
from micrometeorology.stats.metrics import ALL_METRICS

STRICT_PANDAS_MIGRATION = pytest.mark.filterwarnings("error::pandas.errors.PandasChangeWarning")


def _series_csv(path: Path, start: str, values: list[float]) -> Path:
    """Export a single-column hourly series exactly as ``labmim-sensor-process`` does."""
    index = pd.date_range(start, periods=len(values), freq="1h")
    return export_csv(pd.DataFrame({"Temp1": values}, index=index), path)


@pytest.mark.parametrize("freq", ["15min", "30min", "30s"])
def test_a_sub_hourly_index_is_refused_rather_than_written_without_its_minute(
    tmp_path: Path, freq: str
) -> None:
    """year/month/day/hour cannot carry the stamp of a sub-hourly frame.

    Every row inside one clock hour would land on the same key, so the four
    distinct means of a 15-minute aggregation come back from ``read_dataset``
    as four rows with an identical timestamp and no warning anywhere.
    """
    index = pd.date_range("2025-06-25 12:00", periods=4, freq=freq)
    frame = pd.DataFrame({"Temp1": [25.0, 26.0, 27.0, 28.0]}, index=index)

    with pytest.raises(ValueError, match="year/month/day/hour"):
        export_csv(frame, tmp_path / "sub_hourly.csv", include_datetime_columns=True)


def test_an_hourly_index_still_exports_its_datetime_columns(tmp_path: Path) -> None:
    index = pd.date_range("2025-06-25 12:00", periods=2, freq="1h")
    frame = pd.DataFrame({"Temp1": [25.0, 26.0]}, index=index)

    path = export_csv(frame, tmp_path / "hourly.csv", include_datetime_columns=True)

    assert path.read_text(encoding="utf-8").splitlines()[1] == "2025,6,25,12,25.000"


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
        """Pairing is by timestamp: positional alignment scores a lag as RMSE 0."""
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


class TestCompareAllVariables:
    def test_the_table_is_indexed_by_every_metric_name_with_real_values(self) -> None:
        """Rows are the metric names and the numbers are the metrics, not NaN.

        A ``compare_all_variables`` that dropped a metric, or a ``compute_all``
        that returned NaN unconditionally, would still produce a
        correctly-shaped table; the pinned values are what rule both out.
        """
        index = pd.date_range("2020-01-01", periods=5, freq="1h")
        observed = [1.0, 2.0, 3.0, 4.0, 5.0]
        paired = pd.DataFrame(
            {
                "T_obs": observed,
                "T_model": [v + 1.0 for v in observed],
                "RH_obs": observed,
                "RH_model": observed,
            },
            index=index,
        )

        table = compare_all_variables(paired)

        assert list(table.index) == ["n", *ALL_METRICS]
        assert list(table.columns) == ["RH", "T"]
        assert table.loc["RMSE", "T"] == pytest.approx(1.0)
        assert table.loc["MBE", "T"] == pytest.approx(1.0)
        assert table.loc["R²", "T"] == pytest.approx(0.5)
        assert table.loc["RMSE", "RH"] == pytest.approx(0.0)

    def test_an_unpaired_column_is_left_out_of_the_table(self) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="1h")
        paired = pd.DataFrame(
            {"T_obs": [1.0, 2.0, 3.0], "T_model": [1.0, 2.0, 3.0], "RH_obs": [4.0, 5.0, 6.0]},
            index=index,
        )

        assert list(compare_all_variables(paired).columns) == ["T"]


def test_reading_and_scoring_a_dataset_does_not_import_matplotlib() -> None:
    """``labmim-metrics`` pays ~0.5 s per run for a plot it never draws.

    ``stats.comparison`` is pure pandas; the figure lives in
    ``stats.comparison_plots``. Asserting on the module registry of a fresh
    interpreter is the only check that survives another test importing
    matplotlib first.
    """
    probe = (
        "import sys, micrometeorology.stats.comparison, micrometeorology.cli.compute_metrics; "
        "print(any(m == 'matplotlib' or m.startswith('matplotlib.') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", result.stdout


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


class TestTheTimeIndexIsReadOrRefusedClearly:
    """``labmim-comparison`` pairs by TIME; ``labmim-metrics`` may fall back to
    row order. The two must not share one failure mode."""

    @staticmethod
    def _named_index_csv(path: Path) -> Path:
        index = pd.date_range("2022-07-01", periods=6, freq="h", name="timestamp")
        pd.DataFrame({"T": range(6)}, index=index).to_csv(path)
        return path

    def test_a_named_timestamp_column_is_recognised(self, tmp_path: Path) -> None:
        """``DataFrame.to_csv`` writes the index's NAME, so the most ordinary way
        to produce one of these files yields a named time column, not an unnamed
        one — and the reader has to consume it rather than leave a RangeIndex.
        """
        frame = read_dataset(str(self._named_index_csv(tmp_path / "named.csv")))

        assert isinstance(frame.index, pd.DatetimeIndex)
        assert "timestamp" not in frame.columns

    def test_a_frame_with_no_time_index_is_still_returned(self, tmp_path: Path) -> None:
        """Positional alignment is a supported mode of labmim-metrics; the reader
        must not refuse it on that CLI's behalf."""
        path = tmp_path / "plain.csv"
        path.write_text("T\n1.0\n2.0\n", encoding="utf-8")

        frame = read_dataset(str(path))

        assert not isinstance(frame.index, pd.DatetimeIndex)
        assert len(frame) == 2


class TestWindDirectionIsScoredCircularly:
    """``compare_all_variables`` has to reach the circular path by column name.

    The unit is covered in ``test_metrics``; what this pins is the wiring —
    ``compare_variables`` deciding, from the variable name alone, that a
    published RMSE must not be the linear one.
    """

    @staticmethod
    def _paired() -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        return pd.DataFrame(
            {
                "WD_obs": [350.0, 350.0, 10.0, 10.0],
                "WD_model": [10.0, 10.0, 350.0, 350.0],
                "WS_obs": [1.0, 2.0, 3.0, 4.0],
                "WS_model": [1.5, 2.5, 2.5, 4.5],
            },
            index=index,
        )

    def test_the_published_error_is_the_short_way_round(self):
        metrics = compare_all_variables(self._paired())
        assert metrics.loc["MAE", "WD"] == pytest.approx(20.0)
        assert metrics.loc["RMSE", "WD"] == pytest.approx(20.0)

    def test_mean_normalised_metrics_are_blank_for_the_bearing_only(self):
        metrics = compare_all_variables(self._paired())
        bearing = metrics["WD"].astype(float).to_dict()
        speed = metrics["WS"].astype(float).to_dict()
        for name in ("R²", "r", "d", "IOA", "NRMSE"):
            assert np.isnan(bearing[name]), name
            assert np.isfinite(speed[name]), name


class TestTheSampleSizeIsPublished:
    """The printed row count is not the denominator of any metric.

    ``pair_dataframes`` merges LEFT, so the frame keeps one row per observation
    even when no model row matched, and each column then loses its own gaps.
    Measured on the real pair (station archive vs. the operational WRF run) the
    frame has 83,857 rows while pressure is scored over 12,678 — a reader taking
    the printed count as the sample size is out by 6.6x.
    """

    @staticmethod
    def _paired() -> pd.DataFrame:
        index = pd.date_range("2020-01-01", periods=6, freq="h")
        return pd.DataFrame(
            {
                "T_obs": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
                "T_model": [1.5, 2.5, 3.5, 4.5, 5.5, np.nan],
                "pressure_obs": [1010.0, 1011.0, 1012.0, 1013.0, 1014.0, 1015.0],
                "pressure_model": [1010.5, 1011.5, np.nan, np.nan, np.nan, np.nan],
            },
            index=index,
        )

    def test_n_is_per_variable_not_the_row_count(self):
        table = compare_all_variables(self._paired())
        assert len(self._paired()) == 6
        assert table.loc["n", "T"] == 4
        assert table.loc["n", "pressure"] == 2

    def test_n_is_the_population_the_metrics_used(self):
        """Reconciliation: the published n must reproduce the published MBE."""
        paired = self._paired()
        table = compare_all_variables(paired)
        finite = paired[["T_obs", "T_model"]].dropna()
        assert table.loc["n", "T"] == len(finite)
        assert table.loc["MBE", "T"] == pytest.approx(
            float((finite["T_model"] - finite["T_obs"]).mean())
        )
