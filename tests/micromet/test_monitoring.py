"""Tests for the monitoring contract and its exporter.

Offline: synthetic frames only. These pin what the interactive page reads, so a
change that would break the browser fails here instead.

The tests that matter most are the ones around the WRF resolution. The model
file gains columns over time (precipitation is expected but absent today), and
the whole design is that such an addition is a *data* change: the chart starts
drawing it with no edit here. Two tests hold that promise from both sides — the
absent case must be reported, and the present case must be picked up.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from micrometeorology.cli.export_monitoring import PAYLOAD_FILENAME, PAYLOAD_FORMAT
from micrometeorology.cli.export_monitoring import app as monitoring_app
from micrometeorology.sensors.monitoring import (
    MONITORING_CHARTS,
    MonitoringSeries,
    resolve_wrf_column,
)

runner = CliRunner()

STATION_COLUMNS = [
    "T",
    "ur",
    "pressure",
    "precip",
    "WS",
    "WD",
    "Net_CNR1",
    "Sw_dw",
    "Sw_up",
    "Lw_dw",
    "Lw_up",
    "Sw_dif",
    "Sw_par",
]


def _frame(periods: int, freq: str, columns: list[str], *, seed: int = 3) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    index = pd.date_range("2022-07-01", periods=periods, freq=freq, name="timestamp")
    return pd.DataFrame(
        {name: generator.uniform(0.0, 30.0, periods) for name in columns}, index=index
    )


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A week of synthetic archive at both cadences, as labmim-archive writes it."""
    directory = tmp_path / "archive"
    directory.mkdir()
    _frame(2016, "5min", STATION_COLUMNS).to_parquet(directory / "station_5min_qc.parquet")
    _frame(168, "h", STATION_COLUMNS, seed=4).to_parquet(directory / "station_hourly.parquet")
    return directory


class TestChartCatalogue:
    def test_ids_are_unique(self) -> None:
        ids = [chart.id for chart in MONITORING_CHARTS]
        assert len(ids) == len(set(ids))

    def test_series_ids_are_unique_within_a_chart(self) -> None:
        for chart in MONITORING_CHARTS:
            ids = [series.id for series in chart.series]
            assert len(ids) == len(set(ids)), chart.id

    def test_every_station_column_exists_in_the_archive_schema(self) -> None:
        """A typo here would render an empty chart with no error anywhere."""
        for chart in MONITORING_CHARTS:
            for series in chart.series:
                assert series.station in STATION_COLUMNS, f"{chart.id}/{series.id}"

    def test_kinds_and_encodings_stay_within_the_contract(self) -> None:
        for chart in MONITORING_CHARTS:
            assert chart.kind in {"line", "bar", "scatter"}, chart.id
            for series in chart.series:
                assert series.hue in {"net", "shortwave", "longwave"}, series.id
                assert series.direction in {"down", "up"}, series.id

    def test_balance_uses_three_hues_for_five_series(self) -> None:
        """The direction channel is what keeps the palette at three validated hues."""
        balance = next(chart for chart in MONITORING_CHARTS if chart.id == "balanco")
        assert len(balance.series) == 5
        assert len({series.hue for series in balance.series}) == 3
        keys = {(series.hue, series.direction) for series in balance.series}
        assert len(keys) == 5, "hue+direction must identify a series on its own"


class TestResolveWrfColumn:
    def test_first_present_candidate_wins(self) -> None:
        series = MonitoringSeries("x", "X", "T", ("missing", "second", "third"))
        assert resolve_wrf_column(series, pd.Index(["third", "second"])) == "second"

    def test_absent_returns_none(self) -> None:
        series = MonitoringSeries("x", "X", "T", ("nope",))
        assert resolve_wrf_column(series, pd.Index(["T"])) is None

    def test_no_candidates_declared_returns_none(self) -> None:
        assert resolve_wrf_column(MonitoringSeries("x", "X", "Sw_up"), pd.Index(["Sw_up"])) is None


class TestExporter:
    def _payload(self, archive: Path, tmp_path: Path, *args: str) -> dict:
        result = runner.invoke(
            monitoring_app,
            ["-i", str(archive), "-o", str(tmp_path / "out"), *args, "--log-level", "WARNING"],
        )
        assert result.exit_code == 0, result.output
        payload: dict = json.loads(
            (tmp_path / "out" / PAYLOAD_FILENAME).read_text(encoding="utf-8")
        )
        return payload

    def test_writes_the_declared_format(self, archive: Path, tmp_path: Path) -> None:
        payload = self._payload(archive, tmp_path)
        assert payload["format"] == PAYLOAD_FORMAT
        assert len(payload["charts"]) == len(MONITORING_CHARTS)

    def test_axis_is_start_plus_step_not_one_stamp_per_sample(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """The implicit time axis is the reason the document is ~120 kB, not ~170."""
        chart = self._payload(archive, tmp_path)["charts"][0]
        raw = chart["layers"]["raw"]
        assert raw["axis"]["step_minutes"] == 5
        assert raw["axis"]["count"] == len(next(iter(raw["series"].values())))
        assert chart["layers"]["hourly"]["axis"]["step_minutes"] == 60

    def test_without_a_model_file_every_chart_reports_the_wrf_layer_absent(
        self, archive: Path, tmp_path: Path
    ) -> None:
        for chart in self._payload(archive, tmp_path)["charts"]:
            assert chart["layers"]["wrf"] is None
            declared = {s["id"] for s in chart["series"]}
            pending = set(chart["wrf_pending"])
            assert pending <= declared

    def test_precipitation_records_the_names_it_looked_for(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """Today there is no WRF rain column; the payload must say so, and say
        which spellings would be picked up when the extraction grows one."""
        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        candidates = charts["precipitacao"]["wrf_pending"]["precip"]
        assert "RAINNC" in candidates
        assert "precip" in candidates

    def test_gaps_serialise_as_null(self, archive: Path, tmp_path: Path) -> None:
        frame = pd.read_parquet(archive / "station_hourly.parquet")
        frame.loc[frame.index[5], "T"] = np.nan
        frame.to_parquet(archive / "station_hourly.parquet")
        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        assert charts["temperatura"]["layers"]["hourly"]["series"]["t"][5] is None

    def test_window_is_honoured(self, archive: Path, tmp_path: Path) -> None:
        payload = self._payload(archive, tmp_path, "--days", "2", "--end", "2022-07-05")
        hourly = payload["charts"][0]["layers"]["hourly"]
        assert hourly["axis"]["start"].startswith("2022-07-03")
        assert hourly["axis"]["count"] == 49  # inclusive of both endpoints

    def test_precipitation_keeps_the_tipping_bucket_quantum(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """Rounding rain to two decimals would erase the 0,254 mm bucket."""
        frame = pd.read_parquet(archive / "station_5min_qc.parquet")
        frame["precip"] = 0.0
        frame.loc[frame.index[10], "precip"] = 0.254
        frame.to_parquet(archive / "station_5min_qc.parquet")
        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        assert charts["precipitacao"]["layers"]["raw"]["series"]["precip"][10] == 0.254
