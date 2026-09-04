"""Tests for the monitoring contract and its exporter.

Offline: synthetic frames only. These pin what the interactive page reads, so a
change that would break the browser fails here instead.

The tests around the WRF resolution carry the most weight. The model file gains
columns over time (precipitation is expected but absent today), and the design
is that such an addition is a *data* change: the chart starts drawing it with no
edit here. Two tests hold that promise from both sides — the absent case must be
reported, and the present case must be picked up.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from micrometeorology.cli.export_monitoring import (
    PAYLOAD_FILENAME,
    PAYLOAD_FORMAT,
    _axis,
    _regular,
)
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

    def test_no_candidates_declared_returns_none(self) -> None:
        assert resolve_wrf_column(MonitoringSeries("x", "X", "Sw_up"), pd.Index(["Sw_up"])) is None


@pytest.mark.parametrize("cumulative_field", ["RAINNC", "RAINC"])
def test_an_accumulated_wrf_rain_field_is_not_adopted_as_the_precipitation_layer(
    cumulative_field: str,
) -> None:
    """WRF writes RAINC and RAINNC accumulated since the run's initialisation.

    The card's station layers are the rain of each interval, and nothing between
    the extraction file and the chart differences the model column, so adopting
    one would draw a run's running total against a per-hour total under the same
    "mm" label — and withdraw the page's own note that the layer is missing.
    """
    precipitation = next(chart for chart in MONITORING_CHARTS if chart.id == "precipitacao")

    resolved = resolve_wrf_column(precipitation.series[0], pd.Index([cumulative_field]))

    assert resolved is None


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

    @staticmethod
    def _wrf_dat(tmp_path: Path) -> Path:
        """A model file carrying temperature and nothing else."""
        index = pd.date_range("2022-07-01", periods=169, freq="h")
        dat = tmp_path / "series_operacional.dat"
        pd.DataFrame(
            {
                "year": index.year,
                "month": index.month,
                "day": index.day,
                "hour": index.hour,
                "T": np.arange(len(index), dtype=float),
            }
        ).to_csv(dat, index=False)
        return dat

    def test_precipitation_records_the_names_it_looked_for(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """A record written before the producer existed carries no rain column;
        the payload must say so, and name the column that would be picked up.

        A model file is loaded on purpose: ``wrf_pending`` is a statement about
        what the extraction writes, so it is meaningful only once there is an
        extraction to compare against.
        """
        payload = self._payload(archive, tmp_path, "-w", str(self._wrf_dat(tmp_path)))
        charts = {chart["id"]: chart for chart in payload["charts"]}
        candidates = charts["precipitacao"]["wrf_pending"]["precip"]
        assert "precip_mm" in candidates
        assert "RAINNC" not in candidates, "an accumulated field is not a per-interval layer"

    def test_a_column_present_but_empty_this_window_is_reported_as_pending(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """An extraction that stops writing a variable empties one column rather
        than shifting every column after it — the designed signal. Resolved on
        presence alone, that column produced neither the model line (``_layer``
        drops an all-null series) nor the pending note, which is the one case
        the note exists for."""
        index = pd.date_range("2022-07-01", periods=169, freq="h")
        dat = tmp_path / "series_operacional.dat"
        pd.DataFrame(
            {
                "year": index.year,
                "month": index.month,
                "day": index.day,
                "hour": index.hour,
                "T": np.full(len(index), np.nan),
            }
        ).to_csv(dat, index=False)

        payload = self._payload(archive, tmp_path, "-w", str(dat))

        charts = {chart["id"]: chart for chart in payload["charts"]}
        assert charts["temperatura"]["layers"]["wrf"] is None
        assert "t" in charts["temperatura"]["wrf_pending"]

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
        # Right-open: an hourly mean labelled at T covers [T, T+1h), so the last
        # one INSIDE a window ending at 2022-07-05 00:00 starts at 07-04 23:00.
        # A 49th point would have been the mean of the hour after the window.
        assert hourly["axis"]["count"] == 48

    def test_a_missing_row_lands_at_its_true_position(self, archive: Path, tmp_path: Path) -> None:
        """A logger outage is an absent row, not a NaN one.

        The axis is an origin plus a step, so the page rebuilds every abscissa as
        ``start + i * step``. Serialising only the rows that exist would draw
        every later sample early by the length of the gap; the exporter must put
        the frame back on its grid first.
        """
        frame = pd.read_parquet(archive / "station_5min_qc.parquet")
        marker = 12.5
        frame["T"] = marker
        # A two-hour outage: 24 five-minute rows simply are not there.
        gapped = frame.drop(frame.index[100:124])
        gapped.to_parquet(archive / "station_5min_qc.parquet")

        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        raw = charts["temperatura"]["layers"]["raw"]
        values = raw["series"]["t"]

        assert raw["axis"]["step_minutes"] == 5
        assert raw["axis"]["count"] == len(frame)  # the grid, not the surviving rows
        assert values[100:124] == [None] * 24
        assert values[99] == marker
        assert values[124] == marker

    def test_the_wrf_layer_survives_the_spin_up_hour_drop(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """``read_wrf_series`` drops hour 21 of every day, so the model index has
        a hole per day by construction. The published axis must still land the
        last sample on its real timestamp."""
        index = pd.date_range("2022-07-01", periods=169, freq="h")
        dat = tmp_path / "series_operacional.dat"
        pd.DataFrame(
            {
                "year": index.year,
                "month": index.month,
                "day": index.day,
                "hour": index.hour,
                "T": np.arange(len(index), dtype=float),
            }
        ).to_csv(dat, index=False)

        charts = {
            chart["id"]: chart
            for chart in self._payload(archive, tmp_path, "-w", str(dat))["charts"]
        }
        wrf = charts["temperatura"]["layers"]["wrf"]
        axis = wrf["axis"]
        last = pd.Timestamp(axis["start"]) + (axis["count"] - 1) * pd.Timedelta(
            minutes=axis["step_minutes"]
        )

        # This model file runs to 2022-07-08 00:00, an hour past the newest raw
        # sample (23:55), so the window reaches forward to cover it. The
        # aggregated layers are right-open, which puts the last model hour at
        # 23:00 — seven of these slots are the dropped spin-up hour.
        assert axis["step_minutes"] == 60
        assert axis["count"] == 168
        assert last == pd.Timestamp("2022-07-07 23:00")
        # Every dropped spin-up hour is published as a hole at its own slot.
        assert wrf["series"]["t"][21] is None
        assert wrf["series"]["t"][22] == 22.0

    def test_an_irregular_axis_is_refused(self) -> None:
        """The encoding cannot express two cadences; publishing one anyway draws
        the layer out of phase."""
        irregular = pd.DatetimeIndex(["2022-07-01 00:00", "2022-07-01 01:00", "2022-07-01 03:00"])
        with pytest.raises(ValueError, match="one cadence"):
            _axis(irregular)

    def test_an_off_grid_sample_is_refused_rather_than_dropped(self) -> None:
        """Reindexing a frame whose stamps are not on the cadence would delete
        them silently, trading one wrong axis for missing data."""
        index = pd.DatetimeIndex(["2022-07-01 00:00", "2022-07-01 00:02", "2022-07-01 00:05"])
        frame = pd.DataFrame({"T": [1.0, 2.0, 3.0]}, index=index)
        with pytest.raises(ValueError, match="do not sit on the 5min grid"):
            _regular(frame, index, "5min")

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

    def test_no_model_file_does_not_accuse_the_extraction(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """`wrf_pending` means "the extraction does not write this yet".

        The page states exactly that to the reader, so a run given no model file
        must not populate it: a forgotten `-w` in a cron would otherwise publish
        a false claim about the pipeline for every variable the extraction
        demonstrably does write.
        """
        for chart in self._payload(archive, tmp_path)["charts"]:
            assert chart["layers"]["wrf"] is None
            assert chart["wrf_pending"] == {}, chart["id"]

    def test_a_model_that_lacks_a_variable_still_reports_it(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """The other side of the rule above: with a model present, a variable it
        does not carry is still named."""
        charts = {
            chart["id"]: chart
            for chart in self._payload(archive, tmp_path, "-w", str(self._wrf_dat(tmp_path)))[
                "charts"
            ]
        }

        assert charts["temperatura"]["wrf_pending"] == {}
        # This model carries no rain column, which is the case the field exists for.
        assert "precip" in charts["precipitacao"]["wrf_pending"]

    def test_a_series_with_no_value_is_omitted_not_published_as_nulls(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """An all-null array is not the same as an absent key to the page.

        `layerPoints` returns null only when `layer.series[id]` is MISSING; an
        array of nulls passes that guard, builds a dataset and puts an entry in
        the legend for a line the reader cannot see. Live case: the Gill
        thermohygrometer railed in December 2025 and its readings are masked, so
        the default operational window carries no air temperature at all.
        """
        frame = pd.read_parquet(archive / "station_hourly.parquet")
        frame["T"] = np.nan
        frame.to_parquet(archive / "station_hourly.parquet")
        raw = pd.read_parquet(archive / "station_5min_qc.parquet")
        raw["T"] = np.nan
        raw.to_parquet(archive / "station_5min_qc.parquet")

        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        temperature = charts["temperatura"]

        assert temperature["layers"]["raw"] is None
        assert temperature["layers"]["hourly"] is None
        # The declaration stays: the page still knows the chart HAS a temperature
        # series, it just has nothing to draw for it in this window.
        assert [s["id"] for s in temperature["series"]] == ["t"]
        # A chart whose other columns still have data is untouched.
        assert charts["pressao"]["layers"]["hourly"] is not None

    def test_one_empty_series_does_not_remove_its_healthy_neighbours(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """The balance chart has five series; one dead channel must not blank it."""
        for name in ("station_hourly.parquet", "station_5min_qc.parquet"):
            frame = pd.read_parquet(archive / name)
            frame["Sw_up"] = np.nan
            frame.to_parquet(archive / name)

        charts = {chart["id"]: chart for chart in self._payload(archive, tmp_path)["charts"]}
        hourly = charts["balanco"]["layers"]["hourly"]

        assert hourly is not None
        assert "sw_up" not in hourly["series"]
        assert {"net", "sw_down", "lw_down", "lw_up"} <= set(hourly["series"])

    def test_the_window_reaches_forward_to_cover_an_accumulating_model(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """The operational extraction grows forward, so an hour with a WRF value
        and no observation is the normal state, not a fault.

        Anchoring the window's end on the newest SAMPLE would clip exactly the
        part of the model worth looking at. The start still anchors on the
        station, so the reader keeps the record behind them and gains the
        forecast ahead.
        """
        hourly = pd.read_parquet(archive / "station_hourly.parquet")
        # The anchor is the newest RAW sample, which is what the window's end
        # tracks on the default path.
        station_last = pd.read_parquet(archive / "station_5min_qc.parquet").index.max()
        # Three days of model beyond the newest observation.
        index = pd.date_range(hourly.index.min(), station_last + pd.Timedelta(days=3), freq="h")
        dat = tmp_path / "accumulating.dat"
        pd.DataFrame(
            {
                "year": index.year,
                "month": index.month,
                "day": index.day,
                "hour": index.hour,
                "T": np.arange(len(index), dtype=float),
            }
        ).to_csv(dat, index=False)

        payload = self._payload(archive, tmp_path, "-w", str(dat))
        window = payload["window"]

        # The end tracks what the document CARRIES — here the model's own last
        # hour, three days past the newest observation.
        assert pd.Timestamp(window["end"]) == index.max()
        assert pd.Timestamp(window["end"]) > station_last
        # Published beside it so the page can anchor its visible span on the
        # record instead of on the forecast — without it, a model three days
        # ahead pushes three days of real observations out of the default view.
        assert pd.Timestamp(window["station_end"]) == station_last

        temperature = next(c for c in payload["charts"] if c["id"] == "temperatura")
        model = temperature["layers"]["wrf"]
        model_last = pd.Timestamp(model["axis"]["start"]) + pd.Timedelta(
            minutes=model["axis"]["step_minutes"] * (model["axis"]["count"] - 1)
        )
        assert model_last > station_last, "the model's forward extent must publish"
        assert model_last <= pd.Timestamp(window["end"])

        # The STATION's hourly layer still stops at its own last complete hour.
        # Trimming against the window's end instead publishes the station's
        # trailing PARTIAL hour as a full hourly mean -- an average over however
        # many minutes the logger had written, drawn beside hours built from
        # twelve samples.
        hourly_layer = temperature["layers"]["hourly"]
        hourly_last = pd.Timestamp(hourly_layer["axis"]["start"]) + pd.Timedelta(
            minutes=hourly_layer["axis"]["step_minutes"] * (hourly_layer["axis"]["count"] - 1)
        )
        assert hourly_last <= station_last - pd.Timedelta(hours=1)

    def test_station_end_is_measured_inside_the_window_not_in_the_archive(
        self, archive: Path, tmp_path: Path
    ) -> None:
        """With an explicit --end the newest sample in the file can be far past
        the window, and an anchor outside the window is worse than none."""
        payload = self._payload(archive, tmp_path, "--days", "2", "--end", "2022-07-05")
        window = payload["window"]

        assert pd.Timestamp(window["station_end"]) <= pd.Timestamp(window["end"])
        assert pd.Timestamp(window["station_end"]) >= pd.Timestamp(window["start"])
