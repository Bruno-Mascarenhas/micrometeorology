"""Tests for the labmim-site-graphs monitoring-page producer.

Offline and fast: synthetic hourly CSVs via ``tmp_path``, the headless Agg
backend (set on import of the CLI module), and the Typer app driven through
:class:`~typer.testing.CliRunner`.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from typer.testing import CliRunner

from micrometeorology.cli import generate_station_graphs
from micrometeorology.cli.plot_station_graphs import (
    DEFAULT_BALANCE_COMPONENTS,
    DEFAULT_COLUMNS,
    DEFAULT_DIRECTION_COMPONENTS,
    DEFAULT_WRF_COLUMNS,
    GRAPH_SPECS,
    _plot_balance,
    app,
    render_site_graphs,
)
from micrometeorology.sensors.monitoring import MONITORING_CHARTS
from micrometeorology.sensors.plotting import BALANCE_COMPONENT_COLORS, create_figure

runner = CliRunner()

# The nine fixed filenames the monitoring page reads by exact name.
CONTRACT_PNGS = tuple(spec.filename for spec in GRAPH_SPECS)


def test_balance_uses_okabe_ito_colors_and_negates_upward_channels():
    """The four streams stay distinguishable and retain their physical signs."""
    index = pd.date_range("2026-06-01", periods=2, freq="1h")
    net = pd.Series([1.0, 2.0], index=index)
    components = {
        channel: pd.Series([offset, offset + 1.0], index=index)
        for offset, channel in enumerate(BALANCE_COMPONENT_COLORS, start=1)
    }
    fig, ax = create_figure()

    try:
        _plot_balance(ax, net, components)
        lines = {line.get_label(): line for line in ax.get_lines()}

        assert set(lines) == {"Rn", "SW_dw", "SW_up", "LW_dw", "LW_up"}
        assert to_hex(lines["Rn"].get_color()) == "#000000"

        palette = {to_hex(color) for color in matplotlib.color_sequences["okabe_ito"]}
        component_colors = {
            to_hex(lines[label].get_color()) for label in ("SW_dw", "SW_up", "LW_dw", "LW_up")
        }
        assert len(component_colors) == 4
        assert component_colors <= palette
        assert "#000000" not in component_colors
        assert {channel: to_hex(color) for channel, color in BALANCE_COMPONENT_COLORS.items()} == {
            "sw_down": "#e69f00",
            "sw_up": "#56b4e9",
            "lw_down": "#009e73",
            "lw_up": "#cc79a7",
        }
        for label, channel in {
            "SW_dw": "sw_down",
            "SW_up": "sw_up",
            "LW_dw": "lw_down",
            "LW_up": "lw_up",
        }.items():
            assert to_hex(lines[label].get_color()) == to_hex(BALANCE_COMPONENT_COLORS[channel])
        np.testing.assert_array_equal(lines["SW_up"].get_ydata(), -components["sw_up"])
        np.testing.assert_array_equal(lines["LW_up"].get_ydata(), -components["lw_up"])
    finally:
        plt.close(fig)


def test_legacy_balance_uses_shared_palette_and_negates_upward_channels(monkeypatch, tmp_path):
    """The datalogger graph uses the same accessible four-stream semantics."""
    index = pd.date_range("2026-06-01", periods=2, freq="1h")
    columns = {
        "CG3Up_Wm2Cr_Avg": [1.0, 2.0],
        "CM3Up_Wm2_Avg": [3.0, 4.0],
        "CG3Dn_Wm2Cr_Avg": [5.0, 6.0],
        "CM3Dn_Wm2_Avg": [7.0, 8.0],
    }
    frame = pd.DataFrame(columns, index=index)
    monkeypatch.setattr(generate_station_graphs, "save_figure", lambda *_args, **_kwargs: None)

    generate_station_graphs._plot_balanco(
        frame,
        frame,
        tmp_path,
        # Naive on purpose; it only feeds the figure's caption label.
        pd.Timestamp(2026, 6, 1, 1),
    )
    fig = plt.gcf()

    try:
        lines = {line.get_label(): line for line in fig.axes[0].get_lines() if line.get_label()}
        expected_channels = {
            "SW_dw": "sw_down",
            "SW_up": "sw_up",
            "LW_dw": "lw_down",
            "LW_up": "lw_up",
        }
        for label, channel in expected_channels.items():
            assert to_hex(lines[label].get_color()) == to_hex(BALANCE_COMPONENT_COLORS[channel])
        np.testing.assert_array_equal(lines["SW_up"].get_ydata(), -frame["CM3Dn_Wm2_Avg"])
        np.testing.assert_array_equal(lines["LW_up"].get_ydata(), -frame["CG3Dn_Wm2Cr_Avg"])
    finally:
        plt.close(fig)


def _write_hourly_csv(path: Path, *, columns: dict[str, str] | None = None, days: int = 10) -> Path:
    """Write a synthetic hourly processed-sensor CSV with all contract columns.

    ``columns`` overrides the default logical→CSV column names so tests can
    exercise renamed loggers; ``None`` uses :data:`DEFAULT_COLUMNS`.
    """
    mapping = columns if columns is not None else DEFAULT_COLUMNS
    idx = pd.date_range("2026-06-01", periods=days * 24, freq="1h")
    n = len(idx)
    rng = np.random.default_rng(7)
    step = np.arange(n)
    values = {
        "temperatura": 25.0 + 3.0 * np.sin(step / 6.0),
        "umidade": 70.0 + 10.0 * np.cos(step / 6.0),
        "pressao": 1013.0 + rng.normal(0, 1, n),
        "precipitacao": np.clip(rng.normal(0, 0.5, n), 0, None),
        "velocidade": np.abs(rng.normal(3, 1, n)),
        "direcao": rng.uniform(0, 360, n),
        "balanco": 300.0 * np.sin(step / 12.0),
        "radiacao_difusa": np.clip(400.0 * np.sin(step / 12.0), 0, None),
        "radiacao_par": np.clip(200.0 * np.sin(step / 12.0), 0, None),
    }
    frame = {mapping[key]: series for key, series in values.items()}
    df = pd.DataFrame(frame, index=idx)
    df.to_csv(path)
    return path


@pytest.fixture
def hourly_csv(tmp_path: Path) -> Path:
    """A processed hourly CSV carrying every contract column."""
    return _write_hourly_csv(tmp_path / "hourly.csv")


class TestSiteCommand:
    def test_produces_exactly_the_nine_contract_pngs(self, hourly_csv, tmp_path):
        out = tmp_path / "graphs"
        result = runner.invoke(
            app, ["site", "-i", str(hourly_csv), "-o", str(out), "--log-level", "WARNING"]
        )

        assert result.exit_code == 0, result.output
        produced = sorted(p.name for p in out.glob("*.png"))
        assert produced == sorted(CONTRACT_PNGS)
        # Every image is a real, non-empty PNG.
        for name in produced:
            assert (out / name).stat().st_size > 0

    def test_last_days_clips_the_window(self, tmp_path):
        # 10 days of data, ask for 3; graphs still emit, no crash on the clip.
        csv = _write_hourly_csv(tmp_path / "long.csv", days=10)
        out = tmp_path / "g"
        result = runner.invoke(
            app,
            ["site", "-i", str(csv), "-o", str(out), "--last-days", "3", "--log-level", "WARNING"],
        )
        assert result.exit_code == 0, result.output
        assert len(list(out.glob("*.png"))) == len(CONTRACT_PNGS)

    def test_missing_column_warns_and_skips_but_exits_zero(self, tmp_path):
        # Drop the temperature source column only.
        df = pd.read_csv(_write_hourly_csv(tmp_path / "src.csv"), index_col=0, parse_dates=True)
        df = df.drop(columns=[DEFAULT_COLUMNS["temperatura"]])
        csv = tmp_path / "no_temp.csv"
        df.to_csv(csv)

        out = tmp_path / "g"
        result = runner.invoke(
            app, ["site", "-i", str(csv), "-o", str(out), "--log-level", "WARNING"]
        )

        assert result.exit_code == 0, result.output
        produced = sorted(p.name for p in out.glob("*.png"))
        assert "temperatura.png" not in produced
        assert len(produced) == len(CONTRACT_PNGS) - 1
        assert "temperatura" in result.output  # the skip is reported

    def test_strict_makes_a_missing_column_fail(self, tmp_path):
        df = pd.read_csv(_write_hourly_csv(tmp_path / "src.csv"), index_col=0, parse_dates=True)
        df = df.drop(columns=[DEFAULT_COLUMNS["radiacao_par"]])
        csv = tmp_path / "no_par.csv"
        df.to_csv(csv)

        out = tmp_path / "g"
        result = runner.invoke(
            app,
            ["site", "-i", str(csv), "-o", str(out), "--strict", "--log-level", "WARNING"],
        )

        assert result.exit_code != 0

    def test_col_override_retargets_a_renamed_logger_column(self, tmp_path):
        # Logger renamed the temperature column; --col points the graph at it
        # without any code change.
        renamed = dict(DEFAULT_COLUMNS)
        renamed["temperatura"] = "AirT2_C_Avg"
        csv = _write_hourly_csv(tmp_path / "renamed.csv", columns=renamed)

        out = tmp_path / "g"
        # Without the override the default column is missing -> skipped.
        base = runner.invoke(
            app, ["site", "-i", str(csv), "-o", str(out), "--log-level", "WARNING"]
        )
        assert base.exit_code == 0, base.output
        assert not (out / "temperatura.png").exists()

        out2 = tmp_path / "g2"
        overridden = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(csv),
                "-o",
                str(out2),
                "--col",
                "temperatura=AirT2_C_Avg",
                "--log-level",
                "WARNING",
            ],
        )
        assert overridden.exit_code == 0, overridden.output
        assert (out2 / "temperatura.png").stat().st_size > 0

    def test_config_yaml_overrides_columns(self, tmp_path):
        renamed = dict(DEFAULT_COLUMNS)
        renamed["umidade"] = "RH_probe2"
        csv = _write_hourly_csv(tmp_path / "cfg.csv", columns=renamed)

        config = tmp_path / "cols.yaml"
        config.write_text("columns:\n  umidade: RH_probe2\n", encoding="utf-8")

        out = tmp_path / "g"
        result = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(csv),
                "-o",
                str(out),
                "--config",
                str(config),
                "--log-level",
                "WARNING",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "umidade.png").stat().st_size > 0

    def test_direction_reconstructed_from_uv_components(self, tmp_path):
        # No direct WindDir column, but U/V present -> direction graph still made.
        df = pd.read_csv(_write_hourly_csv(tmp_path / "src.csv"), index_col=0, parse_dates=True)
        df = df.drop(columns=[DEFAULT_COLUMNS["direcao"]])
        df["u"] = -np.sin(np.radians(45.0))
        df["v"] = -np.cos(np.radians(45.0))
        csv = tmp_path / "uv.csv"
        df.to_csv(csv)

        out = tmp_path / "g"
        result = runner.invoke(
            app, ["site", "-i", str(csv), "-o", str(out), "--log-level", "WARNING"]
        )
        assert result.exit_code == 0, result.output
        assert (out / "direcao.png").stat().st_size > 0


def _write_raw_parquet(path: Path, hourly: Path) -> Path:
    """A 5-minute record spanning the same window as the hourly CSV."""
    reference = pd.read_csv(hourly, index_col=0, parse_dates=True)
    index = pd.date_range(reference.index.min(), reference.index.max(), freq="5min")
    generator = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            column: generator.normal(float(np.nanmean(reference[column])), 0.5, len(index))
            for column in reference.columns
        },
        index=index,
    )
    frame.to_parquet(path)
    return path


def _write_wrf_dat(path: Path, hourly: Path) -> Path:
    """A minimal series_operacional.dat over the same window."""
    reference = pd.read_csv(hourly, index_col=0, parse_dates=True)
    index = pd.date_range(reference.index.min(), reference.index.max(), freq="1h")
    generator = np.random.default_rng(12)
    frame = pd.DataFrame(
        {
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "T": generator.normal(25.0, 2.0, len(index)),
            "Swdw": np.clip(generator.normal(300.0, 100.0, len(index)), 0, None),
        }
    )
    frame.to_csv(path, index=False)
    return path


class TestThreeLayers:
    """The researcher's requirement: raw under hourly, and WRF wherever it exists.

    The static producer and the interactive page must show the same three
    layers, so this is pinned here as well as in ``test_monitoring.py``.
    """

    def test_raw_and_wrf_layers_reach_every_graph_that_has_them(self, hourly_csv, tmp_path):
        out = tmp_path / "three"
        result = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(hourly_csv),
                "-o",
                str(out),
                "--raw",
                str(_write_raw_parquet(tmp_path / "raw.parquet", hourly_csv)),
                "--wrf",
                str(_write_wrf_dat(tmp_path / "wrf.dat", hourly_csv)),
                "--log-level",
                "WARNING",
            ],
        )
        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in out.glob("*.png")) == sorted(CONTRACT_PNGS)

    def test_without_the_optional_layers_the_output_is_unchanged(self, hourly_csv, tmp_path):
        """The legacy single-layer invocation must keep working untouched."""
        out = tmp_path / "one"
        result = runner.invoke(
            app, ["site", "-i", str(hourly_csv), "-o", str(out), "--log-level", "WARNING"]
        )
        assert result.exit_code == 0, result.output
        assert sorted(p.name for p in out.glob("*.png")) == sorted(CONTRACT_PNGS)

    def test_a_wrf_column_the_extraction_lacks_is_skipped_not_fatal(self, hourly_csv, tmp_path):
        """`series_operacional.dat` has no PAR and no rain; that is normal."""
        from micrometeorology.cli.plot_station_graphs import _wrf_series

        wrf = pd.read_csv(_write_wrf_dat(tmp_path / "w.dat", hourly_csv))
        series, resolved = _wrf_series(wrf, "radiacao_par", DEFAULT_WRF_COLUMNS)
        assert series is None
        assert resolved is None

    def test_the_model_overlay_borrows_the_hue_of_the_series_it_mirrors(self):
        """On the balance chart hue means the physical family, so the WRF
        shortwave must not arrive as a sixth colour — it is dotted instead."""
        from micrometeorology.cli.plot_station_graphs import _plot_wrf

        index = pd.date_range("2026-06-01", periods=3, freq="1h")
        fig, ax = create_figure()
        try:
            _plot_wrf(
                ax,
                pd.Series([1.0, 2.0, 3.0], index=index),
                label="WRF",
                color=BALANCE_COMPONENT_COLORS["sw_down"],
            )
            (line,) = ax.get_lines()
            assert to_hex(line.get_color()) == to_hex(BALANCE_COMPONENT_COLORS["sw_down"])
            assert line.get_linestyle() not in {"-", "solid"}
        finally:
            plt.close(fig)


class TestColumnsCommand:
    def test_generic_mode_writes_legacy_named_files(self, hourly_csv, tmp_path):
        out = tmp_path / "adhoc"
        result = runner.invoke(
            app,
            [
                "columns",
                "-i",
                str(hourly_csv),
                "-o",
                str(out),
                "-v",
                "AirT1_C_Avg",
                "-v",
                "RH1",
                "--last-days",
                "5",
                "--log-level",
                "WARNING",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out / "AirT1_C_Avg_last_5d.png").stat().st_size > 0
        assert (out / "RH1_last_5d.png").stat().st_size > 0

    def test_unknown_column_warns_and_is_skipped(self, hourly_csv, tmp_path):
        out = tmp_path / "adhoc"
        result = runner.invoke(
            app,
            [
                "columns",
                "-i",
                str(hourly_csv),
                "-o",
                str(out),
                "-v",
                "NoSuchColumn",
                "--log-level",
                "WARNING",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "not found" in result.output
        assert not (out / "NoSuchColumn_last_7d.png").exists()


class TestTheTwoProducersAgree:
    """The PNG and the interactive payload draw the same nine charts.

    Both docstrings say they come "from the same archive" and must "be read the
    same way", but each carried its own literals and they had already drifted in
    two places: the PNG's rain candidates had lost ``rain_total``, and its
    humidity frame clipped at 100 % while the payload declared 105 % with a
    caveat explaining that the model exceeds saturation and those points must
    stay visible.
    """

    @staticmethod
    def _payload_limits() -> dict[str, tuple[float, float] | None]:
        return {chart.id: chart.y_limits for chart in MONITORING_CHARTS}

    def test_the_two_catalogues_cover_the_same_charts(self) -> None:
        assert {spec.key for spec in GRAPH_SPECS} == {chart.id for chart in MONITORING_CHARTS}

    def test_y_limits_agree_chart_by_chart(self) -> None:
        payload = self._payload_limits()
        for spec in GRAPH_SPECS:
            expected = payload[spec.key]
            actual = tuple(float(bound) for bound in spec.ylim) if spec.ylim else None
            assert actual == expected, (
                f"{spec.key}: PNG frames {actual} while the payload declares {expected}; "
                "the same data must not be clipped in one product and shown in the other"
            )

    def test_the_rain_candidates_are_one_list_not_two(self) -> None:
        """The ordered tuple is what makes a new spelling a data change."""
        precipitation = next(chart for chart in MONITORING_CHARTS if chart.id == "precipitacao")
        assert DEFAULT_WRF_COLUMNS["precipitacao"] == precipitation.series[0].wrf


class TestAnEmptyStationColumnDoesNotBecomeALegendEntry:
    """A column that exists but holds no value must not be announced.

    Plotting an all-NaN series draws nothing and still registers a legend entry,
    so the chart named a series the reader cannot see — which reads as a line off
    the scale, or as an instrument stuck flat. Two independent causes reach this
    state: a column belonging to a different instrument era, and a channel the
    station genuinely did not record that year.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range("2023-08-26", periods=48, freq="h")
        return pd.DataFrame(
            {
                # Present and populated.
                "PL01_mm_Tot": np.zeros(48),
                # Present and entirely empty — the case under test.
                "AirT1_C_Avg": np.full(48, np.nan),
                "RH1": np.full(48, np.nan),
                "BP1_mbar_Avg": np.full(48, np.nan),
                "WS_ms": np.full(48, np.nan),
                "WindDir": np.full(48, np.nan),
                "Net_Wm2_Avg": np.full(48, np.nan),
                "PSP_Wm2_Avg": np.full(48, np.nan),
                "PAR_Wm2_Avg": np.full(48, np.nan),
            },
            index=index,
        )

    def test_the_empty_columns_are_reported_and_not_drawn(self, tmp_path: Path) -> None:
        written, missing, empty = render_site_graphs(
            self._frame(),
            tmp_path,
            DEFAULT_COLUMNS,
            DEFAULT_BALANCE_COMPONENTS,
            DEFAULT_DIRECTION_COMPONENTS,
        )

        # Only precipitation had data, so only it can be published; the eight
        # empty ones drew no layer at all and must not overwrite a good image.
        assert [p.name for p in written] == ["precipitacao.png"]
        assert set(empty) >= {"temperatura", "umidade", "pressao", "velocidade", "direcao"}
        # `missing` is reserved for a column the frame does not CARRY, which is a
        # configuration error; every column here is present and simply has no
        # value, which is the station being down. Only the first fails --strict.
        assert missing == []

    def test_a_model_layer_still_publishes_without_the_station(self, tmp_path: Path) -> None:
        """The chart keeps whatever layer does have data — it just stops
        claiming the one that does not."""
        frame = self._frame()
        model = pd.DataFrame({"pressure": np.linspace(1010, 1020, 48)}, index=frame.index)

        written, _missing, empty = render_site_graphs(
            frame,
            tmp_path,
            DEFAULT_COLUMNS,
            DEFAULT_BALANCE_COMPONENTS,
            DEFAULT_DIRECTION_COMPONENTS,
            wrf=model,
        )

        assert "pressao.png" in [p.name for p in written]
        assert "pressao" in empty

    def test_strict_fails_on_a_missing_column_but_not_on_an_empty_one(self, tmp_path: Path) -> None:
        """The distinction is a broken configuration against a broken instrument.

        The Gill thermohygrometer railed in December 2025, so temperature and
        humidity are empty in every current window. Failing `--strict` on that
        would make the hourly cron exit non-zero every run until the instrument
        is repaired, freezing the site on its last good artifacts — for a
        condition these images exist to report, not to hide.
        """
        frame = self._frame()
        csv = tmp_path / "empty.csv"
        frame.to_csv(csv)

        empty_run = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(csv),
                "-o",
                str(tmp_path / "a"),
                "--strict",
                "--log-level",
                "WARNING",
            ],
        )
        assert empty_run.exit_code == 0, empty_run.output
        assert "without the station layer" in empty_run.output

        # Drop a column outright: that IS a configuration error.
        frame.drop(columns=[DEFAULT_COLUMNS["precipitacao"]]).to_csv(csv)
        missing_run = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(csv),
                "-o",
                str(tmp_path / "b"),
                "--strict",
                "--log-level",
                "WARNING",
            ],
        )
        assert missing_run.exit_code == 1, missing_run.output
