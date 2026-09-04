"""Tests for the labmim-site-graphs monitoring-page producer.

Offline and fast: synthetic hourly CSVs via ``tmp_path``, the headless Agg
backend (set on import of the CLI module), and the Typer app driven through
:class:`~typer.testing.CliRunner`.
"""

from collections.abc import Iterator
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from typer.testing import CliRunner

from micrometeorology.cli import generate_station_graphs, plot_station_graphs
from micrometeorology.cli.plot_station_graphs import (
    DEFAULT_BALANCE_COMPONENTS,
    DEFAULT_COLUMNS,
    DEFAULT_DIRECTION_COMPONENTS,
    DEFAULT_WRF_COLUMNS,
    GRAPH_SPECS,
    _plot_balance,
    app,
    load_graph_config,
    load_hourly_csv,
    render_site_graphs,
    resolve_column,
)
from micrometeorology.sensors.monitoring import MONITORING_CHARTS
from micrometeorology.sensors.plotting import (
    BALANCE_COMPONENT_COLORS,
    create_figure,
    setup_date_axis,
)

runner = CliRunner()

# The nine fixed filenames the monitoring page reads by exact name.
CONTRACT_PNGS = tuple(spec.filename for spec in GRAPH_SPECS)


@pytest.fixture
def balance_chart() -> Iterator[tuple[dict[str, Line2D], dict[str, pd.Series]]]:
    """The balance axes as ``_plot_balance`` draws them, keyed by legend label."""
    index = pd.date_range("2026-06-01", periods=2, freq="1h")
    net = pd.Series([1.0, 2.0], index=index)
    components = {
        channel: pd.Series([offset, offset + 1.0], index=index)
        for offset, channel in enumerate(BALANCE_COMPONENT_COLORS, start=1)
    }
    fig, ax = create_figure()
    try:
        _plot_balance(ax, net, components)
        yield {str(line.get_label()): line for line in ax.get_lines()}, components
    finally:
        plt.close(fig)


def test_the_balance_draws_the_net_in_black_beside_its_four_components(balance_chart):
    lines, _components = balance_chart

    assert set(lines) == {"Rn", "SW_dw", "SW_up", "LW_dw", "LW_up"}
    assert to_hex(lines["Rn"].get_color()) == "#000000"


def test_the_four_components_stay_distinguishable_on_the_shared_palette(balance_chart):
    """Four Okabe-Ito hues, none of them the net's black."""
    lines, _components = balance_chart
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


def test_the_upward_channels_are_drawn_negative(balance_chart):
    """Upwelling leaves the surface, so it is drawn below the axis."""
    lines, components = balance_chart

    np.testing.assert_array_equal(lines["SW_up"].get_ydata(), -components["sw_up"])
    np.testing.assert_array_equal(lines["LW_up"].get_ydata(), -components["lw_up"])


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


def _logger_name(key: str) -> str:
    """The column ``labmim-sensor-process`` exports for a contract graph.

    The TAIL of the candidate chain. The head is the unified name that only
    ``labmim-archive`` builds, and a processed hourly CSV never carries it.
    """
    return DEFAULT_COLUMNS[key][-1]


def _write_hourly_csv(path: Path, *, columns: dict[str, str] | None = None, days: int = 10) -> Path:
    """Write a synthetic hourly processed-sensor CSV with all contract columns.

    ``columns`` overrides the default logical→CSV column names so tests can
    exercise renamed loggers; ``None`` uses the logger spellings of
    :data:`DEFAULT_COLUMNS`.
    """
    mapping = columns if columns is not None else {k: _logger_name(k) for k in DEFAULT_COLUMNS}
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
        for name in produced:
            assert (out / name).stat().st_size > 0

    def test_last_days_clips_the_window(self, tmp_path):
        csv = _write_hourly_csv(tmp_path / "long.csv", days=10)
        out = tmp_path / "g"
        result = runner.invoke(
            app,
            ["site", "-i", str(csv), "-o", str(out), "--last-days", "3", "--log-level", "WARNING"],
        )
        assert result.exit_code == 0, result.output
        assert len(list(out.glob("*.png"))) == len(CONTRACT_PNGS)


class TestTheLoadedWindow:
    """Nine filenames come out whatever the window, so the clip is pinned here."""

    def test_the_window_reaches_exactly_last_days_back_from_the_newest_row(self, tmp_path):
        csv = _write_hourly_csv(tmp_path / "long.csv", days=10)

        clipped = load_hourly_csv(csv, 3)

        newest = clipped.index.max()
        assert newest == pd.Timestamp("2026-06-10 23:00:00")
        assert clipped.index.min() == newest - pd.Timedelta(days=3)
        assert len(clipped) == 3 * 24 + 1

    @pytest.mark.parametrize("last_days", [0, -1])
    def test_a_window_of_zero_days_or_less_keeps_the_whole_file(self, tmp_path, last_days):
        """The flag's documented 'keep everything' escape hatch."""
        csv = _write_hourly_csv(tmp_path / "long.csv", days=10)

        assert len(load_hourly_csv(csv, last_days)) == 10 * 24

    def test_a_parquet_archive_loads_to_the_same_frame_as_the_csv(self, tmp_path):
        """`labmim-archive` writes parquet and only it carries the unified names,
        so the branch that reads it must not be the untested one.
        """
        csv = _write_hourly_csv(tmp_path / "long.csv", days=4)
        parquet = tmp_path / "station_hourly.parquet"
        pd.read_csv(csv, index_col=0, parse_dates=True).to_parquet(parquet)

        pd.testing.assert_frame_equal(load_hourly_csv(parquet, 2), load_hourly_csv(csv, 2))

    def test_missing_column_warns_and_skips_but_exits_zero(self, tmp_path):
        df = pd.read_csv(_write_hourly_csv(tmp_path / "src.csv"), index_col=0, parse_dates=True)
        df = df.drop(columns=[_logger_name("temperatura")])
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
        assert "temperatura" in result.output, "the skip must be reported"

    def test_strict_makes_a_missing_column_fail(self, tmp_path):
        df = pd.read_csv(_write_hourly_csv(tmp_path / "src.csv"), index_col=0, parse_dates=True)
        df = df.drop(columns=[_logger_name("radiacao_par")])
        csv = tmp_path / "no_par.csv"
        df.to_csv(csv)

        out = tmp_path / "g"
        result = runner.invoke(
            app,
            ["site", "-i", str(csv), "-o", str(out), "--strict", "--log-level", "WARNING"],
        )

        assert result.exit_code != 0

    def test_col_override_retargets_a_renamed_logger_column(self, tmp_path):
        """A logger that renames its temperature column must be reachable by
        ``--col`` alone, with no code change: without the override the default
        column is missing and the graph is skipped."""
        renamed = {key: _logger_name(key) for key in DEFAULT_COLUMNS}
        renamed["temperatura"] = "AirT2_C_Avg"
        csv = _write_hourly_csv(tmp_path / "renamed.csv", columns=renamed)

        out = tmp_path / "g"
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
        renamed = {key: _logger_name(key) for key in DEFAULT_COLUMNS}
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
        df = df.drop(columns=[_logger_name("direcao")])
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

    def test_raw_and_wrf_layers_reach_every_graph_that_has_them(
        self, hourly_csv, tmp_path, monkeypatch
    ):
        """Counting the nine filenames proves nothing: the command writes the same
        nine with no --raw and no --wrf at all. The layers are read off the axes
        of each figure as it is saved.
        """
        drawn: dict[str, list[str]] = {}
        original = plot_station_graphs.save_figure

        def _record(fig, path):
            drawn[Path(path).name] = list(fig.axes[0].get_legend_handles_labels()[1])
            return original(fig, path)

        monkeypatch.setattr(plot_station_graphs, "save_figure", _record)

        result = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(hourly_csv),
                "-o",
                str(tmp_path / "three"),
                "--raw",
                str(_write_raw_parquet(tmp_path / "raw.parquet", hourly_csv)),
                "--wrf",
                str(_write_wrf_dat(tmp_path / "wrf.dat", hourly_csv)),
                "--log-level",
                "WARNING",
            ],
        )

        assert result.exit_code == 0, result.output
        assert sorted(drawn) == sorted(CONTRACT_PNGS)
        for filename, labels in drawn.items():
            assert "bruto 5 min" in labels, filename
        # The synthetic .dat carries T and Swdw only, so exactly the two graphs
        # whose WRF candidate resolves may announce a model layer.
        with_model = {name for name, labels in drawn.items() if any("WRF 1h" in x for x in labels)}
        assert with_model == {"temperatura.png", "balanco.png"}

    def test_the_raw_direction_layer_is_wrapped_onto_the_compass(self, tmp_path, monkeypatch):
        """The raw record is degrees from a vane and can carry 360 or a negative
        residue; the scatter axis is [0, 360), so the layer is taken modulo 360
        before it is drawn. Unwrapped, the points leave the axis silently.
        """
        drawn: dict[str, np.ndarray] = {}
        original = plot_station_graphs.save_figure

        def _record(fig, path):
            (raw_line,) = [
                line for line in fig.axes[0].get_lines() if line.get_label() == "bruto 5 min"
            ]
            drawn[Path(path).name] = np.asarray(raw_line.get_ydata())
            return original(fig, path)

        monkeypatch.setattr(plot_station_graphs, "save_figure", _record)

        index = pd.date_range("2026-06-01", periods=48, freq="h")
        hourly = pd.DataFrame({_logger_name("direcao"): np.linspace(0.0, 359.0, 48)}, index=index)
        raw = pd.DataFrame({_logger_name("direcao"): np.linspace(-45.0, 405.0, 48)}, index=index)

        render_site_graphs(
            hourly,
            tmp_path / "compass",
            DEFAULT_COLUMNS,
            DEFAULT_BALANCE_COMPONENTS,
            DEFAULT_DIRECTION_COMPONENTS,
            raw=raw,
        )

        points = drawn["direcao.png"]
        assert points.min() >= 0.0
        assert points.max() < 360.0

    def test_a_wrf_column_the_extraction_lacks_is_skipped_not_fatal(self, hourly_csv, tmp_path):
        """`series_operacional.dat` has no PAR and no rain; that is normal."""
        from micrometeorology.cli.plot_station_graphs import _wrf_series

        wrf = pd.read_csv(_write_wrf_dat(tmp_path / "w.dat", hourly_csv))
        series, resolved = _wrf_series(wrf, "radiacao_par", DEFAULT_WRF_COLUMNS)
        assert series is None
        assert resolved is None

    def test_the_balance_overlay_borrows_the_shortwave_hue_and_other_charts_do_not(
        self, tmp_path, monkeypatch
    ):
        """``render_site_graphs`` is what decides the overlay hue, per chart kind.

        On the balance chart the model line mirrors incoming shortwave and takes
        that channel's hue; every other chart keeps the model's own colour, which
        is what keeps the two readable side by side on the page.
        """
        overlay: dict[str, str] = {}
        original = plot_station_graphs.save_figure

        def _record(fig, path):
            for line in fig.axes[0].get_lines():
                if str(line.get_label()).startswith("WRF 1h"):
                    overlay[Path(path).name] = to_hex(line.get_color())
            return original(fig, path)

        monkeypatch.setattr(plot_station_graphs, "save_figure", _record)

        index = pd.date_range("2026-06-01", periods=24, freq="h")
        shape = np.clip(np.sin((index.hour.to_numpy() - 6) / 12.0 * np.pi), 0.0, None)
        station = pd.DataFrame(
            {"Net_Wm2_Avg": 500.0 * shape, "BP1_mbar_Avg": np.full(24, 1013.0)}, index=index
        )
        model = pd.DataFrame(
            {"swdown_w_m2": 800.0 * shape, "psfc_hpa": np.full(24, 1012.0)}, index=index
        )

        render_site_graphs(
            station,
            tmp_path / "hues",
            DEFAULT_COLUMNS,
            DEFAULT_BALANCE_COMPONENTS,
            DEFAULT_DIRECTION_COMPONENTS,
            wrf=model,
        )

        assert overlay["balanco.png"] == to_hex(BALANCE_COMPONENT_COLORS["sw_down"])
        assert overlay["pressao.png"] == to_hex(plot_station_graphs._WRF_COLOR)

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

    Both are documented as coming "from the same archive" and as being read the
    same way, so their literals must not drift apart. The humidity frame is the
    sharp case: the payload declares 105 % because the model exceeds saturation
    and those points have to stay visible, so a PNG clipped at 100 % would hide
    in one product what the other publishes.
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

        assert [p.name for p in written] == ["precipitacao.png"], (
            "only the column with data can be published; the empty ones drew no "
            "layer at all and must not overwrite a good image"
        )
        assert set(empty) == {spec.key for spec in GRAPH_SPECS} - {"precipitacao"}
        assert missing == [], (
            "`missing` is reserved for a column the frame does not CARRY: every "
            "column here is present and simply has no value"
        )

    def test_a_model_layer_still_publishes_without_the_station(self, tmp_path: Path) -> None:
        """The chart keeps whatever layer does have data — it just stops
        claiming the one that does not."""
        frame = self._frame()
        model = pd.DataFrame({"psfc_hpa": np.linspace(1010, 1020, 48)}, index=frame.index)

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

    def test_the_balance_draws_its_four_components_when_only_net_is_empty(
        self, tmp_path: Path
    ) -> None:
        """`balanco` is the one multi-series chart, and the emptiness gate read
        only its aggregate Net column: a logger that stopped writing Rn dropped
        the four measured streams with it, even fully populated."""
        frame = self._frame()
        hours = pd.DatetimeIndex(frame.index).hour.to_numpy()
        shape = np.clip(np.sin((hours - 6) / 12.0 * np.pi), 0.0, None)
        frame["CM3Up_Wm2_Avg"] = 900.0 * shape
        frame["CM3Dn_Wm2_Avg"] = 180.0 * shape
        frame["CG3Up_Wm2Cr_Avg"] = np.full(48, 400.0)
        frame["CG3Dn_Wm2Cr_Avg"] = np.full(48, 450.0)

        written, _missing, empty = render_site_graphs(
            frame,
            tmp_path,
            DEFAULT_COLUMNS,
            DEFAULT_BALANCE_COMPONENTS,
            DEFAULT_DIRECTION_COMPONENTS,
        )

        assert "balanco.png" in [path.name for path in written]
        assert "balanco" not in empty

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

        frame.drop(columns=[_logger_name("precipitacao")]).to_csv(csv)
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

    def test_both_producers_plot_the_same_physical_longwave(self, monkeypatch, tmp_path) -> None:
        """The two producers of ``balanco.png`` must read the same quantity.

        ``CG3*_Wm2_Avg`` is the pyrgeometer's raw thermopile signal, missing the
        sigma*T_body^4 case term; ``CG3*_Wm2Cr_Avg`` is the flux. Plotting the
        first gives a downwelling longwave of about -42 W/m2 — a value this
        station cannot measure — under the same filename and title as the
        interactive chart showing +406.

        The correction adds the same term to both channels, so net longwave is
        identical either way: a chart that still sums to Rn is no evidence that
        the two individual values are right, and here both are off by ~447 W/m2.
        """
        index = pd.date_range("2026-06-01", periods=2, freq="1h")
        frame = pd.DataFrame(
            {
                "CM3Up_Wm2_Avg": [700.0, 750.0],
                "CM3Dn_Wm2_Avg": [90.0, 95.0],
                "CG3Up_Wm2_Avg": [-42.0, -40.0],
                "CG3Up_Wm2Cr_Avg": [406.0, 410.0],
                "CG3Dn_Wm2_Avg": [-30.0, -28.0],
                "CG3Dn_Wm2Cr_Avg": [450.0, 455.0],
            },
            index=index,
        )
        monkeypatch.setattr(generate_station_graphs, "save_figure", lambda *_args, **_kwargs: None)

        generate_station_graphs._plot_balanco(frame, frame, tmp_path, pd.Timestamp(2026, 6, 1, 1))
        fig = plt.gcf()

        try:
            lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
            np.testing.assert_array_equal(lines["LW_dw"].get_ydata(), frame["CG3Up_Wm2Cr_Avg"])
            np.testing.assert_array_equal(lines["LW_up"].get_ydata(), -frame["CG3Dn_Wm2Cr_Avg"])
        finally:
            plt.close(fig)

        for channel in ("lw_down", "lw_up"):
            column = DEFAULT_BALANCE_COMPONENTS[channel][-1]
            assert column.endswith("Cr_Avg"), f"{channel} must be the corrected flux, got {column}"


class TestTheUnifiedChannelWinsWhenTheFrameHasIt:
    """Which instrument carries a quantity changes over the record.

    The shade ring was off the PSP from 2019-09 to 2025-05 and the CMP21 carried
    diffuse instead, so reading ``PSP_Wm2_Avg`` unconditionally publishes
    near-global irradiance for any window replayed inside those six years, under
    the title "Radiação Difusa" -- 100.3 W/m2 for the 2022-07 reference week
    against the 67.6 the interactive page publishes for the very same hours from
    ``Sw_dif``.
    """

    @staticmethod
    def _both_names() -> pd.DataFrame:
        """A frame as ``labmim-archive`` writes it: unified name beside its source."""
        idx = pd.date_range("2022-07-01", periods=48, freq="1h")
        step = np.arange(len(idx))
        return pd.DataFrame(
            {
                # The unshaded PSP reads essentially global irradiance here.
                "PSP_Wm2_Avg": np.clip(400.0 * np.sin(step / 12.0), 0, None),
                # The era-correct diffuse channel is the CMP21 behind the ring.
                "Sw_dif": np.clip(150.0 * np.sin(step / 12.0), 0, None),
            },
            index=idx,
        )

    def test_the_era_mapped_channel_is_what_gets_plotted(self, tmp_path):
        frame = self._both_names()

        written, missing, _empty = render_site_graphs(
            frame,
            tmp_path / "g",
            DEFAULT_COLUMNS,
            {},
            DEFAULT_DIRECTION_COMPONENTS,
        )

        assert "radiacao_difusa" not in missing
        assert any(path.name == "radiacao_difusa.png" for path in written)
        assert resolve_column(frame, DEFAULT_COLUMNS["radiacao_difusa"]) == "Sw_dif"

    def test_the_logger_column_still_works_when_it_is_all_there_is(self, tmp_path):
        """The processed-CSV path has no unified names and must keep working."""
        frame = self._both_names().drop(columns=["Sw_dif"])

        assert resolve_column(frame, DEFAULT_COLUMNS["radiacao_difusa"]) == "PSP_Wm2_Avg"

        written, missing, _empty = render_site_graphs(
            frame,
            tmp_path / "g",
            DEFAULT_COLUMNS,
            {},
            DEFAULT_DIRECTION_COMPONENTS,
        )

        assert "radiacao_difusa" not in missing
        assert any(path.name == "radiacao_difusa.png" for path in written)

    def test_an_override_replaces_the_whole_chain(self):
        """`--col` names ONE column; the shipped alternatives must not outrank it."""
        columns, _balance, _direction = load_graph_config(None, ["radiacao_difusa=PSP_Wm2_Avg"])

        assert columns["radiacao_difusa"] == ("PSP_Wm2_Avg",)
        assert resolve_column(self._both_names(), columns["radiacao_difusa"]) == "PSP_Wm2_Avg"


class TestTheDateAxisSurvivesALongWindow:
    """A fixed ``DayLocator`` does not thin -- past 1000 ticks matplotlib drops them.

    The archive's own hourly frame spans 2016-2026, which asks for 3,844 daily
    ticks; the axis then renders as an unreadable black band.
    """

    @staticmethod
    def _axis(days: int):
        fig, ax = create_figure()
        index = pd.date_range("2020-01-01", periods=days * 24, freq="1h")
        ax.plot(index, np.arange(len(index), dtype=float))
        setup_date_axis(ax)
        return fig, ax

    def test_a_week_keeps_one_tick_per_day(self):
        """The contract graphs plot a week; that look must not change."""
        fig, ax = self._axis(7)
        try:
            assert isinstance(ax.xaxis.get_major_locator(), mdates.DayLocator)
            labels = [t for t in ax.xaxis.get_majorticklabels() if t.get_text()]
            assert 6 <= len(labels) <= 9, [t.get_text() for t in labels]
        finally:
            plt.close(fig)

    def test_a_decade_stays_readable(self):
        fig, ax = self._axis(365 * 10)
        try:
            ticks = ax.xaxis.get_major_locator()()
            assert len(ticks) <= 12, len(ticks)
            assert len(ticks) >= 3, len(ticks)
        finally:
            plt.close(fig)


class TestLoadGraphConfigRefusals:
    """Every documented `Raises` branch of the config loader was unexercised, so a
    malformed override could have started passing silently and framed the graphs
    against columns nobody declared."""

    @pytest.mark.parametrize(
        "override",
        ["temperatura", "temperatura=", "temperature=AirT2"],
    )
    def test_a_malformed_or_unknown_override_is_refused(self, override: str) -> None:
        with pytest.raises(ValueError, match="--col"):
            load_graph_config(None, [override])

    def test_a_direction_pair_that_is_not_a_pair_is_refused(self, tmp_path: Path) -> None:
        config = tmp_path / "graphs.yaml"
        config.write_text("direction_components: [u]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="\\[u, v\\] pair"):
            load_graph_config(config, [])

    def test_a_valid_override_reaches_the_returned_chain(self) -> None:
        columns, _balance, _direction = load_graph_config(None, ["temperatura=AirT2_C_Avg"])

        assert columns["temperatura"] == ("AirT2_C_Avg",)


class TestTheEmptyWindowExit:
    """The sibling producer's own test cites this exit as precedent, and it was
    itself untested: a cron chain must be able to tell an empty window from a
    fresh publication, and neither leaves a PNG behind."""

    @staticmethod
    def _header_only(tmp_path: Path) -> Path:
        path = tmp_path / "hourly.csv"
        path.write_text("timestamp,AirT1_C_Avg\n", encoding="utf-8")
        return path

    @pytest.mark.parametrize(("strict", "expected"), [(False, 0), (True, 1)])
    def test_an_empty_window_exits_by_the_strict_flag(
        self, tmp_path: Path, strict: bool, expected: int
    ) -> None:
        out = tmp_path / "out"

        result = runner.invoke(
            app,
            [
                "site",
                "-i",
                str(self._header_only(tmp_path)),
                "-o",
                str(out),
                *(["--strict"] if strict else []),
            ],
        )

        assert result.exit_code == expected, result.output
        assert "nothing to plot" in result.output
        assert not list(out.glob("*.png")) if out.exists() else True
