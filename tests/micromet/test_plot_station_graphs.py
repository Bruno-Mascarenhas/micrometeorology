"""Tests for the labmim-site-graphs monitoring-page producer.

Offline and fast: synthetic hourly CSVs via ``tmp_path``, the headless Agg
backend (set on import of the CLI module), and the Typer app driven through
:class:`~typer.testing.CliRunner`.
"""

from __future__ import annotations

from datetime import datetime
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
    DEFAULT_COLUMNS,
    GRAPH_SPECS,
    _plot_balance,
    app,
)
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
        datetime(2026, 6, 1, 1),
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
