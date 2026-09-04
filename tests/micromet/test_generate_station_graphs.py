"""Tests for the labmim-station-graphs datalogger graph producer.

The datalogger header changes whenever a sensor is unplugged, so the command
must report exactly the PNGs it actually wrote -- an unattended cron job has
nothing else to go on.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from typer.testing import CliRunner

from micrometeorology.cli.generate_station_graphs import app

runner = CliRunner()

# Every source column the ten graphs know how to draw.
FULL_LENTA_COLUMNS = (
    "CM3Up_Wm2_Avg",
    "CM3Dn_Wm2_Avg",
    "CG3Up_Wm2Cr_Avg",
    "CG3Dn_Wm2Cr_Avg",
    "Net_Wm2_Avg",
    "PAR_Wm2_Avg",
    "Temp1_Avg",
    "RH1_Avg",
    "Pmb_WXT",
    "WS_WXT_Avg",
    "WD_WXT_Avg",
)

# What survives when the radiation, pressure, direction and rain sensors are
# unplugged: only temperatura, umidade and velocidade can still be drawn.
SPARSE_LENTA_COLUMNS = ("Temp1_Avg", "RH1_Avg", "WS_WXT_Avg")

START_DATE = "2026-07-21"
SAMPLES = 576  # two days of 5-minute records


def _write_toa5(path: Path, columns: tuple[str, ...]) -> Path:
    """Write a synthetic TOA5 file with the real 4-line header structure."""
    index = np.arange(SAMPLES)
    names = ",".join(f'"{c}"' for c in ("TIMESTAMP", "RECORD", *columns))
    units = ",".join(f'"{u}"' for u in ("TS", "RN", *["W/meter^2"] * len(columns)))
    aggs = ",".join(f'"{a}"' for a in ("", "", *["Avg"] * len(columns)))
    lines = [
        '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","CPU:PRG_LABMIM.CR5","49836","LBM"',
        names,
        units,
        aggs,
    ]
    stamps = np.datetime64(f"{START_DATE}T00:00:00") + index * np.timedelta64(5, "m")
    for row, stamp in enumerate(stamps):
        cells = ",".join(f"{20.0 + (row % 7) + i:.3f}" for i in range(len(columns)))
        lines.append(f'"{stamp}",{row},{cells}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _invoke(lenta: Path, rain: Path, out: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "-l",
            str(lenta),
            "-r",
            str(rain),
            "-o",
            str(out),
            "--start-date",
            START_DATE,
            "--days",
            "2",
            "--log-level",
            "WARNING",
            *extra,
        ],
    )


def _claimed_ok(output: str) -> set[str]:
    return set(re.findall(r"^  \[ok\] (\S+\.png)$", output, flags=re.MULTILINE))


@pytest.fixture
def full_station(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_toa5(tmp_path / "lenta.dat", FULL_LENTA_COLUMNS),
        _write_toa5(tmp_path / "rain.dat", ("PL01_mm_Tot",)),
    )


@pytest.fixture
def sparse_station(tmp_path: Path) -> tuple[Path, Path]:
    """A logger whose radiation, pressure, direction and rain channels are gone."""
    return (
        _write_toa5(tmp_path / "lenta.dat", SPARSE_LENTA_COLUMNS),
        _write_toa5(tmp_path / "rain.dat", ("PL02_mm_Tot",)),
    )


def test_every_ok_claim_matches_a_file_on_disk(sparse_station, tmp_path):
    lenta, rain = sparse_station
    out = tmp_path / "graphs"

    result = _invoke(lenta, rain, out)

    assert result.exit_code == 0, result.output
    assert _claimed_ok(result.output) == {p.name for p in out.glob("*.png")}
    assert _claimed_ok(result.output) == {"temperatura.png", "umidade.png", "velocidade.png"}
    assert "[skip] radiacao_liq.png" in result.output
    assert "3/10 graphs written" in result.output


def test_absent_sources_leave_no_blank_png_behind(sparse_station, tmp_path):
    """A blank overwrite is worse than a skip: the page would show an empty graph."""
    lenta, rain = sparse_station
    out = tmp_path / "graphs"
    out.mkdir()
    stale = out / "balanco.png"
    stale.write_bytes(b"previous good image")

    result = _invoke(lenta, rain, out)

    assert result.exit_code == 0, result.output
    assert stale.read_bytes() == b"previous good image"
    assert plt.get_fignums() == []


def test_strict_turns_a_skipped_graph_into_a_non_zero_exit(sparse_station, tmp_path):
    lenta, rain = sparse_station
    out = tmp_path / "graphs"

    result = _invoke(lenta, rain, out, "--strict")

    assert result.exit_code == 1


def test_a_complete_logger_still_writes_all_ten_graphs(full_station, tmp_path):
    lenta, rain = full_station
    out = tmp_path / "graphs"

    result = _invoke(lenta, rain, out, "--strict")

    assert result.exit_code == 0, result.output
    assert len(_claimed_ok(result.output)) == 10
    assert _claimed_ok(result.output) == {p.name for p in out.glob("*.png")}
    assert "[ok] All graphs saved" in result.output
    assert plt.get_fignums() == []


@pytest.mark.parametrize("bad", ["", "nan", "NaT", "today", "now", "not-a-date"])
def test_an_unparseable_start_date_fails_instead_of_writing_nothing(full_station, tmp_path, bad):
    """A bad ``--start-date`` must exit non-zero, not filter every row away.

    ``pd.to_datetime`` resolves ``""``/``nan``/``NaT`` to ``NaT`` and ``today``/
    ``now`` to the current clock, both of which sail past an explicit ``format=``.
    Left unguarded, a cron wrapper passing an unset ``--start-date "$VAR"`` gets a
    successful-looking run that silently wrote no graphs and left the monitoring
    page stale.
    """
    lenta, rain = full_station
    out = tmp_path / "figs"
    result = runner.invoke(
        app,
        [
            "-l",
            str(lenta),
            "-r",
            str(rain),
            "-o",
            str(out),
            "--start-date",
            bad,
            "--days",
            "2",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code != 0, f"--start-date {bad!r} exited 0: {result.output}"
    assert not out.exists() or not list(out.glob("*.png"))


def test_an_empty_window_honours_strict_instead_of_reporting_success(full_station, tmp_path):
    """A stalled logger is the case where EVERY graph is skipped.

    That path bailed with an unconditional ``sys.exit(0)`` which never consulted
    ``--strict``, so the run that rewrote none of the ten PNGs was the one run
    that could not fail — a cron chain saw green over week-old images. The
    sibling producer of the same nine site images already exits 1 here.
    """
    lenta, rain = full_station
    out = tmp_path / "graphs"

    # A window that predates the file: the date filter selects zero rows.
    strict = _invoke(lenta, rain, out, "--strict", "--start-date", "2001-01-01")
    assert strict.exit_code == 1, strict.output
    assert "nothing to plot" in strict.output
    assert not list(out.glob("*.png")) if out.exists() else True

    # Without --strict the historical behaviour is unchanged.
    lenient = _invoke(lenta, rain, out, "--start-date", "2001-01-01")
    assert lenient.exit_code == 0, lenient.output


def test_the_diffuse_graph_draws_the_models_diffuse_and_not_only_its_global():
    """The figure named radiacao_difusa carried one model curve, the GLOBAL flux.

    Its legend said ``SW_dw-wrf``, but it was the only model line on an axes
    whose other curves are the measured diffuse, so the site read the model's
    global irradiance as its diffuse. Read off the axes the producer actually
    draws, not off the constants it is configured with: the strokes the two
    curves get are decided at the call site, and asserting on literals the test
    itself passed to ``_plot_wrf_overlay`` never reached that decision.
    """
    from micrometeorology.cli.generate_station_graphs import _draw_radiacao_difusa

    index = pd.to_datetime(["2026-01-01 12:00", "2026-01-01 13:00"])
    measured = pd.DataFrame(
        {"CM3Up_Wm2_Avg": [900.0, 950.0], "PSP_Wm2_Avg": [130.0, 150.0]}, index=index
    )
    model = pd.DataFrame(
        {"swdown_w_m2": [800.0, 900.0], "swddif_w_m2": [120.0, 140.0]}, index=index
    )

    figure, axes = plt.subplots()
    try:
        _draw_radiacao_difusa(axes, measured, measured, model)
        drawn = {line.get_label(): line for line in axes.get_lines()}

        assert {"SW_dw-wrf 1h", "SW_df-wrf 1h"} <= set(drawn)
        np.testing.assert_allclose(
            np.asarray(drawn["SW_dw-wrf 1h"].get_ydata(), dtype=float), [800.0, 900.0]
        )
        np.testing.assert_allclose(
            np.asarray(drawn["SW_df-wrf 1h"].get_ydata(), dtype=float), [120.0, 140.0]
        )
        # Two dashed black lines are how the global came to be read as the diffuse.
        assert (drawn["SW_dw-wrf 1h"].get_linestyle(), drawn["SW_dw-wrf 1h"].get_color()) != (
            drawn["SW_df-wrf 1h"].get_linestyle(),
            drawn["SW_df-wrf 1h"].get_color(),
        )
    finally:
        plt.close(figure)


def test_the_two_shipped_producers_agree_on_which_column_the_diffuse_is():
    """One page, two producers: a rename in only one of them republishes the
    static figure and the interactive one from different model channels.
    """
    from micrometeorology.cli.generate_station_graphs import WRF_COLUMNS
    from micrometeorology.cli.plot_station_graphs import DEFAULT_WRF_COLUMNS
    from micrometeorology.wrf.columns import SWDDIF_W_M2, SWDOWN_W_M2

    assert WRF_COLUMNS["radiacao_difusa"] == SWDDIF_W_M2
    assert WRF_COLUMNS["radiacao_global"] == SWDOWN_W_M2
    assert DEFAULT_WRF_COLUMNS["radiacao_difusa"] == (WRF_COLUMNS["radiacao_difusa"],)


def test_the_model_overlay_never_backtracks_in_time():
    """``series_operacional.dat`` is an append-only log of successive runs.

    Read without sorting it plots a line that runs backwards every time a new
    run is appended, with the initialisation hour's identically-zero fluxes
    still on it.
    """
    from micrometeorology.cli.generate_station_graphs import read_wrf_series

    source = Path("data/series_operacional.dat")
    if not source.exists():
        pytest.skip("the operational record is not on this machine")
    index = pd.DatetimeIndex(read_wrf_series(source).index)
    assert index.is_monotonic_increasing
    assert not index.duplicated().any()
    assert not (index.hour == 21).any()


def test_a_diffuse_channel_that_is_a_hard_zero_is_not_drawn_as_the_diffuse():
    """``CMP21_Wm2_Avg`` is a hard 0.0 in every v22 row (calibrations.yaml), yet
    the graph drew it as ``SW_df 1h`` beside the PSP that measures the diffuse."""
    from micrometeorology.cli.generate_station_graphs import _draw_radiacao_difusa

    index = pd.date_range("2026-01-01 10:00", periods=3, freq="h")
    hourly = pd.DataFrame(
        {
            "CM3Up_Wm2_Avg": [500.0, 600.0, 550.0],
            "CMP21_Wm2_Avg": [0.0, 0.0, 0.0],
            "PSP_Wm2_Avg": [120.0, 130.0, 125.0],
        },
        index=index,
    )
    figure, axes = plt.subplots()
    try:
        _draw_radiacao_difusa(axes, hourly, hourly, None)
        labels = axes.get_legend_handles_labels()[1]
    finally:
        plt.close(figure)

    assert "SW_df 1h (PSP)" in labels
    assert "5 min (PSP)" in labels
    assert not any("CMP21" in label for label in labels)
    assert "SW_df 1h" not in labels


def _write_rows(path: Path, column: str, rows: list[tuple[str, float]]) -> Path:
    """A TOA5 table with explicit stamps and values, for the join tests below."""
    lines = [
        '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","CPU:PRG_LABMIM.CR5","49836","LBM"',
        f'"TIMESTAMP","RECORD","{column}"',
        '"TS","RN","mm"',
        '"","","Tot"',
    ]
    lines += [f'"{stamp}",{row},{value:.3f}' for row, (stamp, value) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_rain_at_a_stamp_the_lenta_table_lacks_still_enters_the_hourly_sum(tmp_path, monkeypatch):
    """The rain table rode along in the lenta frame, reindexed onto lenta's own
    stamps: every rain sample at a stamp lenta lacks vanished from the hourly
    bars while the 5-minute trace, drawn from the rain table itself, still showed
    it. Measured on the archive: 2022 lost 601.2 of 874.5 mm."""
    from micrometeorology.cli import generate_station_graphs as module

    stamps = [f"2026-07-21 00:{minute:02d}:00" for minute in range(0, 60, 5)]
    lenta = _write_rows(tmp_path / "lenta.dat", "Temp1_Avg", [(s, 25.0) for s in stamps[:6]])
    rain = _write_rows(tmp_path / "rain.dat", "PL01_mm_Tot", [(s, 1.0) for s in stamps])

    seen: dict[str, pd.DataFrame] = {}
    original = module._plot_precipitacao

    def capture(raw_rain, hourly, out, graph_dt, **kwargs):
        seen["hourly"] = hourly
        return original(raw_rain, hourly, out, graph_dt, **kwargs)

    monkeypatch.setattr(module, "_plot_precipitacao", capture)

    result = _invoke(lenta, rain, tmp_path / "out")

    assert result.exit_code == 0, result.output
    assert seen["hourly"]["PL01_mm_Tot"].sum() == pytest.approx(12.0)


def test_the_wrf_overlay_reaches_the_graphs_end_to_end(full_station, tmp_path):
    """`--wrf` is the whole model half of this command and no test drove it: the
    overlay could have stopped reaching the figures with every assertion green."""
    lenta, rain = full_station
    index = pd.date_range(START_DATE, periods=48, freq="h")
    wrf_dat = tmp_path / "series_operacional.dat"
    pd.DataFrame(
        {
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
            "t2_c": np.linspace(20.0, 30.0, len(index)),
            "swdown_w_m2": np.linspace(0.0, 900.0, len(index)),
            "swddif_w_m2": np.linspace(0.0, 300.0, len(index)),
        }
    ).to_csv(wrf_dat, index=False)

    result = _invoke(lenta, rain, tmp_path / "out", "-w", str(wrf_dat))

    assert result.exit_code == 0, result.output
    assert "temperatura.png" in _claimed_ok(result.output)


def test_the_stamp_is_the_newest_sample_drawn_not_the_wall_clock():
    """Without `--start-date`, `date_end` is today, so a record that stopped
    months ago would be published under today's date. The rule was stated in a
    comment and pinned by nothing."""
    from micrometeorology.cli.generate_station_graphs import newest_plotted_stamp

    lenta = pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2026-07-22 23:55"]))
    rain = pd.DataFrame({"y": [1.0]}, index=pd.DatetimeIndex(["2026-07-22 12:00"]))

    assert newest_plotted_stamp(lenta, rain) == pd.Timestamp("2026-07-22 23:55").to_pydatetime()


def test_the_stamp_reads_the_rain_table_when_it_runs_later():
    from micrometeorology.cli.generate_station_graphs import newest_plotted_stamp

    lenta = pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2026-07-22 12:00"]))
    rain = pd.DataFrame({"y": [1.0]}, index=pd.DatetimeIndex(["2026-07-22 23:55"]))

    assert newest_plotted_stamp(lenta, rain) == pd.Timestamp("2026-07-22 23:55").to_pydatetime()
