"""Tests for the labmim-station-graphs datalogger graph producer.

The datalogger header changes whenever a sensor is unplugged, so the command
must report exactly the PNGs it actually wrote -- an unattended cron job has
nothing else to go on.
"""

import re
from pathlib import Path

import numpy as np
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
