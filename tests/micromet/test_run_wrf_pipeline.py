"""Phase interaction contracts for the local three-phase WRF pipeline."""

from typer.testing import CliRunner

from micrometeorology.cli import run_wrf_pipeline
from micrometeorology.wrf import batch
from tests.micromet.test_wrf_jobs import NT, _write_full_wrf_file


def test_height_specific_poteolico_is_collapsed_for_figures_only(tmp_path, monkeypatch):
    """The figure phase has no poteolico renderer; the JSON phase is height-specific."""
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_pipeline_pot.nc"
    _write_full_wrf_file(wrf, seed=5)

    monkeypatch.setattr(
        batch,
        "run_figure_tasks",
        lambda tasks, *_args, **_kwargs: [task.output_path for task in tasks],
    )

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        [
            "-d",
            str(wrf),
            "-o",
            str(tmp_path / "out"),
            "-v",
            "poteolico50,poteolico100",
            "-w",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "⚠ Skipping poteolico (no figure renderer)" in result.output
    assert "POTEOLICO50" not in result.output
    # The JSON phase keeps both requested heights and neither of the other two.
    written = sorted(p.name for p in (tmp_path / "out" / "JSON").glob("*.json"))
    assert written == sorted(
        [f"D02_POT_EOLICO_{h}M_{i:03d}.json" for h in (50, 100) for i in range(NT)]
        + [f"D02_POT_EOLICO_{h}M.summary.json" for h in (50, 100)]
        + ["manifest.json"]
    )


def test_failed_figures_still_produce_json_but_exit_non_zero(tmp_path, monkeypatch):
    """Phase 2 owns the front-end byte contract: a broken PNG must not skip it."""
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_pipeline_fail.nc"
    _write_full_wrf_file(wrf, seed=5)

    monkeypatch.setattr(batch, "run_figure_tasks", lambda *_args, **_kwargs: [])

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        ["-d", str(wrf), "-o", str(tmp_path / "out"), "-v", "temperature", "-w", "1"],
    )

    assert result.exit_code == 1
    assert f"✗ {NT} figures" in result.output
    assert sorted(p.name for p in (tmp_path / "out" / "JSON").glob("*.json")) == sorted(
        [f"D02_TEMP_{i:03d}.json" for i in range(NT)] + ["D02_TEMP.summary.json", "manifest.json"]
    )


def test_an_output_file_id_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """``-v TSK`` would publish raw Kelvin into the files skin_temperature owns.

    The reject list lives with the export CLI, and this pipeline must apply all
    of it: a token refused by ``labmim-wrf-geojson`` is refused here too.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_pipeline_tsk.nc"
    _write_full_wrf_file(wrf, seed=5)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        ["-d", str(wrf), "-o", str(out), "-v", "TSK", "-w", "1"],
    )

    assert result.exit_code == 2, result.output
    assert "TSK is the output file id of skin_temperature" in result.output
    assert not out.exists()


def test_two_files_of_one_domain_are_a_usage_error_not_a_traceback(tmp_path, monkeypatch):
    """The same-domain guard lives in ``jobs.build_units`` and raises ValueError."""
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    for suffix in ("00", "06"):
        _write_full_wrf_file(wrf_dir / f"wrfout_d02_2026-05-03_{suffix}:00:00", seed=5)

    monkeypatch.setattr(
        batch,
        "run_figure_tasks",
        lambda tasks, *_args, **_kwargs: [task.output_path for task in tasks],
    )

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        [
            "--wrf-dir",
            str(wrf_dir),
            "--date",
            "20260503",
            "-o",
            str(tmp_path / "out"),
            "-v",
            "temperature",
            "--no-figures",
            "-w",
            "1",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "A run publishes one set of files per domain" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_a_malformed_date_is_refused_by_the_pipeline(tmp_path):
    """Every wrfout-resolving CLI shares one ``--date`` validator."""
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        ["--wrf-dir", str(wrf_dir), "--date", "may3", "-o", str(tmp_path / "out")],
    )

    assert result.exit_code == 2, result.output
    assert "--date must be YYYYMMDD" in result.output


def test_successful_figure_run_exits_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_pipeline_ok.nc"
    _write_full_wrf_file(wrf, seed=5)

    monkeypatch.setattr(
        batch,
        "run_figure_tasks",
        lambda tasks, *_args, **_kwargs: [task.output_path for task in tasks],
    )

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        [
            "-d",
            str(wrf),
            "-o",
            str(tmp_path / "out"),
            "-v",
            "temperature",
            "--no-geojson",
            "-w",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"✓ {NT} figures generated" in result.output


def test_also_video_with_no_figures_is_a_usage_error_not_a_silent_success(tmp_path, monkeypatch):
    """Phase 3 encodes the PNGs Phase 1 renders, so the pair is unsatisfiable.

    ``--no-figures`` empties ``png_paths``, which makes the video gate
    ``also_video and png_paths`` unconditionally false. Without this guard the
    run exits 0 having produced none of the videos it was asked for.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_novideo.nc"
    _write_full_wrf_file(wrf, seed=5)
    out = tmp_path / "out"

    result = CliRunner().invoke(
        run_wrf_pipeline.app,
        [
            "-d",
            str(wrf),
            "-o",
            str(out),
            "-v",
            "temperature",
            "-w",
            "1",
            "--no-figures",
            "--also-video",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "--also-video" in result.output
    assert not (out / "videos").exists()
