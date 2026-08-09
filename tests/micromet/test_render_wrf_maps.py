"""Characterization + CLI contracts for the WRF figure command.

``_build_tasks_for_domain`` decides the filename, title, colour scale and
overlay of every published PNG/WebM frame, so the snapshot below pins that
whole surface. Nothing else in the suite covers it.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
from typer.testing import CliRunner

from micrometeorology.cli import render_wrf_maps
from micrometeorology.wrf import jobs, reader, value_source
from micrometeorology.wrf.batch import FigureTask
from micrometeorology.wrf.value_source import ValueFrameSource
from tests.micromet.test_wrf_jobs import NT, _write_full_wrf_file


def _build_tasks(wrf_path: Path, var_list: list[str], output_dir: Path) -> list[FigureTask]:
    with reader.WRFDataset(wrf_path) as ds:
        return render_wrf_maps._build_tasks_for_domain(
            ds,
            var_list,
            output_dir,
            None,
            skip_first=0,
            dpi=100,
        )


class _Snapshot(NamedTuple):
    """Everything about a ``FigureTask`` that reaches the published PNG."""

    title: str
    vmin: float
    vmax: float
    cmap_name: str
    overlay_levels: list[float] | None
    has_overlay: bool
    has_wind_vectors: bool
    dpi: int
    saturation: float
    data_shape: tuple[int, ...]


def _snapshot(task: FigureTask) -> _Snapshot:
    assert (task.u is None) == (task.v is None)
    return _Snapshot(
        title=task.title,
        vmin=task.vmin,
        vmax=task.vmax,
        cmap_name=task.cmap_name,
        overlay_levels=task.overlay_levels,
        has_overlay=task.overlay_data is not None,
        has_wind_vectors=task.u is not None,
        dpi=task.dpi,
        saturation=task.saturation,
        data_shape=task.data.shape,
    )


def _expected(title: str, vmin: float, vmax: float, cmap_name: str) -> _Snapshot:
    return _Snapshot(
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        overlay_levels=None,
        has_overlay=False,
        has_wind_vectors=False,
        dpi=100,
        saturation=2.0,
        data_shape=(4, 5),
    )


def test_default_variable_set_builds_the_pinned_figure_task_surface(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_figures.nc"
    _write_full_wrf_file(wrf, seed=5)

    tasks = _build_tasks(wrf, list(render_wrf_maps.DEFAULT_VARS), tmp_path / "figs")

    # LH and GLW are absent from the synthetic file and are skipped with a
    # warning; the other ten variables each emit one task per time step.
    by_variable: dict[str, list[str]] = {}
    for task in tasks:
        by_variable.setdefault(Path(task.output_path).name.split("_D02_")[0], []).append(
            Path(task.output_path).name
        )
    assert sorted(by_variable) == [
        "HFX",
        "PRES",
        "RAIN",
        "RH2",
        "SWDOWN",
        "TEMP",
        "TSK",
        "VAPOR",
        "WIND",
        "WIND_POWER_DENSITY_10M",
    ]
    for names in by_variable.values():
        assert len(names) == NT

    label = "\nInício Análise: 03/05/2026 09 (UTC)\nPrevisão: 03/05/2026 06HL (Domingo)"
    first_frames = {
        Path(task.output_path).name.split("_D02_")[0]: _snapshot(task)
        for task in tasks
        if task.output_path.endswith("_000.png")
    }
    # Titles, colormaps and scale bounds are baked into the published PNGs, so
    # every one of them is pinned. The generic-scalar arm (HFX/PRES/VAPOR)
    # titles from the NetCDF suffix rather than a per-variable phrase, and only
    # temperature carries the pressure-contour overlay.
    assert first_frames == {
        "TEMP": _expected(
            f"Temperature (°C){label}", 16.863793945312523, 31.418969726562523, "hot_r"
        )._replace(overlay_levels=[880, 900, 950, 1000, 1013], has_overlay=True),
        "TSK": _expected(
            f"Skin Temperature (°C){label}", 15.146752929687523, 35.90810546875002, "hot_r"
        ),
        "RH2": _expected(f"Relative Humidity 2m (%){label}", 42.125858306884766, 100.0, "YlGnBu"),
        "WIND": _expected(
            f"Wind 10m (m/s){label}", 0.36391976475715637, 10.99991512298584, "PuBu"
        )._replace(has_wind_vectors=True),
        "RAIN": _expected(f"Rain (mm){label}", 0.31824588775634766, 4.724166393280029, "afmhot_r"),
        "SWDOWN": _expected(f"SWDOWN (W/m²){label}", 7.392277240753174, 878.29052734375, "hot_r"),
        "WIND_POWER_DENSITY_10M": _expected(
            f"Wind Power Density 10m (W/m²){label}",
            0.028288070112466812,
            572.7734985351562,
            "YlOrRd",
        ),
        "HFX": _expected(f"HFX{label}", -14.924610137939453, 380.68243408203125, "jet"),
        "PRES": _expected(f"PRES{label}", 990.17140625, 1018.40984375, "Blues"),
        "VAPOR": _expected(f"VAPOR{label}", 10.211358778178692, 19.896673038601875, "YlGnBu"),
    }


def test_swdown_drops_night_steps_but_other_variables_keep_them(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_night.nc"
    # 19..23 UTC = 16..20 local: the 6-18h daylight gate keeps three steps.
    _write_full_wrf_file(wrf, seed=31, start_hour_utc=19)

    tasks = _build_tasks(wrf, ["temperature", "SWDOWN"], tmp_path / "figs")

    names = [Path(task.output_path).name for task in tasks]
    assert [n for n in names if n.startswith("SWDOWN")] == [
        f"SWDOWN_D02_{i:03d}.png" for i in range(3)
    ]
    assert [n for n in names if n.startswith("TEMP")] == [
        f"TEMP_D02_{i:03d}.png" for i in range(NT)
    ]


def test_skip_first_drops_leading_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_skip.nc"
    _write_full_wrf_file(wrf, seed=5)

    with reader.WRFDataset(wrf) as ds:
        tasks = render_wrf_maps._build_tasks_for_domain(
            ds, ["temperature"], tmp_path / "figs", None, skip_first=2, dpi=100
        )

    assert [Path(t.output_path).name for t in tasks] == [
        f"TEMP_D02_{i:03d}.png" for i in range(2, NT)
    ]


def test_missing_swdown_warns_and_skips_instead_of_aborting(tmp_path, monkeypatch):
    """A truncated wrfout must not kill the whole cron run."""
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_no_swdown.nc"
    _write_full_wrf_file(wrf, seed=5)
    _drop_variable(wrf, "SWDOWN")

    tasks = _build_tasks(wrf, ["SWDOWN", "temperature"], tmp_path / "figs")

    assert [Path(t.output_path).name for t in tasks] == [f"TEMP_D02_{i:03d}.png" for i in range(NT)]


# ---------------------------------------------------------------------------
# One dispatcher, two products
# ---------------------------------------------------------------------------


def test_figure_values_come_from_the_shared_value_frame_source(tmp_path, monkeypatch):
    """The figure path must not carry a private copy of the extractor dispatch.

    Values and scale bounds are read from the same dispatcher the values-JSON
    work units use, so replacing that dispatcher changes what every PNG is
    drawn from. A second copy in this module would ignore it and keep
    publishing its own numbers.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_shared.nc"
    _write_full_wrf_file(wrf, seed=5)

    sentinel_frame = np.arange(20, dtype=np.float64).reshape(4, 5)
    monkeypatch.setattr(
        render_wrf_maps,
        "build_value_frame_source",
        lambda _dataset, _variable_name: ValueFrameSource(
            frame_for_step=lambda _index: sentinel_frame,
            scale_min=-1.5,
            scale_max=2.5,
        ),
    )

    tasks = _build_tasks(wrf, ["temperature", "HFX"], tmp_path / "figs")

    assert len(tasks) == 2 * NT
    for task in tasks:
        np.testing.assert_array_equal(task.data, sentinel_frame)
        assert (task.vmin, task.vmax) == (-1.5, 2.5)


def _written_step_names(result: jobs.UnitResult) -> list[str]:
    return sorted(
        Path(f).name for f in result.files if not f.endswith((".series.bin", ".summary.json"))
    )


def test_the_swdown_daylight_window_has_one_owner_for_figures_and_json(tmp_path, monkeypatch):
    """Widening or narrowing the daylight window must move both products.

    The rule used to be spelled out twice — once per product — so a change to
    one silently left the other publishing the old window.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    monkeypatch.setattr(value_source, "FIRST_DAYLIGHT_HOUR", 9)
    wrf = tmp_path / "wrfout_d02_window.nc"
    # 09..13 UTC = 06..10 local; only the last two steps survive a 09h floor.
    _write_full_wrf_file(wrf, seed=5, start_hour_utc=9)

    tasks = _build_tasks(wrf, ["SWDOWN"], tmp_path / "figs")
    assert [Path(task.output_path).name for task in tasks] == [
        "SWDOWN_D02_003.png",
        "SWDOWN_D02_004.png",
    ]

    json_dir = tmp_path / "json"
    json_dir.mkdir()
    result = jobs.process_unit(
        jobs.WorkUnit(
            kind="values_json",
            wrf_path=str(wrf),
            variable="SWDOWN",
            json_dir=str(json_dir),
            geojson_dir=str(tmp_path / "geo"),
        )
    )
    assert result.error is None
    assert _written_step_names(result) == ["D02_SWDOWN_003.json", "D02_SWDOWN_004.json"]


def _drop_variable(wrf_path: Path, name: str) -> None:
    """Rewrite *wrf_path* without variable *name*."""
    import netCDF4

    trimmed = wrf_path.with_suffix(".trimmed.nc")
    with netCDF4.Dataset(wrf_path) as src, netCDF4.Dataset(trimmed, "w") as dst:
        dst.setncatts({key: src.getncattr(key) for key in src.ncattrs()})
        for dim_name, dim in src.dimensions.items():
            dst.createDimension(dim_name, len(dim))
        for var_name, var in src.variables.items():
            if var_name == name:
                continue
            clone = dst.createVariable(var_name, var.datatype, var.dimensions)
            clone[:] = var[:]
    trimmed.replace(wrf_path)


# ---------------------------------------------------------------------------
# Duplicate suppression and CLI exit status
# ---------------------------------------------------------------------------


def test_duplicate_variable_requests_do_not_render_the_same_png_twice(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_dup.nc"
    _write_full_wrf_file(wrf, seed=5)

    tasks = _build_tasks(wrf, ["temperature", "temperature"], tmp_path / "figs")

    assert len({task.output_path for task in tasks}) == len(tasks) == NT


def test_cli_exits_non_zero_when_every_figure_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_cli.nc"
    _write_full_wrf_file(wrf, seed=5)

    monkeypatch.setattr(
        render_wrf_maps,
        "run_figure_tasks",
        lambda *_args, **_kwargs: [],
    )

    result = CliRunner().invoke(
        render_wrf_maps.app,
        ["-d", str(wrf), "-o", str(tmp_path / "figs"), "-v", "temperature", "-w", "2"],
    )

    assert result.exit_code == 1
    assert f"✗ {NT} figures" in result.output


def test_cli_exits_zero_when_every_figure_renders(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_cli_ok.nc"
    _write_full_wrf_file(wrf, seed=5)

    monkeypatch.setattr(
        render_wrf_maps,
        "run_figure_tasks",
        lambda tasks, *_args, **_kwargs: [task.output_path for task in tasks],
    )

    result = CliRunner().invoke(
        render_wrf_maps.app,
        ["-d", str(wrf), "-o", str(tmp_path / "figs"), "-v", "temperature", "-w", "2"],
    )

    assert result.exit_code == 0, result.output
    assert f"✓ Generated {NT} figures" in result.output


def test_an_output_file_id_is_refused_before_any_frame_is_rendered(tmp_path, monkeypatch):
    """``-v TSK`` rendered raw KELVIN into the PNGs skin_temperature owns.

    ``TSK`` is not an input variable, it is the output file id of
    ``skin_temperature``. Reaching the raw-NetCDF passthrough with it produced
    ``TSK_D0X_nnn.png`` titled "TSK" on a Kelvin colour bar, under the exact
    names the °C product publishes; and with both ids listed the output-path
    dedup kept whichever came first, so flag ORDER decided which frames shipped.
    The other two WRF CLIs already refuse it — this one did not.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_figures_tsk.nc"
    _write_full_wrf_file(wrf, seed=5)
    figures = tmp_path / "figs"

    result = CliRunner().invoke(
        render_wrf_maps.app,
        ["-d", str(wrf), "-o", str(figures), "-v", "TSK", "-w", "1"],
    )

    assert result.exit_code == 2, result.output
    assert "TSK is the output file id of skin_temperature" in result.output
    assert not list(figures.glob("*.png")) if figures.exists() else True


def test_a_wrfout_missing_a_bespoke_input_is_skipped_not_a_traceback(tmp_path):
    """``build_value_frame_source`` documents ``None`` for an absent variable.

    Eight of its ten branches reached ``get_variable``, a bare dict lookup that
    raises ``KeyError``, so a wrfout without ``Q2`` killed the whole run — every
    later variable, every later domain and the video phase — while a missing raw
    field of the same kind was skipped with a warning.
    """
    wrf = tmp_path / "wrfout_d02_partial.nc"
    _write_partial_wrf_file(wrf)

    with reader.WRFDataset(str(wrf)) as dataset:
        for variable in ("relative_humidity", "vapor", "rain", "wind", "skin_temperature"):
            assert value_source.build_value_frame_source(dataset, variable) is None, variable
        # The fields the file does carry still build normally.
        assert value_source.build_value_frame_source(dataset, "temperature") is not None
        assert value_source.build_value_frame_source(dataset, "pressure") is not None


def _write_partial_wrf_file(path: Path) -> None:
    """A wrfout carrying only Times/XLAT/XLONG/T2/PSFC."""
    import netCDF4

    with netCDF4.Dataset(str(path), "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        ds.DX, ds.DY = 3000.0, 3000.0
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list("2026-05-03_00:00:00")], dtype="S1")
        for name in ("XLAT", "XLONG"):
            variable = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            variable[:] = np.zeros((1, 3, 4), dtype="f4")
        ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))[:] = np.full(
            (1, 3, 4), 300.0, dtype="f4"
        )
        ds.createVariable("PSFC", "f4", ("Time", "south_north", "west_east"))[:] = np.full(
            (1, 3, 4), 101325.0, dtype="f4"
        )
