"""Synthetic tests for the WRF figure-rendering worker backend."""

import json
import logging
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from micrometeorology.wrf import batch
from micrometeorology.wrf.batch import (
    FigureMemmapTask,
    FigureTask,
    MapConfig,
    run_figure_tasks,
)


def _fake_render_figure_memmap_payload(task: FigureMemmapTask) -> str:
    """Picklable stand-in for ``_render_figure_memmap`` (runs in pool workers)."""
    out = Path(task.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(np.load(task.data_path, mmap_mode="r"))
    out.write_text(
        json.dumps({"title": task.title, "values": data.tolist()}),
        encoding="utf-8",
    )
    return task.output_path


def _figure_task(output_path: Path, value: float = 1.0) -> FigureTask:
    return FigureTask(
        lon=np.zeros((2, 2), dtype=np.float32),
        lat=np.zeros((2, 2), dtype=np.float32),
        data=np.full((2, 2), value, dtype=np.float32),
        vmin=0.0,
        vmax=1.0,
        cmap_name="viridis",
        overlay_data=None,
        overlay_levels=None,
        u=None,
        v=None,
        title=f"frame {value}",
        output_path=str(output_path),
        map_config=MapConfig("D01", 0.0, 1.0, 0.0, 1.0, 1, 1, False, None),
        dpi=50,
        saturation=1.0,
    )


def test_figure_memmap_backend_with_provided_executor_matches_owned_pool(monkeypatch, tmp_path):
    tmp_dir = tmp_path / "tmp"

    monkeypatch.setattr(
        "micrometeorology.wrf.batch._render_figure_memmap",
        _fake_render_figure_memmap_payload,
    )

    def build_tasks(prefix: str) -> list[FigureTask]:
        return [_figure_task(tmp_path / f"{prefix}_{i}.png", value=float(i)) for i in range(2)]

    run_figure_tasks(build_tasks("owned"), workers=2, backend="memmap", tmp_dir=tmp_dir)
    with ProcessPoolExecutor(max_workers=2) as pool:
        paths = run_figure_tasks(
            build_tasks("provided"),
            workers=2,
            backend="memmap",
            tmp_dir=tmp_dir,
            executor=pool,
        )
        # The provided executor must remain usable (not shut down).
        assert pool.submit(len, [1, 2, 3]).result() == 3

    assert sorted(paths) == [str(tmp_path / "provided_0.png"), str(tmp_path / "provided_1.png")]
    for i in range(2):
        provided = (tmp_path / f"provided_{i}.png").read_bytes()
        owned = (tmp_path / f"owned_{i}.png").read_bytes()
        assert provided == owned


def test_figure_memmap_backend_cleans_payload_directory(monkeypatch, tmp_path):
    tmp_dir = tmp_path / "tmp"
    out_path = tmp_path / "figure.png"

    def fake_render(task):
        Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(task.output_path).write_text("ok", encoding="utf-8")
        return task.output_path

    monkeypatch.setattr("micrometeorology.wrf.batch._render_figure_memmap", fake_render)

    paths = run_figure_tasks([_figure_task(out_path)], workers=1, backend="memmap", tmp_dir=tmp_dir)

    assert paths == [str(out_path)]
    assert out_path.exists()
    assert tmp_dir.exists()
    assert not list(tmp_dir.iterdir())


def test_broken_process_pool_propagates_instead_of_being_swallowed(tmp_path):
    """A broken pool dooms every remaining task: it must raise, not log-and-drop."""

    class _BrokenPoolExecutor(ProcessPoolExecutor):
        """Every submission comes back already broken.

        ``__init__`` deliberately skips the real one: the collector only ever
        calls ``submit`` on the executor it is handed, so no worker process,
        call queue, or pipe needs to exist.
        """

        def __init__(self) -> None:
            """Build no pool at all."""

        def submit[**P, T](
            self, _fn: Callable[P, T], /, *_args: P.args, **_kwargs: P.kwargs
        ) -> Future[T]:
            future: Future[T] = Future()
            future.set_exception(BrokenProcessPool("worker died"))
            return future

    task = _figure_task(tmp_path / "figure.png")

    with pytest.raises(BrokenProcessPool):
        run_figure_tasks(
            [task],
            workers=2,
            backend="memmap",
            tmp_dir=tmp_path / "memmap-tmp",
            executor=_BrokenPoolExecutor(),
        )


def test_failed_render_is_logged_with_the_frame_it_lost(tmp_path, caplog):
    """A batch-local ordinal is unresolvable; the operator needs the PNG name."""

    class _FailingPoolExecutor(ProcessPoolExecutor):
        """Every submission fails inside the worker (see ``_BrokenPoolExecutor``)."""

        def __init__(self) -> None:
            """Build no pool at all."""

        def submit[**P, T](
            self, _fn: Callable[P, T], /, *_args: P.args, **_kwargs: P.kwargs
        ) -> Future[T]:
            future: Future[T] = Future()
            future.set_exception(RuntimeError("cartopy is unhappy"))
            return future

    task = _figure_task(tmp_path / "TEMP_D03_007.png")

    with caplog.at_level(logging.ERROR, logger="micrometeorology.wrf.batch"):
        paths = run_figure_tasks(
            [task],
            workers=2,
            backend="memmap",
            tmp_dir=tmp_path / "memmap-tmp",
            executor=_FailingPoolExecutor(),
        )

    assert paths == []
    assert str(tmp_path / "TEMP_D03_007.png") in caplog.text
    assert "1 of 1 figure tasks failed to render" in caplog.text


# ---------------------------------------------------------------------------
# In-process rendering behaviour
# ---------------------------------------------------------------------------


class _StopBeforeSavefigError(Exception):
    """Raised from a stubbed colormap lookup to end a render early."""


def _stop_before_savefig(*_args, **_kwargs):
    raise _StopBeforeSavefigError


def _inner_domain_task(output_path: Path, shapes_dir: Path | None) -> FigureTask:
    return _figure_task(output_path)._replace(
        map_config=MapConfig(
            "D03", 0.0, 1.0, 0.0, 1.0, 2, 2, True, str(shapes_dir) if shapes_dir else None
        )
    )


def test_municipality_outlines_are_drawn_from_the_shapes_dir(tmp_path, monkeypatch):
    """--shapes-dir was accepted and pickled to every worker but never read."""
    from cartopy.mpl import geoaxes

    from micrometeorology.wrf import plotting

    requested: list[str] = []
    drawn: list[tuple] = []

    def fake_geometries(shp_path: str) -> tuple[object, ...]:
        requested.append(shp_path)
        return ("municipality-mesh",)

    monkeypatch.setattr(batch, "_municipality_geometries", fake_geometries)
    monkeypatch.setattr(
        geoaxes.GeoAxes,
        "add_geometries",
        lambda _self, geoms, _crs, **kwargs: drawn.append((tuple(geoms), kwargs)),
    )
    monkeypatch.setattr(plotting, "saturated_cmap", _stop_before_savefig)

    with pytest.raises(_StopBeforeSavefigError):
        batch._render_figure(_inner_domain_task(tmp_path / "figure.png", tmp_path / "shapes"))

    assert requested == [str(tmp_path / "shapes" / "BRMUE250GC_SIR.shp")]
    assert drawn == [
        (
            ("municipality-mesh",),
            {"facecolor": "none", "edgecolor": "gray", "linewidth": 0.5},
        )
    ]


@pytest.mark.parametrize("shapes_dir", [None, "unset"])
def test_municipality_outlines_are_skipped_without_shapes_or_inner_domain(
    tmp_path, monkeypatch, shapes_dir
):
    from cartopy.mpl import geoaxes

    from micrometeorology.wrf import plotting

    drawn: list[tuple] = []
    monkeypatch.setattr(
        geoaxes.GeoAxes,
        "add_geometries",
        lambda _self, geoms, _crs, **kwargs: drawn.append((tuple(geoms), kwargs)),
    )
    monkeypatch.setattr(plotting, "saturated_cmap", _stop_before_savefig)

    task = (
        _inner_domain_task(tmp_path / "figure.png", None)
        if shapes_dir is None
        else _figure_task(tmp_path / "figure.png")  # D01: no municipality overlay
    )
    with pytest.raises(_StopBeforeSavefigError):
        batch._render_figure(task)

    assert drawn == []


def test_missing_shapefile_warns_once_per_process_and_draws_nothing(tmp_path, caplog):
    batch._municipality_geometries.cache_clear()
    shp_path = str(tmp_path / "absent" / "BRMUE250GC_SIR.shp")

    with caplog.at_level(logging.WARNING, logger="micrometeorology.wrf.batch"):
        assert batch._municipality_geometries(shp_path) == ()
        assert batch._municipality_geometries(shp_path) == ()

    assert caplog.text.count("Municipality shapefile not found") == 1


def test_wind_figure_with_pressure_overlay_renders_end_to_end(tmp_path):
    """Real cartopy + saturated_cmap draw path: catches a bad import prune."""
    import cartopy

    natural_earth = Path(cartopy.config["data_dir"]) / "shapefiles" / "natural_earth"
    required = (
        natural_earth / "physical" / "ne_10m_coastline.shp",
        natural_earth / "cultural" / "ne_10m_admin_1_states_provinces_lines.shp",
    )
    if not all(shapefile.exists() for shapefile in required):
        pytest.skip("Natural Earth 10m data is not pre-downloaded")

    lon, lat = np.meshgrid(
        np.linspace(-38.6, -38.2, 8, dtype=np.float32),
        np.linspace(-13.2, -12.8, 8, dtype=np.float32),
    )
    wind = np.full((8, 8), 3.0, dtype=np.float32)
    out_path = tmp_path / "WIND_D05_000.png"
    task = _figure_task(out_path)._replace(
        lon=lon,
        lat=lat,
        data=wind,
        u=wind,
        v=wind,
        overlay_data=np.full((8, 8), 1000.0, dtype=np.float32),
        overlay_levels=[880, 900, 950, 1000, 1013],
        map_config=MapConfig("D05", -38.6, -38.2, -13.2, -12.8, 3, 2, True, None),
    )

    assert batch._render_figure(task) == str(out_path)
    assert out_path.stat().st_size > 0


def test_render_figure_closes_its_pyplot_figure_when_the_render_fails(tmp_path):
    """pyplot's global registry holds a strong reference to every open figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = _figure_task(tmp_path / "figure.png")._replace(cmap_name="NOT_A_REAL_CMAP")

    open_before = set(plt.get_fignums())
    with pytest.raises(KeyError):
        batch._render_figure(task)

    assert set(plt.get_fignums()) == open_before
