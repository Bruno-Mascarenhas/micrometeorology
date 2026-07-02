"""Synthetic WRF reader tests."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from micrometeorology.wrf.reader import (
    LazyWRFDataset,
    WRFDataset,
    open_wrf_dataset,
    parse_chunks,
    resolve_wrfout_paths,
)
from micrometeorology.wrf.variables import (
    compute_air_density,
    compute_relative_humidity,
    extract_scalar,
    materialize_2d,
)


def _write_tiny_wrf_file(path: Path, n_times: int = 2) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", n_times)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 3)
        ds.createDimension("DateStrLen", 19)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 2000.0)

        lon = ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        lat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))

        lon[:] = np.array(
            [[[-38.0, -37.5, -37.0], [-38.0, -37.5, -37.0]]] * n_times,
            dtype=np.float32,
        )
        lat[:] = np.array(
            [[[-13.0, -13.0, -13.0], [-12.5, -12.5, -12.5]]] * n_times,
            dtype=np.float32,
        )
        t2[:] = np.arange(6 * n_times, dtype=np.float32).reshape(n_times, 2, 3)
        times[:] = np.array(
            [list(f"2024-01-01_{h:02d}:00:00") for h in range(n_times)],
            dtype="S1",
        )


def test_wrf_reader_handles_tiny_synthetic_netcdf():
    path = Path("scratch") / "wrfout_d01_synthetic_reader.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(path)

        with WRFDataset(path) as wrf:
            lon_grid, lat_grid = wrf.read_grid()

            assert lon_grid.shape == (2, 3)
            assert lat_grid.shape == (2, 3)
            assert wrf.grid_bounds() == (-38.0, -37.0, -13.0, -12.5)
            assert [dt.hour for dt in wrf.parse_times()] == [0, 1]
    finally:
        path.unlink(missing_ok=True)


def test_lazy_wrf_reader_defers_variable_loading_until_requested():
    path = Path("scratch") / "wrfout_d01_synthetic_lazy_reader.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(path)

        with LazyWRFDataset(path) as wrf:
            t2 = wrf.get_variable("T2")

            assert wrf.has_variable("T2")
            assert isinstance(t2, xr.DataArray)
            assert t2.shape == (2, 2, 3)
            np.testing.assert_array_equal(t2.isel(Time=0).to_numpy(), np.arange(6).reshape(2, 3))
    finally:
        path.unlink(missing_ok=True)


def test_open_wrf_dataset_lazy_matches_eager_for_synthetic_file():
    path = Path("scratch") / "wrfout_d01_synthetic_lazy_equivalence.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(path)

        with open_wrf_dataset(path, reader="eager") as eager:
            eager_lon, eager_lat = eager.read_grid()
            eager_times = eager.parse_times()
            eager_t2 = eager.get_variable("T2")

        with open_wrf_dataset(path, reader="lazy", chunks=parse_chunks("none")) as lazy:
            lazy_lon, lazy_lat = lazy.read_grid()
            lazy_times = lazy.parse_times()
            lazy_t2 = lazy.get_variable("T2")

        np.testing.assert_array_equal(lazy_lon, eager_lon)
        np.testing.assert_array_equal(lazy_lat, eager_lat)
        assert lazy_times == eager_times
        assert isinstance(lazy_t2, xr.DataArray)
        np.testing.assert_array_equal(lazy_t2.to_numpy(), eager_t2)
    finally:
        path.unlink(missing_ok=True)


def test_parse_chunks_accepts_none_auto_and_explicit_pairs():
    assert parse_chunks(None) is None
    assert parse_chunks("none") is None
    assert parse_chunks("auto") == "auto"
    assert parse_chunks("Time=1,south_north=128") == {"Time": 1, "south_north": 128}


def test_lazy_scalar_extractor_keeps_dataarray_until_slice_materialization():
    path = Path("scratch") / "wrfout_d01_synthetic_lazy_scalar.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(path)

        with open_wrf_dataset(path, reader="lazy", chunks=parse_chunks("none")) as lazy:
            var_data, vmin, vmax = extract_scalar(lazy, "T2")
            first_step = materialize_2d(var_data[0:1, :, :])

        assert isinstance(var_data, xr.DataArray)
        assert vmin == 6.0
        assert vmax == 10.9
        np.testing.assert_array_equal(first_step, np.arange(6).reshape(2, 3))
    finally:
        path.unlink(missing_ok=True)


def test_relative_humidity_uses_q2_t2_psfc_units():
    q2 = np.array([[[0.010]]], dtype=np.float64)
    t2 = np.array([[[293.15]]], dtype=np.float64)
    psfc = np.array([[[101325.0]]], dtype=np.float64)

    rh = compute_relative_humidity(q2, t2, psfc)

    assert rh.shape == q2.shape
    assert np.isclose(float(rh[0, 0, 0]), 68.60, atol=0.05)


def test_air_density_uses_virtual_temperature():
    t2 = np.array([[[300.0]]], dtype=np.float64)
    psfc = np.array([[[100000.0]]], dtype=np.float64)
    q2 = np.array([[[0.010]]], dtype=np.float64)

    rho = compute_air_density(t2, psfc, q2)

    assert np.isclose(float(rho[0, 0, 0]), 1.154, atol=0.001)


def test_resolve_wrfout_paths_matches_exact_domain_set(tmp_path):
    for d in (1, 2, 3, 4):
        (tmp_path / f"wrfout_d{d:02d}_2026-01-01_00:00:00").touch()

    def names(domains):
        return [p.name for p in resolve_wrfout_paths(tmp_path, "20260101", domains)]

    assert names((1, 4)) == [
        "wrfout_d01_2026-01-01_00:00:00",
        "wrfout_d04_2026-01-01_00:00:00",
    ]
    assert names((2,)) == ["wrfout_d02_2026-01-01_00:00:00"]
    assert names(None) == [f"wrfout_d{d:02d}_2026-01-01_00:00:00" for d in (1, 2, 3, 4)]


def test_get_variable_keeps_time_axis_for_single_timestep_file(tmp_path):
    path = tmp_path / "wrfout_d01_single_step.nc"
    _write_tiny_wrf_file(path, n_times=1)

    with WRFDataset(path) as eager:
        t2 = eager.get_variable("T2")
        assert t2.shape == (1, 2, 3)
        np.testing.assert_array_equal(t2[0], np.arange(6).reshape(2, 3))

    with LazyWRFDataset(path) as lazy:
        t2_lazy = lazy.get_variable("T2")
        assert isinstance(t2_lazy, xr.DataArray)
        assert "Time" in t2_lazy.dims
        assert t2_lazy.shape == (1, 2, 3)
        np.testing.assert_array_equal(t2_lazy.isel(Time=0).to_numpy(), np.arange(6).reshape(2, 3))


def test_get_variable_block_reads_unsqueezed_time_slabs():
    path = Path("scratch") / "wrfout_d01_synthetic_block_reader.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(path)

        with WRFDataset(path) as wrf:
            assert wrf.n_time_steps == 2

            block = wrf.get_variable_block("T2", 0, 1)
            assert block.shape == (1, 2, 3)
            np.testing.assert_array_equal(block[0], np.arange(6).reshape(2, 3))

            # t_stop past the end is clamped; values match the eager full read.
            tail = wrf.get_variable_block("T2", 1, 99)
            assert tail.shape == (1, 2, 3)
            full = np.asarray(wrf.dataset.variables["T2"][:])
            np.testing.assert_array_equal(tail, full[1:2])

            with pytest.raises(ValueError, match="Invalid time block"):
                wrf.get_variable_block("T2", 1, 1)
    finally:
        path.unlink(missing_ok=True)
