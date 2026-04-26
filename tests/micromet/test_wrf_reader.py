"""Synthetic WRF reader tests."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from micrometeorology.wrf.reader import LazyWRFDataset, WRFDataset, open_wrf_dataset, parse_chunks


def _write_tiny_wrf_file(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 2)
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
            [[[-38.0, -37.5, -37.0], [-38.0, -37.5, -37.0]]] * 2,
            dtype=np.float32,
        )
        lat[:] = np.array(
            [[[-13.0, -13.0, -13.0], [-12.5, -12.5, -12.5]]] * 2,
            dtype=np.float32,
        )
        t2[:] = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        times[:] = np.array(
            [list("2024-01-01_00:00:00"), list("2024-01-01_01:00:00")],
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
            assert t2.shape == (2, 2, 3)
            np.testing.assert_array_equal(t2[0], np.arange(6).reshape(2, 3))
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
        np.testing.assert_array_equal(lazy_t2, eager_t2)
    finally:
        path.unlink(missing_ok=True)


def test_parse_chunks_accepts_none_auto_and_explicit_pairs():
    assert parse_chunks(None) is None
    assert parse_chunks("none") is None
    assert parse_chunks("auto") == "auto"
    assert parse_chunks("Time=1,south_north=128") == {"Time": 1, "south_north": 128}
