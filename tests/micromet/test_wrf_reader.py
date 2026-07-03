"""Synthetic WRF reader tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pytest

from micrometeorology.wrf.reader import WRFDataset, resolve_wrfout_paths
from micrometeorology.wrf.variables import (
    compute_air_density,
    compute_relative_humidity,
    extract_scalar,
    materialize_2d,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_wrf_reader_handles_tiny_synthetic_netcdf(tmp_path):
    path = tmp_path / "wrfout_d01_synthetic_reader.nc"
    _write_tiny_wrf_file(path)

    with WRFDataset(path) as wrf:
        lon_grid, lat_grid = wrf.read_grid()

        assert lon_grid.shape == (2, 3)
        assert lat_grid.shape == (2, 3)
        assert wrf.grid_bounds() == (-38.0, -37.0, -13.0, -12.5)
        assert [dt.hour for dt in wrf.parse_times()] == [0, 1]


def test_scalar_extractor_bounds_and_first_step_materialization(tmp_path):
    path = tmp_path / "wrfout_d01_synthetic_scalar.nc"
    _write_tiny_wrf_file(path)

    with WRFDataset(path) as wrf:
        var_data, vmin, vmax = extract_scalar(wrf, "T2")
        first_step = materialize_2d(var_data[0:1, :, :])

    assert isinstance(var_data, np.ndarray)
    assert vmin == 6.0
    # float32 data: the 98th percentile lands a ULP below the decimal literal.
    assert vmax == pytest.approx(10.9, abs=1e-5)
    np.testing.assert_array_equal(first_step, np.arange(6).reshape(2, 3))


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


def test_get_variable_block_reads_unsqueezed_time_slabs(tmp_path):
    path = tmp_path / "wrfout_d01_synthetic_block_reader.nc"
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
