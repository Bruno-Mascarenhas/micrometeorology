"""Bitwise equivalence of block-streamed wind extraction vs the eager path."""

from __future__ import annotations

from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import pytest

from micrometeorology.wrf import variables as vmod
from micrometeorology.wrf.reader import WRFDataset
from tests.micromet._reference import (
    compute_adjusted_heights,
    compute_wind_vectors_at_height,
    interpolate_speed_to_height,
)

if TYPE_CHECKING:
    from pathlib import Path

NT, NZ, NY, NX = 7, 5, 4, 5


def _write_wind_wrf_file(path: Path, *, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", NT)
        ds.createDimension("bottom_top", NZ)
        ds.createDimension("bottom_top_stag", NZ + 1)
        ds.createDimension("south_north", NY)
        ds.createDimension("south_north_stag", NY + 1)
        ds.createDimension("west_east", NX)
        ds.createDimension("west_east_stag", NX + 1)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 1000.0)

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        v = ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        ph = ds.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
        phb = ds.createVariable(
            "PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))

        u[:] = rng.uniform(-20, 20, size=(NT, NZ, NY, NX + 1)).astype(np.float32)
        v[:] = rng.uniform(-20, 20, size=(NT, NZ, NY + 1, NX)).astype(np.float32)
        # Geopotential increasing with level, like real WRF output.
        base = np.cumsum(
            rng.uniform(200, 600, size=(NT, NZ + 1, NY, NX)).astype(np.float32), axis=1
        )
        ph[:] = (base * 0.1).astype(np.float32)
        phb[:] = (base * 9.0).astype(np.float32)
        hgt[:] = rng.uniform(0, 80, size=(NT, NY, NX)).astype(np.float32)


def _eager_reference(ds: WRFDataset, targets: tuple[int, ...]) -> dict[int, dict]:
    """Reproduce the pre-deletion CLI WIND_POTENTIAL branch exactly (frozen oracles)."""
    u_central, v_central, height_adjusted, speed_4d = compute_adjusted_heights(ds)
    out: dict[int, dict] = {}
    for target in targets:
        speed_3d = interpolate_speed_to_height(speed_4d, height_adjusted, target)
        steps = [vmod.materialize_2d(speed_3d[i : i + 1, :, :]) for i in range(speed_3d.shape[0])]
        vectors = [
            compute_wind_vectors_at_height(
                u_central[i : i + 1],
                v_central[i : i + 1],
                height_adjusted[i : i + 1],
                target,
                downsampling=4,
            )
            for i in range(speed_3d.shape[0])
        ]
        # Scale bounds follow the site-wide convention (get_low_high):
        # skip the spin-up first step, cap the max at the 98th percentile.
        vmin, vmax = vmod.get_low_high(speed_3d)
        out[target] = {
            "vmin": vmin,
            "vmax": vmax,
            "steps": steps,
            "vectors": vectors,
        }
    return out


@pytest.mark.parametrize("block_steps", [3, 64])
def test_stream_wind_at_heights_matches_eager_path_bitwise(tmp_path, block_steps):
    path = tmp_path / "wrfout_d03_stream_synth.nc"
    _write_wind_wrf_file(path)

    targets = (50, 100, 150)
    with WRFDataset(path) as ds:
        reference = _eager_reference(ds, targets)
        series = vmod.stream_wind_at_heights(ds, targets, block_steps=block_steps)

    assert [s.target for s in series] == list(targets)
    for s in series:
        ref = reference[s.target]
        assert s.vmin == ref["vmin"]
        assert s.vmax == ref["vmax"]
        # Bounds pin the site-wide convention: skip step 0, 98th-pct max.
        assert s.vmin == float(np.nanmin(s.speed_steps[1:]))
        assert s.vmax == float(np.nanpercentile(s.speed_steps[1:].ravel(), 98))
        assert s.speed_steps.dtype == ref["steps"][0].dtype
        for i, ref_step in enumerate(ref["steps"]):
            assert np.array_equal(s.speed_steps[i], ref_step, equal_nan=True)
        # Wind vector payloads embed unrounded floats in the values JSON —
        # exact equality, not approx.
        assert s.wind_vectors == ref["vectors"]


def test_stream_wind_block_boundary_independence(tmp_path):
    path = tmp_path / "wrfout_d03_stream_blocks.nc"
    _write_wind_wrf_file(path, seed=23)

    with WRFDataset(path) as ds:
        a = vmod.stream_wind_at_heights(ds, (100,), block_steps=2)
        b = vmod.stream_wind_at_heights(ds, (100,), block_steps=7)

    assert np.array_equal(a[0].speed_steps, b[0].speed_steps, equal_nan=True)
    assert a[0].wind_vectors == b[0].wind_vectors
    assert (a[0].vmin, a[0].vmax) == (b[0].vmin, b[0].vmax)
