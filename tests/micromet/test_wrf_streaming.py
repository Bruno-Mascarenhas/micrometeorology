"""Bitwise equivalence of block-streamed wind extraction vs the eager path."""

from pathlib import Path

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

        # The map rotation, which every real wrfout carries. Without it here the
        # block path could cross a 3-step block with the file's full time axis
        # and nothing would notice until the operational run: real files are
        # Mercator, so the values are the identity but the SHAPES are not.
        cos_alpha = ds.createVariable("COSALPHA", "f4", ("Time", "south_north", "west_east"))
        sin_alpha = ds.createVariable("SINALPHA", "f4", ("Time", "south_north", "west_east"))
        cos_alpha[:] = np.ones((NT, NY, NX), dtype=np.float32)
        sin_alpha[:] = np.zeros((NT, NY, NX), dtype=np.float32)


def _eager_reference(ds: WRFDataset, targets: tuple[int, ...]) -> dict[int, dict]:
    """Frozen oracle: the eager CLI WIND_POTENTIAL branch, step for step."""
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
        # Scale bounds follow the site-wide convention (percentile_scale_bounds):
        # skip the spin-up first step, cap the max at the 98th percentile.
        vmin, vmax = vmod.percentile_scale_bounds(speed_3d)
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
        assert s.vmin == float(np.nanmin(s.speed_steps[1:]))
        assert s.vmax == float(np.nanpercentile(s.speed_steps[1:].ravel(), 98))
        assert s.speed_steps.dtype == ref["steps"][0].dtype
        for i, ref_step in enumerate(ref["steps"]):
            assert np.array_equal(s.speed_steps[i], ref_step, equal_nan=True)
        # The overlay payload rounds angles to 1dp and magnitudes to 2dp, so the
        # comparison against the rounded reference is exact, not approximate.
        for i, ref_vec in enumerate(ref["vectors"]):
            got = s.wind_vectors[i]
            assert got is not None, f"wind vector packaging failed for step {i}"
            assert got["downsampled_linear_indices"] == ref_vec["downsampled_linear_indices"]
            assert got["downsampled_angles"] == np.round(ref_vec["downsampled_angles"], 1).tolist()
            assert (
                got["downsampled_magnitudes"]
                == np.round(ref_vec["downsampled_magnitudes"], 2).tolist()
            )


def _write_offsubgrid_defects_wrf_file(path: Path, *, seed: int = 41) -> None:
    """Wind file whose NaN and non-monotonic columns all lie OFF the stride-4
    wind-vector subgrid (rows/cols 0, 4, 8), so the full-grid interpolator
    falls back to the argsort reference while the subgrid keeps its fast path.
    """
    nt, nz, ny, nx = 5, 6, 9, 11
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", nt)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("bottom_top_stag", nz + 1)
        ds.createDimension("south_north", ny)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("west_east_stag", nx + 1)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 1000.0)

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        v = ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        ph = ds.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
        phb = ds.createVariable(
            "PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))

        u_values = rng.uniform(-20, 20, size=(nt, nz, ny, nx + 1)).astype(np.float32)
        u_values[0, 2, 1, 2] = np.nan  # feeds u_c columns x=1 and x=2 only
        u[:] = u_values
        v[:] = rng.uniform(-20, 20, size=(nt, nz, ny + 1, nx)).astype(np.float32)

        base = np.cumsum(
            rng.uniform(200, 600, size=(nt, nz + 1, ny, nx)).astype(np.float32), axis=1
        )
        base[1, 3, 2, 3] = base[1, 1, 2, 3]  # non-monotonic column at (y=2, x=3)
        ph[:] = (base * 0.1).astype(np.float32)
        phb[:] = (base * 9.0).astype(np.float32)
        hgt[:] = rng.uniform(0, 80, size=(nt, ny, nx)).astype(np.float32)


def _full_grid_wind_reference(
    ds: WRFDataset, targets: tuple[int, ...], downsampling: int = 4
) -> dict[int, dict]:
    """Reference path: chained staggering, full-grid u/v interpolation, then
    ``np.mgrid`` sampling of the trigonometry."""
    from micrometeorology.wrf.interpolation import VerticalInterpolator

    n_t = ds.n_time_steps
    u_raw = ds.get_variable_block("U", 0, n_t)
    u_c = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0
    v_raw = ds.get_variable_block("V", 0, n_t)
    v_c = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0
    ph = ds.get_variable_block("PH", 0, n_t)
    phb = ds.get_variable_block("PHB", 0, n_t)
    height = (ph + phb) / 9.81
    height_c = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0
    hgt = ds.get_variable_block("HGT", 0, n_t)
    height_adjusted = height_c - hgt[:, np.newaxis, :, :]

    speed_4d = np.hypot(u_c, v_c)
    ny, nx = speed_4d.shape[2], speed_4d.shape[3]
    interpolator = VerticalInterpolator(height_adjusted, axis=1)
    # Precondition of the fixture: the two grids must take DIFFERENT internal
    # routes, otherwise the comparison below proves nothing.
    assert not interpolator._fast_ok
    assert np.isnan(u_c).any()
    assert not np.isnan(u_c[:, :, ::downsampling, ::downsampling]).any()
    subgrid_heights = np.ascontiguousarray(height_adjusted[:, :, ::downsampling, ::downsampling])
    assert VerticalInterpolator(subgrid_heights, axis=1)._fast_ok

    out: dict[int, dict] = {}
    for target in targets:
        speed_3d = interpolator.interpolate(speed_4d, float(target))
        u_3d = interpolator.interpolate(u_c, float(target))
        v_3d = interpolator.interpolate(v_c, float(target))
        vectors = []
        for k in range(u_3d.shape[0]):
            magnitude = np.hypot(u_3d[k], v_3d[k])
            angle = np.arctan2(u_3d[k], v_3d[k]) * 180.0 / np.pi
            angle = np.where(angle < 0, angle + 360.0, angle)
            i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
            i_flat, j_flat = i_idx.ravel(), j_idx.ravel()
            angles_flat = angle[i_flat, j_flat]
            mags_flat = magnitude[i_flat, j_flat]
            valid = ~np.isnan(angles_flat)
            vectors.append(
                {
                    "downsampled_angles": np.round(
                        angles_flat[valid].astype(np.float64), 1
                    ).tolist(),
                    "downsampled_magnitudes": np.round(
                        mags_flat[valid].astype(np.float64), 2
                    ).tolist(),
                    "downsampled_linear_indices": (i_flat * nx + j_flat)[valid].tolist(),
                }
            )
        out[target] = {"speed": speed_3d, "vectors": vectors}
    return out


@pytest.mark.parametrize("block_steps", [2, 5])
def test_stream_wind_subgrid_matches_full_grid_interpolation(tmp_path, block_steps):
    """Interpolating u/v on the stride-4 subgrid must reproduce the full-grid
    interpolation sampled afterwards, bit for bit, even when the NaN and
    non-monotonic columns that flip the interpolator's fast path are outside
    the subgrid (so the two grids take different internal routes)."""
    path = tmp_path / "wrfout_d03_stream_subgrid.nc"
    _write_offsubgrid_defects_wrf_file(path)

    targets = (50, 100, 150)
    with WRFDataset(path) as ds:
        reference = _full_grid_wind_reference(ds, targets)
        series = vmod.stream_wind_at_heights(ds, targets, block_steps=block_steps)

    for wind_series in series:
        ref = reference[wind_series.target]
        assert np.array_equal(wind_series.speed_steps, ref["speed"], equal_nan=True)
        assert wind_series.wind_vectors == ref["vectors"]


def test_stream_wind_block_boundary_independence(tmp_path):
    path = tmp_path / "wrfout_d03_stream_blocks.nc"
    _write_wind_wrf_file(path, seed=23)

    with WRFDataset(path) as ds:
        a = vmod.stream_wind_at_heights(ds, (100,), block_steps=2)
        b = vmod.stream_wind_at_heights(ds, (100,), block_steps=7)

    assert np.array_equal(a[0].speed_steps, b[0].speed_steps, equal_nan=True)
    assert a[0].wind_vectors == b[0].wind_vectors
    assert (a[0].vmin, a[0].vmax) == (b[0].vmin, b[0].vmax)
