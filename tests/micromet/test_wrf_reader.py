"""Synthetic WRF reader tests."""

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from micrometeorology.common.types import GridLevel
from micrometeorology.wrf.reader import (
    WRFDataset,
    assert_one_file_per_domain,
    detect_grid_level,
    normalize_run_date,
    resolve_wrfout_paths,
)
from micrometeorology.wrf.series import extract_point_series
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


def test_point_series_reads_the_cell_nearest_the_target_in_time_order(tmp_path):
    path = tmp_path / "wrfout_d01_point_series.nc"
    _write_tiny_wrf_file(path)

    frame = extract_point_series([path], -12.5, -37.0, ["T2"])

    assert list(frame.columns) == ["T2"]
    assert [str(stamp) for stamp in frame.index] == [
        "2024-01-01 00:00:00",
        "2024-01-01 01:00:00",
    ]
    # Row 1, column 2 of the 2x3 grid is the nearest cell to (-12.5, -37.0).
    assert frame["T2"].tolist() == [5.0, 11.0]


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


def test_a_time_block_starting_past_the_last_step_names_the_step_count(tmp_path):
    path = tmp_path / "wrfout_d01_block_past_the_end.nc"
    _write_tiny_wrf_file(path)

    with WRFDataset(path) as wrf, pytest.raises(ValueError, match="past the 2 steps"):
        wrf.get_variable_block("T2", 7, 99)


def test_build_date_metadata_uses_pinned_product_timezone(tmp_path, monkeypatch):
    """Local datetimes must come from the pinned product timezone, never the
    host OS setting — a UTC-configured job host must not shift the forecast."""
    from datetime import timedelta

    path = tmp_path / "wrfout_d01_synthetic_tz.nc"
    _write_tiny_wrf_file(path)

    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    with WRFDataset(path) as wrf:
        entries = wrf.build_date_metadata()
    assert entries[0]["datetime_local"].utcoffset() == timedelta(hours=-3)
    # 2024-01-01 00:00 UTC is 2023-12-31 21:00 in America/Bahia.
    assert entries[0]["datetime_local"].strftime("%d/%m/%Y %H") == "31/12/2023 21"

    monkeypatch.setenv("LABMIM_TIMEZONE", "UTC")
    with WRFDataset(path) as wrf:
        entries = wrf.build_date_metadata()
    assert entries[0]["datetime_local"].utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "spelling",
    ["20260503", "2026-05-03", "2026/05/03", "2026050300"],
)
def test_normalize_run_date_keeps_the_digits_of_every_accepted_spelling(spelling):
    assert normalize_run_date(spelling) == spelling.replace("-", "").replace("/", "")


@pytest.mark.parametrize("typo", ["2026-5-3", "2026050", "may3", "2026-05-0x"])
def test_normalize_run_date_refuses_anything_short_of_a_full_day(typo):
    """Slicing a mistyped date would report it as a day WRF produced no files for."""
    with pytest.raises(ValueError, match="--date must be YYYYMMDD"):
        normalize_run_date(typo)


def test_a_date_with_no_wrfout_warns_which_pattern_found_nothing(tmp_path, caplog):
    assert resolve_wrfout_paths(tmp_path, "20260503", (2,)) == []
    assert "wrfout_d02_2026-05-03*" in caplog.text


def test_detect_grid_level_reads_the_token_from_any_name_shape():
    assert detect_grid_level("wrfout_d03.nc") is GridLevel.D03
    assert detect_grid_level("/archive/WRFOUT_D03_2026-07-27") is GridLevel.D03
    assert detect_grid_level(Path("wrfout_d01_2013-07-01_01_00_00-003_")) is GridLevel.D01
    assert detect_grid_level("wrfout_2026-07-27_00_00_00") is None
    assert detect_grid_level("wrfout_d06_2026-07-27") is None


def test_undetectable_domain_refuses_to_publish_as_d01(tmp_path):
    """Guessing D01 would republish this file's grid and values over the real
    D01 products, since every output name is built from the detected level."""
    path = tmp_path / "wrfout_2026-07-27_00_00_00.nc"
    _write_tiny_wrf_file(path)

    with pytest.raises(ValueError, match="Could not detect grid level"):
        WRFDataset(path)


def test_untokenized_open_leaves_no_dangling_netcdf_handle(tmp_path):
    """The refusal happens before the open, so no HDF5 handle outlives it —
    ``__exit__`` never runs for a constructor that raised."""
    path = tmp_path / "wrfout_d06_2026-07-27.nc"
    _write_tiny_wrf_file(path)

    with pytest.raises(ValueError, match="Could not detect grid level"):
        WRFDataset(path)
    # A second writer can only take the file if nothing still holds it open.
    with netCDF4.Dataset(path, "a") as ds:
        ds.setncattr("REOPENED", 1)


def test_assert_one_file_per_domain_names_the_colliding_files():
    first = "/wrf/wrfout_d02_2026-05-03_09:00:00"
    second = "/wrf/wrfout_d02_2026-05-04_09:00:00"

    assert_one_file_per_domain([first, "/wrf/wrfout_d03_2026-05-03_09:00:00"])

    with pytest.raises(ValueError, match="would overwrite each other") as excinfo:
        assert_one_file_per_domain([first, second, "/wrf/wrfout_d03_2026-05-03_09:00:00"])
    message = str(excinfo.value)
    assert "D02" in message
    assert Path(first).name in message
    assert Path(second).name in message
    assert "D03" not in message


def test_assert_one_file_per_domain_ignores_untokenized_names():
    """Those units fail individually in the worker, so they never collide."""
    assert_one_file_per_domain(["/wrf/renamed_a.nc", "/wrf/renamed_b.nc"])


def test_build_date_metadata_flags_skipped_steps(tmp_path):
    path = tmp_path / "wrfout_d01_synthetic_skip.nc"
    _write_tiny_wrf_file(path)

    with WRFDataset(path) as wrf:
        entries = wrf.build_date_metadata(skip_first_n=1)
    assert [e["skip"] for e in entries] == [True, False]
    assert [e["index"] for e in entries] == [0, 1]


def test_an_unwritten_record_is_refused_instead_of_publishing_9e36(tmp_path):
    """``set_auto_mask(False)`` hands back the fill value as a plain float, so
    ``np.isfinite`` says True and every reduction downstream — colour scales,
    published cell values — treats 9.96921e+36 as a measurement."""
    path = tmp_path / "wrfout_d02_2026-05-03_00:00:00"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 3)
        ds.createDimension("DateStrLen", 19)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 1000.0)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        hfx = ds.createVariable("HFX", "f4", ("Time", "south_north", "west_east"))
        for step in range(3):
            stamp = f"2026-05-03_0{step}:00:00"
            times[step] = np.array(list(stamp), dtype="S1")
        # The Time dimension is extended by writing Times but not HFX for the
        # last step: the shape of a run interrupted partway through a record.
        hfx[0:2] = 1.0

    with WRFDataset(path) as wrf:
        assert np.isfinite(np.asarray(wrf.dataset.variables["HFX"][:])).all()
        with pytest.raises(ValueError, match="fill value"):
            wrf.get_variable("HFX")
        with pytest.raises(ValueError, match="fill value"):
            wrf.get_variable_block("HFX", 0, 3)


def test_a_fully_written_variable_still_reads(tmp_path):
    path = tmp_path / "wrfout_d02_2026-05-04_00:00:00"
    _write_tiny_wrf_file(path, n_times=2)

    with WRFDataset(path) as wrf:
        assert wrf.get_variable("T2").shape[0] == 2
        assert wrf.get_variable_block("T2", 0, 2).shape[0] == 2
