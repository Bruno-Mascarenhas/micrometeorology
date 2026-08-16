"""The sea-level reduction and the isobar overlay published from it.

The reduction is pinned against a standard atmosphere rather than against its own
output: its whole purpose is that two columns of ONE air mass reduce to ONE
pressure however high the ground under them sits, and nothing but that property
distinguishes a working reduction from surface pressure with extra arithmetic.

The overlay is pinned on the guarantee the site depends on — every published step
carries at least one line — because a step that silently published none would
read on the page as a domain with no weather rather than as a missing artifact.
"""

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from micrometeorology.wrf import isobars, jobs
from micrometeorology.wrf.sea_level_pressure import (
    MAX_TRUSTWORTHY_TERRAIN_M,
    sea_level_pressure_hpa,
    terrain_reduction_caveat,
)

SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
STANDARD_LAPSE_RATE = 0.0065


def _standard_atmosphere_column(
    terrain_m: float, *, levels: int = 60, thickness_m: float = 300.0
) -> tuple[np.ndarray, ...]:
    """One ICAO-standard column starting at *terrain_m*, as ``(K, 1, 1)`` fields."""
    height = terrain_m + np.arange(levels) * thickness_m
    temperature = SEA_LEVEL_TEMPERATURE_K - STANDARD_LAPSE_RATE * height
    pressure = SEA_LEVEL_PRESSURE_PA * (
        1.0 - STANDARD_LAPSE_RATE * height / SEA_LEVEL_TEMPERATURE_K
    ) ** (9.80665 / (287.04 * STANDARD_LAPSE_RATE))
    shape = (levels, 1, 1)
    return (
        pressure.reshape(shape),
        height.reshape(shape),
        temperature.reshape(shape),
        np.zeros(shape),
    )


def test_a_column_that_already_starts_at_sea_level_is_returned_unchanged():
    """With nothing to extrapolate through, the reduction must be the identity."""
    reduced = sea_level_pressure_hpa(*_standard_atmosphere_column(0.0))

    assert reduced[0, 0] == pytest.approx(SEA_LEVEL_PRESSURE_PA / 100.0, abs=1e-6)


@pytest.mark.parametrize("terrain_m", [400.0, 800.0, 1200.0])
def test_one_air_mass_reduces_to_one_pressure_whatever_terrain_it_sits_on(terrain_m):
    """The property that separates a reduction from surface pressure.

    Surface pressure over 1200 m is 136 hPa below its sea-level value; after the
    reduction the two must be the same weather.
    """
    pressure, height, temperature, vapor = _standard_atmosphere_column(terrain_m)

    reduced = sea_level_pressure_hpa(pressure, height, temperature, vapor)[0, 0]

    assert pressure[0, 0, 0] / 100.0 < 1000.0
    assert reduced == pytest.approx(SEA_LEVEL_PRESSURE_PA / 100.0, abs=0.1)


def test_a_tropical_afternoon_over_high_terrain_stays_in_a_physical_range():
    """Exercises the RIP hot-surface branch, which exists to bound exactly this."""
    pressure, height, _temperature, vapor = _standard_atmosphere_column(1200.0)
    scorching = 320.0 - STANDARD_LAPSE_RATE * (height - height[0])

    reduced = sea_level_pressure_hpa(pressure, height, scorching, vapor)[0, 0]

    assert 950.0 < reduced < 1080.0


def test_a_column_whose_top_never_clears_the_reference_level_refuses_to_guess():
    """A shallow file cannot define the reduction, and must not invent one."""
    shallow = _standard_atmosphere_column(0.0, levels=3, thickness_m=50.0)

    with pytest.raises(ValueError, match="100 hPa above the surface"):
        sea_level_pressure_hpa(*shallow)


def test_fields_that_disagree_on_shape_are_refused():
    pressure, height, temperature, vapor = _standard_atmosphere_column(0.0)

    with pytest.raises(ValueError, match="one common"):
        sea_level_pressure_hpa(pressure, height[:-1], temperature, vapor)


def test_terrain_past_the_trustworthy_limit_is_reported_to_the_operator():
    caveat = terrain_reduction_caveat(np.array([10.0, MAX_TRUSTWORTHY_TERRAIN_M + 250.0]))

    assert caveat is not None
    assert "1750" in caveat


def test_terrain_inside_the_trustworthy_range_earns_no_caveat():
    assert terrain_reduction_caveat(np.array([0.0, MAX_TRUSTWORTHY_TERRAIN_M - 1.0])) is None


def test_a_column_bracketed_by_a_missing_level_refuses_rather_than_returning_nan():
    pressure, height, temperature, vapor = _standard_atmosphere_column(0.0)
    pressure = pressure.copy()
    # The level just under the one that first clears 100 hPa above the surface,
    # so the gap lands inside the interpolation bracket rather than beside it.
    pressure[2] = np.nan

    with pytest.raises(ValueError, match="not finite"):
        sea_level_pressure_hpa(pressure, height, temperature, vapor)


def test_the_spacing_is_the_coarsest_that_still_fills_a_typical_step():
    """A 12 hPa run takes 2 hPa lines; anything coarser leaves the map nearly bare."""
    assert isobars.choose_interval_hpa(12.0) == 2.0
    assert isobars.choose_interval_hpa(1.4) == 0.2


@pytest.mark.parametrize("unusable", [np.nan, 0.0, -3.0])
def test_an_unusable_pressure_range_refuses_to_pick_a_spacing(unusable):
    with pytest.raises(ValueError, match="finite and positive"):
        isobars.choose_interval_hpa(unusable)


def test_every_level_falls_strictly_inside_the_field_so_each_one_draws():
    field = np.linspace(1008.3, 1016.7, 64).reshape(8, 8)

    levels = isobars.isobar_levels_hpa(field, 2.0)

    np.testing.assert_allclose(levels, [1010.0, 1012.0, 1014.0, 1016.0])


@pytest.mark.parametrize(
    ("low_hpa", "high_hpa"),
    [(1016.02, 1016.09), (1013.994, 1013.998)],
    ids=["flatter_than_the_spacing", "narrower_than_the_rounding_step"],
)
def test_a_field_too_flat_for_a_round_level_still_yields_one_inside_its_own_range(
    low_hpa, high_hpa
):
    """The guarantee the page rests on: no step is ever published lineless.

    The narrower field is thinner than the rounding step, where a rounded midpoint
    lands back outside the field it came from: the level traces nothing, the entry
    is dropped as empty, and the step ships `"isobars": []` — the guarantee failing
    in exactly the case it exists for.
    """
    barely_varying = np.linspace(low_hpa, high_hpa, 64).reshape(8, 8)

    (level,) = isobars.isobar_levels_hpa(barely_varying, 1.0)

    assert barely_varying.min() < level < barely_varying.max()


def test_a_field_with_no_variation_at_all_has_no_isobars_and_says_so():
    with pytest.raises(ValueError, match="constant"):
        isobars.isobar_levels_hpa(np.full((4, 4), 1013.0), 1.0)


def _ramp_grid(size: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(
        np.linspace(-40.0, -37.0, size), np.linspace(-14.0, -11.0, size), indexing="xy"
    )
    field = 1010.0 + 6.0 * (lon - lon.min()) / (lon.max() - lon.min())
    return lon, lat, field


def test_paths_are_published_as_lon_lat_pairs_on_the_grids_own_coordinates():
    lon, lat, field = _ramp_grid()

    payload = isobars.create_isobars_json(
        lon,
        lat,
        field,
        levels_hpa=np.array([1013.0]),
        interval_hpa=1.0,
        date_time="03/05/2026 12:00:00",
    )

    (line,) = payload["isobars"]
    assert line["level"] == 1013.0
    # The field ramps with longitude alone, so the 1013 hPa isobar is the meridian
    # halfway across. Checking only that the points land somewhere on the map
    # would pass just as happily with the field transposed.
    drawn = np.asarray([point for path in line["paths"] for point in path])
    np.testing.assert_allclose(drawn[:, 0], -38.5, atol=1e-3)
    assert drawn[:, 1].min() == pytest.approx(lat.min(), abs=1e-3)
    assert drawn[:, 1].max() == pytest.approx(lat.max(), abs=1e-3)


def test_a_level_the_field_never_reaches_is_not_published_as_an_empty_line():
    lon, lat, field = _ramp_grid()

    payload = isobars.create_isobars_json(
        lon,
        lat,
        field,
        levels_hpa=np.array([1013.0, 1400.0]),
        interval_hpa=1.0,
        date_time="03/05/2026 12:00:00",
    )

    assert [line["level"] for line in payload["isobars"]] == [1013.0]


def test_a_field_with_missing_cells_still_publishes_a_strictly_parseable_document():
    """The site parses strictly: a bare NaN token loses the whole document.

    A field with holes is the case that could put one there, so the hole is real
    here rather than the payload being checked against a field that has none.
    """
    lon, lat, field = _ramp_grid()
    field = field.copy()
    field[4:9, 4:9] = np.nan

    payload = isobars.create_isobars_json(
        lon,
        lat,
        field,
        levels_hpa=isobars.isobar_levels_hpa(field, 1.0),
        interval_hpa=1.0,
        date_time="03/05/2026 12:00:00",
    )

    assert payload["isobars"]
    json.dumps(payload, allow_nan=False)


def test_the_overlay_targets_name_ids_the_writers_actually_publish_under():
    from micrometeorology.common.types import WRFVariable

    generated = {
        jobs.values_output_id(WRFVariable.WIND),
        jobs.values_output_id(WRFVariable.RAIN),
        jobs.values_output_id(WRFVariable.TEMPERATURE),
        jobs.values_output_id(WRFVariable.WIND_POWER_DENSITY_10M),
        *(jobs.poteolico_output_id(height) for height in jobs.POTEOLICO_ALL_HEIGHTS),
    }

    assert set(isobars.ISOBAR_OVERLAY_VARIABLES) == generated


def _write_standard_atmosphere_wrfout(
    path: Path,
    *,
    terrain_max_m: float = 900.0,
    flat_steps: tuple[int, ...] = (),
) -> None:
    """A wrfout whose 3-D fields are one ICAO-standard air mass over a ridge.

    *flat_steps* strips the horizontal pressure gradient from those steps, which
    over flat ground leaves a field with no contour at all — the degenerate step
    the unit has to survive.
    """
    nt, nz, ny, nx = 3, 40, 12, 12
    with netCDF4.Dataset(path, "w") as ds:
        for name, size in (
            ("Time", nt),
            ("DateStrLen", 19),
            ("bottom_top", nz),
            ("bottom_top_stag", nz + 1),
            ("south_north", ny),
            ("west_east", nx),
        ):
            ds.createDimension(name, size)
        ds.setncattr("DX", 3000.0)
        ds.setncattr("DY", 3000.0)

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list(f"2026-05-03_{9 + i:02d}:00:00") for i in range(nt)], dtype="S1")
        lon = ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        lat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        lon[:] = np.broadcast_to(np.linspace(-39.0, -38.0, nx), (nt, ny, nx))
        lat[:] = np.broadcast_to(np.linspace(-13.5, -12.5, ny)[:, None], (nt, ny, nx))

        # A ridge across the domain, so the reduction has real terrain to undo,
        # and a west-east pressure gradient, so the isobars have something to trace.
        terrain = np.broadcast_to(np.linspace(0.0, terrain_max_m, nx), (ny, nx))
        gradient = np.tile(np.linspace(0.0, 400.0, nx), (nt, ny, 1))
        for step in flat_steps:
            gradient[step] = 0.0
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        hgt[:] = np.broadcast_to(terrain, (nt, ny, nx))

        staggered = np.zeros((nt, nz + 1, ny, nx))
        pressure = np.zeros((nt, nz, ny, nx))
        theta = np.zeros((nt, nz, ny, nx))
        for k in range(nz + 1):
            staggered[:, k] = terrain + k * 300.0
        for k in range(nz):
            height = 0.5 * (staggered[:, k] + staggered[:, k + 1])
            temperature = SEA_LEVEL_TEMPERATURE_K - STANDARD_LAPSE_RATE * height
            pressure[:, k] = (SEA_LEVEL_PRESSURE_PA + gradient) * (
                1.0 - STANDARD_LAPSE_RATE * height / SEA_LEVEL_TEMPERATURE_K
            ) ** (9.80665 / (287.04 * STANDARD_LAPSE_RATE))
            theta[:, k] = temperature * (1.0e5 / pressure[:, k]) ** (2.0 / 7.0) - 300.0

        for name, values, dims in (
            ("PH", staggered * 9.81, ("Time", "bottom_top_stag", "south_north", "west_east")),
            (
                "PHB",
                np.zeros_like(staggered),
                ("Time", "bottom_top_stag", "south_north", "west_east"),
            ),
            ("P", pressure, ("Time", "bottom_top", "south_north", "west_east")),
            ("PB", np.zeros_like(pressure), ("Time", "bottom_top", "south_north", "west_east")),
            ("T", theta, ("Time", "bottom_top", "south_north", "west_east")),
            ("QVAPOR", np.zeros_like(pressure), ("Time", "bottom_top", "south_north", "west_east")),
        ):
            variable = ds.createVariable(name, "f4", dims)
            variable[:] = values.astype(np.float32)


def _isobars_unit(tmp_path: Path, wrf: Path) -> jobs.WorkUnit:
    return jobs.WorkUnit(
        kind="isobars",
        wrf_path=str(wrf),
        variable="isobars",
        json_dir=str(tmp_path / "json"),
        geojson_dir=str(tmp_path / "geo"),
    )


def test_the_published_overlay_reduces_a_standard_atmosphere_to_its_real_pressure(tmp_path):
    """End to end: what lands on disk must be the weather, not the ridge under it."""
    wrf = tmp_path / "wrfout_d03_isobars.nc"
    _write_standard_atmosphere_wrfout(wrf)
    unit = _isobars_unit(tmp_path, wrf)

    result = jobs.process_unit(unit)

    assert result.error is None
    assert result.files
    for written in result.files:
        payload = json.loads(Path(written).read_text(encoding="utf-8"))
        assert payload["isobars"], f"{Path(written).name} published no line"
        levels = [line["level"] for line in payload["isobars"]]
        assert min(levels) > 1010.0
        assert max(levels) < 1020.0


def test_the_manifest_tells_the_page_which_variables_the_overlay_belongs_over(tmp_path):
    """Fed the writer's OWN filenames, so a name the manifest cannot index fails here.

    `availability` is keyed off a regex over the written names; hand-typed inputs
    would let a writer emit something that regex drops and still pass.
    """
    wrf = tmp_path / "wrfout_d03_manifest.nc"
    _write_standard_atmosphere_wrfout(wrf)
    unit = _isobars_unit(tmp_path, wrf)
    result = jobs.process_unit(unit)
    assert result.error is None

    manifest_path = jobs.write_run_manifest(tmp_path / "json", [result])

    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlay = manifest["features"]["isobar_overlay"]
    assert overlay["variable"] == "ISOBARS"
    assert "WIND" in overlay["draw_over"]
    assert "SWDOWN" not in overlay["draw_over"]
    # Every written step was indexed: a name the regex missed would leave the
    # timeline empty instead.
    assert manifest["index_min"] == 0
    assert manifest["index_max"] == len(result.files) - 1


def test_a_step_with_no_contour_is_dropped_without_taking_the_domain_with_it(tmp_path):
    wrf = tmp_path / "wrfout_d03_partial.nc"
    _write_standard_atmosphere_wrfout(wrf, terrain_max_m=0.0, flat_steps=(1,))
    unit = _isobars_unit(tmp_path, wrf)

    result = jobs.process_unit(unit)

    assert result.error is None
    assert len(result.files) == 2
    assert any("step 001" in warning for warning in result.warnings)
    assert not (tmp_path / "json" / "D03_ISOBARS_001.json").exists()
