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

#: MM5/RIP4 seaprs: TC, the ceiling the "ridiculous" correction bends back to.
HOT_SURFACE_THRESHOLD_K = 273.16 + 17.5


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


def _reduced_by_hand(surface_temperature_k: float, terrain_m: float, *, correct: bool) -> float:
    """RIP4 seaprs for one dry ICAO column, written out from the algorithm.

    The reference level is found by inverting the analytic pressure profile
    instead of interpolating between model levels, so this is an oracle and not
    a second reading of the code under test.
    """
    lowest_pressure = SEA_LEVEL_PRESSURE_PA * (
        1.0 - STANDARD_LAPSE_RATE * terrain_m / SEA_LEVEL_TEMPERATURE_K
    ) ** (9.80665 / (287.04 * STANDARD_LAPSE_RATE))
    reference_pressure = lowest_pressure - 10000.0
    reference_height = (SEA_LEVEL_TEMPERATURE_K / STANDARD_LAPSE_RATE) * (
        1.0
        - (reference_pressure / SEA_LEVEL_PRESSURE_PA) ** (287.04 * STANDARD_LAPSE_RATE / 9.80665)
    )
    reference_temperature = surface_temperature_k - STANDARD_LAPSE_RATE * (
        reference_height - terrain_m
    )

    extrapolated_surface = reference_temperature * (lowest_pressure / reference_pressure) ** (
        STANDARD_LAPSE_RATE * 287.04 / 9.81
    )
    sea_level_temperature = reference_temperature + STANDARD_LAPSE_RATE * reference_height
    if correct and extrapolated_surface > HOT_SURFACE_THRESHOLD_K:
        sea_level_temperature = (
            HOT_SURFACE_THRESHOLD_K - 0.005 * (extrapolated_surface - HOT_SURFACE_THRESHOLD_K) ** 2
        )

    reduced_pa = lowest_pressure * np.exp(
        2.0 * 9.81 * terrain_m / (287.04 * (sea_level_temperature + extrapolated_surface))
    )
    return float(reduced_pa / 100.0)


def test_a_tropical_afternoon_over_high_terrain_is_bent_back_by_the_hot_surface_branch():
    """Exercises the RIP hot-surface branch, which exists to bound exactly this.

    Uncorrected, a 320 K surface extrapolates to a sea-level temperature of
    327.8 K and reduces to 995.4 hPa — a fictitious low over the Chapada. The
    branch caps it at 286.1 K and gives 1004.1 hPa. Both sit inside any window
    wide enough to be called 'physical', so the assertion is on the number.
    """
    pressure, height, _temperature, vapor = _standard_atmosphere_column(1200.0)
    scorching = 320.0 - STANDARD_LAPSE_RATE * (height - height[0])

    reduced = sea_level_pressure_hpa(pressure, height, scorching, vapor)[0, 0]

    assert reduced == pytest.approx(_reduced_by_hand(320.0, 1200.0, correct=True), abs=0.01)
    assert _reduced_by_hand(320.0, 1200.0, correct=False) == pytest.approx(995.37, abs=0.01)


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


@pytest.mark.parametrize(
    ("typical_range_hpa", "interval_hpa"),
    [
        (1.14, 1.0),
        (4.9, 1.0),
        (9.9, 1.0),
        (10.0, 2.0),
        (12.0, 2.0),
        (19.9, 2.0),
        (20.0, 4.0),
    ],
)
def test_the_spacing_is_the_coarsest_that_still_fills_a_typical_step(
    typical_range_hpa, interval_hpa
):
    """The rungs are the published legend, so both sides of each are pinned.

    1.14 hPa is the innermost domain's median range and takes the 1 hPa floor;
    a 12 hPa run takes 2 hPa lines, and only
    from 20 hPa on is 4 coarse enough to still fill the map.
    """
    assert isobars.choose_interval_hpa(typical_range_hpa) == interval_hpa


def test_the_contouring_smoother_damps_a_cell_scale_spike_without_moving_the_field():
    """The sigma is in CELLS, so one cell is damped and the field's mass stays put."""
    field = np.full((21, 21), 1010.0)
    field[10, 10] = 1020.0

    smoothed = isobars.smooth_for_contouring(field)

    assert smoothed.shape == field.shape
    assert smoothed[10, 10] < 1013.0
    assert smoothed.mean() == pytest.approx(field.mean(), abs=1e-9)


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
    [(1016.02, 1016.09), (1013.0, 1013.0)],
    ids=["flatter_than_the_spacing", "no_variation_at_all"],
)
def test_a_field_too_flat_to_name_a_round_level_draws_nothing(low_hpa, high_hpa):
    """A line at the midpoint would assert a gradient the field does not resolve."""
    barely_varying = np.linspace(low_hpa, high_hpa, 64).reshape(8, 8)

    assert isobars.isobar_levels_hpa(barely_varying, 1.0).size == 0


def test_a_reduction_that_produced_nothing_finite_is_an_error_not_an_empty_step():
    """A failed reduction and a flat field are different claims about the world."""
    with pytest.raises(ValueError, match="reduction failed"):
        isobars.isobar_levels_hpa(np.full((4, 4), np.nan), 1.0)


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


def test_the_published_document_is_the_whole_schema_the_page_reads():
    """``map-manager.renderIsobars`` reads ``metadata.interval`` and draws
    ``isobars[].paths``; the document is a byte contract, so it is frozen whole
    here rather than key by key. A five-cell ramp gives one straight meridian,
    which is short enough to write out.
    """
    lon, lat, field = _ramp_grid(size=5)

    payload = isobars.create_isobars_json(
        lon,
        lat,
        field,
        levels_hpa=np.array([1013.0]),
        interval_hpa=1.0,
        date_time="03/05/2026 12:00:00",
    )

    reread = json.loads(json.dumps(payload))
    # `1013 == 1013.0` in Python, so equality alone would accept an integer level
    # and the page's `toFixed` would then read a different string.
    assert isinstance(reread["isobars"][0]["level"], float)
    assert reread == {
        "metadata": {
            "date_time": "03/05/2026 12:00:00",
            "unit": "hPa",
            "interval": 1.0,
            "smoothing_sigma_cells": 1.5,
        },
        "isobars": [
            {
                "level": 1013.0,
                "paths": [
                    [
                        [-38.5, -11.0],
                        [-38.5, -11.75],
                        [-38.5, -12.5],
                        [-38.5, -13.25],
                        [-38.5, -14.0],
                    ]
                ],
            }
        ],
    }


def test_the_coordinates_are_published_rounded_to_four_decimals():
    """Four decimals is about 10 m at this latitude and keeps the step files
    small; an unrounded float64 would triple them and change every byte.
    """
    lon, lat, field = _ramp_grid(size=7)
    off_grid = lon - 0.000123456

    payload = isobars.create_isobars_json(
        off_grid,
        lat,
        field,
        levels_hpa=np.array([1013.0]),
        interval_hpa=1.0,
        date_time="03/05/2026 12:00:00",
    )

    drawn = [point for line in payload["isobars"] for path in line["paths"] for point in path]
    assert drawn
    for longitude, latitude in drawn:
        assert longitude == round(longitude, 4)
        assert latitude == round(latitude, 4)


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
        levels = sorted({line["level"] for line in payload["isobars"]})
        assert levels == [1014.0, 1015.0, 1016.0, 1017.0]


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


def test_a_step_with_no_contour_publishes_an_empty_overlay(tmp_path):
    """The step is published and says it drew nothing, rather than inventing a line."""
    wrf = tmp_path / "wrfout_d03_partial.nc"
    _write_standard_atmosphere_wrfout(wrf, terrain_max_m=0.0, flat_steps=(1,))
    unit = _isobars_unit(tmp_path, wrf)

    result = jobs.process_unit(unit)

    assert result.error is None
    assert len(result.files) == 3
    flat = json.loads((tmp_path / "json" / "D03_ISOBARS_001.json").read_text(encoding="utf-8"))
    assert flat["isobars"] == []
    assert flat["metadata"]["unit"] == "hPa"
    drawn = json.loads((tmp_path / "json" / "D03_ISOBARS_000.json").read_text(encoding="utf-8"))
    assert drawn["isobars"]


def test_a_humid_column_reduces_lower_than_the_same_column_dry():
    """`vapor_mixing_ratio` reaches the reduction only through the virtual
    temperature. Moist air is lighter, so a warmer virtual column needs
    less pressure below it: the reduced value must come out lower."""
    pressure, height, temperature, dry = _standard_atmosphere_column(1200.0)
    humid = np.full_like(dry, 0.018)  # 18 g/kg, a moist tropical column

    reduced_dry = sea_level_pressure_hpa(pressure, height, temperature, dry)[0, 0]
    reduced_humid = sea_level_pressure_hpa(pressure, height, temperature, humid)[0, 0]

    assert reduced_humid < reduced_dry
    # 0.608 * 0.018 = 1.09% on the virtual temperature, which over 1200 m of
    # hypsometric extrapolation is about 1.5 hPa.
    assert reduced_dry - reduced_humid == pytest.approx(1.5, abs=0.5)
