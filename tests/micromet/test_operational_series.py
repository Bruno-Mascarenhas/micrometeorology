"""The operational point series: schema evolution, stations, windows and the v1 repair."""

import math
import re
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from micrometeorology.cli.export_operational_series import app
from micrometeorology.common.physics import STEFAN_BOLTZMANN
from micrometeorology.wrf.operational_record import (
    DEFAULT_HEADER,
    DEFAULT_STATION,
    OPERATIONAL_CATALOG,
    V1_COLUMNS,
    V1_TO_V2,
    Station,
    append_block,
    build_columns,
    extend_header,
    format_row,
    legacy_spellings,
    migrate_to_v2,
    parse_station,
    read_header,
    read_stations,
    rename_v1_columns,
)
from micrometeorology.wrf.operational_series import (
    assign_domains,
    extract_operational_block,
)
from micrometeorology.wrf.reader import WRFDataset

# The v1 header line, verbatim, minus its twelve trailing empty fields. Every
# reader of series_operacional.dat resolved its columns by these names and the
# file is positional, so this order is the contract the migration must preserve.
V1_HEADER = (
    "year,month,day,hour,T,ur,pressure,e,es,q,WS,WD,u,v,Swdw,Swdw_b,Swdw_farms,"
    "Swup_b,Swup_calc,Swdf,Swdf_farms,Swdr,Swdr_farms,Lwdw_glw,Lwdw_b,Lwup_b,"
    "Lwup_calc,ALBD,EMISS,H,LE,G,ustar,PBLH,TSM"
)

SURFACE_FIELDS = {
    "T2": 300.0,
    "PSFC": 101300.0,
    "Q2": 0.016,
    "U10": 3.0,
    "V10": -4.0,
    "COSALPHA": 1.0,
    "SINALPHA": 0.0,
    "SWDOWN": 800.0,
    "SWDNB": 800.0,
    "SWUPB": 120.0,
    "SWDDIF": 200.0,
    "SWDDIR": 600.0,
    "GLW": 400.0,
    "TSK": 305.0,
    "ALBEDO": 0.15,
    "EMISS": 0.88,
    "HFX": 250.0,
    "LH": 30.0,
    "GRDFLX": -60.0,
    "UST": 0.4,
    "PBLH": 900.0,
    "SST": 296.15,
}


def _write_wrfout(
    path: Path,
    n_times: int = 4,
    stamps: list[str] | None = None,
    omit: tuple[str, ...] = (),
    extra: dict[str, float] | None = None,
    rain: list[float] | None = None,
    cold_start: bool = False,
    lat: tuple[float, float, float] = (-13.1, -13.0, -12.9),
    lon: tuple[float, float, float] = (-38.6, -38.5, -38.4),
    dx: float = 1000.0,
) -> None:
    stamps = stamps or [f"2026-08-08_{hour:02d}:00:00" for hour in range(n_times)]
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", n_times)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 3)
        ds.createDimension("DateStrLen", 19)
        ds.setncattr("DX", dx)
        ds.setncattr("DY", dx)

        dims = ("Time", "south_north", "west_east")
        ds.createVariable("XLAT", "f4", dims)[:] = np.broadcast_to(
            np.array(lat, dtype=np.float32)[None, :, None], (n_times, 3, 3)
        )
        ds.createVariable("XLONG", "f4", dims)[:] = np.broadcast_to(
            np.array(lon, dtype=np.float32)[None, None, :], (n_times, 3, 3)
        )
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(
            [list(stamp) for stamp in stamps], dtype="S1"
        )

        for name, value in {**SURFACE_FIELDS, **(extra or {})}.items():
            if name in omit:
                continue
            # A per-step ramp so a column that silently reused one step would
            # show up as a constant instead of matching the source.
            ramp = np.arange(n_times, dtype=np.float32)[:, None, None]
            variable = ds.createVariable(name, "f4", dims)
            variable[:] = value + ramp * (value * 0.01)
            if cold_start and name == "GLW":
                variable[0, :, :] = 0.0

        accumulated = np.array(rain if rain is not None else [0.0] * n_times, dtype=np.float32)
        for name in ("RAINC", "RAINNC"):
            if name in omit:
                continue
            ds.createVariable(name, "f4", dims)[:] = np.broadcast_to(
                accumulated[:, None, None] / 2.0, (n_times, 3, 3)
            )


@pytest.fixture
def wrfout(tmp_path: Path) -> Path:
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path)
    return path


@pytest.fixture
def nested_run(tmp_path: Path) -> list[Path]:
    """A parent and a nest, the nest covering only the middle of the parent."""
    parent = tmp_path / "wrfout_d01_2026-08-08_00:00:00"
    nest = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(parent, lat=(-16.0, -13.0, -10.0), lon=(-42.0, -38.5, -35.0), dx=27000.0)
    _write_wrfout(nest, lat=(-13.1, -13.0, -12.9), lon=(-38.6, -38.5, -38.4), dx=1000.0)
    return [parent, nest]


def _block(path: Path, **kwargs):
    with WRFDataset(path) as ds:
        return extract_operational_block(ds, -13.0, -38.5, **kwargs)


def _datasets(paths: list[Path]) -> list[str]:
    return [token for path in paths for token in ("-d", str(path))]


def test_the_v1_header_maps_to_v2_field_by_field_in_the_same_order():
    assert list(V1_COLUMNS) == V1_HEADER.split(",")
    assert [new for _, new in V1_TO_V2] == list(DEFAULT_HEADER[:-1])


def test_v2_adds_exactly_one_column_to_the_v1_schema():
    assert DEFAULT_HEADER[-1] == "precip_mm"
    assert len(DEFAULT_HEADER) == len(V1_COLUMNS) + 1


def test_every_v2_name_carries_its_unit_or_is_dimensionless():
    dimensionless = {"year", "month", "day", "hour", "albedo", "emissivity"}
    units = ("_c", "_pct", "_hpa", "_pa", "_m_s", "_deg", "_w_m2", "_m", "_mm", "_g_kg")
    unsuffixed = [
        name for name in DEFAULT_HEADER if name not in dimensionless and not name.endswith(units)
    ]

    assert unsuffixed == []


def test_every_catalogue_column_appears_exactly_once_in_the_header():
    names = [column.name for column in OPERATIONAL_CATALOG]

    assert len(names) == len(set(names))


def test_a_row_is_rendered_against_the_header_it_is_given_not_the_catalogue_order():
    row = format_row(
        pd.Timestamp("2026-08-08 07:00:00"),
        {"t2_c": 25.5, "q2_g_kg": 16.0},
        ["hour", "q2_g_kg", "year", "t2_c"],
    )

    assert row == "7,16.0000,2026,25.5000"


def test_a_header_field_this_run_cannot_fill_is_written_as_no_value():
    row = format_row(
        pd.Timestamp("2026-08-08 07:00:00"), {"t2_c": 25.5}, ["t2_c", "swdown_farms_w_m2"]
    )

    assert row == "25.5000,nan"


def test_a_variable_the_wrfout_lost_empties_its_column_without_shifting_the_others(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, omit=("SWDDIF",))

    frame = _block(path, hours=4).frame

    assert frame["swddif_w_m2"].isna().all()
    assert frame["swddir_w_m2"].notna().all()
    assert list(frame.columns) == list(DEFAULT_HEADER[4:])


def test_hour_21_local_is_the_run_initialisation_at_00_utc(wrfout):
    index = _block(wrfout, hours=4).frame.index

    assert list(index.hour) == [21, 22, 23, 0]
    assert index[0] == pd.Timestamp("2026-08-07 21:00:00")


def test_a_window_longer_than_the_file_is_refused(wrfout):
    with pytest.raises(ValueError, match="holds 4 steps"):
        _block(wrfout, hours=24)


def test_a_non_hourly_output_interval_is_refused(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=2, stamps=["2026-08-08_00:00:00", "2026-08-08_00:30:00"])

    with pytest.raises(ValueError, match="from the hour"):
        _block(path, hours=2)


def test_a_gap_in_the_window_is_refused(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(
        path,
        n_times=3,
        stamps=["2026-08-08_00:00:00", "2026-08-08_01:00:00", "2026-08-08_03:00:00"],
    )

    with pytest.raises(ValueError, match="not contiguous hourly"):
        _block(path, hours=3)


def test_a_few_seconds_of_model_clock_drift_still_stamps_a_whole_hour(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=2, stamps=["2026-08-08_00:00:00", "2026-08-08_01:00:02"])

    index = _block(path, hours=2).frame.index

    assert list(index.minute) == [0, 0]
    assert list(index.second) == [0, 0]


def test_a_coordinate_outside_the_domain_is_refused(wrfout):
    with WRFDataset(wrfout) as ds, pytest.raises(ValueError, match="outside"):
        extract_operational_block(ds, -20.0, -38.5, hours=2)


def test_the_serving_cell_is_the_nearest_centre_and_its_distance_is_reported(wrfout):
    with WRFDataset(wrfout) as ds:
        block = extract_operational_block(ds, -12.91, -38.41, hours=2)

    assert (block.latitude, block.longitude) == pytest.approx((-12.9, -38.4))
    assert block.distance_km < 2.0


def test_vapour_pressure_uses_the_mixing_ratio_conversion_wrf_q2_calls_for(wrfout):
    # From the float32 the file actually stores, not the literal it was built
    # from: the assertion is about the FORMULA, not about netCDF rounding.
    mixing = float(np.float32(SURFACE_FIELDS["Q2"]))
    pressure = float(np.float32(SURFACE_FIELDS["PSFC"])) / 100.0
    expected = mixing * pressure / (0.622 + mixing)

    assert _block(wrfout, hours=1).frame["e_hpa"].iloc[0] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ("u10", "v10", "bearing"),
    [(0.0, -3.0, 0.0), (-3.0, 0.0, 90.0), (0.0, 3.0, 180.0), (3.0, 0.0, 270.0)],
)
def test_wind_direction_is_the_bearing_the_wind_blows_from(tmp_path, u10, v10, bearing):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=1, extra={"U10": u10, "V10": v10})

    frame = _block(path, hours=1).frame

    assert frame["wind_dir_deg"].iloc[0] == pytest.approx(bearing)
    assert frame["wind_speed_m_s"].iloc[0] == pytest.approx(3.0)


def test_a_calm_publishes_no_bearing_rather_than_a_confident_one(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=1, extra={"U10": 0.0, "V10": 0.0})

    assert np.isnan(_block(path, hours=1).frame["wind_dir_deg"].iloc[0])


def test_relative_humidity_is_published_unclipped_above_saturation(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=1)
    with netCDF4.Dataset(path, "a") as ds:
        ds.variables["Q2"][:] = 0.030
        ds.variables["T2"][:] = 295.0

    assert _block(path, hours=1).frame["rh_pct"].iloc[0] > 100.0


def test_albedo_and_emissivity_are_published_as_the_dimensionless_model_values(wrfout):
    frame = _block(wrfout, hours=1).frame

    assert frame["albedo"].iloc[0] == pytest.approx(0.15)
    assert frame["emissivity"].iloc[0] == pytest.approx(0.88)
    assert frame["swup_w_m2"].iloc[0] == pytest.approx(0.15 * 800.0)


def test_screen_level_emission_is_evaluated_in_kelvin(wrfout):
    emission = _block(wrfout, hours=1).frame["lwup_air_w_m2"].iloc[0]

    assert emission == pytest.approx(0.88 * STEFAN_BOLTZMANN * 300.0**4)


def test_the_bottom_of_atmosphere_diagnostics_win_over_the_reconstruction(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=1, extra={"LWDNB": 411.0, "LWUPB": 466.0})

    frame = _block(path, hours=1).frame

    assert frame["lwdnb_w_m2"].iloc[0] == pytest.approx(411.0)
    assert frame["lwup_w_m2"].iloc[0] == pytest.approx(466.0)
    assert frame["glw_w_m2"].iloc[0] == pytest.approx(400.0)


def test_without_them_the_downward_flux_falls_back_to_glw(wrfout):
    frame = _block(wrfout, hours=1).frame

    assert frame["lwdnb_w_m2"].iloc[0] == frame["glw_w_m2"].iloc[0]


def test_the_farms_columns_are_published_as_no_value(wrfout):
    frame = _block(wrfout, hours=4).frame

    farms = ["swdown_farms_w_m2", "swddif_farms_w_m2", "swddir_farms_w_m2"]
    assert frame[farms].isna().all().all()


def test_precipitation_is_the_hourly_increment_not_the_accumulation(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=4, rain=[0.0, 2.0, 2.0, 5.0])

    precip = _block(path, hours=4).frame["precip_mm"]

    assert list(precip[1:]) == pytest.approx([2.0, 0.0, 3.0])


def test_the_first_step_of_a_run_has_no_increment_to_publish(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=4, rain=[0.0, 2.0, 2.0, 5.0])

    assert np.isnan(_block(path, hours=4).frame["precip_mm"].iloc[0])


def test_a_window_starting_mid_run_differences_against_its_real_predecessor(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=4, rain=[0.0, 2.0, 2.0, 5.0])

    precip = _block(path, hours=2, start_step=2).frame["precip_mm"]

    assert list(precip) == pytest.approx([0.0, 3.0])


def test_the_step_before_the_first_radiation_call_publishes_no_physics(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=3, cold_start=True)

    frame = _block(path, hours=3).frame

    physics = ["swdown_w_m2", "glw_w_m2", "lwup_w_m2", "hfx_w_m2", "pblh_m", "albedo", "ustar_m_s"]
    assert frame[physics].iloc[0].isna().all()
    assert frame[physics].iloc[1:].notna().all().all()


def test_the_state_the_run_was_initialised_from_survives_the_cold_start(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=3, cold_start=True)

    frame = _block(path, hours=3).frame

    dynamics = ["t2_c", "rh_pct", "psfc_hpa", "e_hpa", "q2_g_kg", "wind_speed_m_s", "sst_c"]
    assert frame[dynamics].iloc[0].notna().all()


def test_a_continuation_run_keeps_its_radiation_and_is_not_blanked(wrfout):
    assert _block(wrfout, hours=4).frame["glw_w_m2"].notna().all()


def test_the_default_station_is_the_tower_and_names_its_own_file():
    assert DEFAULT_STATION.filename == "labmim_series_operacional.dat"


def test_a_station_token_is_name_lat_lon():
    station = parse_station("ilheus:-14.7889:-39.0339")

    assert (station.name, station.latitude, station.longitude) == ("ilheus", -14.7889, -39.0339)


@pytest.mark.parametrize(
    ("token", "message"),
    [
        ("ilheus:-14.7889", "expected name:lat:lon"),
        ("ilheus:-14.7889:-39.0339:3", "expected name:lat:lon"),
        ("ilheus:x:-39.0", "could not convert"),
        ("../etc:1:2", "file name"),
    ],
)
def test_a_malformed_station_token_is_refused(token, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_station(token)


def test_a_station_name_may_not_escape_the_output_directory():
    with pytest.raises(ValueError, match="file name"):
        Station(name="../../etc/passwd", latitude=-13.0, longitude=-38.5)


def test_a_coordinate_off_the_globe_is_refused():
    with pytest.raises(ValueError, match="latitude/longitude"):
        Station(name="nowhere", latitude=-913.0, longitude=-38.5)


def test_a_station_list_reads_with_or_without_a_header(tmp_path):
    path = tmp_path / "estacoes.csv"
    path.write_text("name,lat,lon\nlabmim,-13.0055,-38.5089\nilheus,-14.7889,-39.0339\n")

    stations = read_stations(path)

    assert [station.name for station in stations] == ["labmim", "ilheus"]


def test_a_station_listed_twice_is_refused_because_both_would_write_one_file(tmp_path):
    path = tmp_path / "estacoes.csv"
    path.write_text("labmim,-13.0,-38.5\nlabmim,-14.0,-39.0\n")

    with pytest.raises(ValueError, match="more than once"):
        read_stations(path)


def test_a_malformed_row_stops_the_list_rather_than_being_skipped(tmp_path):
    path = tmp_path / "estacoes.csv"
    path.write_text("labmim,-13.0,-38.5\nilheus,-14.7889\n")

    with pytest.raises(ValueError, match="expected name,lat,lon"):
        read_stations(path)


def test_a_station_is_served_by_the_finest_domain_that_contains_it(nested_run):
    inside = Station(name="labmim", latitude=-13.0, longitude=-38.5)
    outside = Station(name="sul", latitude=-15.5, longitude=-41.0)

    assignments, uncovered = assign_domains([inside, outside], nested_run)

    assert not uncovered
    served = {a.station.name: a.dx_m for a in assignments}
    assert served == {"labmim": 1000.0, "sul": 27000.0}


def test_a_station_no_domain_reaches_is_reported_rather_than_snapped_to_an_edge(nested_run):
    far = Station(name="manaus", latitude=-3.1, longitude=-60.0)

    assignments, uncovered = assign_domains([far], nested_run)

    assert not assignments
    assert [station.name for station in uncovered] == ["manaus"]


def test_appending_to_a_file_still_on_the_v1_schema_is_refused(tmp_path, wrfout):
    target = tmp_path / "serie.dat"
    _v1_file(target, [_v1_row()])
    before = target.read_bytes()

    with pytest.raises(ValueError, match="still on the v1 schema"):
        append_block(target, _block(wrfout, hours=2).frame, read_header(target))

    assert target.read_bytes() == before


def test_the_time_fields_alone_do_not_make_a_header_look_like_v1():
    assert legacy_spellings(DEFAULT_HEADER) == ()
    assert "T" in legacy_spellings(V1_COLUMNS)


def test_cli_refuses_to_append_before_the_record_has_been_migrated(tmp_path, wrfout):
    target = tmp_path / "series"
    target.mkdir()
    record = target / "labmim_series_operacional.dat"
    _v1_file(record, [_v1_row()])
    before = record.read_bytes()

    result = CliRunner().invoke(app, ["run", "-d", str(wrfout), "-o", str(target), "--hours", "2"])

    assert result.exit_code != 0
    assert "still on the v1 schema" in result.output
    assert record.read_bytes() == before


def test_a_v1_spelling_is_refused_as_a_raw_column_name(wrfout):
    with WRFDataset(wrfout) as ds, pytest.raises(KeyError, match="v1 spelling"):
        build_columns(["T"], ds.has_variable)


def test_a_raw_column_is_blanked_at_the_cold_start_rather_than_guessed(tmp_path):
    path = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    _write_wrfout(path, n_times=3, cold_start=True)

    with WRFDataset(path) as ds:
        columns = build_columns(["TSK"], ds.has_variable)
        frame = extract_operational_block(ds, -13.0, -38.5, hours=3, columns=columns).frame

    assert np.isnan(frame["TSK"].iloc[0])
    assert frame["TSK"].iloc[1:].notna().all()


def test_migration_refuses_a_truncated_row_instead_of_padding_it(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [",".join(_v1_row().split(",")[:20])])

    with pytest.raises(ValueError, match="20 fields"):
        migrate_to_v2(path)


def test_a_station_list_whose_header_follows_a_comment_is_still_a_header(tmp_path):
    path = tmp_path / "estacoes.csv"
    path.write_text("# estacoes do litoral\n\nname,lat,lon\nlabmim,-13.0055,-38.5089\n")

    assert [station.name for station in read_stations(path)] == ["labmim"]


def test_append_creates_the_file_with_the_header_it_was_given(tmp_path, wrfout):
    target = tmp_path / "novo.dat"

    written = append_block(target, _block(wrfout, hours=2).frame, DEFAULT_HEADER)

    assert written == 2
    assert read_header(target) == DEFAULT_HEADER
    assert len(target.read_text().splitlines()) == 3


def test_a_block_already_in_the_file_is_skipped(tmp_path, wrfout):
    target = tmp_path / "novo.dat"
    frame = _block(wrfout, hours=2).frame
    append_block(target, frame, DEFAULT_HEADER)

    assert append_block(target, frame, DEFAULT_HEADER) == 0
    assert len(target.read_text().splitlines()) == 3


def test_force_appends_a_block_that_is_already_in_the_file(tmp_path, wrfout):
    target = tmp_path / "novo.dat"
    frame = _block(wrfout, hours=2).frame
    append_block(target, frame, DEFAULT_HEADER)

    assert append_block(target, frame, DEFAULT_HEADER, force=True) == 2
    assert len(target.read_text().splitlines()) == 5


def test_a_file_left_without_a_trailing_newline_does_not_swallow_the_first_row(tmp_path, wrfout):
    target = tmp_path / "novo.dat"
    target.write_text(",".join(DEFAULT_HEADER) + "\n2026,1,1,0" + ",0.0" * 32)

    append_block(target, _block(wrfout, hours=2).frame, DEFAULT_HEADER)

    lines = target.read_text().splitlines()
    assert len(lines) == 4
    assert lines[1].startswith("2026,1,1,0")


def test_extending_the_header_leaves_every_row_already_written_untouched(tmp_path, wrfout):
    target = tmp_path / "novo.dat"
    append_block(target, _block(wrfout, hours=2).frame, DEFAULT_HEADER)
    before = target.read_text().splitlines()[1:]

    header = extend_header(target, ("olr_w_m2",))

    assert header == (*DEFAULT_HEADER, "olr_w_m2")
    assert target.read_text().splitlines()[1:] == before


def test_a_column_added_later_reads_back_as_absent_for_the_older_rows(tmp_path, wrfout):
    target = tmp_path / "novo.dat"
    append_block(target, _block(wrfout, hours=2).frame, DEFAULT_HEADER)
    extend_header(target, ("olr_w_m2",))

    frame = pd.read_csv(target)

    assert frame["olr_w_m2"].isna().all()
    assert frame["t2_c"].notna().all()


def test_an_unknown_column_name_is_refused_rather_than_dropped(wrfout):
    with WRFDataset(wrfout) as ds, pytest.raises(KeyError, match="Espresso"):
        build_columns(["t2_c", "Espresso"], ds.has_variable)


def test_a_wrfout_variable_outside_the_catalogue_becomes_a_column_of_its_own(wrfout):
    with WRFDataset(wrfout) as ds:
        columns = build_columns(["t2_c", "TSK"], ds.has_variable)
        frame = extract_operational_block(ds, -13.0, -38.5, hours=2, columns=columns).frame

    assert list(frame.columns) == ["t2_c", "TSK"]
    assert frame["TSK"].iloc[0] == pytest.approx(305.0)


_V1_TEMPERATURE_C = 25.0
_V1_MIXING_RATIO_G_KG = 16.0
_V1_PRESSURE_HPA = 1013.0
_V1_SHORTWAVE = 800.0
_V1_ALBEDO = 0.15
_V1_EMISSIVITY = 0.88
_KELVIN = 273.15
_REPAIRED = {"albedo", "emissivity", "swup_w_m2", "lwup_air_w_m2", "e_hpa", "rh_pct"}


def _v1_row(cold_start: bool = False, wide: bool = False) -> str:
    """One row exactly as the v1 extraction would have written it."""
    saturation_pa = 611.2 * math.exp(17.67 * _V1_TEMPERATURE_C / (_V1_TEMPERATURE_C + 243.5))
    mixing = _V1_MIXING_RATIO_G_KG / 1000.0
    # The defect: the specific-humidity denominator applied to a mixing ratio.
    vapor_hpa = mixing * _V1_PRESSURE_HPA / (0.622 + 0.378 * mixing)
    broken_albedo = _V1_ALBEDO - _KELVIN
    broken_emissivity = _V1_EMISSIVITY - _KELVIN

    values = dict.fromkeys(V1_COLUMNS, "nan")
    values.update(
        year="2022",
        month="6",
        day="16",
        hour="21",
        T=f"{_V1_TEMPERATURE_C:.4f}",
        ur=f"{100 * vapor_hpa * 100 / saturation_pa:.4f}",
        pressure=f"{_V1_PRESSURE_HPA:.4f}",
        e=f"{vapor_hpa:.4f}",
        es=f"{saturation_pa:.4f}",
        q=f"{_V1_MIXING_RATIO_G_KG:.4f}",
        Swdw=f"{_V1_SHORTWAVE:.4f}",
        Lwdw_glw="0.0000" if cold_start else "400.0000",
        H="0.0000" if cold_start else "250.0000",
        PBLH="0.0000" if cold_start else "900.0000",
        ALBD=f"{broken_albedo:.4f}",
        EMISS=f"{broken_emissivity:.4f}",
        TSM="20.0000",
    )
    reflected = f"{broken_albedo * _V1_SHORTWAVE:.4f}"
    emission = f"{broken_emissivity * STEFAN_BOLTZMANN * _V1_TEMPERATURE_C**4:.4f}"
    if wide:
        # The pre-2022-10-07 layout: the two derived fluxes at the END of the row.
        return ",".join(values[name] for name in V1_COLUMNS) + "," * 11 + f"{reflected},{emission}"
    values["Swup_calc"] = reflected
    values["Lwup_calc"] = emission
    return ",".join(values[name] for name in V1_COLUMNS)


def _v1_file(path: Path, rows: list[str]) -> None:
    path.write_text(V1_HEADER + "," * 12 + "\n" + "\n".join(rows) + "\n")


def test_the_v1_header_reads_as_thirty_five_columns(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])

    assert read_header(path) == V1_COLUMNS


def test_migration_repairs_the_kelvin_subtraction_on_the_dimensionless_columns(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])

    migrate_to_v2(path)

    frame = pd.read_csv(path)
    assert frame["albedo"].iloc[0] == pytest.approx(_V1_ALBEDO, abs=1e-4)
    assert frame["emissivity"].iloc[0] == pytest.approx(_V1_EMISSIVITY, abs=1e-4)


def test_migration_repairs_the_two_fluxes_the_broken_constants_propagated_into(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])

    migrate_to_v2(path)

    frame = pd.read_csv(path)
    assert frame["swup_w_m2"].iloc[0] == pytest.approx(_V1_ALBEDO * _V1_SHORTWAVE, abs=1e-3)
    expected = _V1_EMISSIVITY * STEFAN_BOLTZMANN * (_V1_TEMPERATURE_C + _KELVIN) ** 4
    assert frame["lwup_air_w_m2"].iloc[0] == pytest.approx(expected, abs=1e-3)


def test_migration_repairs_the_vapour_pressure_and_the_humidity_that_followed_it(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])

    migrate_to_v2(path)

    frame = pd.read_csv(path)
    mixing = _V1_MIXING_RATIO_G_KG / 1000.0
    expected = mixing * _V1_PRESSURE_HPA / (0.622 + mixing)
    assert frame["e_hpa"].iloc[0] == pytest.approx(expected, abs=1e-4)
    assert frame["rh_pct"].iloc[0] == pytest.approx(
        100 * expected * 100 / frame["es_pa"].iloc[0], abs=1e-3
    )


def test_migration_no_values_the_physics_of_the_cold_start_step(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row(cold_start=True)])

    report = migrate_to_v2(path)

    frame = pd.read_csv(path)
    assert frame[["glw_w_m2", "hfx_w_m2", "pblh_m", "swdown_w_m2", "albedo"]].iloc[0].isna().all()
    assert frame[["t2_c", "psfc_hpa", "q2_g_kg", "sst_c"]].iloc[0].notna().all()
    assert report.blanked["pblh_m"] == 1


def test_migration_moves_the_trailing_values_into_their_columns_before_repairing(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row(wide=True)])

    report = migrate_to_v2(path)

    assert report.recovered == 2
    frame = pd.read_csv(path)
    assert frame["swup_w_m2"].iloc[0] == pytest.approx(_V1_ALBEDO * _V1_SHORTWAVE, abs=1e-3)


def test_migration_leaves_the_file_uniformly_as_wide_as_its_header(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row(), _v1_row(wide=True)])
    migrate_to_v2(path)

    widths = {len(line.split(",")) for line in path.read_text().splitlines()}

    assert widths == {len(DEFAULT_HEADER)}


def test_migration_passes_untouched_cells_through_verbatim(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])
    original = dict(zip(V1_COLUMNS, path.read_text().splitlines()[1].split(","), strict=False))

    migrate_to_v2(path)

    migrated = dict(zip(DEFAULT_HEADER, path.read_text().splitlines()[1].split(","), strict=True))
    untouched = {new: migrated[new] for _, new in V1_TO_V2 if new not in _REPAIRED}
    assert untouched == {new: original[old] for old, new in V1_TO_V2 if new not in _REPAIRED}


def test_migration_keeps_a_copy_of_the_file_it_rewrote(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])
    before = path.read_text()

    migrate_to_v2(path)

    assert (tmp_path / "serie.dat.bak").read_text() == before


def test_migrating_an_already_migrated_file_is_refused(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row()])
    migrate_to_v2(path)

    with pytest.raises(ValueError, match="already on the v2 schema"):
        migrate_to_v2(path)


def test_migration_refuses_a_header_that_is_neither_schema(tmp_path):
    path = tmp_path / "serie.dat"
    path.write_text("hour,year,month,day\n0,2022,6,16\n")

    with pytest.raises(ValueError, match="neither v1 nor v2"):
        migrate_to_v2(path)


def test_migration_refuses_a_row_of_an_unknown_width(tmp_path):
    path = tmp_path / "serie.dat"
    _v1_file(path, [",".join(["0"] * 40)])

    with pytest.raises(ValueError, match="40 fields"):
        migrate_to_v2(path)


def test_migration_refuses_a_flux_the_v1_formula_does_not_explain(tmp_path):
    path = tmp_path / "serie.dat"
    row = _v1_row().split(",")
    row[V1_COLUMNS.index("Lwup_calc")] = "-500.0000"
    _v1_file(path, [",".join(row)])

    with pytest.raises(ValueError, match="implies an emissivity"):
        migrate_to_v2(path)


def test_a_header_naming_the_same_column_twice_is_refused(tmp_path):
    path = tmp_path / "serie.dat"
    path.write_text("year,month,day,hour,T,T\n")

    with pytest.raises(ValueError, match="more than once"):
        read_header(path)


def test_a_frame_read_from_a_v1_file_takes_the_v2_names():
    frame = pd.DataFrame({"T": [25.0], "ur": [80.0], "Swdw": [700.0]})

    renamed = rename_v1_columns(frame)

    assert list(renamed.columns) == ["t2_c", "rh_pct", "swdown_w_m2"]


def test_a_frame_already_on_v2_is_returned_untouched():
    frame = pd.DataFrame({"t2_c": [25.0]})

    assert rename_v1_columns(frame) is frame


def test_a_migrated_file_reads_back_through_the_climatology_reader(tmp_path, wrfout):
    from micrometeorology.cli.export_climatology import read_wrf_series

    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row(cold_start=True)])
    migrate_to_v2(path)
    append_block(path, _block(wrfout, hours=4).frame, read_header(path))

    frame = read_wrf_series(path)

    assert "precip_mm" in frame.columns
    assert not [name for name in frame.columns if name.startswith("Unnamed")]
    assert frame.index.is_monotonic_increasing


def test_a_file_still_on_v1_reads_back_under_the_v2_names(tmp_path):
    from micrometeorology.cli.export_climatology import read_wrf_series

    path = tmp_path / "serie.dat"
    _v1_file(path, [_v1_row().replace(",21,", ",10,", 1)])

    frame = read_wrf_series(path)

    assert "t2_c" in frame.columns
    assert "T" not in frame.columns


def test_cli_dry_run_writes_nothing(tmp_path, wrfout):
    target = tmp_path / "series"

    result = CliRunner().invoke(
        app, ["run", "-d", str(wrfout), "-o", str(target), "--hours", "2", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_cli_writes_one_file_per_station_named_after_it(tmp_path, nested_run):
    target = tmp_path / "series"

    result = CliRunner().invoke(
        app,
        [
            "run",
            *_datasets(nested_run),
            "-o",
            str(target),
            "--hours",
            "2",
            "-s",
            "labmim:-13.0:-38.5",
            "-s",
            "sul:-15.5:-41.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in target.iterdir()) == [
        "labmim_series_operacional.dat",
        "sul_series_operacional.dat",
    ]


def test_cli_serves_each_station_from_the_finest_domain_that_reaches_it(tmp_path, nested_run):
    result = CliRunner().invoke(
        app,
        [
            "run",
            *_datasets(nested_run),
            "-o",
            str(tmp_path / "series"),
            "--hours",
            "2",
            "-s",
            "labmim:-13.0:-38.5",
            "-s",
            "sul:-15.5:-41.0",
        ],
    )

    assert "labmim: wrfout_d04" in result.output
    assert "sul: wrfout_d01" in result.output


def test_cli_defaults_to_the_tower_when_no_station_is_named(tmp_path, wrfout):
    target = tmp_path / "series"

    result = CliRunner().invoke(app, ["run", "-d", str(wrfout), "-o", str(target), "--hours", "2"])

    assert result.exit_code == 0, result.output
    assert (target / "labmim_series_operacional.dat").exists()


def test_cli_exits_non_zero_when_a_station_falls_outside_every_domain(tmp_path, wrfout):
    result = CliRunner().invoke(
        app,
        [
            "run",
            "-d",
            str(wrfout),
            "-o",
            str(tmp_path / "series"),
            "--hours",
            "2",
            "-s",
            "manaus:-3.1:-60.0",
        ],
    )

    assert result.exit_code == 1
    assert "fora de todos os dominios" in result.output


def test_cli_still_writes_the_stations_it_can_serve(tmp_path, wrfout):
    target = tmp_path / "series"

    CliRunner().invoke(
        app,
        [
            "run",
            "-d",
            str(wrfout),
            "-o",
            str(target),
            "--hours",
            "2",
            "-s",
            "labmim:-13.0:-38.5",
            "-s",
            "manaus:-3.1:-60.0",
        ],
    )

    assert (target / "labmim_series_operacional.dat").exists()
    assert not (target / "manaus_series_operacional.dat").exists()


def test_cli_reads_its_stations_from_a_list(tmp_path, nested_run):
    listing = tmp_path / "estacoes.csv"
    listing.write_text("name,lat,lon\nlabmim,-13.0,-38.5\nsul,-15.5,-41.0\n")
    target = tmp_path / "series"

    result = CliRunner().invoke(
        app,
        [
            "run",
            *_datasets(nested_run),
            "-o",
            str(target),
            "--hours",
            "2",
            "--stations",
            str(listing),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(target.iterdir())) == 2


def test_cli_refuses_a_new_column_when_the_header_may_not_grow(tmp_path, wrfout):
    target = tmp_path / "series"
    target.mkdir()
    append_block(
        target / "labmim_series_operacional.dat", _block(wrfout, hours=2).frame, DEFAULT_HEADER
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "-d",
            str(wrfout),
            "-o",
            str(target),
            "--hours",
            "2",
            "-v",
            "t2_c,TSK",
            "--no-extend-header",
        ],
    )

    assert result.exit_code != 0
    assert "TSK" in result.output


def test_cli_reports_an_unreadable_wrfout_by_name(tmp_path):
    broken = tmp_path / "wrfout_d04_2026-08-08_00:00:00"
    broken.write_bytes(b"not netcdf")

    result = CliRunner().invoke(app, ["run", "-d", str(broken), "-o", str(tmp_path / "series")])

    assert result.exit_code != 0
    assert "could not be read as NetCDF" in result.output


def test_a_partial_last_row_left_by_an_interrupted_append_is_cut_and_written_again(tmp_path):
    """A row cut mid-write still parsed as a stamp on its first four fields, so
    its hour counted as already present and the cron line skipped it forever."""
    target = tmp_path / "novo.dat"
    complete = "2026,8,8,22" + ",0.0" * (len(DEFAULT_HEADER) - 4)
    target.write_text(",".join(DEFAULT_HEADER) + "\n" + complete + "\n2026,8,8,23,25.5")
    frame = pd.DataFrame({DEFAULT_HEADER[4]: [25.5]}, index=pd.DatetimeIndex(["2026-08-08 23:00"]))

    written = append_block(target, frame, DEFAULT_HEADER)

    lines = target.read_text().splitlines()
    assert written == 1
    assert len(lines) == 3
    assert lines[2].startswith("2026,8,8,23,25.5")
