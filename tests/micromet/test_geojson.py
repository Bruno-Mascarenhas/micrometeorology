"""Tests for the GeoJSON / JSON generation pipeline.

Covers:
- ``write_grid_geojson_stream`` → byte/structure identity vs the frozen
  reference oracles in ``tests.micromet._reference``
- ``write_values_json_stream`` → matches the reference in-memory payload
- ``create_wind_vectors_json`` → standalone wind vector file schema
"""

import errno
import inspect
import json
import re
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import pytest

from micrometeorology.wrf import jobs
from micrometeorology.wrf.geojson import (
    create_wind_vectors_json,
    write_grid_compact_json_stream,
    write_grid_geojson_stream,
    write_values_json_stream,
)
from tests.micromet._reference import (
    create_grid_geojson,
    create_values_json,
    reference_write_grid_geojson_stream,
)


@pytest.fixture
def sample_grid() -> tuple[np.ndarray, np.ndarray]:
    """Small 4x5 lon/lat grid for testing."""
    ny, nx = 4, 5
    lon = np.linspace(-40, -38, nx)[np.newaxis, :].repeat(ny, axis=0)
    lat = np.linspace(-14, -12, ny)[:, np.newaxis].repeat(nx, axis=1)
    return lon, lat


@pytest.fixture
def sample_values_2d() -> np.ndarray:
    """4x5 array with some NaN values."""
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    arr[0, 0] = np.nan
    arr[2, 3] = np.nan
    return arr


@pytest.fixture
def sample_wind_2d() -> tuple[np.ndarray, np.ndarray]:
    """4x5 U/V wind component arrays."""
    rng = np.random.default_rng(42)
    u = rng.uniform(-5, 5, size=(4, 5))
    v = rng.uniform(-5, 5, size=(4, 5))
    return u, v


class TestCreateValuesJson:
    def test_nan_becomes_none(self, tmp_path, sample_values_2d):
        written = write_values_json_stream(
            tmp_path / "values.json", sample_values_2d, 0.0, 20.0, "N/A"
        )
        result = json.loads(Path(written).read_text(encoding="utf-8"))
        assert result["values"][0] is None
        # Index (2,3) = flat index 2*5+3 = 13
        assert result["values"][13] is None

    def test_values_are_rounded_to_2dp(self, tmp_path):
        arr = np.array([[1.23456, 2.789]], dtype=np.float64)
        written = write_values_json_stream(tmp_path / "rounded.json", arr, 0.0, 3.0, "N/A")
        result = json.loads(Path(written).read_text(encoding="utf-8"))
        assert result["values"][0] == 1.23
        assert result["values"][1] == 2.79

    def test_streamed_values_json_matches_in_memory_payload(self, tmp_path, sample_values_2d):
        out = tmp_path / "values.json"
        expected = create_values_json(sample_values_2d, 0.0, 20.0, None)
        write_values_json_stream(
            out,
            sample_values_2d,
            0.0,
            20.0,
            "N/A",
            chunk_size=3,
        )
        with open(out, encoding="utf-8") as f:
            actual = json.load(f)

        assert actual == expected

    def test_jobs_values_writer_uses_streaming_payload(self):
        source = inspect.getsource(jobs._atomic_values_json)

        assert "write_values_json_stream" in source
        assert ".tolist()" not in source

    def test_whole_floats_serialized_as_integers(self, tmp_path):
        """0.0 -> 0 (and -0.0 -> -0) in the serialized text; parsed values
        are unchanged, including across chunk boundaries."""
        arr = np.array([[0.0, -0.0, 2.0, 1.25, np.nan]], dtype=np.float64)
        out = tmp_path / "vals.json"
        write_values_json_stream(out, arr, 0.0, 5.0, "N/A", chunk_size=2)

        text = out.read_text(encoding="utf-8")
        assert '"values":[0,-0,2,1.25,null]' in text
        with open(out, encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed["values"] == [0, 0, 2, 1.25, None]

    def test_int_formatting_does_not_touch_fractional_values(self, tmp_path):
        arr = np.array([[10.05, 100.0, -0.25]], dtype=np.float64)
        out = tmp_path / "vals.json"
        write_values_json_stream(out, arr, 0.0, 5.0, "N/A")
        assert '"values":[10.05,100,-0.25]' in out.read_text(encoding="utf-8")

    @pytest.mark.parametrize("boundary_value", [0.0, -0.0, 7.0, 1.25, np.nan, -1e17, 1e-3], ids=str)
    def test_whole_float_stripping_survives_chunk_boundaries(self, tmp_path, boundary_value):
        """The last element of a chunk is the boundary case: every token type
        must round-trip there, not only mid-chunk."""
        arr = np.array([[3.0, boundary_value, 0.0, -0.0, 2.5]], dtype=np.float64)
        out = tmp_path / "vals.json"
        write_values_json_stream(out, arr, 0.0, 5.0, "N/A", chunk_size=2)

        with open(out, encoding="utf-8") as f:
            parsed = json.load(f)
        expected = [None if np.isnan(v) else v for v in np.round(arr, 2).ravel().tolist()]
        assert parsed["values"] == expected

    def test_values_text_matches_whole_float_regex_reference(self, tmp_path):
        """The compact ``.0``-stripping must match the lookahead-regex oracle
        built below exactly, including adjacent whole floats and nulls."""
        whole_float_re = re.compile(r"(-?\d+)\.0(?=,|$)")
        rng = np.random.default_rng(3)
        fields = [
            np.round(rng.uniform(288.0, 310.0, 1000), 2),
            np.round(np.where(rng.random(1000) < 0.9, 0.0, rng.uniform(0, 30, 1000)), 2),
            np.where(rng.random(1000) < 0.2, np.nan, np.round(rng.normal(0, 1e17, 1000), 2)),
            np.zeros(1000),
            np.full(1000, -0.0),
        ]
        for index, field in enumerate(fields):
            out = tmp_path / f"vals_{index}.json"
            write_values_json_stream(out, field, -1.0, 1.0, "N/A", chunk_size=97)
            text = out.read_text(encoding="utf-8").split('"values":[', 1)[1][:-2]

            chunk = np.round(field.astype(np.float64, copy=False), 2)
            values: list[float | None] = chunk.tolist()
            for idx in np.flatnonzero(~np.isfinite(chunk)):
                values[idx] = None
            reference = ",".join(
                whole_float_re.sub(
                    r"\1", json.dumps(values[s : s + 97], separators=(",", ":"))[1:-1]
                )
                for s in range(0, len(values), 97)
            )
            assert text == reference


class TestPublishedGridsAreWrittenAtomically:
    """A failed grid write must leave the file the site is serving untouched."""

    @pytest.mark.parametrize(
        ("writer", "name"),
        [
            (write_grid_geojson_stream, "grid.geojson"),
            (write_grid_compact_json_stream, "grid.grid.json"),
        ],
    )
    def test_a_write_that_dies_partway_leaves_the_previous_grid_in_place(
        self, tmp_path, sample_grid, monkeypatch, writer, name
    ):
        lon, lat = sample_grid
        published = tmp_path / name
        writer(published, lon, lat, 1000.0, 2000.0)
        intact = published.read_bytes()

        def _die(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr("micrometeorology.wrf.geojson.json.dump", _die)
        with pytest.raises(OSError, match="No space left on device"):
            writer(published, lon, lat, 1000.0, 2000.0)

        assert published.read_bytes() == intact

    @pytest.mark.parametrize(
        ("writer", "name"),
        [
            (write_grid_geojson_stream, "grid.geojson"),
            (write_grid_compact_json_stream, "grid.grid.json"),
        ],
    )
    def test_a_failed_write_leaves_no_temporary_file_behind(
        self, tmp_path, sample_grid, monkeypatch, writer, name
    ):
        lon, lat = sample_grid

        def _die(*_args, **_kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr("micrometeorology.wrf.geojson.json.dump", _die)
        with pytest.raises(OSError, match="No space left on device"):
            writer(tmp_path / name, lon, lat, 1000.0, 2000.0)

        assert list(tmp_path.iterdir()) == []


class TestWriteGridCompactJsonStream:
    # 7-decimal vs 10-decimal rounding of the SAME corner value can differ by
    # at most 0.5e-7 + 0.5e-10.
    ROUNDING_TOL = 5.05e-8

    def _corner_sets(self, feature):
        ring = feature["geometry"]["coordinates"][0]
        return sorted({p[0] for p in ring}), sorted({p[1] for p in ring})

    def test_separable_grid_uses_edges_format_and_matches_geojson(self, tmp_path, sample_grid):
        lon, lat = sample_grid
        geo = tmp_path / "grid.geojson"
        compact = tmp_path / "grid.grid.json"
        write_grid_geojson_stream(geo, lon, lat, 1000.0, 2000.0)
        write_grid_compact_json_stream(compact, lon, lat, 1000.0, 2000.0)

        with open(compact, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["format"] == "grid-edges-v1"
        assert payload["shape"] == [4, 5]
        assert payload["metadata"] == {"resolucao_m": [1000.0, 2000.0]}
        assert len(payload["lon_edges"]) == 5 + 1
        assert len(payload["lat_edges"]) == 4 + 1

        with open(geo, encoding="utf-8") as f:
            features = json.load(f)["features"]
        n_cols = payload["shape"][1]
        for k, feature in enumerate(features):
            assert feature["properties"]["linear_index"] == k
            i, j = divmod(k, n_cols)
            lons, lats = self._corner_sets(feature)
            edge_lons = sorted([payload["lon_edges"][j], payload["lon_edges"][j + 1]])
            edge_lats = sorted([payload["lat_edges"][i], payload["lat_edges"][i + 1]])
            for got, want in zip(edge_lons, lons, strict=True):
                assert abs(got - want) <= self.ROUNDING_TOL
            for got, want in zip(edge_lats, lats, strict=True):
                assert abs(got - want) <= self.ROUNDING_TOL

    def test_non_separable_grid_falls_back_to_bounds_format(self, tmp_path):
        ny, nx = 3, 4
        lon = np.linspace(-40, -38, nx)[np.newaxis, :].repeat(ny, axis=0)
        lat = np.linspace(-14, -12, ny)[:, np.newaxis].repeat(nx, axis=1)
        lon = lon + np.linspace(0, 0.01, ny)[:, np.newaxis]  # skew: rows differ

        geo = tmp_path / "g.geojson"
        compact = tmp_path / "g.grid.json"
        write_grid_geojson_stream(geo, lon, lat, 500.0, 500.0)
        write_grid_compact_json_stream(compact, lon, lat, 500.0, 500.0)

        with open(compact, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["format"] == "grid-bounds-v1"
        assert payload["shape"] == [ny, nx]
        assert len(payload["bounds"]) == ny * nx

        with open(geo, encoding="utf-8") as f:
            features = json.load(f)["features"]
        for k, feature in enumerate(features):
            lon_left, lat_bottom, lon_right, lat_top = payload["bounds"][k]
            lons, lats = self._corner_sets(feature)
            for got, want in zip(sorted([lon_left, lon_right]), lons, strict=True):
                assert abs(got - want) <= self.ROUNDING_TOL
            for got, want in zip(sorted([lat_bottom, lat_top]), lats, strict=True):
                assert abs(got - want) <= self.ROUNDING_TOL

    def test_float32_masked_grid_matches_production_reader_path(self, tmp_path):
        """MaskedArray float32 input (what WRFDataset.read_grid returns) must take
        the same corner-arithmetic path as a plain float32 array.

        ``np.ma`` arithmetic promotes ``/ 2`` to float64, which moves the
        published edges in their last decimals.
        """
        ny, nx = 4, 3
        lon = np.linspace(-49.6, -49.0, nx, dtype=np.float32)[np.newaxis, :].repeat(ny, axis=0)
        lat = np.linspace(-20.2, -19.6, ny, dtype=np.float32)[:, np.newaxis].repeat(nx, axis=1)

        masked = tmp_path / "masked.grid.json"
        plain = tmp_path / "plain.grid.json"
        write_grid_compact_json_stream(
            masked, np.ma.MaskedArray(lon), np.ma.MaskedArray(lat), 27000.0, 27000.0
        )
        write_grid_compact_json_stream(plain, lon, lat, 27000.0, 27000.0)

        payload = json.loads(masked.read_text(encoding="utf-8"))
        assert payload == json.loads(plain.read_text(encoding="utf-8"))
        assert payload["format"] == "grid-edges-v1"


class TestCreateWindVectorsJson:
    def test_date_in_metadata(self, sample_wind_2d):
        u, v = sample_wind_2d
        # Naive on purpose (the writer drops tzinfo anyway); pandas keeps it so.
        dt = pd.Timestamp(2024, 3, 15, 9, 0, 0)
        result = create_wind_vectors_json(u, v, dt, downsampling=2)
        assert result["metadata"]["date_time"] == "15/03/2024 09:00:00"

    def test_nan_values_excluded(self):
        """NaN grid cells should be excluded from downsampled output."""
        u = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        v = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float64)
        result = create_wind_vectors_json(u, v, None, downsampling=1)
        # (0,1) is NaN so should be excluded
        assert len(result["downsampled_angles"]) == 3
        assert len(result["downsampled_magnitudes"]) == 3
        assert len(result["downsampled_linear_indices"]) == 3

    # Several seeds because the float32 ULP the note below describes lands on the
    # other side of the 1-decimal rounding for only a couple of cells in a couple
    # of grids: one seed does not exercise the claim.
    @pytest.mark.parametrize("seed", [19, 41, 57, 83, 101, 137])
    @pytest.mark.parametrize("downsampling", [1, 2, 3, 4])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_downsampled_payload_equals_full_grid_then_sample(self, downsampling, dtype, seed):
        """Striding before hypot/arctan2 must be exact, not merely close: the
        payload has to equal the full-grid computation sampled afterwards, for
        strides that do not divide the grid and for NaN cells."""
        rng = np.random.default_rng(seed)
        ny, nx = 11, 7
        u = rng.uniform(-25, 25, size=(ny, nx)).astype(dtype)
        v = rng.uniform(-25, 25, size=(ny, nx)).astype(dtype)
        u[rng.random((ny, nx)) < 0.05] = np.nan
        v[rng.random((ny, nx)) < 0.05] = np.nan

        # In the INPUT dtype and the same operand order the writer uses: in
        # float32 `np.degrees` and `* 180.0 / np.pi` differ by an ULP that lands
        # on the other side of the 1-decimal rounding for a handful of cells.
        angle = np.arctan2(u, v) * 180.0 / np.pi
        angle = np.where(angle < 0, angle + 360.0, angle)
        i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
        i_flat, j_flat = i_idx.ravel(), j_idx.ravel()
        angles = np.round(angle[i_flat, j_flat].astype(np.float64), 1)
        mags = np.round(np.hypot(u, v)[i_flat, j_flat].astype(np.float64), 2)
        valid = ~np.isnan(angles)

        result = create_wind_vectors_json(u, v, None, downsampling=downsampling)
        assert result["downsampled_angles"] == angles[valid].tolist()
        assert result["downsampled_magnitudes"] == mags[valid].tolist()
        assert result["downsampled_linear_indices"] == (i_flat * nx + j_flat)[valid].tolist()


def _non_uniform_float32_grid(ny: int, nx: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Curvilinear float32 grid with negative coords and irregular spacing."""
    rng = np.random.default_rng(seed)
    lon_axis = np.sort(rng.uniform(-41.0, -37.0, nx)).astype(np.float32)
    lat_axis = np.sort(rng.uniform(-15.0, -11.0, ny))[::-1].astype(np.float32)
    lon = np.repeat(lon_axis[np.newaxis, :], ny, axis=0)
    lat = np.repeat(lat_axis[:, np.newaxis], nx, axis=1)
    # Small perturbation so rows/columns are not identical (curvilinear grid).
    lon = lon + rng.uniform(-0.01, 0.01, size=lon.shape).astype(np.float32)
    lat = lat + rng.uniform(-0.01, 0.01, size=lat.shape).astype(np.float32)
    assert lon.dtype == np.float32
    assert lat.dtype == np.float32
    return lon, lat


class TestGridGeoJsonStreamByteIdentity:
    """The vectorized writer must produce byte-identical files to the frozen
    per-feature loop in ``tests.micromet._reference``.

    Performance note (no timing assertion, CI-robust): the vectorized writer
    renders a 99x99 grid in ~10-20 ms vs ~780 ms for the per-feature
    ``_grid_cell_feature`` + ``json.dump`` reference loop.
    """

    def _assert_stream_bytes_match_reference(
        self,
        tmp_path: Path,
        lon: np.ndarray,
        lat: np.ndarray,
    ) -> bytes:
        ref_path = tmp_path / "reference.geojson"
        new_path = tmp_path / "vectorized.geojson"
        reference_write_grid_geojson_stream(ref_path, lon, lat, 3000.0, 3000.0)
        write_grid_geojson_stream(new_path, lon, lat, 3000.0, 3000.0)
        ref_bytes = ref_path.read_bytes()
        new_bytes = new_path.read_bytes()
        assert new_bytes == ref_bytes
        return new_bytes

    def test_bytes_identical_4x5_float32_non_uniform(self, tmp_path):
        lon, lat = _non_uniform_float32_grid(4, 5, seed=1)
        self._assert_stream_bytes_match_reference(tmp_path, lon, lat)

    def test_bytes_identical_7x3_float32_non_uniform(self, tmp_path):
        lon, lat = _non_uniform_float32_grid(7, 3, seed=2)
        self._assert_stream_bytes_match_reference(tmp_path, lon, lat)

    def test_bytes_identical_2x2_minimal_grid(self, tmp_path):
        """2x2 grid: every cell hits only the edge formulas."""
        lon = np.array([[-40.5, -38.25], [-40.4, -38.15]], dtype=np.float32)
        lat = np.array([[-12.1, -12.2], [-13.9, -14.05]], dtype=np.float32)
        self._assert_stream_bytes_match_reference(tmp_path, lon, lat)

    def test_bytes_identical_99x99_dense_random_float32(self, tmp_path):
        """Dense random float32 grid (the fallback for the round-tie case).

        A true builtin-round vs np.round tie at the 10th decimal is impossible
        for float32 inputs: any float32 value at geographic magnitude is
        m * 2**e with m < 2**24, so v * 1e10 = m * 5**10 * 2**(e+10) has at
        most ~48 significant bits and is EXACT in float64 — np.round and
        builtin round then agree everywhere (verified empirically over 8e8
        random float32 samples). Hence the spec fallback: byte-equality on a
        dense random 99x99 float32 grid.
        """
        rng = np.random.default_rng(99)
        lon = rng.uniform(-45.0, -35.0, size=(99, 99)).astype(np.float32)
        lat = rng.uniform(-16.0, -10.0, size=(99, 99)).astype(np.float32)
        self._assert_stream_bytes_match_reference(tmp_path, lon, lat)

    def test_bytes_identical_masked_array_float32(self, tmp_path):
        """WRF readers return float32 MaskedArrays (mask all False).

        np.ma arithmetic promotes ``/ 2`` to float64, unlike the per-element
        float32 scalar path, so the corner math must not run on the MaskedArray
        or edge cells drift in the 6th decimal.
        """
        lon, lat = _non_uniform_float32_grid(6, 4, seed=3)
        lon_ma = np.ma.MaskedArray(lon, mask=False)
        lat_ma = np.ma.MaskedArray(lat, mask=False)
        self._assert_stream_bytes_match_reference(tmp_path, lon_ma, lat_ma)

    def test_bytes_identical_float64_round_tie_grid(self, tmp_path):
        """float64 grid pinned at values where round() and np.round disagree.

        For float64 inputs v * 1e10 is inexact, so np.round(v, 10) can land on
        an exact .5 and round half-to-even while builtin round(v, 10) rounds
        the true decimal expansion correctly. A constant grid keeps every
        corner exactly at the tie value ((t + t) / 2 == t, t - (t - t) / 2 == t).
        """
        tie_lat = -14.000000000050001
        tie_lon = -38.000000000050001
        # The trap must be real: builtin round and np.round have to disagree here.
        assert round(tie_lat, 10) != float(np.round(tie_lat, 10))
        assert round(tie_lon, 10) != float(np.round(tie_lon, 10))

        lat = np.full((3, 4), tie_lat, dtype=np.float64)
        lon = np.full((3, 4), tie_lon, dtype=np.float64)
        data = self._assert_stream_bytes_match_reference(tmp_path, lon, lat)
        # Builtin-round digits must appear in the output (np.round would
        # have written -14.0 / -38.0 instead).
        assert b"-14.0000000001" in data
        assert b"-38.0000000001" in data


def test_grid_geojson_stream_matches_reference_dict(tmp_path, sample_grid):
    lon, lat = sample_grid
    expected = create_grid_geojson(lon, lat, 3000.0, 3000.0, "")
    out = tmp_path / "D01.geojson"
    write_grid_geojson_stream(out, lon, lat, 3000.0, 3000.0)
    with open(out, encoding="utf-8") as f:
        actual = json.load(f)

    assert actual == expected


GRID_WRITERS = [write_grid_geojson_stream, write_grid_compact_json_stream]
NON_FINITE_VALUES = [np.nan, np.inf, -np.inf]


def _raise_on_non_finite_token(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token: {token}")


def _parse_as_a_browser_would(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_raise_on_non_finite_token)


@pytest.mark.parametrize("writer", GRID_WRITERS, ids=lambda w: w.__name__)
@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=str)
def test_grid_writers_refuse_a_non_finite_longitude(tmp_path, sample_grid, writer, value):
    """A published file must never carry a bare NaN/Infinity token, which every
    site consumer's strict JSON parser rejects."""
    lon, lat = sample_grid
    lon = lon.copy()
    lon[1, 2] = value

    with pytest.raises(ValueError, match=r"(?i)non-finite"):
        writer(tmp_path / "grid.json", lon, lat, 1000.0, 2000.0)


@pytest.mark.parametrize("writer", GRID_WRITERS, ids=lambda w: w.__name__)
@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=str)
def test_grid_writers_refuse_a_non_finite_latitude(tmp_path, sample_grid, writer, value):
    lon, lat = sample_grid
    lat = lat.copy()
    lat[2, 0] = value

    with pytest.raises(ValueError, match=r"(?i)non-finite"):
        writer(tmp_path / "grid.json", lon, lat, 1000.0, 2000.0)


@pytest.mark.parametrize("writer", GRID_WRITERS, ids=lambda w: w.__name__)
@pytest.mark.parametrize("value", NON_FINITE_VALUES, ids=str)
def test_grid_writers_refuse_a_non_finite_resolution(tmp_path, sample_grid, writer, value):
    lon, lat = sample_grid

    with pytest.raises(ValueError, match=r"(?i)non-finite"):
        writer(tmp_path / "grid.json", lon, lat, value, 2000.0)


@pytest.mark.parametrize("writer", GRID_WRITERS, ids=lambda w: w.__name__)
def test_grid_writers_publish_nothing_when_they_refuse(tmp_path, sample_grid, writer):
    lon, lat = sample_grid
    lat = lat.copy()
    lat[0, 0] = np.nan
    out = tmp_path / "grid.json"

    with pytest.raises(ValueError, match=r"(?i)non-finite"):
        writer(out, lon, lat, 1000.0, 2000.0)

    assert not out.exists()


def test_masked_grid_with_a_non_finite_fill_is_refused(tmp_path, sample_grid):
    """A WRF MaskedArray hides the poisoned cell from ``isfinite`` on the mask,
    but the corner arithmetic reads the underlying data all the same."""
    lon, lat = sample_grid
    poisoned = lon.copy()
    poisoned[1, 1] = np.nan
    masked_lon = np.ma.MaskedArray(poisoned, mask=(np.arange(lon.size) == 6).reshape(lon.shape))

    with pytest.raises(ValueError, match=r"(?i)non-finite"):
        write_grid_geojson_stream(tmp_path / "grid.geojson", masked_lon, lat, 1000.0, 2000.0)


def test_values_writer_publishes_a_masked_cell_as_null(tmp_path):
    frame = np.ma.masked_array(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        mask=[[False, True], [False, False]],
    )
    out = tmp_path / "masked.json"

    write_values_json_stream(out, frame, 0.0, 5.0, "N/A")

    with open(out, encoding="utf-8") as f:
        assert json.load(f)["values"] == [1, None, 3, 4]


@pytest.mark.parametrize("value", [np.inf, -np.inf], ids=str)
def test_values_writer_publishes_an_infinite_value_as_null(tmp_path, value):
    out = tmp_path / "infinite.json"

    write_values_json_stream(out, np.array([[1.0, value]]), 0.0, 5.0, "N/A")

    with open(out, encoding="utf-8") as f:
        assert json.load(f)["values"] == [1, None]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_values_writer_refuses_a_non_positive_chunk_size(tmp_path, sample_values_2d, chunk_size):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        write_values_json_stream(
            tmp_path / "chunked.json",
            sample_values_2d,
            0.0,
            20.0,
            "N/A",
            chunk_size=chunk_size,
        )


def test_wind_vectors_refuse_components_of_different_shapes():
    with pytest.raises(ValueError, match="wind vector shapes differ"):
        create_wind_vectors_json(np.zeros((2, 3)), np.zeros((3, 2)), None)


@pytest.mark.parametrize("writer", GRID_WRITERS, ids=lambda w: w.__name__)
def test_grid_writers_refuse_a_grid_thinner_than_two_cells(tmp_path, writer):
    """The corner arithmetic takes each edge cell's outer half-width from its
    neighbour, which a single-row grid has none of."""
    lon = np.linspace(-40.0, -38.0, 5)[np.newaxis, :]
    lat = np.full((1, 5), -13.0)

    with pytest.raises(ValueError, match="at least a 2x2 grid"):
        writer(tmp_path / "thin.json", lon, lat, 1000.0, 1000.0)


def test_grid_geojson_bytes_parse_under_strict_json(tmp_path, sample_grid):
    lon, lat = sample_grid
    out = write_grid_geojson_stream(tmp_path / "D01.geojson", lon, lat, 1000.0, 2000.0)

    assert len(_parse_as_a_browser_would(out)["features"]) == lon.size


def test_compact_grid_bytes_parse_under_strict_json(tmp_path, sample_grid):
    lon, lat = sample_grid
    out = write_grid_compact_json_stream(tmp_path / "D01.grid.json", lon, lat, 1000.0, 2000.0)

    assert _parse_as_a_browser_would(out)["format"] == "grid-edges-v1"


def test_values_json_bytes_parse_under_strict_json(tmp_path, sample_values_2d):
    out = write_values_json_stream(tmp_path / "D01_t00.json", sample_values_2d, 0.0, 20.0, "N/A")

    assert _parse_as_a_browser_would(out)["values"][0] is None
