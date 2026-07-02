"""Tests for the GeoJSON / JSON generation pipeline.

Covers:
- ``create_grid_geojson`` → correct FeatureCollection structure and linear_index
- ``create_values_json`` → vectorized NaN→None handling and rounding
- ``create_wind_vectors_json`` → standalone wind vector file schema
"""

from __future__ import annotations

import inspect
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from micrometeorology.wrf.batch import _write_json_payload
from micrometeorology.wrf.geojson import (
    _write_grid_geojson_stream_reference,
    create_grid_geojson,
    create_values_json,
    create_wind_vectors_json,
    save_geojson,
    write_grid_geojson_stream,
    write_values_json_stream,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# create_grid_geojson
# ---------------------------------------------------------------------------


class TestCreateGridGeoJson:
    def test_feature_collection_type(self, sample_grid):
        lon, lat = sample_grid
        result = create_grid_geojson(lon, lat, 3000.0, 3000.0, "hot_r")
        assert result["type"] == "FeatureCollection"

    def test_feature_count_matches_grid(self, sample_grid):
        lon, lat = sample_grid
        ny, nx = lon.shape
        result = create_grid_geojson(lon, lat, 3000.0, 3000.0, "hot_r")
        assert len(result["features"]) == ny * nx

    def test_linear_index_sequential(self, sample_grid):
        lon, lat = sample_grid
        result = create_grid_geojson(lon, lat, 3000.0, 3000.0, "hot_r")
        indices = [f["properties"]["linear_index"] for f in result["features"]]
        assert indices == list(range(lon.shape[0] * lon.shape[1]))

    def test_each_feature_is_polygon(self, sample_grid):
        lon, lat = sample_grid
        result = create_grid_geojson(lon, lat, 3000.0, 3000.0, "hot_r")
        for f in result["features"]:
            assert f["type"] == "Feature"
            assert f["geometry"]["type"] == "Polygon"
            # Each polygon should be closed (first == last coord)
            coords = f["geometry"]["coordinates"][0]
            assert len(coords) == 5  # 4 corners + closing point
            assert coords[0] == coords[-1]

    def test_metadata_resolution(self, sample_grid):
        lon, lat = sample_grid
        result = create_grid_geojson(lon, lat, 3000.0, 5000.0, "hot_r")
        assert result["metadata"]["resolucao_m"] == [3000.0, 5000.0]


# ---------------------------------------------------------------------------
# create_values_json
# ---------------------------------------------------------------------------


class TestCreateValuesJson:
    def test_values_length_matches_flat_array(self, sample_values_2d):
        result = create_values_json(sample_values_2d, 0.0, 20.0, None)
        assert len(result["values"]) == sample_values_2d.size

    def test_nan_becomes_none(self, sample_values_2d):
        result = create_values_json(sample_values_2d, 0.0, 20.0, None)
        # Index (0,0) = flat index 0 was set to NaN
        assert result["values"][0] is None
        # Index (2,3) = flat index 2*5+3 = 13
        assert result["values"][13] is None

    def test_values_are_rounded_to_2dp(self):
        arr = np.array([[1.23456, 2.789]], dtype=np.float64)
        result = create_values_json(arr, 0.0, 3.0, None)
        assert result["values"][0] == 1.23
        assert result["values"][1] == 2.79

    def test_masked_array_support(self):
        data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False]).reshape(1, 3)
        result = create_values_json(data, 0.0, 3.0, None)
        assert result["values"][0] == 1.0
        assert result["values"][1] is None  # masked → NaN → None
        assert result["values"][2] == 3.0

    def test_scale_values_count(self, sample_values_2d):
        result = create_values_json(sample_values_2d, 10.0, 30.0, None)
        assert len(result["metadata"]["scale_values"]) == 6

    def test_date_formatting(self, sample_values_2d):
        dt = datetime(2024, 6, 15, 12, 30, 45)
        result = create_values_json(sample_values_2d, 0.0, 1.0, dt)
        # Minutes/seconds should be zeroed
        assert result["metadata"]["date_time"] == "15/06/2024 12:00:00"

    def test_wind_data_included_when_provided(self, sample_values_2d):
        wind = {"downsampled_angles": [180.0], "downsampled_magnitudes": [5.0]}
        result = create_values_json(sample_values_2d, 0.0, 1.0, None, wind_data=wind)
        assert "wind" in result["metadata"]
        assert result["metadata"]["wind"] == wind

    def test_wind_data_absent_when_none(self, sample_values_2d):
        result = create_values_json(sample_values_2d, 0.0, 1.0, None)
        assert "wind" not in result["metadata"]

    def test_streamed_values_json_matches_in_memory_payload(self, sample_values_2d):
        root = Path("scratch") / f"stream-values-{uuid.uuid4().hex}"
        out = root / "values.json"
        root.mkdir(parents=True, exist_ok=True)
        try:
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
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_batch_json_writer_uses_streaming_payload(self):
        source = inspect.getsource(_write_json_payload)

        assert "write_values_json_stream" in source
        assert ".tolist()" not in source


# ---------------------------------------------------------------------------
# create_wind_vectors_json
# ---------------------------------------------------------------------------


class TestCreateWindVectorsJson:
    def test_output_has_required_keys(self, sample_wind_2d):
        u, v = sample_wind_2d
        result = create_wind_vectors_json(u, v, None, downsampling=2)
        assert "metadata" in result
        assert "downsampled_angles" in result
        assert "downsampled_magnitudes" in result
        assert "downsampled_linear_indices" in result

    def test_downsampling_reduces_count(self, sample_wind_2d):
        u, v = sample_wind_2d
        full = create_wind_vectors_json(u, v, None, downsampling=1)
        ds = create_wind_vectors_json(u, v, None, downsampling=2)
        assert len(ds["downsampled_angles"]) < len(full["downsampled_angles"])

    def test_angles_in_valid_range(self, sample_wind_2d):
        u, v = sample_wind_2d
        result = create_wind_vectors_json(u, v, None, downsampling=1)
        for angle in result["downsampled_angles"]:
            assert 0 <= angle < 360

    def test_magnitudes_non_negative(self, sample_wind_2d):
        u, v = sample_wind_2d
        result = create_wind_vectors_json(u, v, None, downsampling=1)
        for mag in result["downsampled_magnitudes"]:
            assert mag >= 0

    def test_linear_indices_within_grid(self, sample_wind_2d):
        u, v = sample_wind_2d
        ny, nx = u.shape
        result = create_wind_vectors_json(u, v, None, downsampling=1)
        for idx in result["downsampled_linear_indices"]:
            assert 0 <= idx < ny * nx

    def test_magnitude_consistency(self):
        """Magnitude should match np.hypot for known inputs."""
        u = np.array([[3.0, 0.0]], dtype=np.float64)
        v = np.array([[4.0, 5.0]], dtype=np.float64)
        result = create_wind_vectors_json(u, v, None, downsampling=1)
        assert result["downsampled_magnitudes"][0] == pytest.approx(5.0, abs=0.01)
        assert result["downsampled_magnitudes"][1] == pytest.approx(5.0, abs=0.01)

    def test_date_in_metadata(self, sample_wind_2d):
        u, v = sample_wind_2d
        dt = datetime(2024, 3, 15, 9, 0, 0)
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


# ---------------------------------------------------------------------------
# write_grid_geojson_stream — byte identity vs the reference per-feature loop
# ---------------------------------------------------------------------------


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
    """The vectorized writer must produce byte-identical files to the old loop.

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
        _write_grid_geojson_stream_reference(ref_path, lon, lat, 3000.0, 3000.0)
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

        Regression: np.ma arithmetic promotes ``/ 2`` to float64, unlike the
        per-element float32 scalar path — corner math must not run on the
        MaskedArray or edge cells drift in the 6th decimal.
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
        # The trap must be real: old path (builtin round) differs from np.round.
        assert round(tie_lat, 10) != float(np.round(tie_lat, 10))
        assert round(tie_lon, 10) != float(np.round(tie_lon, 10))

        lat = np.full((3, 4), tie_lat, dtype=np.float64)
        lon = np.full((3, 4), tie_lon, dtype=np.float64)
        data = self._assert_stream_bytes_match_reference(tmp_path, lon, lat)
        # Builtin-round digits must appear in the output (np.round would
        # have written -14.0 / -38.0 instead).
        assert b"-14.0000000001" in data
        assert b"-38.0000000001" in data


def test_save_geojson_stream_matches_in_memory_geojson(sample_grid):
    root = Path("scratch") / f"stream-geojson-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    lon, lat = sample_grid
    try:
        expected = create_grid_geojson(lon, lat, 3000.0, 3000.0, "")
        out = save_geojson(root, "D01", lon, lat, 3000.0, 3000.0)
        with open(out, encoding="utf-8") as f:
            actual = json.load(f)

        assert actual == expected
    finally:
        shutil.rmtree(root, ignore_errors=True)
