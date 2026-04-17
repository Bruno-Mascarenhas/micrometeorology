"""Tests for the GeoJSON / JSON generation pipeline.

Covers:
- ``create_grid_geojson`` → correct FeatureCollection structure and linear_index
- ``create_values_json`` → vectorized NaN→None handling and rounding
- ``create_wind_vectors_json`` → standalone wind vector file schema
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from labmim_micrometeorology.wrf.geojson import (
    create_grid_geojson,
    create_values_json,
    create_wind_vectors_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_grid() -> tuple[np.ndarray, np.ndarray]:
    """Small 4×5 lon/lat grid for testing."""
    ny, nx = 4, 5
    lon = np.linspace(-40, -38, nx)[np.newaxis, :].repeat(ny, axis=0)
    lat = np.linspace(-14, -12, ny)[:, np.newaxis].repeat(nx, axis=1)
    return lon, lat


@pytest.fixture()
def sample_values_2d() -> np.ndarray:
    """4×5 array with some NaN values."""
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    arr[0, 0] = np.nan
    arr[2, 3] = np.nan
    return arr


@pytest.fixture()
def sample_wind_2d() -> tuple[np.ndarray, np.ndarray]:
    """4×5 U/V wind component arrays."""
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
