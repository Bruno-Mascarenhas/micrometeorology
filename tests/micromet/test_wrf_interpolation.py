"""Synthetic tests for WRF interpolation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from micrometeorology.wrf import interpolation
from micrometeorology.wrf.interpolation import (
    VerticalInterpolator,
    vertical_interpolate,
)


def _monotonic_heights(shape, axis, seed=0, dtype=np.float32):
    """Strictly increasing NaN-free heights (cumsum of positive steps)."""
    rng = np.random.default_rng(seed)
    steps = rng.uniform(1.0, 50.0, size=shape).astype(dtype)
    return np.cumsum(steps, axis=axis, dtype=dtype)


def _random_values(shape, seed, dtype=np.float32):
    return np.random.default_rng(seed).normal(size=shape).astype(dtype)


def _install_fallback_spy(monkeypatch):
    """Record calls routed through the module-level ``vertical_interpolate``.

    ``VerticalInterpolator`` resolves ``vertical_interpolate`` through the
    module global, so patching the module attribute observes only the
    fallback route.  The direct import in this test module keeps pointing at
    the original function for computing references.
    """
    calls: list[tuple] = []
    original = interpolation.vertical_interpolate

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "vertical_interpolate", _spy)
    return calls


def test_vertical_interpolator_fast_path_matches_reference_float32(monkeypatch):
    shape = (4, 12, 9, 7)
    heights = _monotonic_heights(shape, axis=1, seed=1)
    values = _random_values(shape, seed=2)
    # Below all heights, interior, exactly-representable, above all heights.
    targets = [-5.0, 0.5, 37.0, 123.4, 1.0e6]

    expected = {target: vertical_interpolate(values, heights, target, axis=1) for target in targets}

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    for target in targets:
        result = interp.interpolate(values, target)
        assert result.dtype == expected[target].dtype
        assert result.shape == expected[target].shape
        assert np.array_equal(result, expected[target], equal_nan=True)
    assert not calls


def test_vertical_interpolator_target_exactly_on_level():
    levels = np.array([10.0, 50.0, 100.0, 250.0, 500.0], dtype=np.float32)
    heights = np.broadcast_to(levels[None, :, None, None], (2, 5, 4, 3)).copy()
    values = _random_values(heights.shape, seed=4)

    interp = VerticalInterpolator(heights, axis=1)
    # First, interior, and last level exercise the strict `>` tie behavior.
    for target in [10.0, 100.0, 500.0]:
        expected = vertical_interpolate(values, heights, target, axis=1)
        result = interp.interpolate(values, target)
        assert np.array_equal(result, expected, equal_nan=True)


def test_vertical_interpolator_duplicate_heights_fall_back(monkeypatch):
    shape = (3, 8, 5, 5)
    heights = _monotonic_heights(shape, axis=1, seed=5)
    # Duplicate the top level: non-strictly increasing columns, and targets
    # above all heights hit the h2 == h1 -> frac 0 path in the reference.
    heights[:, -1, ...] = heights[:, -2, ...]
    values = _random_values(shape, seed=6)
    targets = [80.0, 1.0e6]

    expected = {target: vertical_interpolate(values, heights, target, axis=1) for target in targets}

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    for target in targets:
        result = interp.interpolate(values, target)
        assert np.array_equal(result, expected[target], equal_nan=True)
    assert len(calls) == len(targets)


def test_vertical_interpolator_nan_values_fall_back(monkeypatch):
    shape = (3, 10, 6, 6)
    heights = _monotonic_heights(shape, axis=1, seed=7)
    values = _random_values(shape, seed=8)
    values[0, 3, 2, 2] = np.nan
    values[1, :, 4, 1] = np.nan  # fully invalid column
    target = 120.0

    expected = vertical_interpolate(values, heights, target, axis=1)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    result = interp.interpolate(values, target)
    assert len(calls) == 1
    assert np.array_equal(result, expected, equal_nan=True)


def test_vertical_interpolator_nan_heights_fall_back(monkeypatch):
    shape = (2, 9, 5, 4)
    heights = _monotonic_heights(shape, axis=1, seed=9)
    heights[1, 5, 3, 2] = np.nan
    values = _random_values(shape, seed=10)
    target = 95.0

    expected = vertical_interpolate(values, heights, target, axis=1)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    result = interp.interpolate(values, target)
    assert len(calls) == 1
    assert np.array_equal(result, expected, equal_nan=True)


def test_vertical_interpolator_single_valid_level_falls_back(monkeypatch):
    shape = (2, 6, 4, 4)
    heights = _monotonic_heights(shape, axis=1, seed=11)
    rng = np.random.default_rng(12)
    values = np.full(shape, np.nan, dtype=np.float32)
    # Exactly one valid value per column, at a random level.
    flat = values.reshape(shape[0], shape[1], -1)
    keep = rng.integers(0, shape[1], size=(shape[0], flat.shape[2]))
    for t in range(shape[0]):
        for col in range(flat.shape[2]):
            flat[t, keep[t, col], col] = rng.normal()
    target = 60.0

    expected = vertical_interpolate(values, heights, target, axis=1)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    result = interp.interpolate(values, target)
    assert len(calls) == 1
    assert np.array_equal(result, expected, equal_nan=True)


def test_vertical_interpolator_float64_inputs(monkeypatch):
    shape = (3, 11, 6, 5)
    heights64 = _monotonic_heights(shape, axis=1, seed=13, dtype=np.float64)
    values64 = _random_values(shape, seed=14, dtype=np.float64)
    values32 = _random_values(shape, seed=15, dtype=np.float32)
    target = 140.0

    expected64 = vertical_interpolate(values64, heights64, target, axis=1)
    expected_mixed = vertical_interpolate(values32, heights64, target, axis=1)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights64, axis=1)

    result64 = interp.interpolate(values64, target)
    assert result64.dtype == np.float64
    assert np.array_equal(result64, expected64, equal_nan=True)

    result_mixed = interp.interpolate(values32, target)
    assert result_mixed.dtype == np.float64
    assert np.array_equal(result_mixed, expected_mixed, equal_nan=True)
    assert not calls


def test_vertical_interpolator_axis0_3d(monkeypatch):
    shape = (14, 7, 6)
    heights = _monotonic_heights(shape, axis=0, seed=16)
    values = _random_values(shape, seed=17)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=0)
    for target in [3.0, 88.0, 5.0e5]:
        expected = vertical_interpolate(values, heights, target, axis=0)
        result = interp.interpolate(values, target)
        assert result.shape == expected.shape
        assert np.array_equal(result, expected, equal_nan=True)
    assert not calls


def test_vertical_interpolator_cache_across_fields_and_targets(monkeypatch):
    shape = (3, 10, 6, 6)
    heights = _monotonic_heights(shape, axis=1, seed=18)
    u = _random_values(shape, seed=19)
    v = _random_values(shape, seed=20)
    targets = [25.0, 90.0]

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    for target in targets:
        for field in (u, v):
            expected = vertical_interpolate(field, heights, target, axis=1)
            result = interp.interpolate(field, target)
            assert np.array_equal(result, expected, equal_nan=True)
    assert not calls
    assert sorted(interp._bracket_cache) == targets

    # dtype mismatch at a cached target must recompute, not reuse stale brackets.
    u64 = u.astype(np.float64)
    expected64 = vertical_interpolate(u64, heights, targets[0], axis=1)
    result64 = interp.interpolate(u64, targets[0])
    assert result64.dtype == np.float64
    assert np.array_equal(result64, expected64, equal_nan=True)


def test_vertical_interpolator_shape_mismatch_raises():
    heights = _monotonic_heights((2, 5, 3, 3), axis=1, seed=21)
    values = _random_values((2, 5, 3, 4), seed=22)
    interp = VerticalInterpolator(heights, axis=1)
    with pytest.raises(ValueError, match="same shape"):
        interp.interpolate(values, 50.0)


def test_vertical_interpolator_perf_sanity_large_block():
    shape = (8, 30, 40, 40)
    heights = _monotonic_heights(shape, axis=1, seed=23)
    u = _random_values(shape, seed=24)
    v = _random_values(shape, seed=25)

    interp = VerticalInterpolator(heights, axis=1)
    for target in [10.0, 50.0, 100.0, 200.0]:
        for field in (u, v):
            expected = vertical_interpolate(field, heights, target, axis=1)
            result = interp.interpolate(field, target)
            assert np.array_equal(result, expected, equal_nan=True)
