"""Synthetic tests for WRF interpolation utilities."""

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

    ``VerticalInterpolator`` resolves ``vertical_interpolate`` through the module
    global, so patching the module attribute observes the fallback route alone.
    This module's direct import still points at the original function, which is
    what the references are computed with.
    """
    calls: list[tuple] = []
    original = interpolation.vertical_interpolate

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "vertical_interpolate", _spy)
    return calls


#: Every other test here, the frozen oracle in ``_reference.py`` and the streaming
#: and jobs suites all compare against ``vertical_interpolate`` itself, so a wrong
#: weight is invisible everywhere. These columns are small enough to interpolate on
#: paper: the expected value is written from the two bracketing levels, never read
#: back from the function.
@pytest.mark.parametrize(
    ("heights", "values", "target", "expected"),
    [
        ([0.0, 100.0], [0.0, 10.0], 25.0, 2.5),
        ([0.0, 100.0], [0.0, 10.0], 75.0, 7.5),
        ([120.0, 500.0], [4.0, 23.0], 310.0, 13.5),
        # Unequal levels: the weight is a fraction of THIS bracket, not of the column.
        ([0.0, 10.0, 1000.0], [0.0, 1.0, 100.0], 5.0, 0.5),
        ([0.0, 10.0, 1000.0], [0.0, 1.0, 100.0], 505.0, 50.5),
        # Below the lowest and above the highest level: the bracket at that end is
        # extended, so both ends extrapolate along the nearest pair's slope.
        ([100.0, 200.0], [10.0, 20.0], 50.0, 5.0),
        ([100.0, 200.0], [10.0, 20.0], 250.0, 25.0),
    ],
)
def test_a_two_level_column_interpolates_to_the_value_computed_by_hand(
    heights, values, target, expected
):
    column = vertical_interpolate(
        np.array(values, dtype=np.float64).reshape(-1, 1),
        np.array(heights, dtype=np.float64).reshape(-1, 1),
        target,
    )

    assert column[0] == pytest.approx(expected, rel=1e-12)


def test_a_column_with_one_usable_level_returns_it_without_extrapolating():
    """With no second level there is no slope to extend, and inventing one would
    publish a wind at 80 m from a single reading 300 m up.
    """
    column = vertical_interpolate(np.array([[7.5], [np.nan]]), np.array([[300.0], [np.nan]]), 80.0)

    assert column[0] == pytest.approx(7.5)


def test_a_column_with_no_usable_level_is_nan_not_a_number():
    column = vertical_interpolate(
        np.array([[np.nan], [np.nan]]), np.array([[np.nan], [np.nan]]), 80.0
    )

    assert np.isnan(column[0])


def test_a_collapsed_bracket_takes_the_lower_level_rather_than_dividing_by_zero():
    """Two levels at one height give a zero denominator; the documented answer
    is the lower level's value, not an infinity propagating into the field.
    """
    column = vertical_interpolate(np.array([[3.0], [9.0]]), np.array([[150.0], [150.0]]), 150.0)

    assert column[0] == pytest.approx(3.0)


def test_vertical_interpolator_fast_path_matches_reference_float32(monkeypatch):
    shape = (4, 12, 9, 7)
    heights = _monotonic_heights(shape, axis=1, seed=1)
    values = _random_values(shape, seed=2)
    # Below all heights, interior, exactly-representable, above all heights.
    targets = [-5.0, 37.0, 123.4, 1.0e6]

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


def test_interpolate_many_matches_per_target_interpolate():
    shape = (3, 12, 7, 5)
    heights = _monotonic_heights(shape, axis=1, seed=26)
    values = _random_values(shape, seed=27)
    targets = (-3.0, 50.0, 137.5, 1.0e6)

    interp = VerticalInterpolator(heights, axis=1)
    many = interp.interpolate_many(values, targets)

    assert len(many) == len(targets)
    for result, target in zip(many, targets, strict=True):
        expected = vertical_interpolate(values, heights, target, axis=1)
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert np.array_equal(result, expected, equal_nan=True)


def test_interpolate_many_preserves_a_singleton_time_axis():
    """A one-step block must keep its time axis: only the vertical axis is
    dropped, so downstream ``result[k]`` step indexing still works."""
    shape = (1, 12, 5, 5)
    heights = _monotonic_heights(shape, axis=1, seed=28)
    values = _random_values(shape, seed=29)

    interp = VerticalInterpolator(heights, axis=1)
    (result,) = interp.interpolate_many(values, (75.0,))

    assert result.shape == (1, 5, 5)
    assert np.array_equal(result, vertical_interpolate(values, heights, 75.0, axis=1))


def test_interpolate_many_falls_back_once_per_target(monkeypatch):
    shape = (2, 9, 4, 4)
    heights = _monotonic_heights(shape, axis=1, seed=30)
    values = _random_values(shape, seed=31)
    values[0, 2, 1, 1] = np.nan
    targets = (30.0, 90.0)

    calls = _install_fallback_spy(monkeypatch)
    interp = VerticalInterpolator(heights, axis=1)
    results = interp.interpolate_many(values, targets)

    assert len(calls) == len(targets)
    for result, target in zip(results, targets, strict=True):
        expected = vertical_interpolate(values, heights, target, axis=1)
        assert np.array_equal(result, expected, equal_nan=True)


def test_interpolate_many_keeps_the_block_memory_guard(monkeypatch):
    """The >16 GiB guard runs once per field rather than once per target, with
    the block context and multiplier unchanged."""
    shape = (2, 6, 4, 4)
    heights = _monotonic_heights(shape, axis=1, seed=32)
    values = _random_values(shape, seed=33)

    guard_calls: list[tuple] = []
    original = interpolation.assert_reasonable_array_size

    def _spy(guarded_shape, dtype, **kwargs):
        guard_calls.append((guarded_shape, kwargs))
        original(guarded_shape, dtype, **kwargs)

    monkeypatch.setattr(interpolation, "assert_reasonable_array_size", _spy)
    interp = VerticalInterpolator(heights, axis=1)
    interp.interpolate_many(values, (10.0, 50.0, 100.0))

    assert guard_calls == [(shape, {"context": "vertical interpolation block", "multiplier": 6.0})]


def test_vertical_interpolator_large_block_matches_reference():
    """The fast path must still agree with the reference on a realistic block."""
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
