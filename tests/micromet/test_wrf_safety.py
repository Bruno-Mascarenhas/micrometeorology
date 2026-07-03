"""Regression tests for the WRF memory guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from micrometeorology.wrf.safety import assert_reasonable_array_size, estimate_array_nbytes


def test_estimate_array_nbytes_multiplies_elements_by_itemsize():
    assert estimate_array_nbytes((3, 4), np.float64) == 96
    assert estimate_array_nbytes([2, 5], np.float32) == 40
    assert estimate_array_nbytes((), np.float64) == 8
    assert estimate_array_nbytes((0, 10), np.float64) == 0


def test_estimate_array_nbytes_rejects_negative_dimensions():
    with pytest.raises(ValueError, match="negative dimension"):
        estimate_array_nbytes((2, -1), np.float64)


def test_memory_guard_fails_before_large_allocation():
    with pytest.raises(MemoryError, match="test allocation"):
        assert_reasonable_array_size(
            (1024, 1024),
            np.float64,
            max_gb=0.001,
            context="test allocation",
        )


def test_memory_guard_accepts_reasonable_sizes():
    assert_reasonable_array_size(
        (64, 64),
        np.float64,
        max_gb=0.001,
        context="small allocation",
    )


def test_memory_guard_multiplier_scales_the_estimate():
    # 1024x1024 float64 = 8 MiB: below a 0.008 GiB limit on its own, but a
    # 2x working-set multiplier pushes the estimate over it.
    assert_reasonable_array_size(
        (1024, 1024),
        np.float64,
        max_gb=0.008,
        context="unmultiplied allocation",
    )
    with pytest.raises(MemoryError, match="multiplied allocation"):
        assert_reasonable_array_size(
            (1024, 1024),
            np.float64,
            max_gb=0.008,
            multiplier=2.0,
            context="multiplied allocation",
        )
