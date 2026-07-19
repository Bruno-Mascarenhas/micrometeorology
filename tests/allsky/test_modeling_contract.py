"""Torch-free contract tests for allsky.modeling: lazy imports and group_slices.

These do not import torch (the subprocess checks assert it stays out), so the
module deliberately avoids ``pytest.importorskip('torch')`` at import time.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from allsky.features import FEATURE_GROUPS, active_feature_groups, resolve_feature_set
from allsky.modeling.contracts import group_slices


def test_import_allsky_modeling_is_torch_free():
    """Contract: importing allsky.modeling must not pull torch (lazy __getattr__)."""
    code = (
        "import sys\n"
        "import allsky.modeling\n"
        "import allsky.modeling.contracts\n"
        "from allsky.modeling import group_slices\n"  # torch-free name
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_accessing_torch_name_imports_lazily():
    """Accessing a torch-bearing name resolves through the submodule."""
    import allsky.modeling as modeling

    assert callable(modeling.build_model)
    assert modeling.SensorEncoder.__name__ == "SensorEncoder"


def test_unknown_attribute_raises_attribute_error():
    import allsky.modeling as modeling

    with pytest.raises(AttributeError, match="does_not_exist"):
        modeling.does_not_exist  # noqa: B018 - triggering __getattr__ on purpose


def test_group_slices_covers_safe_features_exactly_once():
    feature_columns = resolve_feature_set("safe")
    groups = active_feature_groups("safe")
    slices = group_slices(feature_columns, groups)

    covered = [i for indices in slices.values() for i in indices]
    assert sorted(covered) == list(range(len(feature_columns)))
    assert len(covered) == len(set(covered))  # each column exactly once


def test_group_slices_preserves_group_order():
    feature_columns = resolve_feature_set("safe")
    groups = active_feature_groups("safe")
    slices = group_slices(feature_columns, groups)
    assert list(slices) == list(groups)  # insertion order preserved


def test_group_slices_skips_empty_groups():
    """A safe feature vector paired with the full groups drops radiometry_aux."""
    feature_columns = resolve_feature_set("safe")
    slices = group_slices(feature_columns, FEATURE_GROUPS)
    assert "radiometry_aux" not in slices  # no extended members present


def test_group_slices_indices_map_to_named_columns():
    feature_columns = resolve_feature_set("safe")
    groups = active_feature_groups("safe")
    slices = group_slices(feature_columns, groups)
    for group, indices in slices.items():
        assert [feature_columns[i] for i in indices] == sorted(
            groups[group], key=feature_columns.index
        )
