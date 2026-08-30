"""Sky conditions from Kt and the cumulative curve the page reads them off.

The partition is Escobedo et al. (2009) sec. 3.1 and lives in
``allsky.data.contracts``; these tests pin that this module applies it with the
same closed upper edge as the all-sky manifest, so the two pipelines cannot
label the same clearness index differently.
"""

import numpy as np
import pytest

from labmim_core.sky import SKY_CLASS_KT_UPPER_BOUNDS
from micrometeorology.stats.sky_condition import (
    classify_sky_condition,
    cumulative_fractions,
    sky_condition_summary,
)


def test_a_clearness_index_exactly_on_a_bound_belongs_to_the_lower_condition():
    """Escobedo et al. (2009) sec. 3.1 publishes the bands with a CLOSED upper edge."""
    on_the_bounds = np.array(SKY_CLASS_KT_UPPER_BOUNDS, dtype=float)

    assert classify_sky_condition(on_the_bounds).tolist() == [0, 1, 2]


def test_the_classification_matches_the_all_sky_manifest_on_the_boundaries_themselves():
    """One partition, two pipelines: a drift here relabels the published record.

    The manifest classifier is private, so it is imported here rather than in
    production code — the point is to pin the semantics, not to couple them.
    """
    from allsky.data.manifest import _classify_sky

    bounds = np.array(SKY_CLASS_KT_UPPER_BOUNDS, dtype=float)
    kt = np.concatenate(
        [
            np.linspace(0.0, 1.0, 501),
            bounds,
            np.nextafter(bounds, 0.0),  # one ulp below each bound
            np.nextafter(bounds, 1.0),  # one ulp above each bound
        ]
    )

    mine = classify_sky_condition(kt)
    theirs = _classify_sky(kt, labelable=np.ones(kt.shape, dtype=bool))

    assert mine.tolist() == theirs.tolist()


def test_a_non_finite_clearness_index_is_labelled_unusable_rather_than_clear():
    labels = classify_sky_condition(np.array([np.nan, np.inf, -np.inf, 0.8]))

    assert labels.tolist() == [-1, -1, -1, 3]


def test_the_cumulative_is_the_running_sum_of_the_bars_it_is_drawn_beside():
    counts = [2, 3, 5]

    assert cumulative_fractions(counts, total=10) == [0.2, 0.5, 1.0]


def test_the_cumulative_denominator_is_the_whole_subset_not_only_the_bars():
    """Samples outside the binned range still count, so F stops short of 1."""
    assert cumulative_fractions([2, 3, 5], total=20) == [0.1, 0.25, 0.5]


def test_an_empty_subset_yields_no_cumulative_instead_of_dividing_by_zero():
    assert cumulative_fractions([0, 0], total=0) == []


def test_the_class_fractions_cover_the_labelable_sample_exactly_once():
    kt = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, np.nan])

    summary = sky_condition_summary(kt)

    assert summary["n"] == 7
    assert sum(c["count"] for c in summary["conditions"]) == 7
    assert sum(c["fraction"] for c in summary["conditions"]) == pytest.approx(1.0)


def test_the_summary_publishes_the_bounds_and_both_numberings():
    summary = sky_condition_summary(np.array([0.5]))

    assert summary["kt_upper_bounds"] == list(SKY_CLASS_KT_UPPER_BOUNDS)
    assert [c["condition"] for c in summary["conditions"]] == [1, 2, 3, 4]
    assert [c["id"] for c in summary["conditions"]] == ["i", "ii", "iii", "iv"]
    assert summary["conditions"][0]["kt_range"] == [None, 0.35]
    assert summary["conditions"][3]["kt_range"] == [0.65, None]


def test_a_sample_with_nothing_labelable_publishes_no_fraction_rather_than_zero():
    summary = sky_condition_summary(np.array([np.nan, np.nan]))

    assert summary["n"] == 0
    assert all(c["fraction"] is None for c in summary["conditions"])
    assert all(c["count"] == 0 for c in summary["conditions"])
