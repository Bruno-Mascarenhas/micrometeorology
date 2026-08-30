"""K-index stratification and the baseline references of the evaluator.

Three defects with one root — which k-index a number is measured against:

- the ``kindex_band`` stratum binned the frozen ``target_kindex`` column with the
  published Escobedo bounds, which are defined on the clearness index Kt, so a
  ``kstar`` dataset was labelled against bounds on another quantity;
- the clear-sky reference of the k-index head was hardcoded to ``1``, the
  clear-sky value of k* and of no clearness index;
- the persistence reference was rebuilt from whatever rows it was handed, so a
  stratified row scored the model against a "previous observation" that could be
  a day away, and the reference RMSE was divided into a model RMSE measured over
  a different set of rows.

A later round of review added three more, with the same root:

- ``kindex_band`` moved to ``target_kt``, a column older manifests do not carry,
  so evaluating one died with ``KeyError: 'target_kt'`` after full inference;
- a manifest without a ``.meta.json`` sidecar lost its k-index clear-sky
  baseline without a word;
- the skill score published a reference RMSE and a whole-split model RMSE that
  do not reconcile to it, because the numerator is measured over the paired rows.

Torch-free: the helpers run over hand-built manifest rows.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from allsky.clearsky import haurwitz_ghi_from_cos_zenith
from allsky.config import SITE_TZ, SITE_UTC_OFFSET_HOURS
from allsky.data.manifest import _classify_sky
from allsky.evaluation.evaluator import (
    _add_strata,
    _build_predictions_frame,
    _clearsky_reference,
    _stratified_metrics,
    _target_metrics,
)
from labmim_core.site import SiteConfig
from labmim_core.sky import (
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLEAR,
    SKY_CLOUDY,
    SKY_PARTLY_CLOUDY_CLEAR,
    SKY_PARTLY_CLOUDY_DIFFUSE,
)
from labmim_core.solar import cos_zenith, extraterrestrial_ghi

_SITE = SiteConfig()


def _manifest_rows(local_times: pd.DatetimeIndex, **columns: object) -> pd.DataFrame:
    """Manifest slice carrying the columns the strata and the targets read."""
    n = len(local_times)
    zenith_deg = np.degrees(np.arccos(cos_zenith(local_times, _SITE, SITE_UTC_OFFSET_HOURS)))
    rows: dict[str, object] = {
        "sample_id": [f"allsky-{ts:%Y%m%d-%H%M}" for ts in local_times],
        "day_id": [f"{ts:%Y-%m-%d}" for ts in local_times],
        "timestamp_utc": local_times.tz_localize(SITE_TZ).tz_convert("UTC"),
        "solar_elevation": 90.0 - zenith_deg,
        "solar_zenith": zenith_deg,
        "sky_class": np.zeros(n, dtype=np.int64),
        "qc_flags": np.zeros(n, dtype=np.int64),
        "target_dhi": np.full(n, 100.0),
        "target_kindex": np.full(n, 0.7),
        "target_kt": np.full(n, 0.5),
    }
    rows.update(columns)
    return pd.DataFrame(rows)


def _two_days_of_noon_and_afternoon() -> pd.DatetimeIndex:
    """Two days of 10-minute frames, six per day, in two separate hours."""
    starts = ("2025-03-20 12:00", "2025-03-20 13:00", "2025-03-21 12:00", "2025-03-21 13:00")
    return pd.DatetimeIndex(
        [
            pd.Timestamp(start) + pd.Timedelta(minutes=10 * step)
            for start in starts
            for step in range(3)
        ]
    )


def _stratified_row(
    stratified: pd.DataFrame, *, stratum_kind: str, stratum: str, metric: str
) -> pd.Series:
    selected = stratified[
        (stratified["stratum_kind"] == stratum_kind)
        & (stratified["stratum"] == stratum)
        & (stratified["metric"] == metric)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


#: The sky classes each published band may hold, so a Kt sitting exactly on a
#: bound cannot be labelled one condition by the manifest and another by the
#: evaluator's band.
_BAND_SKY_CLASSES: dict[str, set[int]] = {
    "overcast_le0.35": {SKY_CLOUDY},
    "partial_0.35-0.65": {SKY_PARTLY_CLOUDY_DIFFUSE, SKY_PARTLY_CLOUDY_CLEAR},
    "clear_gt0.65": {SKY_CLEAR},
}


@pytest.mark.parametrize("bound", SKY_CLASS_KT_UPPER_BOUNDS)
def test_a_kt_on_a_published_bound_lands_in_the_band_that_holds_its_sky_class(
    bound: float,
) -> None:
    sky_class = _classify_sky(np.array([bound]), labelable=np.array([True]))
    split_df = _manifest_rows(
        pd.date_range("2025-03-20 12:00", periods=1), target_kt=[bound], sky_class=sky_class
    )
    frame = pd.DataFrame({"sample_id": split_df["sample_id"]})

    _add_strata(frame, split_df)

    band = str(frame["kindex_band"].iloc[0])
    assert int(sky_class[0]) in _BAND_SKY_CLASSES.get(band, set())


@pytest.mark.parametrize(
    ("target_kt", "target_kindex", "expected_band"),
    [(0.30, 0.42, "overcast_le0.35"), (0.50, 0.70, "partial_0.35-0.65")],
)
def test_kindex_band_bins_the_index_its_bounds_are_published_on(
    target_kt: float, target_kindex: float, expected_band: str
) -> None:
    split_df = _manifest_rows(
        pd.date_range("2025-03-20 12:00", periods=1),
        target_kt=[target_kt],
        target_kindex=[target_kindex],
    )
    frame = pd.DataFrame({"sample_id": split_df["sample_id"]})

    _add_strata(frame, split_df)

    assert frame["kindex_band"].tolist() == [expected_band]


def test_clearsky_reference_of_a_kt_dataset_is_the_clear_sky_clearness_index() -> None:
    local_times = pd.date_range("2025-03-20 08:00", periods=5, freq="2h")
    frame = _manifest_rows(local_times)
    expected = haurwitz_ghi_from_cos_zenith(
        cos_zenith(local_times, _SITE, SITE_UTC_OFFSET_HOURS)
    ) / extraterrestrial_ghi(local_times, _SITE, SITE_UTC_OFFSET_HOURS)

    reference = _clearsky_reference(frame, "kindex", kindex_kind="kt")

    assert reference == pytest.approx(expected)


def test_clearsky_reference_of_a_kstar_dataset_is_unity() -> None:
    frame = _manifest_rows(pd.date_range("2025-03-20 08:00", periods=5, freq="2h"))

    reference = _clearsky_reference(frame, "kindex", kindex_kind="kstar")

    assert reference == pytest.approx(np.ones(len(frame)))


def test_no_clearsky_reference_is_invented_when_the_manifest_kindex_kind_is_unknown() -> None:
    frame = _manifest_rows(pd.date_range("2025-03-20 08:00", periods=5, freq="2h"))

    assert _clearsky_reference(frame, "kindex", kindex_kind=None) is None


def test_persistence_reference_is_the_previous_observation_of_the_same_day() -> None:
    local_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-03-20 12:00"),
            pd.Timestamp("2025-03-20 12:10"),
            pd.Timestamp("2025-03-20 12:20"),
            pd.Timestamp("2025-03-21 12:00"),
            pd.Timestamp("2025-03-21 12:10"),
        ]
    )
    observed = np.array([100.0, 110.0, 120.0, 500.0, 510.0])
    split_df = _manifest_rows(local_times, target_dhi=observed)

    frame = _build_predictions_frame(split_df, {"dhi": observed}, ["dhi"], kindex_kind="kstar")

    np.testing.assert_array_equal(
        frame["persistence_dhi"].to_numpy(dtype=np.float64),
        np.array([np.nan, 100.0, 110.0, np.nan, 500.0]),
    )


def test_stratified_persistence_is_the_split_cadence_not_the_stratum_neighbour() -> None:
    local_times = _two_days_of_noon_and_afternoon()
    observed = np.array(
        [100.0, 110.0, 120.0, 200.0, 210.0, 220.0, 500.0, 510.0, 520.0, 600.0, 610.0, 620.0]
    )
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": observed + 5.0}, ["dhi"], kindex_kind="kstar"
    )

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    noon = _stratified_row(
        stratified, stratum_kind="hour_of_day", stratum="12", metric="rmse_persistence"
    )
    assert noon["value"] == pytest.approx(10.0)


def test_reference_rows_count_the_pairs_the_reference_was_scored_over() -> None:
    local_times = _two_days_of_noon_and_afternoon()
    observed = np.array(
        [100.0, 110.0, 120.0, 200.0, 210.0, 220.0, 500.0, 510.0, 520.0, 600.0, 610.0, 620.0]
    )
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": observed + 5.0}, ["dhi"], kindex_kind="kstar"
    )

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    noon = _stratified_row(
        stratified, stratum_kind="hour_of_day", stratum="12", metric="rmse_persistence"
    )
    assert int(noon["n"]) == 4


def test_skill_scores_model_and_reference_over_the_same_rows() -> None:
    local_times = pd.date_range("2025-03-20 12:00", periods=3, freq="10min")
    observed = np.array([100.0, 110.0, 120.0])
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": np.array([1000.0, 115.0, 125.0])}, ["dhi"], kindex_kind="kstar"
    )

    metrics = _target_metrics(frame, "dhi")

    assert metrics["skill_persistence"] == pytest.approx(0.5)


def test_kindex_band_is_absent_for_a_manifest_built_before_target_kt() -> None:
    split_df = _manifest_rows(pd.date_range("2025-03-20 12:00", periods=2, freq="10min")).drop(
        columns=["target_kt"]
    )
    frame = pd.DataFrame({"sample_id": split_df["sample_id"]})

    _add_strata(frame, split_df)

    assert "kindex_band" not in frame.columns


def test_a_manifest_built_before_target_kt_warns_that_it_loses_the_kindex_breakdown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    split_df = _manifest_rows(pd.date_range("2025-03-20 12:00", periods=2, freq="10min")).drop(
        columns=["target_kt"]
    )
    frame = pd.DataFrame({"sample_id": split_df["sample_id"]})

    with caplog.at_level(logging.WARNING, logger="allsky.evaluation.evaluator"):
        _add_strata(frame, split_df)

    assert any("kindex_band breakdown is absent" in record.message for record in caplog.records)


def test_a_manifest_with_no_recorded_kindex_kind_warns_that_it_loses_the_clearsky_baseline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_times = pd.date_range("2025-03-20 08:00", periods=5, freq="1h")
    split_df = _manifest_rows(local_times)

    with caplog.at_level(logging.WARNING, logger="allsky.evaluation.evaluator"):
        _build_predictions_frame(
            split_df, {"kindex": np.full(len(local_times), 0.7)}, ["kindex"], kindex_kind=None
        )

    assert [record.getMessage() for record in caplog.records if "kindex_kind" in record.msg] != []


def test_the_lost_clearsky_baseline_is_reported_once_not_once_per_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_times = pd.date_range("2025-03-20 08:00", periods=5, freq="1h")
    split_df = _manifest_rows(local_times)

    with caplog.at_level(logging.WARNING, logger="allsky.evaluation.evaluator"):
        _build_predictions_frame(
            split_df, {"kindex": np.full(len(local_times), 0.7)}, ["kindex"], kindex_kind=None
        )

    assert len([record for record in caplog.records if "kindex_kind" in record.msg]) == 1


def test_a_known_kindex_kind_scores_its_clearsky_baseline_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    local_times = pd.date_range("2025-03-20 08:00", periods=5, freq="1h")
    split_df = _manifest_rows(local_times)

    with caplog.at_level(logging.WARNING, logger="allsky.evaluation.evaluator"):
        _build_predictions_frame(
            split_df, {"kindex": np.full(len(local_times), 0.7)}, ["kindex"], kindex_kind="kstar"
        )

    assert [record for record in caplog.records if "kindex_kind" in record.msg] == []


def _stratified_rows(
    stratified: pd.DataFrame, *, stratum_kind: str, stratum: str, metric: str
) -> pd.DataFrame:
    return stratified[
        (stratified["stratum_kind"] == stratum_kind)
        & (stratified["stratum"] == stratum)
        & (stratified["metric"] == metric)
    ]


def _two_days_whose_noon_frames_all_open_their_day() -> pd.DatetimeIndex:
    """Two days whose only 12:00 frame is the first of the day, plus two at 13:00."""
    return pd.DatetimeIndex(
        [
            pd.Timestamp("2025-03-20 12:00"),
            pd.Timestamp("2025-03-20 13:00"),
            pd.Timestamp("2025-03-20 13:10"),
            pd.Timestamp("2025-03-21 12:00"),
            pd.Timestamp("2025-03-21 13:00"),
            pd.Timestamp("2025-03-21 13:10"),
        ]
    )


@pytest.mark.parametrize(
    "metric", ["rmse_persistence", "rmse_model_persistence", "skill_persistence"]
)
def test_a_stratum_that_pairs_no_persistence_row_publishes_no_persistence_metric(
    metric: str,
) -> None:
    local_times = _two_days_whose_noon_frames_all_open_their_day()
    observed = np.array([100.0, 110.0, 120.0, 500.0, 510.0, 520.0])
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": observed + 5.0}, ["dhi"], kindex_kind="kstar"
    )

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    assert _stratified_rows(
        stratified, stratum_kind="hour_of_day", stratum="12", metric=metric
    ).empty


def test_a_stratum_keeps_its_model_metrics_when_a_reference_pairs_nothing() -> None:
    local_times = _two_days_whose_noon_frames_all_open_their_day()
    observed = np.array([100.0, 110.0, 120.0, 500.0, 510.0, 520.0])
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": observed + 5.0}, ["dhi"], kindex_kind="kstar"
    )

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    noon = _stratified_row(stratified, stratum_kind="hour_of_day", stratum="12", metric="rmse")
    assert int(noon["n"]) == 2


def test_an_unresolvable_clearsky_baseline_publishes_no_clearsky_metric() -> None:
    local_times = pd.date_range("2025-03-20 08:00", periods=5, freq="1h")
    split_df = _manifest_rows(local_times)
    frame = _build_predictions_frame(
        split_df, {"kindex": np.full(len(local_times), 0.7)}, ["kindex"], kindex_kind=None
    )

    stratified = _stratified_metrics(
        frame, ["kindex"], {"kindex": _target_metrics(frame, "kindex")}
    )

    assert stratified[stratified["metric"].str.endswith("_clearsky")].empty


def test_no_stratified_row_is_published_over_zero_pairs() -> None:
    local_times = _two_days_whose_noon_frames_all_open_their_day()
    observed = np.array([100.0, 110.0, 120.0, 500.0, 510.0, 520.0])
    split_df = _manifest_rows(local_times, target_dhi=observed)
    frame = _build_predictions_frame(
        split_df, {"dhi": observed + 5.0}, ["dhi"], kindex_kind="kstar"
    )

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    assert bool((stratified["n"] > 0).all())


def _two_days_with_a_dropped_persistence_row() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A 4-row split whose two day-opening rows have no persistence predecessor."""
    local_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-03-20 12:00"),
            pd.Timestamp("2025-03-20 12:10"),
            pd.Timestamp("2025-03-21 12:00"),
            pd.Timestamp("2025-03-21 12:10"),
        ]
    )
    observed = np.array([100.0, 110.0, 500.0, 510.0])
    predicted = np.array([420.0, 320.0, 380.0, 690.0])
    return _manifest_rows(local_times, target_dhi=observed), observed, predicted


def test_published_model_rmse_is_measured_over_the_rows_paired_with_persistence() -> None:
    split_df, observed, predicted = _two_days_with_a_dropped_persistence_row()
    frame = _build_predictions_frame(split_df, {"dhi": predicted}, ["dhi"], kindex_kind="kstar")
    paired = np.array([False, True, False, True])

    metrics = _target_metrics(frame, "dhi")

    assert metrics["rmse_model_persistence"] == pytest.approx(
        float(np.sqrt(np.mean((predicted[paired] - observed[paired]) ** 2)))
    )


def test_persistence_skill_reconciles_from_the_two_published_rmses() -> None:
    split_df, _observed, predicted = _two_days_with_a_dropped_persistence_row()
    frame = _build_predictions_frame(split_df, {"dhi": predicted}, ["dhi"], kindex_kind="kstar")

    metrics = _target_metrics(frame, "dhi")

    assert metrics["skill_persistence"] == pytest.approx(
        1.0 - metrics["rmse_model_persistence"] / metrics["rmse_persistence"], abs=1e-12
    )


def test_the_paired_model_rmse_row_counts_the_pairs_not_the_whole_split() -> None:
    split_df, _observed, predicted = _two_days_with_a_dropped_persistence_row()
    frame = _build_predictions_frame(split_df, {"dhi": predicted}, ["dhi"], kindex_kind="kstar")

    stratified = _stratified_metrics(frame, ["dhi"], {"dhi": _target_metrics(frame, "dhi")})

    row = _stratified_row(
        stratified, stratum_kind="overall", stratum="all", metric="rmse_model_persistence"
    )
    assert int(row["n"]) == 2
