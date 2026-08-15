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

Torch-free: the helpers run over hand-built manifest rows.
"""

import numpy as np
import pandas as pd
import pytest

from allsky.clearsky import haurwitz_ghi_from_cos_zenith
from allsky.config import SITE_TZ, SITE_UTC_OFFSET_HOURS, SiteConfig
from allsky.evaluation.evaluator import (
    _add_strata,
    _build_predictions_frame,
    _clearsky_reference,
    _stratified_metrics,
    _target_metrics,
)
from allsky.solar import cos_zenith, extraterrestrial_ghi

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


@pytest.mark.parametrize(
    ("target_kt", "target_kindex", "expected_band"),
    [(0.30, 0.42, "overcast_lt0.35"), (0.50, 0.70, "partial_0.35-0.65")],
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
