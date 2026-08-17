"""The statistical gates, and the one distinction the whole design rests on.

A range gate rejects the impossible. These reject the improbable, and the danger
of that is cutting real weather: a convective cold pool IS a large temperature
step. What separates the two is not the size of the jump but whether the series
comes BACK — so the test that matters most here is the one asserting a step that
stays down survives.
"""

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors.quality import mask_persistent_runs, mask_step_excursions


def _frame(column: str, values: list[float], *, freq: str = "5min") -> pd.DataFrame:
    """A five-minute frame on the archive's own microsecond-resolution index."""
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq).astype("datetime64[us]")
    return pd.DataFrame({column: values}, index=index)


def test_a_value_that_leaves_and_returns_is_masked_but_its_neighbours_are_not():
    frame = _frame("BP1_mbar_Avg", [1013.0, 1013.0, 996.0, 1013.0, 1013.0])

    frame, removed = mask_step_excursions(
        frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 2}]
    )

    assert removed == 1
    assert frame["BP1_mbar_Avg"].isna().tolist() == [False, False, True, False, False]


def test_a_step_that_stays_down_is_left_alone():
    """A cold pool does not recover in five minutes; a sensor dropout does.

    Without this the gate cuts the convective events the station exists to record:
    the same threshold on a plain step test flags them at eight times the archive's
    rain-coincidence baseline.
    """
    frame = _frame("Temp1_Avg", [28.0, 28.0, 22.0, 22.0, 22.5])

    frame, removed = mask_step_excursions(
        frame, [{"column": "Temp1_Avg", "threshold": 3.0, "return_tol": 1.5, "max_len": 2}]
    )

    assert removed == 0
    assert not frame["Temp1_Avg"].isna().any()


def test_a_value_that_comes_back_to_the_wrong_level_is_not_an_excursion():
    """Returning near the departure point is what makes it an excursion."""
    frame = _frame("Temp1_Avg", [28.0, 28.0, 22.0, 25.0, 25.0])

    _frame_out, removed = mask_step_excursions(
        frame, [{"column": "Temp1_Avg", "threshold": 3.0, "return_tol": 1.5, "max_len": 2}]
    )

    assert removed == 0


def test_a_gap_in_the_record_is_not_a_step():
    """Two readings either side of missing hours never stood next to each other."""
    frame = _frame("BP1_mbar_Avg", [1013.0, 996.0, 1013.0], freq="1h")

    _frame_out, removed = mask_step_excursions(
        frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 2}]
    )

    assert removed == 0


def test_a_run_longer_than_the_column_tolerates_is_masked():
    frame = _frame("RH1_Avg", [50.0] + [72.47] * 6 + [55.0])

    frame, removed = mask_persistent_runs(frame, [{"column": "RH1_Avg", "min_run": 4}])

    assert removed == 6
    assert frame["RH1_Avg"].notna().tolist() == [True] + [False] * 6 + [True]


def test_a_run_at_the_tolerated_length_survives():
    frame = _frame("RH1_Avg", [50.0] + [72.47] * 4 + [55.0])

    _frame_out, removed = mask_persistent_runs(frame, [{"column": "RH1_Avg", "min_run": 4}])

    assert removed == 0


def test_a_gap_breaks_a_run_rather_than_bridging_it():
    """Identical values either side of missing hours are not evidence of a rail."""
    frame = _frame("RH1_Avg", [72.47] * 6, freq="1h")

    _frame_out, removed = mask_persistent_runs(frame, [{"column": "RH1_Avg", "min_run": 4}])

    assert removed == 0


def test_an_anemometer_frozen_above_calm_is_masked_despite_the_exemption():
    """The 18-day rail this test exists for sat at 0.281 m/s, not at zero."""
    frame = _frame("WS_ms_S_WVT", [0.281] * 8)

    _frame_out, removed = mask_persistent_runs(frame, [{"column": "WS_ms_S_WVT", "min_run": 4}])

    assert removed == 8


def test_a_reading_frozen_at_the_calm_level_is_still_masked():
    """There is no exemption for calm, and the measurement is why.

    Genuine calm runs reach six samples at their longest here; the rails run 110
    to 255 hours. An exemption keyed to the LEVEL would have protected exactly the
    sensors that died reporting zero.
    """
    frame = _frame("WS_WXT_Avg", [0.0] * 8)

    _frame_out, removed = mask_persistent_runs(frame, [{"column": "WS_WXT_Avg", "min_run": 4}])

    assert removed == 8


def test_a_column_the_logger_never_wrote_is_skipped_rather_than_raising():
    """Headers change between eras, and a limit may name a column of another one."""
    frame = _frame("BP1_mbar_Avg", [1013.0, 1013.0])

    _frame_out, removed = mask_step_excursions(
        frame, [{"column": "Pmb_WXT_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 2}]
    )

    assert removed == 0


def test_a_masked_neighbour_does_not_blind_the_excursion_test():
    """The gate before this one usually took one side of the very dropout sought.

    Differencing across that hole made the test blind exactly where it was needed:
    the spike loses the neighbour it would have been measured against.
    """
    frame = _frame("BP1_mbar_Avg", [1013.0, np.nan, 996.0, 1013.0, 1013.0])

    _frame_out, removed = mask_step_excursions(
        frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 2}]
    )

    assert removed == 1


def test_an_excursion_window_shorter_than_one_sample_is_refused():
    """It would detect nothing and report zero, which reads as a clean archive."""
    frame = _frame("BP1_mbar_Avg", [1013.0, 1013.0, 996.0, 1013.0])

    with pytest.raises(ValueError, match="max_len must be at least 1"):
        mask_step_excursions(
            frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 0}]
        )


def test_the_reported_count_never_includes_a_cell_that_was_already_missing():
    """`archive_report.json` tallies must sum to the real raw-to-QC delta."""
    frame = _frame("BP1_mbar_Avg", [1013.0, 996.0, np.nan, 996.0, 1013.0])
    already_missing = int(frame["BP1_mbar_Avg"].isna().sum())

    frame, removed = mask_step_excursions(
        frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 3}]
    )

    assert removed == int(frame["BP1_mbar_Avg"].isna().sum()) - already_missing


def test_missing_samples_neither_start_nor_extend_a_run():
    frame = _frame("RH1_Avg", [72.47, np.nan, 72.47, 72.47, 72.47, 72.47])

    _frame_out, removed = mask_persistent_runs(frame, [{"column": "RH1_Avg", "min_run": 3}])

    assert removed == 4


def test_a_double_dip_does_not_consume_the_reading_between_the_two():
    """The sample that recovers one dip is also a departure from it.

    Measured against the dip as its baseline, the ambient reading looks like the
    excursion and gets masked. Twenty-one good samples of the real archive were
    removed this way.
    """
    frame = _frame("BP1_mbar_Avg", [1013.32, 1013.37, 990.83, 1013.30, 990.79, 1013.30, 1013.30])

    frame, removed = mask_step_excursions(
        frame, [{"column": "BP1_mbar_Avg", "threshold": 1.0, "return_tol": 1.0, "max_len": 2}]
    )

    assert removed == 2
    assert frame["BP1_mbar_Avg"].iloc[3] == 1013.30, "the reading between the dips is ambient"
