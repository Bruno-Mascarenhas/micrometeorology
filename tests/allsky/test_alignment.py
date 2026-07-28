"""Tests for allsky.data.alignment: build-time frame/sensor pairing.

Dataset-level windowing is not here — it lives once in
``allsky.data.datasets.MultimodalEmbeddingDataset`` and is covered by
``test_data_datasets.py``. This module briefly carried a second, unreachable
implementation with different semantics; see ``test_config_bounds.py`` for the
config-level contract on the window names.
"""

from __future__ import annotations

import pandas as pd
import pytest

from allsky.data.alignment import AlignmentResult, AlignmentStrategy, CenterFrame


class TestCenterFramePair:
    def test_pairs_each_frame_to_nearest_sensor(self):
        frames = pd.DatetimeIndex(["2025-03-21 08:00", "2025-03-21 08:30"])
        sensors = pd.date_range("2025-03-21 06:00", "2025-03-21 12:00", freq="5min")
        result = CenterFrame(max_distance_minutes=5.0).pair(frames, sensors)

        assert result.matched.all()
        assert (result.distance_minutes == 0.0).all()
        # 08:00 is the 24th 5-min step from 06:00.
        assert result.sensor_pos[0] == 24

    def test_unmatched_frame_beyond_tolerance(self):
        frames = pd.DatetimeIndex(["2025-03-21 08:00", "2025-03-21 11:40"])
        sensors = pd.DatetimeIndex(["2025-03-21 08:02", "2025-03-21 12:00"])
        result = CenterFrame(max_distance_minutes=5.0).pair(frames, sensors)

        assert bool(result.matched[0])  # 2 min away -> matched
        assert not bool(result.matched[1])  # 20 min away -> unmatched
        assert result.sensor_pos[1] == -1
        assert pd.isna(result.distance_minutes[1])

    def test_resolution_mismatch_still_pairs(self):
        # frame index in ns, sensor index in us: must still align (regression).
        frames = pd.DatetimeIndex(["2025-03-21 08:00"]).as_unit("ns")
        sensors = pd.date_range("2025-03-21 07:00", "2025-03-21 09:00", freq="5min").as_unit("us")
        result = CenterFrame().pair(frames, sensors)
        assert result.matched.all()
        assert result.distance_minutes[0] == pytest.approx(0.0)

    def test_non_monotonic_sensor_raises(self):
        frames = pd.DatetimeIndex(["2025-03-21 08:00"])
        sensors = pd.DatetimeIndex(["2025-03-21 09:00", "2025-03-21 08:00"])
        with pytest.raises(ValueError, match="monotonic"):
            CenterFrame().pair(frames, sensors)

    def test_empty_inputs(self):
        empty = pd.DatetimeIndex([])
        sensors = pd.date_range("2025-03-21 06:00", periods=3, freq="5min")
        result = CenterFrame().pair(empty, sensors)
        assert len(result.sensor_pos) == 0


class TestBuildTimeContract:
    def test_center_frame_satisfies_the_pairing_protocol(self):
        assert isinstance(CenterFrame(), AlignmentStrategy)

    def test_the_protocol_is_the_pairing_contract_the_manifest_builder_needs(self):
        """The protocol used to advertise dataset windowing this module never did.

        ``build_manifest`` only ever calls ``pair``; a protocol declaring
        ``select_frames`` accepted classes that could not build a manifest.
        """
        from typing import get_type_hints

        hints = get_type_hints(AlignmentStrategy.pair)
        assert hints["return"] is AlignmentResult
        assert not hasattr(AlignmentStrategy, "select_frames")
        assert not hasattr(CenterFrame, "select_frames")
