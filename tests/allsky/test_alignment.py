"""Tests for allsky.data.alignment: strategies, pairing, windowing, registry."""

from __future__ import annotations

import pandas as pd
import pytest

from allsky.data.alignment import (
    AlignmentStrategy,
    AttentionPooling,
    CenterFrame,
    MeanEmbedding,
    available_strategies,
    get_strategy,
    register_strategy,
)


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

    def test_select_frames_nearest_single(self):
        frames = pd.date_range("2025-03-21 08:00", periods=5, freq="30min")
        cf = CenterFrame(max_distance_minutes=5.0)
        assert cf.select_frames(pd.Timestamp("2025-03-21 09:01"), frames) == [2]
        assert cf.select_frames(pd.Timestamp("2025-03-21 09:20"), frames) == []


class TestWindowStrategies:
    def test_mean_embedding_returns_all_frames_in_window(self):
        frames = pd.date_range("2025-03-21 08:00", periods=7, freq="5min")
        strategy = MeanEmbedding(window_minutes=20.0)
        positions = strategy.select_frames(pd.Timestamp("2025-03-21 08:15"), frames)
        # window [08:05, 08:25] -> frames at 08:05,08:10,08:15,08:20,08:25.
        assert positions == [1, 2, 3, 4, 5]

    def test_window_positions_time_ordered_for_shuffled_index(self):
        times = pd.to_datetime(["2025-03-21 08:20", "2025-03-21 08:00", "2025-03-21 08:10"])
        frames = pd.DatetimeIndex(times)
        positions = AttentionPooling(window_minutes=60.0).select_frames(
            pd.Timestamp("2025-03-21 08:10"), frames
        )
        # ordered by time: 08:00 (pos1), 08:10 (pos2), 08:20 (pos0).
        assert positions == [1, 2, 0]

    def test_distinct_ids(self):
        assert CenterFrame.id == "center_frame"
        assert MeanEmbedding.id == "mean_embedding"
        assert AttentionPooling.id == "attention_pooling"


class TestRegistry:
    def test_get_strategy_builds_instance(self):
        strategy = get_strategy("center_frame", max_distance_minutes=3.0)
        assert isinstance(strategy, CenterFrame)
        assert strategy.max_distance_minutes == 3.0

    def test_all_builtins_registered(self):
        assert set(available_strategies()) >= {
            "center_frame",
            "mean_embedding",
            "attention_pooling",
        }

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError, match="unknown alignment strategy"):
            get_strategy("does_not_exist")

    def test_register_custom_strategy(self):
        class Fixed:
            id = "fixed_test"

            def select_frames(self, sample_time, frame_index):  # noqa: ARG002
                return [0]

        register_strategy("fixed_test", Fixed)
        strategy = get_strategy("fixed_test")
        assert isinstance(strategy, AlignmentStrategy)  # runtime_checkable protocol
        assert strategy.select_frames(pd.Timestamp("2025-03-21 08:00"), pd.DatetimeIndex([])) == [0]

    def test_center_frame_satisfies_protocol(self):
        assert isinstance(CenterFrame(), AlignmentStrategy)
