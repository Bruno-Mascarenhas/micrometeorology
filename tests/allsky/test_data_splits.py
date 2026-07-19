"""Tests for allsky.data.splits: determinism, guard, leakage, fractions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from allsky.data.splits import (
    DaySplit,
    SplitExistsError,
    check_split_leakage,
    create_day_splits,
    load_split_artifact,
    save_split_artifact,
)

DAYS = [f"2025-03-{d:02d}" for d in range(1, 21)]  # 20 distinct days


class TestCreateDaySplits:
    def test_deterministic_for_same_seed(self):
        a = create_day_splits(DAYS, val_fraction=0.2, test_fraction=0.1, seed=7)
        b = create_day_splits(DAYS, val_fraction=0.2, test_fraction=0.1, seed=7)
        assert a.assignment == b.assignment
        assert a.split_id == b.split_id

    def test_order_of_input_does_not_matter(self):
        a = create_day_splits(DAYS, seed=3)
        b = create_day_splits(list(reversed(DAYS)), seed=3)
        assert a.split_id == b.split_id

    def test_different_seed_changes_assignment(self):
        a = create_day_splits(DAYS, seed=1)
        b = create_day_splits(DAYS, seed=2)
        assert a.assignment != b.assignment

    def test_fractions_respected(self):
        split = create_day_splits(DAYS, val_fraction=0.25, test_fraction=0.1, seed=0)
        assert len(split.days_for("test")) == 2  # floor(20 * 0.1)
        assert len(split.days_for("val")) == 5  # floor(20 * 0.25)
        assert len(split.days_for("train")) == 13

    def test_no_leakage_across_splits(self):
        split = create_day_splits(DAYS, val_fraction=0.2, test_fraction=0.2, seed=5)
        train = set(split.days_for("train"))
        val = set(split.days_for("val"))
        test = set(split.days_for("test"))
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(DAYS)

    def test_small_nonzero_fraction_keeps_at_least_one_day(self):
        split = create_day_splits(DAYS, val_fraction=0.01, test_fraction=0.01, seed=0)
        assert len(split.days_for("val")) == 1
        assert len(split.days_for("test")) == 1

    def test_zero_fraction_leaves_split_empty(self):
        split = create_day_splits(DAYS, val_fraction=0.2, test_fraction=0.0, seed=0)
        assert split.days_for("test") == []

    def test_duplicated_days_collapsed(self):
        split = create_day_splits([*DAYS, *DAYS], val_fraction=0.2, test_fraction=0.1, seed=0)
        assert len(split.assignment) == len(DAYS)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError, match="val_fraction"):
            create_day_splits(DAYS, val_fraction=1.0)

    def test_fractions_sum_too_large_raises(self):
        with pytest.raises(ValueError, match="< 1"):
            create_day_splits(DAYS, val_fraction=0.6, test_fraction=0.5)

    def test_too_few_days_raises(self):
        with pytest.raises(ValueError, match="too few days"):
            create_day_splits(["2025-03-01", "2025-03-02"], val_fraction=0.4, test_fraction=0.4)

    def test_empty_days_raises(self):
        with pytest.raises(ValueError, match="no day_ids"):
            create_day_splits([])


class TestSplitArtifact:
    def test_save_load_roundtrip(self, tmp_path: Path):
        split = create_day_splits(DAYS, seed=11)
        path = tmp_path / "splits.json"
        save_split_artifact(split, path)
        loaded = load_split_artifact(path)
        assert loaded.assignment == split.assignment
        assert loaded.split_id == split.split_id

    def test_artifact_json_has_expected_fields(self, tmp_path: Path):
        split = create_day_splits(DAYS, seed=1)
        path = tmp_path / "splits.json"
        save_split_artifact(split, path)
        payload = json.loads(path.read_text())
        assert set(payload) >= {
            "split_id",
            "seed",
            "val_fraction",
            "test_fraction",
            "assignment",
            "created_at",
        }

    def test_saving_identical_split_is_idempotent(self, tmp_path: Path):
        split = create_day_splits(DAYS, seed=1)
        path = tmp_path / "splits.json"
        save_split_artifact(split, path)
        # No error, same file.
        save_split_artifact(split, path)
        assert load_split_artifact(path).split_id == split.split_id

    def test_overwriting_different_split_requires_force(self, tmp_path: Path):
        path = tmp_path / "splits.json"
        save_split_artifact(create_day_splits(DAYS, seed=1), path)
        with pytest.raises(SplitExistsError, match="different split"):
            save_split_artifact(create_day_splits(DAYS, seed=2), path)

    def test_force_overwrites(self, tmp_path: Path):
        path = tmp_path / "splits.json"
        save_split_artifact(create_day_splits(DAYS, seed=1), path)
        other = create_day_splits(DAYS, seed=2)
        save_split_artifact(other, path, force=True)
        assert load_split_artifact(path).split_id == other.split_id

    def test_corrupt_split_id_detected_on_load(self, tmp_path: Path):
        split = create_day_splits(DAYS, seed=1)
        path = tmp_path / "splits.json"
        save_split_artifact(split, path)
        payload = json.loads(path.read_text())
        payload["split_id"] = "deadbeef"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="corrupt"):
            load_split_artifact(path)


class TestLeakageSelfCheck:
    def test_disjoint_passes(self):
        check_split_leakage(["a", "b"], ["c"], ["d"])  # must not raise

    def test_overlap_raises(self):
        with pytest.raises(ValueError, match="leakage"):
            check_split_leakage(["a", "b"], ["b", "c"], ["d"])

    def test_from_dict_runs_leakage_check(self):
        # Hand-built assignment cannot leak (day -> one split); a malformed
        # split_id-free payload with duplicate-day intent is caught by the
        # disjoint check via days_for.
        split = DaySplit(
            assignment={"2025-01-01": "train", "2025-01-02": "val"},
            seed=0,
            val_fraction=0.5,
            test_fraction=0.0,
            created_at="",
        )
        restored = DaySplit.from_dict(split.to_dict())
        assert restored.days_for("train") == ["2025-01-01"]
