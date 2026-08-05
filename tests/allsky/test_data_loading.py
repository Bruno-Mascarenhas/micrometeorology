"""Tests for allsky.data.loading: the shared artifact loaders' declared types."""

import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from allsky.data.loading import load_split, resolve_against_root
from allsky.data.splits import DaySplit, create_day_splits, save_split_artifact

DAYS = [f"2025-03-{d:02d}" for d in range(1, 11)]


class TestLoadSplit:
    def test_return_type_is_daysplit(self):
        """``-> Any`` here silently disabled checking of every ``split.*`` access."""
        assert get_type_hints(load_split)["return"] is DaySplit

    def test_loads_the_saved_artifact(self, tmp_path: Path):
        split = create_day_splits(DAYS, seed=3)
        path = tmp_path / "splits.json"
        save_split_artifact(split, path)
        loaded = load_split(path)
        assert isinstance(loaded, DaySplit)
        assert loaded.split_id == split.split_id

    def test_missing_artifact_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="split artifact not found"):
            load_split(tmp_path / "absent.json")

    def test_module_imports_without_torch(self):
        """Contract: torch is reached only inside ``default_embedding_reader``."""
        code = (
            "import sys\n"
            "import allsky.data.loading\n"
            "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestResolveAgainstRoot:
    def test_relative_is_joined_absolute_is_kept(self, tmp_path: Path):
        assert resolve_against_root("a/b.parquet", tmp_path) == tmp_path / "a" / "b.parquet"
        absolute = tmp_path / "elsewhere.parquet"
        assert resolve_against_root(absolute, tmp_path / "root") == absolute
