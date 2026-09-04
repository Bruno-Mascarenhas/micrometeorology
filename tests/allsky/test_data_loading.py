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


class TestLoadManifestVerifiesTheSidecarAgainstTheParquet:
    """`write_manifest_parquet` performs two independent atomic writes with no
    transaction linking them, and both hash guards downstream — the evaluator's
    and the resume provenance check — compare a checkpoint's stored string
    against the SIDECAR's stored string, never against the parquet's bytes. A
    parquet left over from a half-completed rewrite passed both."""

    @staticmethod
    def _written(tmp_path: Path):
        import pandas as pd

        from allsky.data.manifest import write_manifest_parquet

        manifest = pd.DataFrame({"sample_id": ["a", "b"], "target_dhi": [1.0, 2.0]})
        path = tmp_path / "manifest.parquet"
        meta = write_manifest_parquet(manifest, {"dataset_version": "v2"}, path)
        return manifest, path, meta

    def test_a_matching_pair_loads(self, tmp_path: Path):
        from allsky.data.loading import load_manifest

        manifest, path, meta = self._written(tmp_path)

        loaded, loaded_meta = load_manifest(path)

        assert len(loaded) == len(manifest)
        assert loaded_meta["manifest_sha256"] == meta["manifest_sha256"]

    def test_a_parquet_that_does_not_match_its_sidecar_is_refused(self, tmp_path: Path):
        """The sidecar stays as written; only the parquet is replaced, which is
        what a crash between the two atomic writes leaves behind."""
        import pandas as pd

        from allsky.data.loading import load_manifest

        _manifest, path, _meta = self._written(tmp_path)
        pd.DataFrame({"sample_id": ["a"], "target_dhi": [9.0]}).to_parquet(path, index=False)

        with pytest.raises(ValueError, match="manifest_sha256"):
            load_manifest(path)


class TestLoadManifestProvenance:
    def test_a_manifest_that_is_not_there_is_named(self, tmp_path: Path):
        from allsky.data.loading import load_manifest

        with pytest.raises(FileNotFoundError, match="manifest parquet not found"):
            load_manifest(tmp_path / "absent.parquet")

    def test_a_manifest_with_no_sidecar_degrades_to_an_empty_meta(self, tmp_path: Path, caplog):
        """The provenance fields the sidecar carries — the hash check, split_id,
        dataset_version — are then unavailable, which has to be said out loud."""
        import pandas as pd

        from allsky.data.loading import load_manifest

        path = tmp_path / "manifest.parquet"
        pd.DataFrame({"sample_id": ["a"]}).to_parquet(path, index=False)

        with caplog.at_level("WARNING"):
            manifest, meta = load_manifest(path)

        assert len(manifest) == 1
        assert meta == {}
        assert "sidecar" in caplog.text
