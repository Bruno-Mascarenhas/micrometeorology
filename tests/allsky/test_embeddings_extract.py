"""Tests for allsky.embeddings.extract: FakeBackbone end-to-end, resume, dry-run.

FakeBackbone.encode is the only torch touch-point, so the whole module is gated
on torch; no DINOv2 / network is ever exercised.
"""

import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.embeddings.backbone import FakeBackbone
from allsky.embeddings.extract import _load_uint8, extract_embeddings
from allsky.embeddings.storage import SafetensorsEmbeddingReader, read_index, read_meta, shard_path

pytest.importorskip("torch")  # FakeBackbone.encode builds a torch tensor


def _make_dataset(tmp_path: Path, n: int = 5, size: int = 16) -> pd.DataFrame:
    """Write *n* tiny JPEGs under ``frames/`` and return a manifest DataFrame."""
    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        name = f"allsky-20250321-09{i * 10:02d}.jpg"
        iio.imwrite(
            frames / name,
            rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            quality=90,
        )
        rows.append({"sample_id": name.removesuffix(".jpg"), "image_path": f"frames/{name}"})
    return pd.DataFrame(rows)


class TestEndToEnd:
    def test_extract_writes_shards_index_meta(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=2,
            shard_size=3,
            config_sha256="deadbeef",
        )
        assert summary["encoded"] == 5
        assert summary["skipped"] == 0
        assert summary["shards_written"] == 2  # ceil(5 / 3)
        assert summary["dim"] == 8
        assert summary["dtype"] == "fp16"

        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert set(index["sample_id"]) == set(manifest["sample_id"])

        meta = read_meta(out)
        assert meta["backbone"] == "fake"
        assert meta["revision"] == "fake-v1"
        assert meta["dim"] == 8
        assert meta["count"] == 5
        assert meta["config_sha256"] == "deadbeef"

    def test_embeddings_are_deterministic_hash_of_frame(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=8
        )
        reader = SafetensorsEmbeddingReader(out)
        backbone = FakeBackbone(dim=8)
        for _, row in manifest.iterrows():
            loaded = _load_uint8(tmp_path / row["image_path"])
            expected = backbone._embed_one(loaded)  # verifying determinism
            np.testing.assert_allclose(
                reader(row["sample_id"]), expected.astype(np.float16), rtol=0
            )

    def test_shard_sizes(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        # 5 samples, shard_size 2 -> shards of 2, 2, 1.
        assert shard_path(out, 0).exists()
        assert shard_path(out, 1).exists()
        assert shard_path(out, 2).exists()
        assert not shard_path(out, 3).exists()


class TestFrameDecoding:
    """Frame bytes are the FakeBackbone/DINOv2 input: the decoder must not shift them."""

    @staticmethod
    def _imageio_recipe(path: Path) -> np.ndarray:
        image = np.asarray(iio.imread(path))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return image.astype(np.uint8, copy=False)

    def test_matches_imageio_recipe(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3, size=32)
        for path in manifest["image_path"]:
            full = tmp_path / str(path)
            loaded = _load_uint8(full)
            expected = self._imageio_recipe(full)
            assert loaded.dtype == expected.dtype
            assert loaded.shape == expected.shape
            np.testing.assert_array_equal(loaded, expected)

    def test_grayscale_is_channel_replicated(self, tmp_path: Path):
        gray = tmp_path / "frames" / "gray.jpg"
        gray.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(12)
        iio.imwrite(gray, rng.integers(0, 256, (24, 24), dtype=np.uint8), quality=90)
        loaded = _load_uint8(gray)
        assert loaded.shape == (24, 24, 3)
        np.testing.assert_array_equal(loaded, self._imageio_recipe(gray))


class TestResume:
    def test_second_run_encodes_nothing(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        first = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=3
        )
        assert first["encoded"] == 5

        shards_before = sorted(p.name for p in out.glob("embeddings-*.safetensors"))
        second = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=3
        )
        assert second["encoded"] == 0
        assert second["skipped"] == 5
        assert second["shards_written"] == 0
        shards_after = sorted(p.name for p in out.glob("embeddings-*.safetensors"))
        assert shards_after == shards_before

    def test_resume_appends_only_new_samples(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        # Grow the dataset with two more frames, keep the originals.
        grown = _make_dataset(tmp_path, n=5)
        summary = extract_embeddings(
            grown, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        assert summary["encoded"] == 2
        assert summary["skipped"] == 3
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert len(set(index["sample_id"])) == 5

    def test_no_resume_reprocesses(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            resume=False,
        )
        assert summary["encoded"] == 3
        assert summary["skipped"] == 0


class TestResumeCompatibility:
    def test_matching_meta_resumes(self, tmp_path: Path):
        """A resume with an identical backbone/config skips already-done work."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        assert summary["encoded"] == 0
        assert summary["skipped"] == 4

    def test_pooling_change_refuses_resume(self, tmp_path: Path):
        """A different pooling on resume must raise rather than mix stores."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        changed = FakeBackbone(dim=8)
        changed.pooling = "cls"  # meta recorded pooling="fake"
        with pytest.raises(RuntimeError, match="resume"):
            extract_embeddings(
                manifest, changed, out, data_root=tmp_path, batch_size=4, shard_size=4
            )

    def test_config_sha_change_refuses_resume(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        with pytest.raises(RuntimeError, match="incompatible"):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="cfg-2",
            )

    def test_a_meta_carrying_the_legacy_digest_and_the_same_pixel_config_resumes(
        self, tmp_path: Path
    ):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
            pixel_config_sha256="pixels-1",
        )

        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="widened-cfg",
            legacy_config_sha256="legacy-cfg",
            pixel_config_sha256="pixels-1",
        )

        assert summary["skipped"] == 4
        assert read_meta(out)["config_sha256"] == "widened-cfg"

    def test_the_legacy_digest_migration_names_the_sections_the_old_gate_never_covered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
            pixel_config_sha256="pixels-1",
        )

        with caplog.at_level(logging.WARNING, logger="allsky.embeddings.extract"):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="widened-cfg",
                legacy_config_sha256="legacy-cfg",
                pixel_config_sha256="pixels-1",
            )

        assert "mask" in caplog.text
        assert "crop" in caplog.text
        assert "resize" in caplog.text

    def test_a_legacy_digest_store_that_records_no_pixel_config_refuses_resume(
        self, tmp_path: Path
    ):
        # Every store a pre-migration release wrote: the legacy digest covered
        # the embeddings section alone, so nothing on disk says which mask /
        # crop / resize / clock shaped the pixels behind these vectors.
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
        )

        with pytest.raises(RuntimeError, match="--no-resume"):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="widened-cfg",
                legacy_config_sha256="legacy-cfg",
                pixel_config_sha256="pixels-1",
            )

    def test_a_legacy_digest_store_whose_pixel_config_moved_refuses_resume(self, tmp_path: Path):
        # horizon_v1.png -> horizon_v2.png: the legacy digest stays equal while
        # every encoded pixel changes.
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
            pixel_config_sha256="pixels-1",
        )

        with pytest.raises(RuntimeError, match="--no-resume"):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="widened-cfg",
                legacy_config_sha256="legacy-cfg",
                pixel_config_sha256="pixels-2",
            )

    def test_a_refused_legacy_store_keeps_its_original_stamp(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
        )

        with pytest.raises(RuntimeError):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="widened-cfg",
                legacy_config_sha256="legacy-cfg",
                pixel_config_sha256="pixels-1",
            )

        assert read_meta(out)["config_sha256"] == "legacy-cfg"

    def test_a_legacy_digest_with_a_changed_pooling_still_refuses_resume(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="legacy-cfg",
        )
        changed = FakeBackbone(dim=8)
        changed.pooling = "other-pooling"

        with pytest.raises(RuntimeError, match="incompatible"):
            extract_embeddings(
                manifest,
                changed,
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="widened-cfg",
                legacy_config_sha256="legacy-cfg",
            )

    def test_incompatible_meta_can_be_overwritten_with_no_resume(self, tmp_path: Path):
        """--no-resume bypasses the compatibility guard (documented escape hatch)."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        changed = FakeBackbone(dim=8)
        changed.pooling = "cls"
        summary = extract_embeddings(
            manifest, changed, out, data_root=tmp_path, batch_size=4, shard_size=4, resume=False
        )
        assert summary["encoded"] == 4


class TestIncrementalIndex:
    """Finding F8: per-shard index parts, consolidated atomically at completion."""

    def test_completion_consolidates_and_removes_parts(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        # A completed run leaves a single consolidated index and no leftover parts.
        assert sorted(out.glob("index.part-*.parquet")) == []
        assert (out / "index.parquet").exists()
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert set(index["sample_id"].astype(str)) == set(manifest["sample_id"].astype(str))

    def test_resume_after_crash_reextracts_only_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import allsky.embeddings.extract as ex

        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"

        # Crash mid-run: fail writing the SECOND shard's index part, so shard 0's
        # part survives (2 ids done) while the rest never get indexed.
        original = ex._write_index_part
        calls = {"n": 0}

        def flaky(out_dir: Path, shard_index: int, part_rows: list) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash mid-run")
            original(out_dir, shard_index, part_rows)

        monkeypatch.setattr(ex, "_write_index_part", flaky)
        with pytest.raises(RuntimeError, match="crash mid-run"):
            extract_embeddings(
                manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
            )

        # Parts survive the crash; the consolidated index was never written.
        parts = sorted(out.glob("index.part-*.parquet"))
        assert parts, "at least the first shard's index part must survive the crash"
        assert not (out / "index.parquet").exists()
        part_ids = set(pd.concat([pd.read_parquet(p) for p in parts])["sample_id"].astype(str))
        assert 0 < len(part_ids) < 5  # a genuine partial state

        # Resume with the real writer: only the un-indexed ids re-extract.
        monkeypatch.setattr(ex, "_write_index_part", original)
        summary = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        assert summary["skipped"] == len(part_ids)
        assert summary["encoded"] == 5 - len(part_ids)

        # Final consolidated index equals the full sample set (== per-part union),
        # with no leftover parts.
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert len(set(index["sample_id"].astype(str))) == 5
        assert set(index["sample_id"].astype(str)) == set(manifest["sample_id"].astype(str))
        assert sorted(out.glob("index.part-*.parquet")) == []

    def test_reader_serves_all_ids_after_resume(self, tmp_path: Path):
        """The consolidated store round-trips through the reader (values intact)."""
        manifest = _make_dataset(tmp_path, n=6)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=3, shard_size=2
        )
        reader = SafetensorsEmbeddingReader(out)
        backbone = FakeBackbone(dim=8)
        for _, row in manifest.iterrows():
            loaded = _load_uint8(tmp_path / row["image_path"])
            expected = backbone._embed_one(loaded)
            np.testing.assert_allclose(
                reader(row["sample_id"]), expected.astype(np.float16), rtol=0
            )


class TestCrashedOverwrite:
    """A crashed --no-resume rerun must never leave an index describing old shards."""

    @staticmethod
    def _crash_after_first_shard(monkeypatch: pytest.MonkeyPatch) -> None:
        import allsky.embeddings.extract as ex

        original = ex.save_shard
        calls = {"n": 0}

        def flaky(path: Path, embeddings: np.ndarray) -> Path:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash mid-run")
            return original(path, embeddings)

        monkeypatch.setattr(ex, "save_shard", flaky)

    def test_crashed_no_resume_then_resume_serves_each_id_its_own_vector(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )

        # Re-extract from scratch with the rows in the opposite order (a re-filtered
        # manifest), and die after the first shard has landed.
        reordered = manifest.iloc[::-1].reset_index(drop=True)
        self._crash_after_first_shard(monkeypatch)
        with pytest.raises(RuntimeError, match="crash mid-run"):
            extract_embeddings(
                reordered,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=2,
                shard_size=2,
                resume=False,
            )
        monkeypatch.undo()

        # The stale consolidated index is gone, so the store fails closed instead
        # of mapping sample_ids onto the shard rows this run overwrote.
        assert not (out / "index.parquet").exists()
        with pytest.raises(FileNotFoundError, match="no embedding index"):
            SafetensorsEmbeddingReader(out)

        extract_embeddings(
            reordered, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        reader = SafetensorsEmbeddingReader(out)
        backbone = FakeBackbone(dim=8)
        for _, row in manifest.iterrows():
            expected = backbone._embed_one(_load_uint8(tmp_path / row["image_path"]))
            np.testing.assert_allclose(
                reader(row["sample_id"]), expected.astype(np.float16), rtol=0
            )

    def test_crashed_run_leaves_provenance_so_a_backbone_change_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        self._crash_after_first_shard(monkeypatch)
        with pytest.raises(RuntimeError, match="crash mid-run"):
            extract_embeddings(
                manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
            )
        monkeypatch.undo()

        assert read_meta(out)["dim"] == 8  # provenance survives the crash
        with pytest.raises(RuntimeError, match="incompatible"):
            extract_embeddings(
                manifest, FakeBackbone(dim=32), out, data_root=tmp_path, batch_size=2, shard_size=2
            )
        # The same backbone still resumes and only re-extracts what is missing.
        summary = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        assert summary["skipped"] == 2
        assert summary["encoded"] == 2

    def test_indexed_store_without_provenance_refuses_resume(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=2)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        (out / "embeddings.meta.json").unlink()  # a store from an older version
        with pytest.raises(RuntimeError, match=r"no embeddings\.meta\.json"):
            extract_embeddings(
                manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
            )


class TestDecodeWorkers:
    def test_bad_decode_workers_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="decode_workers"):
            extract_embeddings(
                _make_dataset(tmp_path, n=1),
                FakeBackbone(dim=8),
                tmp_path / "emb",
                data_root=tmp_path,
                decode_workers=0,
            )

    def test_thread_count_does_not_change_the_store(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=6)
        serial = tmp_path / "serial"
        threaded = tmp_path / "threaded"
        for out, workers in ((serial, 1), (threaded, 4)):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                decode_workers=workers,
            )
        left = read_index(serial)
        right = read_index(threaded)
        assert left is not None
        assert right is not None
        pd.testing.assert_frame_equal(left, right)
        for shard in (0, 1):
            assert (
                shard_path(serial, shard).read_bytes() == shard_path(threaded, shard).read_bytes()
            )


class TestDryRun:
    def test_dry_run_writes_nothing(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=2,
            shard_size=2,
            dry_run=True,
        )
        assert summary["dry_run"] is True
        assert summary["encoded"] == 0
        assert not out.exists(), "dry-run must not create the output directory"

    def test_no_resume_dry_run_destroys_nothing(self, tmp_path: Path):
        """--no-resume drops the stale index, but only once it is really writing."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        leftover = out / "index.part-00007.parquet"
        leftover.write_bytes((out / "index.parquet").read_bytes())
        before = {p.name: p.read_bytes() for p in sorted(out.iterdir())}

        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=2,
            shard_size=2,
            resume=False,
            dry_run=True,
        )
        assert {p.name: p.read_bytes() for p in sorted(out.iterdir())} == before


class TestValidation:
    def test_missing_columns_raise(self, tmp_path: Path):
        out = tmp_path / "emb"
        with pytest.raises(ValueError, match="image_path"):
            extract_embeddings(
                pd.DataFrame({"sample_id": ["a"]}),
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
            )

    def test_bad_batch_size_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size"):
            extract_embeddings(
                _make_dataset(tmp_path, n=1),
                FakeBackbone(dim=8),
                tmp_path / "emb",
                data_root=tmp_path,
                batch_size=0,
            )
