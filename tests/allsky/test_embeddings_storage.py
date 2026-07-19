"""Tests for allsky.embeddings.storage: shard roundtrip, reader, validation, atomicity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from allsky.data.datasets import EmbeddingReader, MultimodalEmbeddingDataset
from allsky.embeddings.storage import (
    EMBEDDINGS_TENSOR_KEY,
    INDEX_FILENAME,
    EmbeddingValidationReport,
    SafetensorsEmbeddingReader,
    load_shard,
    read_index,
    save_shard,
    shard_filename,
    shard_path,
    validate_embeddings,
    write_index,
    write_meta,
)


def _write_store(
    out_dir: Path, sample_ids: list[str], dim: int = 4, *, shard_size: int = 3
) -> np.ndarray:
    """Write a deterministic embedding store (shards + index + meta); return fp32 source."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(sample_ids), dim)).astype(np.float32)
    rows = []
    for shard_index, start in enumerate(range(0, len(sample_ids), shard_size)):
        block = embeddings[start : start + shard_size]
        save_shard(shard_path(out_dir, shard_index), block)
        for row, sid in enumerate(sample_ids[start : start + shard_size]):
            rows.append({"sample_id": sid, "shard": shard_index, "row": row})
    write_index(out_dir, pd.DataFrame(rows, columns=["sample_id", "shard", "row"]))
    write_meta(
        out_dir,
        {
            "backbone": "fake",
            "revision": "fake-v1",
            "pooling": "fake",
            "dim": dim,
            "transform": "identity",
            "config_sha256": None,
            "count": len(sample_ids),
            "dtype": "fp16",
        },
    )
    return embeddings


class TestShardRoundtrip:
    def test_shard_is_fp16_and_roundtrips(self, tmp_path: Path):
        source = np.array([[0.1, -0.2, 0.3], [1.5, 2.5, -3.5]], dtype=np.float32)
        path = save_shard(tmp_path / shard_filename(0), source)
        loaded = load_shard(path)
        assert loaded.dtype == np.float16
        assert loaded.shape == (2, 3)
        # Values roundtrip to fp16 precision (lossy but deterministic).
        np.testing.assert_allclose(loaded.astype(np.float32), source.astype(np.float16), rtol=0)

    def test_save_shard_rejects_non_2d(self, tmp_path: Path):
        with pytest.raises(ValueError, match="2-D"):
            save_shard(tmp_path / shard_filename(0), np.zeros(4, dtype=np.float32))

    def test_index_roundtrip(self, tmp_path: Path):
        _write_store(tmp_path, ["allsky-20250321-0900", "allsky-20250321-0930"], dim=4)
        index = read_index(tmp_path)
        assert index is not None
        assert list(index.columns) == ["sample_id", "shard", "row"]
        assert set(index["sample_id"]) == {"allsky-20250321-0900", "allsky-20250321-0930"}

    def test_read_index_missing_is_none(self, tmp_path: Path):
        assert read_index(tmp_path) is None


class TestReader:
    def test_reader_dim_and_values(self, tmp_path: Path):
        ids = [f"allsky-20250321-09{m:02d}" for m in (0, 10, 20, 30, 40)]
        source = _write_store(tmp_path, ids, dim=4, shard_size=2)
        reader = SafetensorsEmbeddingReader(tmp_path)
        assert reader.dim == 4
        assert len(reader) == len(ids)
        for i, sid in enumerate(ids):
            got = reader(sid)
            assert got.shape == (4,)
            assert got.dtype == np.float32
            np.testing.assert_allclose(got, source[i].astype(np.float16), rtol=0)

    def test_reader_satisfies_embedding_reader_protocol(self, tmp_path: Path):
        _write_store(tmp_path, ["allsky-20250321-0900"], dim=4)
        reader = SafetensorsEmbeddingReader(tmp_path)
        assert isinstance(reader, EmbeddingReader)

    def test_missing_sample_id_raises_keyerror_naming_it(self, tmp_path: Path):
        _write_store(tmp_path, ["allsky-20250321-0900"], dim=4)
        reader = SafetensorsEmbeddingReader(tmp_path)
        with pytest.raises(KeyError, match="allsky-20250321-9999"):
            reader("allsky-20250321-9999")

    def test_reader_no_index_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=INDEX_FILENAME):
            SafetensorsEmbeddingReader(tmp_path)

    def test_lru_bounds_open_shards(self, tmp_path: Path):
        ids = [f"allsky-20250321-09{m:02d}" for m in range(0, 60, 10)]
        _write_store(tmp_path, ids, dim=4, shard_size=1)  # one shard per sample
        reader = SafetensorsEmbeddingReader(tmp_path, cache_size=2)
        for sid in ids:
            reader(sid)
        assert len(reader._cache) <= 2  # asserting the LRU bound


class TestPreload:
    def test_preloaded_matches_lru_vectors(self, tmp_path: Path):
        # Finding F7: preload loads every shard once into one resident array; it
        # must serve byte-identical vectors to the lazy LRU path.
        ids = [f"allsky-20250321-09{m:02d}" for m in range(0, 60, 10)]
        source = _write_store(tmp_path, ids, dim=4, shard_size=2)
        lru = SafetensorsEmbeddingReader(tmp_path)
        preloaded = SafetensorsEmbeddingReader(tmp_path, preload=True)
        assert preloaded.preloaded is True
        assert lru.preloaded is False
        assert preloaded.dim == lru.dim == 4
        assert len(preloaded) == len(ids)
        for i, sid in enumerate(ids):
            got = preloaded(sid)
            assert got.shape == (4,)
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, lru(sid))
            np.testing.assert_allclose(got, source[i].astype(np.float16), rtol=0)

    def test_preloaded_missing_id_raises_naming_it(self, tmp_path: Path):
        _write_store(tmp_path, ["allsky-20250321-0900"], dim=4)
        reader = SafetensorsEmbeddingReader(tmp_path, preload=True)
        with pytest.raises(KeyError, match="allsky-20250321-9999"):
            reader("allsky-20250321-9999")


class TestReaderVsDataset:
    def test_embedding_dataset_uses_reader(self, tmp_path: Path):
        pytest.importorskip("torch")
        ids = ["allsky-20250321-0900", "allsky-20250321-0930", "allsky-20250321-1000"]
        _write_store(tmp_path, ids, dim=4)
        reader = SafetensorsEmbeddingReader(tmp_path)
        rng = np.random.default_rng(1)
        manifest = pd.DataFrame(
            {
                "sample_id": ids,
                "air_temp_c": rng.normal(size=len(ids)),
                "rel_humidity": rng.uniform(40, 90, size=len(ids)),
                "sky_class": [0, 1, 2],
                "target_dhi": [100.0, 200.0, np.nan],
                "target_kindex": [0.8, 0.5, 0.2],
                "cloud_fraction": [np.nan, np.nan, np.nan],
            }
        )
        dataset = MultimodalEmbeddingDataset(
            manifest,
            ["air_temp_c", "rel_humidity"],
            embedding_reader=reader,
            train=True,
        )
        assert dataset.embedding_dim == reader.dim == 4
        item = dataset[1]
        assert item["embedding"].shape == (4,)
        assert set(item) == {
            "features",
            "embedding",
            "dhi",
            "kindex",
            "sky_class",
            "cloud_fraction",
        }


class TestValidation:
    def test_reports_missing_and_duplicate(self):
        index = pd.DataFrame(
            {
                "sample_id": ["a", "b", "b"],  # b duplicated
                "shard": [0, 0, 0],
                "row": [0, 1, 2],
            }
        )
        report = validate_embeddings(index, ["a", "b", "c", "d"])  # c, d missing
        assert isinstance(report, EmbeddingValidationReport)
        assert report.missing == ["c", "d"]
        assert report.duplicate == ["b"]
        assert not report.ok
        with pytest.raises(ValueError, match=r"missing|duplicate"):
            report.raise_if_failed()

    def test_ok_report_passes(self):
        index = pd.DataFrame({"sample_id": ["a", "b"], "shard": [0, 0], "row": [0, 1]})
        report = validate_embeddings(index, ["a", "b"])
        assert report.ok
        report.raise_if_failed()  # must not raise

    def test_validate_against_written_store(self, tmp_path: Path):
        ids = ["allsky-20250321-0900", "allsky-20250321-0930"]
        _write_store(tmp_path, ids, dim=4)
        index = read_index(tmp_path)
        assert index is not None
        report = validate_embeddings(index, [*ids, "allsky-20250321-1000"])
        assert report.missing == ["allsky-20250321-1000"]
        assert report.duplicate == []


class TestAtomicity:
    def test_failed_shard_write_leaves_no_partial_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import safetensors.numpy as st

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(st, "save_file", _boom)
        target = tmp_path / shard_filename(0)
        with pytest.raises(RuntimeError, match="disk full"):
            save_shard(target, np.zeros((2, 4), dtype=np.float32))

        # No final shard, and no temp debris in the directory.
        assert not target.exists()
        leftovers = list(tmp_path.iterdir())
        assert leftovers == [], f"unexpected files after failed write: {leftovers}"

    def test_shard_tensor_key(self, tmp_path: Path):
        import safetensors.numpy as st

        save_shard(tmp_path / shard_filename(0), np.zeros((1, 4), dtype=np.float32))
        loaded = st.load_file(str(tmp_path / shard_filename(0)))
        assert list(loaded) == [EMBEDDINGS_TENSOR_KEY]


class TestPreloadReadOnly:
    def test_preloaded_vectors_are_immutable_views(self, tmp_path: Path):
        # The resident matrix is shared across every caller (and COW-shared
        # across fork workers); a mutable row view would let one consumer
        # silently corrupt the store for all others.
        ids = [f"allsky-20250321-09{m:02d}" for m in range(0, 30, 10)]
        _write_store(tmp_path, ids, dim=4, shard_size=2)
        reader = SafetensorsEmbeddingReader(tmp_path, preload=True)
        vector = reader(ids[0])
        with pytest.raises(ValueError, match="read-only"):
            vector[0] = 123.0
        # Parity with the lazy path is unaffected by the write protection.
        lazy = SafetensorsEmbeddingReader(tmp_path)
        np.testing.assert_array_equal(np.asarray(vector), lazy(ids[0]))
