"""Shared synthetic-experiment builders for the Wave C4b evaluation tests.

Mirrors the tiny fixture approach in ``tests/allsky/test_engine.py`` (C4a) but
lives in its own helper module so the evaluation tests can reuse it without
importing or mutating that test file: a 3-day v2 manifest built with the real
manifest builder, a persisted day split, and a deterministic dict-backed
embedding reader.  Everything is offline, CPU-only and needs no image files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from allsky import solar
from allsky.config import ExperimentConfig, SiteConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet
from allsky.data.splits import create_day_splits, save_split_artifact

if TYPE_CHECKING:
    from pathlib import Path

_MET = {
    "AirT1_C_Avg": (20.0, 30.0),
    "DP1_C_Avg": (10.0, 20.0),
    "RH1": (50.0, 90.0),
    "BP1_mbar_Avg": (1005.0, 1015.0),
    "WS_ms": (0.0, 8.0),
    "WindDir": (0.0, 360.0),
}


class DictEmbeddingReader:
    """Deterministic dict-backed reader implementing the EmbeddingReader protocol."""

    def __init__(self, sample_ids: list[str], dim: int = 8) -> None:
        rng = np.random.default_rng(0)
        self._data = {str(s): rng.standard_normal(dim).astype(np.float32) for s in sample_ids}
        self.dim = dim

    def __call__(self, sample_id: str) -> np.ndarray:
        return self._data[str(sample_id)]

    def sample_ids(self) -> list[str]:
        return list(self._data)


def _sensor(site: SiteConfig, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(first + pd.Timedelta(hours=8), last + pd.Timedelta(hours=19), freq="5min")
    rng = np.random.default_rng(0)
    e0h = solar.extraterrestrial_ghi(index, site)
    data = {k: rng.uniform(lo, hi, len(index)) for k, (lo, hi) in _MET.items()}
    data["CM3Up_Wm2_Avg"] = np.clip(0.7 * e0h, 0.0, None)
    data["PSP_Wm2_Avg"] = np.clip(0.2 * e0h, 0.0, None)
    return pd.DataFrame(data, index=index)


def make_dataset(
    tmp_path: Path,
    *,
    n_days: int = 3,
    per_day: int = 20,
    write_images: bool = False,
    image_px: int = 8,
) -> tuple[Path, pd.DataFrame, object]:
    """Build a tiny manifest + split under ``tmp_path/data``; return (root, manifest, split).

    When *write_images* is True, a tiny ``image_px``x``image_px`` RGB JPEG is
    written at every manifest ``image_path`` so ``input_mode='image'`` experiments
    can actually decode frames (embedding-mode tests leave *write_images* False and
    need no image files).
    """
    site = SiteConfig()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2025-03-20", periods=n_days, freq="D")
    rows = []
    idx = 0
    for day in days:
        for ts in pd.date_range(day + pd.Timedelta(hours=9), periods=per_day, freq="30min"):
            rows.append(
                {
                    "frame_path": f"frames/allsky-{ts:%Y%m%d-%H%M}.jpg",
                    "timestamp": ts,
                    "video": "v.mp4",
                    "index": idx,
                }
            )
            idx += 1
    frames = pd.DataFrame(rows)
    manifest, meta = build_manifest(
        frames, _sensor(site, days[0], days[-1]), site=site, data_root=root
    )
    write_manifest_parquet(manifest, meta, root / "manifest.parquet")
    split = create_day_splits(
        manifest["day_id"].tolist(), val_fraction=0.34, test_fraction=0.0, seed=0
    )
    save_split_artifact(split, root / "splits.json")
    if write_images:
        write_frame_images(root, manifest, image_px=image_px)
    return root, manifest, split


def write_frame_images(root: Path, manifest: pd.DataFrame, *, image_px: int = 8) -> None:
    """Write a tiny deterministic JPEG at every manifest ``image_path`` under *root*."""
    import imageio.v3 as iio

    rng = np.random.default_rng(1)
    for rel in manifest["image_path"].astype(str):
        full = root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(full, rng.integers(0, 256, (image_px, image_px, 3), dtype=np.uint8), quality=90)


def make_embeddings_store(
    root: Path, manifest: pd.DataFrame, *, dim: int = 8, shard_size: int = 16, subdir: str = "emb"
) -> Path:
    """Write a real safetensors embedding store (shards + index + meta) under ``root/subdir``.

    Deterministic per ``sample_id``; returns the store directory.  Lets tests
    exercise the engine's default (non-injected) :class:`SafetensorsEmbeddingReader`
    — including the preload path — over real on-disk shards.
    """
    from allsky.embeddings.storage import save_shard, shard_path, write_index, write_meta

    out = root / subdir
    out.mkdir(parents=True, exist_ok=True)
    ids = [str(s) for s in manifest["sample_id"]]
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(ids), dim)).astype(np.float32)
    rows = []
    for shard_index, start in enumerate(range(0, len(ids), shard_size)):
        block = embeddings[start : start + shard_size]
        save_shard(shard_path(out, shard_index), block)
        for row, sid in enumerate(ids[start : start + shard_size]):
            rows.append({"sample_id": sid, "shard": shard_index, "row": row})
    write_index(out, pd.DataFrame(rows, columns=["sample_id", "shard", "row"]))
    write_meta(
        out,
        {
            "backbone": "fake",
            "revision": "fake-v1",
            "pooling": "fake",
            "dim": dim,
            "transform": "identity",
            "config_sha256": None,
            "count": len(ids),
            "dtype": "fp16",
        },
    )
    return out


def make_config(
    root: Path,
    *,
    model: str = "sensor_only",
    epochs: int = 2,
    batch_size: int = 8,
    targets: dict | None = None,
) -> ExperimentConfig:
    """Build an embedding-mode experiment config over the synthetic dataset."""
    return ExperimentConfig.model_validate(
        {
            "experiment": True,
            "seed": 0,
            "output_dir": str(root / "out"),
            "data": {
                "manifest": "manifest.parquet",
                "data_root": str(root),
                "split_artifact": "splits.json",
                "embeddings_dir": "emb",
                "input_mode": "embedding",
            },
            "features": {"set": "safe"},
            "targets": targets
            or {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
            "model": {"name": model},
            "train": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": {"monitor": "val_loss", "patience": 100},
            },
        }
    )


def reader_for(manifest: pd.DataFrame, dim: int = 8) -> DictEmbeddingReader:
    """Dict-backed embedding reader covering every manifest sample_id."""
    return DictEmbeddingReader([str(s) for s in manifest["sample_id"]], dim=dim)
