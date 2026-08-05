"""Evaluator review findings: inference worker count and k-index provenance.

Two independent defects, both exercised through the real
:func:`allsky.evaluation.evaluator.evaluate_checkpoint`:

- ``_run_inference`` hardcoded ``num_workers=0``, so image-mode evaluation
  decoded the whole split single-threaded with no knob to change it;
- ``targets.kindex.kind`` was written into the checkpoint and never compared
  against the k-index the manifest was actually built with, so a report could
  label k* numbers as k_t.

CPU-only and offline: a tiny conv backbone stands in for DINOv2.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from allsky.config import ExperimentConfig
from allsky.evaluation.evaluator import (
    _check_kindex_kind,
    _manifest_kindex_kind,
    evaluate_checkpoint,
)
from allsky.training.engine import run_experiment
from tests.allsky import _synthetic as synthetic
from tests.allsky.test_engine_findings import TinyConvBackbone, _image_cfg


class TestInferenceWorkers:
    def test_image_mode_predictions_identical_across_worker_counts(self, tmp_path: Path):
        root, _manifest, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=8
        )
        run_dir = tmp_path / "run"
        run_experiment(
            _image_cfg(root, model="film", epochs=1),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=lambda: TinyConvBackbone(dim=12),
        )

        def _evaluate(workers: int) -> pd.DataFrame:
            torch.manual_seed(0)
            return evaluate_checkpoint(
                run_dir / "best.ckpt",
                split="val",
                data_root=root,
                num_workers=workers,
                image_backbone_builder=lambda: TinyConvBackbone(dim=12),
            ).predictions

        pd.testing.assert_frame_equal(_evaluate(0), _evaluate(2))

    def test_embedding_mode_defaults_to_zero_workers(self, tmp_path: Path, monkeypatch):
        # The preloaded resident embedding array must not be copied into workers,
        # so train.num_workers deliberately does NOT carry over in embedding mode.
        import torch.utils.data as torch_data

        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        cfg = synthetic.make_config(root, epochs=1)
        cfg.train.num_workers = 4
        run_dir = tmp_path / "run"
        run_experiment(cfg, data_root=root, output_dir=run_dir, embedding_reader=reader)

        requested: list[int] = []
        real_loader = torch_data.DataLoader

        def recording_loader(*args, **kwargs):
            requested.append(int(kwargs.get("num_workers", 0)))
            return real_loader(*args, **kwargs)

        monkeypatch.setattr(torch_data, "DataLoader", recording_loader)
        evaluate_checkpoint(
            run_dir / "best.ckpt", split="val", data_root=root, embedding_reader=reader
        )
        assert requested == [0]


class TestKIndexKindProvenance:
    @staticmethod
    def _train(tmp_path: Path, kind: str) -> tuple[Path, synthetic.DictEmbeddingReader, Path]:
        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        cfg = synthetic.make_config(
            root,
            epochs=1,
            targets={
                "dhi": {"enabled": True, "loss": "huber"},
                "kindex": {"enabled": True, "kind": kind},
            },
        )
        run_dir = tmp_path / "run"
        run_experiment(cfg, data_root=root, output_dir=run_dir, embedding_reader=reader)
        return root, reader, run_dir / "best.ckpt"

    def test_matching_kind_is_reported_ok(self, tmp_path: Path):
        root, reader, ckpt = self._train(tmp_path, "kstar")
        result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)
        assert result.meta["kindex_kind"] == "kstar"
        assert result.meta["kindex_kind_ok"] is True

    def test_mismatched_kind_warns(self, tmp_path: Path, caplog):
        # The dataset is kstar; the experiment claims kt. The head still scores
        # the frozen kstar column, so the report must say so.
        root, reader, ckpt = self._train(tmp_path, "kt")
        with caplog.at_level("WARNING"):
            result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)
        assert result.meta["kindex_kind_ok"] is False
        assert "kindex_kind='kstar'" in caplog.text

    def test_mismatched_kind_raises_under_strict(self, tmp_path: Path):
        root, reader, ckpt = self._train(tmp_path, "kt")
        with pytest.raises(ValueError, match="kindex_kind"):
            evaluate_checkpoint(
                ckpt, split="val", data_root=root, embedding_reader=reader, strict=True
            )


class TestKIndexKindHelpers:
    def test_meta_wins_over_the_constant_column(self):
        manifest = pd.DataFrame({"kindex_kind": ["kt", "kt"]})
        assert _manifest_kindex_kind(manifest, {"kindex_kind": "kstar"}) == "kstar"

    def test_column_is_the_fallback_when_the_sidecar_lost_it(self):
        manifest = pd.DataFrame({"kindex_kind": ["kt", "kt"]})
        assert _manifest_kindex_kind(manifest, {}) == "kt"

    def test_empty_manifest_does_not_raise(self):
        empty = pd.DataFrame({"kindex_kind": pd.Series(dtype="object")})
        assert _manifest_kindex_kind(empty, {}) is None

    def test_absent_everywhere_is_none(self):
        assert _manifest_kindex_kind(pd.DataFrame({"sample_id": ["a"]}), {}) is None

    def test_unknown_kind_never_raises_even_under_strict(self):
        cfg = ExperimentConfig.model_validate({"targets": {"kindex": {"enabled": True}}})
        assert _check_kindex_kind(cfg, None, strict=True) is False

    def test_disabled_head_is_not_checked(self):
        cfg = ExperimentConfig.model_validate({"targets": {"kindex": {"enabled": False}}})
        assert _check_kindex_kind(cfg, "kt", strict=True) is False
