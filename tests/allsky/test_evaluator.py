"""Torch-gated integration tests for allsky.evaluation.evaluator.

A tiny sensor_only experiment is trained with the C4a engine on a synthetic
3-day manifest (dict-backed embedding reader, CPU, 2 epochs), then evaluated on
the val split: global metrics per enabled target, the stratified breakdown
kinds, denormalized (physical-unit) predictions, and the manifest-hash mismatch
paths (warn by default, error under strict).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from allsky.evaluation.evaluator import evaluate_checkpoint  # noqa: E402
from allsky.training.engine import run_experiment  # noqa: E402
from tests.allsky import _synthetic as synthetic  # noqa: E402


def _train(tmp_path: Path, *, epochs: int = 2, targets: dict | None = None):
    """Train a tiny experiment; return (root, reader, best_ckpt)."""
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    reader = synthetic.reader_for(manifest)
    cfg = synthetic.make_config(root, epochs=epochs, targets=targets)
    run_dir = tmp_path / "run"
    run_experiment(cfg, data_root=root, output_dir=run_dir, embedding_reader=reader)
    return root, reader, run_dir / "best.ckpt"


class TestGlobalMetrics:
    def test_global_metrics_present_for_enabled_targets(self, tmp_path: Path):
        root, reader, ckpt = _train(tmp_path)
        result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)

        assert result.enabled_targets == ["dhi", "sky"]
        assert set(result.global_metrics) == {"dhi", "sky"}
        # regression target carries the normalized errors
        for key in ("rmse", "mae", "bias", "nmae", "nrmse", "n"):
            assert key in result.global_metrics["dhi"]
        # classification target carries a fixed confusion matrix
        assert set(result.global_metrics["sky"]) >= {"accuracy", "macro_f1", "confusion"}
        assert result.confusion is not None
        assert len(result.confusion["matrix"]) == 3
        assert result.meta["manifest_hash_ok"] is True
        assert result.meta["split_id_ok"] is True

    def test_kindex_target_metrics_when_enabled(self, tmp_path: Path):
        targets = {
            "dhi": {"enabled": True, "loss": "mse"},
            "kindex": {"enabled": True, "kind": "kstar"},
        }
        root, reader, ckpt = _train(tmp_path, targets=targets)
        result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)
        assert set(result.global_metrics) == {"dhi", "kindex"}
        assert result.confusion is None  # sky head disabled


class TestStratified:
    def test_stratified_table_has_expected_stratum_kinds(self, tmp_path: Path):
        root, reader, ckpt = _train(tmp_path)
        result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)

        kinds = set(result.stratified["stratum_kind"].unique())
        assert {
            "overall",
            "sky_class",
            "solar_elevation",
            "hour_of_day",
            "month",
            "qc_flags",
            "kindex_band",
        } <= kinds
        # long-form schema
        assert list(result.stratified.columns) == [
            "target",
            "stratum_kind",
            "stratum",
            "metric",
            "value",
            "n",
        ]
        # the overall rows agree with the global metrics
        overall_dhi = result.stratified[
            (result.stratified["stratum_kind"] == "overall")
            & (result.stratified["target"] == "dhi")
            & (result.stratified["metric"] == "rmse")
        ]
        assert len(overall_dhi) == 1
        assert overall_dhi["value"].iloc[0] == pytest.approx(result.global_metrics["dhi"]["rmse"])


class TestDenormalization:
    def test_predictions_in_physical_units_not_normalized_space(self, tmp_path: Path):
        root, reader, ckpt = _train(tmp_path)
        result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)

        preds = result.predictions
        assert {"obs_dhi", "pred_dhi", "obs_sky", "pred_sky"} <= set(preds.columns)
        pred_mean = float(preds["pred_dhi"].mean())
        obs_mean = float(preds["obs_dhi"].mean())
        # Physical diffuse is tens-to-hundreds of W/m2; a normalized prediction
        # would sit near 0. A barely-trained model tracks the target mean.
        assert 5.0 < pred_mean < 2000.0
        assert abs(pred_mean - obs_mean) < obs_mean
        # One prediction per evaluated val sample (exact count depends on the
        # night-elevation drop applied at manifest build; assert the invariant).
        assert result.n_samples == len(preds)
        assert len(preds) > 0


class TestProvenanceChecks:
    def test_manifest_hash_mismatch_warns(self, tmp_path: Path, caplog):
        import logging

        root, reader, ckpt = _train(tmp_path)
        _corrupt_manifest_hash(ckpt)

        with caplog.at_level(logging.WARNING, logger="allsky.evaluation.evaluator"):
            result = evaluate_checkpoint(ckpt, split="val", data_root=root, embedding_reader=reader)
        assert result.meta["manifest_hash_ok"] is False
        assert any("manifest hash mismatch" in record.message for record in caplog.records)

    def test_manifest_hash_mismatch_strict_raises(self, tmp_path: Path):
        root, reader, ckpt = _train(tmp_path)
        _corrupt_manifest_hash(ckpt)
        with pytest.raises(ValueError, match="manifest hash mismatch"):
            evaluate_checkpoint(
                ckpt, split="val", data_root=root, embedding_reader=reader, strict=True
            )

    def test_empty_split_raises(self, tmp_path: Path):
        # make_dataset uses test_fraction=0.0 -> no test days.
        root, reader, ckpt = _train(tmp_path)
        with pytest.raises(ValueError, match="split 'test' has no days"):
            evaluate_checkpoint(ckpt, split="test", data_root=root, embedding_reader=reader)


def _corrupt_manifest_hash(ckpt: Path) -> None:
    """Rewrite the checkpoint's stored manifest hash to force a mismatch."""
    payload = torch.load(ckpt, weights_only=False)
    payload["manifest_sha256"] = "deadbeef" * 8
    torch.save(payload, ckpt)
