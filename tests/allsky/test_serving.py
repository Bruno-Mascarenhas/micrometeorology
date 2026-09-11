"""The serving pin: validation, checkpoint fingerprinting and the watch's ``--serving``."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from allsky.cli import app
from allsky.serving import (
    HeadRoles,
    PinVerificationError,
    RoleSelector,
    ServingConfig,
    ServingConfigError,
    load_serving_config,
    sha256_of_file,
    verify_pinned_checkpoints,
)
from tests.allsky._pins import control, pinned, serving_pin_payload

runner = CliRunner()
REPO_PIN = Path(__file__).resolve().parents[2] / "configs" / "allsky" / "serving" / "ceu.yaml"


def _written(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _pin_dict(tmp_path: Path, *, members: int = 2) -> dict:
    checkpoints = []
    for index in range(members):
        ckpt, digest = _written(tmp_path / f"member{index}" / "best.ckpt", bytes([index]) * 1024)
        checkpoints.append(pinned(ckpt, digest, ckpt.parent / "eval-test"))
    sensor, sensor_digest = _written(
        tmp_path / "controls" / "sensor_only" / "best.ckpt", b"scalars only"
    )
    climatology, climatology_digest = _written(
        tmp_path / "controls" / "climatology" / "best.ckpt", b"train means"
    )
    return serving_pin_payload(
        frame_checkpoints=checkpoints,
        sensor_only=control(sensor, sensor_digest, sensor.parent / "eval-test"),
        climatology=control(climatology, climatology_digest, climatology.parent / "eval-test"),
        dataset=tmp_path / "dataset",
        training_history=tmp_path / "member0" / "metrics.csv",
    )


def _write_pin(tmp_path: Path, payload: dict) -> Path:
    pin = tmp_path / "pin.yaml"
    pin.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return pin


def test_the_repository_pin_loads_and_names_one_report_per_member():
    pin = load_serving_config(REPO_PIN)

    assert all(member.report.name == "eval-test" for member in pin.frame_checkpoints)
    assert pin.attribution_member is pin.frame_checkpoints[pin.attribution_checkpoint]


def test_a_pin_with_a_stray_key_is_refused(tmp_path):
    payload = _pin_dict(tmp_path)
    payload["checkpoint"] = "typo"

    with pytest.raises(ValueError, match="checkpoint"):
        ServingConfig.model_validate(payload)


def test_an_attribution_index_past_the_members_is_refused(tmp_path):
    payload = _pin_dict(tmp_path)
    payload["attribution_checkpoint"] = 2

    with pytest.raises(ValueError, match="names no member"):
        ServingConfig.model_validate(payload)


def test_a_member_without_its_report_is_refused(tmp_path):
    payload = _pin_dict(tmp_path)
    del payload["frame_checkpoints"][1]["report"]

    with pytest.raises(ValueError, match=r"frame_checkpoints\.1\.report"):
        ServingConfig.model_validate(payload)


def test_a_pin_that_is_not_a_mapping_is_refused(tmp_path):
    pin = tmp_path / "pin.yaml"
    pin.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ServingConfigError, match="mapping"):
        load_serving_config(pin)


def test_a_pin_with_a_schema_violation_names_the_field(tmp_path):
    payload = _pin_dict(tmp_path)
    payload["min_elevation_deg"] = "high"
    pin = _write_pin(tmp_path, payload)

    with pytest.raises(ServingConfigError, match="min_elevation_deg"):
        load_serving_config(pin)


def test_watch_stops_with_exit_one_on_a_malformed_pin(tmp_path):
    payload = _pin_dict(tmp_path)
    payload["frame_checkpointz"] = payload.pop("frame_checkpoints")
    pin = _write_pin(tmp_path, payload)

    result = runner.invoke(app, ["watch", "--out", str(tmp_path / "watch"), "--serving", str(pin)])

    assert result.exit_code == 1


def test_verification_returns_the_digests_members_first_then_the_control(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))

    digests = verify_pinned_checkpoints(pin)

    assert digests == [
        *(member.sha256 for member in pin.frame_checkpoints),
        pin.controls.sensor_only.checkpoint.sha256,
        pin.controls.climatology.checkpoint.sha256,
    ]


def test_a_tilde_in_a_pinned_path_names_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    payload = _pin_dict(tmp_path)
    payload["frame_checkpoints"][0]["path"] = "~/member0/best.ckpt"
    payload["reports"]["dataset"] = "~/dataset"

    pin = ServingConfig.model_validate(payload)

    assert pin.frame_checkpoints[0].path == tmp_path / "member0" / "best.ckpt"
    assert pin.reports.dataset == tmp_path / "dataset"


def test_a_rewritten_control_checkpoint_fails_verification_too(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))
    pin.controls.sensor_only.checkpoint.path.write_bytes(b"other bytes")

    with pytest.raises(PinVerificationError, match="sensor_only"):
        verify_pinned_checkpoints(pin)


def test_the_selection_split_defaults_to_validation(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))

    assert pin.selection.selection_split == "val"


def test_the_domain_check_defaults_to_not_pinned(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))

    assert pin.reports.domain_check is None


def test_a_rewritten_checkpoint_fails_verification_by_name(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))
    pin.frame_checkpoints[1].path.write_bytes(b"other bytes")

    with pytest.raises(PinVerificationError, match="member1"):
        verify_pinned_checkpoints(pin)


def test_a_missing_checkpoint_fails_verification_by_name(tmp_path):
    pin = ServingConfig.model_validate(_pin_dict(tmp_path))
    pin.frame_checkpoints[0].path.unlink()

    with pytest.raises(PinVerificationError, match="does not exist"):
        verify_pinned_checkpoints(pin)


def test_the_hash_cache_is_reused_while_the_file_is_unchanged(tmp_path):
    target = tmp_path / "big.ckpt"
    target.write_bytes(b"x" * 4096)
    cache_dir = tmp_path / "state"

    first = sha256_of_file(target, cache_dir=cache_dir)
    cache = json.loads((cache_dir / "checkpoint-sha256.json").read_text())
    cache[str(target.resolve())]["sha256"] = "0" * 64
    (cache_dir / "checkpoint-sha256.json").write_text(json.dumps(cache))
    second = sha256_of_file(target, cache_dir=cache_dir)

    assert first == hashlib.sha256(b"x" * 4096).hexdigest()
    assert second == "0" * 64


def test_the_hash_cache_is_dropped_when_the_file_changes(tmp_path):
    target = tmp_path / "big.ckpt"
    target.write_bytes(b"x" * 4096)
    cache_dir = tmp_path / "state"
    sha256_of_file(target, cache_dir=cache_dir)

    target.write_bytes(b"y" * 8192)
    digest = sha256_of_file(target, cache_dir=cache_dir)

    assert digest == hashlib.sha256(b"y" * 8192).hexdigest()


def test_watch_refuses_a_pin_beside_an_explicit_frame_checkpoint(tmp_path):
    payload = _pin_dict(tmp_path)
    pin = _write_pin(tmp_path, payload)
    explicit = pin.parent / "member0" / "best.ckpt"

    result = runner.invoke(
        app,
        [
            "watch",
            "--out",
            str(tmp_path / "watch"),
            "--serving",
            str(pin),
            "--checkpoint-frame",
            str(explicit),
        ],
    )

    assert result.exit_code == 1


def test_watch_refuses_a_pin_beside_an_explicit_elevation_floor(tmp_path):
    pin = _write_pin(tmp_path, _pin_dict(tmp_path))

    result = runner.invoke(
        app,
        [
            "watch",
            "--out",
            str(tmp_path / "watch"),
            "--serving",
            str(pin),
            "--min-elevation-deg",
            "10",
        ],
    )

    assert result.exit_code == 1


def test_watch_stops_on_a_pin_whose_checkpoint_was_rewritten(tmp_path):
    payload = _pin_dict(tmp_path)
    Path(payload["frame_checkpoints"][0]["path"]).write_bytes(b"rewritten")
    pin = _write_pin(tmp_path, payload)

    result = runner.invoke(app, ["watch", "--out", str(tmp_path / "watch"), "--serving", str(pin)])

    assert result.exit_code == 1


def test_watch_takes_checkpoints_roles_and_floor_from_the_pin(tmp_path, monkeypatch):
    payload = _pin_dict(tmp_path)
    payload["frame_sky_role"] = "best"
    payload["frame_dhi_role"] = "all"
    payload["min_elevation_deg"] = 12.5
    pin = _write_pin(tmp_path, payload)
    seen: dict = {}

    def fake_run_watch(_capture, out_dir, **kwargs):
        seen.update(kwargs, out_dir=out_dir)
        return 0

    import allsky.snapshot
    import allsky.watch

    monkeypatch.setattr(allsky.watch, "run_watch", fake_run_watch)
    monkeypatch.setattr("allsky.cli.watch._build_client", lambda *_args, **_kwargs: object())

    result = runner.invoke(app, ["watch", "--out", str(tmp_path / "watch"), "--serving", str(pin)])

    assert result.exit_code == 0, result.output
    assert [Path(p) for p in seen["checkpoint_frames"]] == [
        Path(m["path"]) for m in payload["frame_checkpoints"]
    ]
    assert seen["frame_roles"] == HeadRoles(sky=RoleSelector.best, dhi=RoleSelector.all)
    assert seen["min_solar_elevation_deg"] == 12.5


def test_watch_stops_when_a_pinned_member_cannot_be_built(tmp_path, monkeypatch):
    pin = _write_pin(tmp_path, _pin_dict(tmp_path))
    import allsky.snapshot

    def refuse(_path, **_kwargs):
        raise ValueError("no backbone weights on this machine")

    monkeypatch.setattr(allsky.snapshot, "load_served_model", refuse)
    monkeypatch.setattr("allsky.cli.watch._build_client", lambda *_args, **_kwargs: object())

    result = runner.invoke(app, ["watch", "--out", str(tmp_path / "watch"), "--serving", str(pin)])

    assert result.exit_code == 1
