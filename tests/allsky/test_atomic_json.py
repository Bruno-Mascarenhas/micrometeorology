"""JSON encoding contract of the atomic writers.

The lenient writer keeps bare ``NaN``/``Infinity`` tokens because a diverged
epoch's metrics history is legitimate telemetry; the strict writer is the one
whose bytes reach the public site, where every consumer parses with a strict
``response.json()`` that rejects those tokens.
"""

import json
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

from labmim_core.atomic import atomic_write_json, atomic_write_strict_json


def _raise_on_non_finite_token(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token: {token}")


def _parse_as_a_browser_would(text: str) -> Any:
    return json.loads(text, parse_constant=_raise_on_non_finite_token)


def test_lenient_writer_keeps_non_finite_tokens_for_metrics_history(tmp_path: Path) -> None:
    history = {"epoch": 40, "train_loss": float("nan")}

    path = atomic_write_json(tmp_path / "metrics.json", history)

    assert '"train_loss": NaN' in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=str)
def test_strict_writer_refuses_to_publish_a_non_finite_float(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        atomic_write_strict_json(tmp_path / "prediction.json", {"ghi_w_m2": value})


def test_strict_writer_output_parses_under_strict_json(tmp_path: Path) -> None:
    payload = {"ghi_w_m2": 812.5, "kt": 0.74, "captured_at": None}

    path = atomic_write_strict_json(tmp_path / "prediction.json", payload)

    assert _parse_as_a_browser_would(path.read_text(encoding="utf-8")) == payload


def test_strict_writer_leaves_the_previous_payload_in_place_when_it_refuses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prediction.json"
    atomic_write_strict_json(path, {"ghi_w_m2": 812.5})

    with pytest.raises(ValueError, match="not JSON compliant"):
        atomic_write_strict_json(path, {"ghi_w_m2": float("nan")})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ghi_w_m2": 812.5}


def test_strict_writer_removes_its_temp_file_when_it_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        atomic_write_strict_json(tmp_path / "prediction.json", {"kt": float("inf")})

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "value", [np.float32("nan"), np.float32("inf"), np.float64("-inf")], ids=str
)
def test_strict_writer_refuses_a_non_finite_numpy_scalar(tmp_path: Path, value: Any) -> None:
    """``numpy.float32`` is not a ``float`` subclass, so it reached ``default=str``
    and was published as the string ``"nan"`` instead of being refused."""
    with pytest.raises(ValueError, match="not JSON compliant"):
        atomic_write_strict_json(tmp_path / "prediction.json", {"ghi_w_m2": value})

    assert list(tmp_path.iterdir()) == []


def test_strict_writer_unwraps_numpy_scalars_to_json_numbers(tmp_path: Path) -> None:
    payload = {"n": np.int64(5), "kt": np.float32(0.5), "ok": np.bool_(True)}

    path = atomic_write_strict_json(tmp_path / "prediction.json", payload)

    assert _parse_as_a_browser_would(path.read_text(encoding="utf-8")) == {
        "n": 5,
        "kt": 0.5,
        "ok": True,
    }
