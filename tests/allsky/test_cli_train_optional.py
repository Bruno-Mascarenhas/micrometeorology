"""``allsky train`` must name the extra when torch is missing.

The lazy-import architecture promises ``allsky --help`` works in a minimal
install, so hitting the torch boundary is an expected user path — and it used to
end in a bare ``ModuleNotFoundError: No module named 'torch'`` naming neither
the extra nor the install command. torch itself is never removed here: the
precondition check is driven by hiding the spec.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from allsky.cli import app
from micrometeorology.common.optional import require

runner = CliRunner()


class TestRequire:
    def test_present_module_passes(self):
        require("json", "allsky")  # never raises

    def test_absent_module_names_the_extra_and_the_command(self):
        with pytest.raises(ImportError, match=r"pip install labmim-micrometeorology\[allsky\]"):
            require("definitely_not_installed_xyz", "allsky")

    def test_does_not_import_the_module_it_checks(self):
        # find_spec must not execute the module, or the lazy-import guarantee
        # (``allsky --help`` in a minimal environment) would be broken.
        sys.modules.pop("wsgiref.validate", None)
        require("wsgiref.validate", "allsky")
        assert "wsgiref.validate" not in sys.modules


def _experiment_config(tmp_path: Path) -> Path:
    config = tmp_path / "exp.yaml"
    config.write_text(
        "experiment: true\nname: guard\ntrain:\n  epochs: 1\n  device: cpu\n", encoding="utf-8"
    )
    return config


class TestTrainTorchGuard:
    def test_missing_torch_reports_the_extra(self, tmp_path: Path, monkeypatch):
        real_find_spec = importlib.util.find_spec

        def hide_torch(name, package=None):
            if name == "torch":
                return None
            return real_find_spec(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", hide_torch)
        result = runner.invoke(app, ["train", "--config", str(_experiment_config(tmp_path))])

        assert result.exit_code != 0
        assert isinstance(result.exception, ImportError)
        assert "pip install labmim-micrometeorology[allsky]" in str(result.exception)
