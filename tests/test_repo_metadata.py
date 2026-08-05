"""Contract tests for repo metadata nothing imports: pyproject knobs and the Colab notebook.

Neither is reached by importing the package, so both drift silently.  The two
regressions pinned here actually shipped: a notebook cell cloning a branch that
does not exist on the remote (every Colab run died at the first substantive
cell), and a pytest config that blanket-ignored ``DeprecationWarning`` /
``UserWarning`` so an upstream removal could only ever surface as a hard failure
after it landed.

The branch check is offline by construction — it pins the notebook's ``BRANCH``
to the ref ``notebooks/README.md`` tells users to open the notebook from.
Confirm against the live remote with ``git ls-remote --heads <REPO_URL>``.
"""

import json
import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_NOTEBOOK = _ROOT / "notebooks" / "allsky_multimodal_colab.ipynb"
_NOTEBOOKS_README = _ROOT / "notebooks" / "README.md"


def _filterwarnings() -> list[str]:
    with open(_PYPROJECT, "rb") as fh:
        config = tomllib.load(fh)
    entries = config["tool"]["pytest"]["ini_options"]["filterwarnings"]
    assert isinstance(entries, list)
    return entries


def _notebook_cell_sources() -> list[str]:
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def _configured_branch() -> str:
    for source in _notebook_cell_sources():
        match = re.search(r'^BRANCH = "([^"]+)"', source, re.MULTILINE)
        if match:
            return match.group(1)
    raise AssertionError("the notebook CONFIG cell has no BRANCH assignment")


def _clone_cell() -> str:
    return next(source for source in _notebook_cell_sources() if '"clone"' in source)


def test_deprecation_warnings_are_visible() -> None:
    """A blanket category ignore hides upstream removals for their whole window."""
    entries = _filterwarnings()
    assert "ignore::DeprecationWarning" not in entries
    assert "ignore::UserWarning" not in entries
    assert "default::DeprecationWarning" in entries
    assert "default::UserWarning" in entries


def test_notebook_clones_the_ref_the_readme_advertises() -> None:
    """Colab fetches the .ipynb over HTTP; a ref the remote lacks kills the run."""
    advertised = re.search(
        r"/blob/([^/]+)/notebooks/", _NOTEBOOKS_README.read_text(encoding="utf-8")
    )
    assert advertised is not None, "notebooks/README.md no longer shows the Colab open URL"
    assert _configured_branch() == advertised.group(1)


def test_clone_cell_takes_its_ref_from_the_branch_knob() -> None:
    """Fresh clone and the fetch/checkout/pull resume arm must read the same knob."""
    cell = _clone_cell()
    assert cell.count("BRANCH") >= 3, cell


def test_every_package_reports_the_distribution_version() -> None:
    """All three packages ship in ONE distribution, so all three must say 1.3.1.

    ``allsky`` and ``solrad_correction`` carried a hand-written ``0.1.0`` while
    the distribution was at 1.3.1, so anything introspecting them — including a
    reproducibility stamp written next to a trained model — recorded a version
    that never existed.  The CI wheel smoke test only ever checked
    ``micrometeorology``.
    """
    from importlib.metadata import version

    import allsky
    import micrometeorology
    import solrad_correction

    distribution = version("labmim-micrometeorology")
    assert micrometeorology.__version__ == distribution
    assert allsky.__version__ == distribution
    assert solrad_correction.__version__ == distribution


def test_project_urls_are_declared() -> None:
    """PyPI-style metadata a published wheel needs; nothing imports it, so it drifted."""
    with open(_PYPROJECT, "rb") as fh:
        urls = tomllib.load(fh)["project"]["urls"]

    assert set(urls) >= {"Homepage", "Repository", "Issues"}
    assert all(value.startswith("https://") for value in urls.values())
