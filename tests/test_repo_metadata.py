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

import importlib.util
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_ENVIRONMENT = _ROOT / "environment.yml"
_NOTEBOOK = _ROOT / "notebooks" / "allsky_multimodal_colab.ipynb"
_NOTEBOOKS_README = _ROOT / "notebooks" / "README.md"
_CITATION = _ROOT / "CITATION.cff"
_ZENODO = _ROOT / ".zenodo.json"
_README = _ROOT / "README.md"
#: Notebooks whose outputs must be stripped. ``legacy/`` predates the rule.
_NOTEBOOKS = sorted(
    path
    for path in (_ROOT / "notebooks").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in path.parts
)


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


def _citation() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(_CITATION.read_text(encoding="utf-8"))
    return loaded


def _zenodo() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_ZENODO.read_text(encoding="utf-8"))
    return loaded


def _pyproject() -> dict:
    with open(_PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def test_the_release_metadata_agrees_on_one_version() -> None:
    """CITATION.cff is what a citing paper reads and pyproject is what the wheel
    reports; a release that moves one and not the other cites code that was
    never published.
    """
    assert _citation()["version"] == _pyproject()["project"]["version"]


def test_the_doi_the_citation_gives_is_the_one_the_readme_advertises() -> None:
    """Two hand-maintained copies of a DOI point readers at two records."""
    doi = _citation()["doi"]

    assert doi in _README.read_text(encoding="utf-8")


def test_every_metadata_file_declares_the_same_licence() -> None:
    """A licence stated four ways is a licence nobody can rely on."""
    assert _citation()["license"] == "MIT"
    assert _zenodo()["license"] == "MIT"
    assert _pyproject()["project"]["license"]["text"] == "MIT"
    assert (_ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_the_citation_and_the_zenodo_record_name_the_same_authors() -> None:
    """Zenodo mints the DOI from .zenodo.json and the CFF is what tooling cites;
    an author on one and not the other is dropped from half the record.
    """
    cited = {
        f"{author['family-names']}, {author['given-names']}"
        for author in _citation()["authors"]
        if "family-names" in author
    }

    assert cited == {creator["name"] for creator in _zenodo()["creators"]}


@pytest.mark.parametrize("notebook", _NOTEBOOKS, ids=lambda path: path.name)
def test_a_tracked_notebook_carries_no_cell_output(notebook: Path) -> None:
    """The house rule asks for stripped outputs before every commit and nothing
    enforced it: a notebook committed with 40 MB of inline images is permanent
    in the history. Checked here rather than by a hook so it costs no new
    dependency and fails in CI like any other test.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]

    carrying = [
        position
        for position, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count"))
    ]
    assert carrying == [], f"{notebook.name}: cells {carrying} carry output"


#: Distributions this repo ships. A notebook importing from one of them names a
#: module that has to exist; anything else is a third-party dependency.
_OWN_PACKAGES = ("allsky", "labmim_core", "micrometeorology", "solrad_correction")


@pytest.mark.parametrize("notebook", _NOTEBOOKS, ids=lambda path: path.name)
def test_a_tracked_notebook_imports_only_modules_this_repo_has(notebook: Path) -> None:
    """A notebook importing a module that was renamed away dies at its first
    substantive cell, and nothing else in the suite reads notebook source. One
    tracked notebook imported ``solrad_correction.utils.plots``, which has never
    existed.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell.get("cell_type") == "code"]

    named = set()
    for source in sources:
        for statement in re.findall(r"^\s*from\s+([\w.]+)\s+import", source, re.MULTILINE):
            if statement.split(".")[0] in _OWN_PACKAGES:
                named.add(statement)
        for statement in re.findall(r"^\s*import\s+([\w.]+)", source, re.MULTILINE):
            if statement.split(".")[0] in _OWN_PACKAGES:
                named.add(statement)

    missing = [module for module in sorted(named) if importlib.util.find_spec(module) is None]
    assert missing == [], f"{notebook.name}: {missing}"


def test_the_notebook_names_configs_that_exist() -> None:
    """The clone cell's BRANCH was pinned after a run died on it; the config
    paths the same notebook passes to the CLI fail the same way, one cell later.
    """
    referenced = set()
    for source in _notebook_cell_sources():
        referenced.update(re.findall(r'"(configs/[^"]+\.yaml)"', source))
        referenced.update(re.findall(r"--config\s+(\S*configs/\S+\.yaml)", source))

    assert referenced, "the notebook names no config at all"
    assert [name for name in sorted(referenced) if not (_ROOT / name).is_file()] == []


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
    """Every package here ships in ONE distribution, so each must report its version.

    ``allsky`` and ``solrad_correction`` carried a hand-written ``0.1.0`` while
    the distribution was at 1.3.1, so anything introspecting them — including a
    reproducibility stamp written next to a trained model — recorded a version
    that never existed.  The CI wheel smoke test only ever checked
    ``micrometeorology``.
    """
    from importlib.metadata import version

    import allsky
    import labmim_core
    import micrometeorology
    import solrad_correction

    distribution = version("labmim-micrometeorology")
    assert micrometeorology.__version__ == distribution
    assert allsky.__version__ == distribution
    assert labmim_core.__version__ == distribution
    assert solrad_correction.__version__ == distribution


def test_project_urls_are_declared() -> None:
    """PyPI-style metadata a published wheel needs; nothing imports it, so it drifted."""
    with open(_PYPROJECT, "rb") as fh:
        urls = tomllib.load(fh)["project"]["urls"]

    assert set(urls) >= {"Homepage", "Repository", "Issues"}
    assert all(value.startswith("https://") for value in urls.values())


def test_the_conda_bootstrap_pins_the_uv_range_the_project_requires() -> None:
    """environment.yml pinned uv below 0.12 while pyproject requires 0.12.5 or
    newer, so the documented first run (conda env create, make install-dev)
    failed on a clean machine with uv refusing the project."""
    with open(_PYPROJECT, "rb") as fh:
        required = tomllib.load(fh)["tool"]["uv"]["required-version"]
    dependencies = yaml.safe_load(_ENVIRONMENT.read_text(encoding="utf-8"))["dependencies"]
    uv_spec = next(
        entry for entry in dependencies if isinstance(entry, str) and entry.startswith("uv")
    )

    assert uv_spec == f"uv{required}"
