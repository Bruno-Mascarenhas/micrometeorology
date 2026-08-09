"""Tests for micrometeorology.stats.climatology_export.

Offline: synthetic samples only. These pin the *contract* the site reads, so a
change that breaks the page fails here rather than in a browser.
"""

import json

import numpy as np
import pytest

from micrometeorology.stats.climatology_export import (
    CLIMATOLOGY_VARIABLES,
    MANIFEST_FORMAT,
    RAIN_BUCKET_MM,
    VARIABLE_FORMAT,
    Atom,
    VariableSpec,
    build_manifest,
    build_variable_payload,
    write_json,
)

SUBSETS = [
    {"id": "observed_all", "source": "observed", "season": "all", "label": "Ano inteiro"},
    {"id": "observed_djf", "source": "observed", "season": "DJF", "label": "Verão"},
]


@pytest.fixture
def wind_spec() -> VariableSpec:
    return next(spec for spec in CLIMATOLOGY_VARIABLES if spec.id == "wind_speed")


@pytest.fixture
def wind_samples() -> dict[str, np.ndarray]:
    generator = np.random.default_rng(4)
    return {
        "observed_all": generator.weibull(1.9, 20_000) * 2.4,
        "observed_djf": generator.weibull(1.9, 5_000) * 2.2,
    }


class TestVariableCatalogue:
    def test_ids_are_unique(self):
        ids = [spec.id for spec in CLIMATOLOGY_VARIABLES]
        assert len(ids) == len(set(ids))

    def test_every_edge_set_is_strictly_increasing(self):
        for spec in CLIMATOLOGY_VARIABLES:
            assert all(b > a for a, b in zip(spec.edges, spec.edges[1:], strict=False)), spec.id

    def test_every_variable_carries_a_caveat(self):
        """The scientific qualifications travel with the data so the page cannot drop them."""
        for spec in CLIMATOLOGY_VARIABLES:
            assert spec.caveats, spec.id

    def test_rain_edges_sit_on_the_tipping_bucket_lattice(self):
        """Bins narrower than the bucket produce a comb that looks like signal."""
        spec = next(s for s in CLIMATOLOGY_VARIABLES if s.id == "precipitation")
        for edge in spec.edges:
            lattice_index = edge / RAIN_BUCKET_MM - 0.5
            assert lattice_index == pytest.approx(round(lattice_index), abs=1e-6), edge

    def test_rain_bins_are_never_narrower_than_one_bucket(self):
        spec = next(s for s in CLIMATOLOGY_VARIABLES if s.id == "precipitation")
        widths = np.diff(spec.edges)
        assert widths.min() >= RAIN_BUCKET_MM - 1e-9


class TestVariablePayload:
    def test_carries_the_declared_format(self, wind_spec, wind_samples):
        payload = build_variable_payload(wind_spec, wind_samples, version="v1")
        assert payload["format"] == VARIABLE_FORMAT
        assert payload["version"] == "v1"

    def test_curve_aligns_one_to_one_with_the_bars(self, wind_spec, wind_samples):
        """This alignment is what lets the page draw both on one categorical axis."""
        payload = build_variable_payload(wind_spec, wind_samples, version="v1")
        subset = payload["subsets"]["observed_all"]
        assert len(subset["curve"]) == len(subset["counts"]) == len(payload["edges"]) - 1

    def test_bars_and_curve_share_one_normalisation(self, wind_spec, wind_samples):
        """Both are densities of the continuous part; the atom is printed separately."""
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.4)]}
        payload = build_variable_payload(wind_spec, wind_samples, version="v1", atoms=atoms)
        subset = payload["subsets"]["observed_all"]
        widths = np.diff(payload["edges"])
        bars = float(np.sum(np.array(subset["density"]) * widths))
        curve = float(np.sum(np.array(subset["curve"]) * widths))
        assert bars == pytest.approx(1.0, abs=1e-9)
        # A 40 % atom must NOT scale the curve down: it would sit at 0.6 here and
        # be drawn flat against the axis under bars that integrate to 1.
        assert curve == pytest.approx(1.0, abs=0.02)

    def test_atoms_are_reported_verbatim(self, wind_spec, wind_samples):
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.037)]}
        payload = build_variable_payload(wind_spec, wind_samples, version="v1", atoms=atoms)
        assert payload["subsets"]["observed_all"]["atoms"] == [
            {"id": "calm", "label": "Calmarias", "fraction": 0.037}
        ]

    def test_empty_subset_is_present_and_empty_not_omitted(self, wind_spec, wind_samples):
        """The page must be able to say "no data" rather than offer fewer options."""
        samples = {**wind_samples, "observed_jja": np.array([])}
        payload = build_variable_payload(wind_spec, samples, version="v1")
        empty = payload["subsets"]["observed_jja"]
        assert empty["n"] == 0
        assert empty["fit"] is None
        assert empty["curve"] is None

    def test_display_range_is_shared_by_every_subset(self, wind_spec, wind_samples):
        """A per-subset axis would silently rescale under the reader between clicks."""
        payload = build_variable_payload(wind_spec, wind_samples, version="v1")
        first, last = payload["display_range"]
        assert 0 <= first < last <= len(payload["edges"]) - 2

    def test_display_range_ignores_a_lone_far_outlier(self, wind_spec):
        """One gust must not stretch the axis over a range blank for everything else."""
        generator = np.random.default_rng(5)
        typical = generator.weibull(2.0, 20_000) * 2.0
        samples = {"observed_all": np.append(typical, 19.5)}
        payload = build_variable_payload(wind_spec, samples, version="v1")
        _first, last = payload["display_range"]
        assert payload["edges"][last + 1] < 15.0

    def test_out_of_range_samples_are_tallied_not_dropped(self, wind_spec):
        samples = {"observed_all": np.array([0.5, 1.0, 999.0])}
        payload = build_variable_payload(wind_spec, samples, version="v1")
        assert payload["subsets"]["observed_all"]["above"] == 1

    def test_rose_variable_carries_sectors_not_edges(self):
        spec = next(s for s in CLIMATOLOGY_VARIABLES if s.chart == "rose")
        generator = np.random.default_rng(6)
        samples = {"observed_all": generator.uniform(0, 360, 5_000)}
        payload = build_variable_payload(spec, samples, version="v1")
        assert "sectors" in payload
        assert "edges" not in payload
        assert sum(payload["subsets"]["observed_all"]["frequencies"]) == pytest.approx(
            1.0, abs=1e-4
        )

    def test_non_finite_numbers_become_null(self, wind_spec):
        """allow_nan=False would otherwise refuse to serialise a degenerate subset."""
        payload = build_variable_payload(wind_spec, {"observed_all": np.array([1.0])}, version="v1")
        encoded = json.dumps(payload, allow_nan=False)
        assert "NaN" not in encoded


class TestManifest:
    def _manifest(self, selector):
        return build_manifest(
            version="v1",
            generated_utc="v1",
            station={"name": "teste"},
            period={"start": "2016-09-29", "end": "2026-04-24"},
            sources=[{"id": "observed", "label": "Estação"}],
            subsets=SUBSETS,
            selector=selector,
            coverage={"years": []},
            variables=list(CLIMATOLOGY_VARIABLES),
        )

    def test_declares_the_format_and_every_variable_file(self):
        manifest = self._manifest(["observed_all"])
        assert manifest["format"] == MANIFEST_FORMAT
        assert [entry["file"] for entry in manifest["variables"]] == [
            f"{spec.id}.json" for spec in CLIMATOLOGY_VARIABLES
        ]

    def test_selector_may_be_narrower_than_the_precomputed_subsets(self):
        """Every source-by-season pair is computed; the page offers a chosen few."""
        manifest = self._manifest(["observed_all"])
        assert len(manifest["selector"]) < len(manifest["subsets"])

    def test_a_selector_naming_an_undeclared_subset_is_rejected(self):
        """It would render a dead option whose only symptom is an empty chart."""
        with pytest.raises(ValueError, match="undeclared subset"):
            self._manifest(["observed_all", "nao_existe"])


class TestWriteJson:
    def test_writes_compact_utf8(self, tmp_path):
        path = write_json(tmp_path / "x.json", {"label": "Precipitação", "n": 1})
        text = path.read_text(encoding="utf-8")
        assert text == '{"label":"Precipitação","n":1}'

    def test_refuses_a_non_finite_number(self, tmp_path):
        """NaN is not valid JSON; failing here beats failing in every browser."""
        with pytest.raises(ValueError, match=r"Out of range|not JSON compliant|NaN"):
            write_json(tmp_path / "x.json", {"v": float("nan")})

    def test_leaves_no_temporary_behind_on_failure(self, tmp_path):
        with pytest.raises(ValueError, match=r"Out of range|not JSON compliant|NaN"):
            write_json(tmp_path / "x.json", {"v": float("inf")})
        assert list(tmp_path.iterdir()) == []

    def test_replaces_atomically(self, tmp_path):
        """A reader mid-run sees the old file or the new one, never a truncated parse."""
        path = tmp_path / "x.json"
        write_json(path, {"v": 1})
        write_json(path, {"v": 2})
        assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
        assert [item.name for item in tmp_path.iterdir()] == ["x.json"]
