"""Tests for micrometeorology.stats.climatology_export.

Offline: synthetic samples only. These pin the *contract* the site reads, so a
change that breaks the page fails here rather than in a browser.
"""

import json
import re

import numpy as np
import pytest

from micrometeorology.stats.climatology_export import (
    CLIMATOLOGY_VARIABLES,
    MANIFEST_FORMAT,
    RAIN_BUCKET_MM,
    REFERENCES,
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
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.037, 740)]}
        payload = build_variable_payload(wind_spec, wind_samples, version="v1", atoms=atoms)
        assert payload["subsets"]["observed_all"]["atoms"] == [
            {"id": "calm", "label": "Calmarias", "fraction": 0.037, "count": 740}
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
        # The rose's caveats cite both of its atoms by id, and a citation that
        # resolves to nothing is refused rather than published half-written.
        atoms = {
            "observed_all": [
                Atom("arithmetic_mean_era", "Era de média aritmética", 0.465, 32588),
                Atom("calm", "Calmarias", 0.052, 2785),
            ]
        }
        payload = build_variable_payload(spec, samples, version="v1", atoms=atoms)
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


class TestReferenceLinks:
    """Every published citation has to LAND on the work it names.

    The links turn a bracketed marker on the page into a source the reader can
    check, so a dead one costs the page its provenance with no visible symptom.
    A Crossref title-search route can answer 200, serve a real page and ignore
    the query entirely — status-only checking cannot see that, and a DOI
    resolver link cannot fail that way.
    """

    def test_every_reference_resolves_through_doi_org(self) -> None:
        for ref in REFERENCES.values():
            assert ref.url.startswith("https://doi.org/10."), ref.key

    def test_no_reference_falls_back_to_a_crossref_search(self) -> None:
        """Especially not the route that answers 200 and searches nothing."""
        for ref in REFERENCES.values():
            assert "search.crossref.org" not in ref.url, ref.key

    def test_every_doi_is_syntactically_resolvable(self) -> None:
        """A DOI is ``10.<registrant>/<suffix>``, and the reserved characters the
        AMS suffixes carry must be percent-encoded or the URL is malformed."""
        for ref in REFERENCES.values():
            doi = ref.url.removeprefix("https://doi.org/")
            assert re.fullmatch(r"10\.\d{4,9}/\S+", doi), ref.key
            assert "<" not in doi, f"{ref.key}: percent-encode < in the URL"
            assert ">" not in doi, f"{ref.key}: percent-encode > in the URL"
            assert " " not in doi, ref.key

    def test_reference_keys_and_markers_agree(self) -> None:
        """A marker with no record renders as raw ``[[key]]`` to the reader."""
        marker = re.compile(r"\[\[([a-z0-9]+)\]\]")
        cited = set()
        for spec in CLIMATOLOGY_VARIABLES:
            for text in (*spec.caveats, spec.family_label):
                cited.update(marker.findall(text))
        assert cited <= set(REFERENCES), sorted(cited - set(REFERENCES))


class TestSubsetTotalsAgree:
    """One subset, one total. The panel prints several numbers about the same
    subset, and a reader who adds them up has to arrive where the page says."""

    def test_n_decomposes_into_the_bars_plus_what_fell_outside(self, wind_spec) -> None:
        """`n == sum(counts) + below + above`, checkable on screen.

        `n` is the whole sample, never the binned count alone: the statistics
        printed beside the bars describe that same `n`, so a value outside every
        bar has to be accounted for by `below`/`above` rather than dropped.
        """
        samples = {"observed_all": np.array([-3.0, 0.5, 1.0, 2.0, 3.0, 999.0])}
        payload = build_variable_payload(wind_spec, samples, version="v1")
        subset = payload["subsets"]["observed_all"]

        assert subset["n"] == sum(subset["counts"]) + subset["below"] + subset["above"]
        assert subset["below"] >= 1
        assert subset["above"] >= 1

    def test_the_statistics_describe_the_same_sample_n_counts(self, wind_spec) -> None:
        """The maximum printed beside the bars may lie outside them — that is
        what `above` is for — but it must belong to the sample `n` counts."""
        samples = {"observed_all": np.array([0.5, 1.0, 2.0, 999.0])}
        payload = build_variable_payload(wind_spec, samples, version="v1")
        subset = payload["subsets"]["observed_all"]

        assert subset["n"] == 4
        assert subset["stats"]["max"] == 999.0

    def test_an_atom_publishes_the_count_its_fraction_is_a_share_of(self, wind_spec) -> None:
        """The fraction's denominator is the subset BEFORE the mass was removed,
        and that number is not the ``n`` printed beside it.

        Without the count the reader cannot resolve the two: multiplying
        "5,2 % calmarias" by the "66.345 observações" on the same panel gives
        3.472, and the mass is 3.663 — a share of 70.008.
        """
        samples = {"observed_all": np.array([1.0, 2.0, 3.0])}
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.25, 1)]}
        payload = build_variable_payload(wind_spec, samples, version="v1", atoms=atoms)
        published = payload["subsets"]["observed_all"]["atoms"][0]

        assert published["count"] == 1
        total = payload["subsets"]["observed_all"]["n"] + published["count"]
        assert published["fraction"] == pytest.approx(published["count"] / total)

    def test_the_rose_publishes_atoms_the_same_way_as_a_histogram(self) -> None:
        """Two serialisers for one field is how they drift; the rose had no count."""
        spec = next(s for s in CLIMATOLOGY_VARIABLES if s.chart == "rose")
        generator = np.random.default_rng(9)
        samples = {"observed_all": generator.uniform(0, 360, 500)}
        # Both atoms, because the shipped wind_direction caveats cite both and an
        # unresolvable citation is a hard failure by design.
        atoms = {
            "observed_all": [
                Atom("arithmetic_mean_era", "Era de média aritmética", 0.465, 32588),
                Atom("calm", "Calmarias", 0.1, 55),
            ]
        }
        payload = build_variable_payload(spec, samples, version="v1", atoms=atoms)

        published = {atom["id"]: atom for atom in payload["subsets"]["observed_all"]["atoms"]}
        assert published["calm"]["count"] == 55
        assert published["arithmetic_mean_era"]["count"] == 32588


class TestCaveatsCarryThePublishedNumbers:
    """A number in prose is a copy, and a copy stops being true.

    Every count, share and fitted parameter a caveat quotes is interpolated from
    the value the same payload publishes, so reprocessing that moves the numbers
    moves the sentences with them and a caveat can never contradict the panel
    printed beside it.
    """

    @staticmethod
    def _spec(caveat: str) -> VariableSpec:
        return VariableSpec(
            id="probe",
            label="Sonda",
            unit="m/s",
            chart="histogram",
            family="weibull",
            family_label="Weibull",
            edges=tuple(float(v) for v in range(21)),
            caveats=(caveat,),
        )

    def test_an_atom_count_comes_from_the_atom(self, wind_samples):
        spec = self._spec("Saem do ajuste {{atom:calm:count}} horas de calmaria.")
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.037, 3663)]}

        payload = build_variable_payload(spec, wind_samples, version="v1", atoms=atoms)

        assert payload["caveats"][0] == "Saem do ajuste 3.663 horas de calmaria."
        assert payload["subsets"]["observed_all"]["atoms"][0]["count"] == 3663

    def test_an_atom_share_comes_from_the_atom(self, wind_samples):
        spec = self._spec("Calmarias: {{atom:calm:share}} % do registro.")
        atoms = {"observed_all": [Atom("calm", "Calmarias", 0.0523, 3663)]}

        payload = build_variable_payload(spec, wind_samples, version="v1", atoms=atoms)

        assert payload["caveats"][0] == "Calmarias: 5,2 % do registro."

    def test_a_fitted_parameter_comes_from_the_fit(self, wind_samples):
        spec = self._spec("Forma estimada: {{param:shape:3}}.")

        payload = build_variable_payload(spec, wind_samples, version="v1")

        shape = payload["subsets"]["observed_all"]["fit"]["params"]["shape"]
        printed = f"{shape:.3f}".replace(".", ",")
        assert payload["caveats"][0] == f"Forma estimada: {printed}."

    def test_an_unpublished_atom_stops_the_export(self, wind_samples):
        """A sentence quoting a number nobody published must not reach a reader."""
        spec = self._spec("Saem {{atom:ausente:count}} horas.")

        with pytest.raises(ValueError, match="ausente"):
            build_variable_payload(spec, wind_samples, version="v1")

    def test_an_unpublished_parameter_stops_the_export(self, wind_samples):
        spec = self._spec("Coeficiente {{param:inexistente:2}}.")

        with pytest.raises(ValueError, match="inexistente"):
            build_variable_payload(spec, wind_samples, version="v1")


class TestTheShippedCaveatsQuoteNoLooseCount:
    """No caveat in the catalogue may hard-code a count the export computes.

    The check is on the CATALOGUE, not on one export, so a literal reintroduced
    by a future edit is caught before it can ship.
    """

    def test_no_shipped_caveat_states_a_bare_hour_count(self):
        # Counts are what go stale; a threshold ("acima de 10°") or a cited
        # range does not, so only "<digits> horas" is refused.
        offenders = [
            f"{spec.id}: {match.group(0)}"
            for spec in CLIMATOLOGY_VARIABLES
            for caveat in spec.caveats
            for match in re.finditer(r"(\d[\d.]*)\s+horas", caveat)
        ]
        assert offenders == [], (
            "hard-coded hour counts in caveat prose; interpolate with "
            "{{atom:<id>:count}} so the sentence and the panel cannot disagree"
        )
