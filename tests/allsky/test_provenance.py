"""The provenance digests every resume gate in the stack keys on."""

import pytest

from allsky.config import PrepareConfig
from allsky.provenance import config_subset_sha256


class TestConfigSubsetRefusesASectionThatIsNotThere:
    """The guard is what makes a renamed section a loud failure instead of a
    digest that quietly stops covering it — and every resume gate in the stack
    keys on that digest."""

    def test_an_unknown_top_level_section_is_named(self):
        with pytest.raises(RuntimeError, match="nao_existe"):
            config_subset_sha256(PrepareConfig(), sections=("mask", "nao_existe"), subject="x")

    def test_an_unknown_nested_field_is_named(self):
        with pytest.raises(RuntimeError, match="nao_existe"):
            config_subset_sha256(
                PrepareConfig(),
                sections=(),
                nested_fields={"video": ("nao_existe",)},
                subject="x",
            )

    def test_the_sections_that_do_exist_still_hash(self):
        assert len(config_subset_sha256(PrepareConfig(), sections=("mask",), subject="x")) == 64
