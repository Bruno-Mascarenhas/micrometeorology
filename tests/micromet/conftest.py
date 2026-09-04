"""Shared helpers for the climatology and distribution suites."""

import inspect

from micrometeorology.stats.distributions import FAMILIES


def required_fit_options(family: str) -> set[str]:
    """Keyword-only parameters of a family's estimator that carry no default.

    Derived from the signature rather than listed by hand, so a new family or a
    new covariate lands in the check without anyone remembering to say so.
    """
    signature = inspect.signature(FAMILIES[family].fit)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
