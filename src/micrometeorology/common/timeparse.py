"""Strict naive-timestamp parsing, shared by every strftime-format entry point.

This package deliberately works in **naive** station-local / WRF time: sensor
frames and WRF outputs both carry naive ``DatetimeIndex``es, and comparing one
against an aware bound raises ``TypeError``. That rules out the timezone-aware
spellings ruff's DTZ rules would otherwise push a ``datetime.strptime`` call
toward, so parsing goes through pandas instead -- same strftime format, same
naive result, no DTZ violation.

The catch, and the reason this lives in one place: ``pd.to_datetime`` is *not* a
drop-in for ``datetime.strptime``. Even with an explicit ``format=`` it has two
escape hatches that ``strptime`` does not have, and both fail open --

- ``""``, ``"nan"``, ``"NaN"``, ``"NaT"`` (any case) parse to ``NaT`` instead of
  raising, and ``NaT`` propagates silently: ``NaT + Timedelta`` is ``NaT``, every
  comparison against it is ``False``, and ``NaT.date()`` returns ``NaT`` rather
  than erroring, so a bad value reads as "no data" rather than "bad input";
- ``"today"`` and ``"now"`` **bypass the format entirely** and return the current
  timestamp, so a malformed value silently becomes *right now*.

:func:`parse_naive_timestamp` closes both, restoring exact ``strptime`` "match
or raise" semantics. Use it for anything parsing an operator- or filename-supplied
date; a silent wrong answer in a cron-driven pipeline is far worse than a crash.
"""

import pandas as pd

__all__ = ["parse_naive_timestamp"]

#: Strings pandas resolves against the clock instead of the supplied format.
_RELATIVE_KEYWORDS = frozenset({"today", "now"})


def parse_naive_timestamp(value: str, fmt: str) -> pd.Timestamp:
    """Parse *value* with *fmt*, or raise ``ValueError``.

    Behaves like ``datetime.strptime(value, fmt)`` -- exact match or error -- but
    returns a naive :class:`pandas.Timestamp` and does not trip ruff's DTZ rules.

    Raises
    ------
    ValueError
        If *value* is not an exact match for *fmt*, including the ``NaT``
        sentinels and the relative keywords pandas would otherwise accept.
    """
    if value.strip().lower() in _RELATIVE_KEYWORDS:
        raise ValueError(f"time data {value!r} does not match format {fmt!r}")
    parsed = pd.to_datetime(value, format=fmt)
    if pd.isna(parsed):
        raise ValueError(f"time data {value!r} does not match format {fmt!r}")
    return parsed
