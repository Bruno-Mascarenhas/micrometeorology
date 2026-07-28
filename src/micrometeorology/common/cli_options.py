"""Parsers for the repeatable, comma-separated option style every CLI accepts.

``-D 1 -D 4``, ``-D 1,4`` and ``-D "1, 4"`` all mean the same thing to every
console script, because cron lines are written by hand and both spellings are
in use. The parsing lived as four near-identical private copies, one of which
had already drifted into not dropping empty tokens.
"""

from __future__ import annotations

import typer


def parse_csv(raw: str | list[str] | None) -> tuple[str, ...]:
    """Flatten repeated and/or comma-separated string options, dropping blanks."""
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    tokens: list[str] = []
    for item in raw:
        tokens.extend(token.strip() for token in item.split(",") if token.strip())
    return tuple(tokens)


def parse_int_csv(raw: str | list[str] | None) -> tuple[int, ...]:
    """Flatten repeated and/or comma-separated integer options, dropping blanks.

    A token that is not an integer is a usage error, so it is reported as one
    instead of escaping as a bare ``ValueError`` traceback.
    """
    numbers: list[int] = []
    for token in parse_csv(raw):
        try:
            numbers.append(int(token))
        except ValueError:
            raise typer.BadParameter(f"expected a whole number, got {token!r}") from None
    return tuple(numbers)
