"""Parsers for the repeatable, comma-separated option style every CLI accepts.

``-D 1 -D 4``, ``-D 1,4`` and ``-D "1, 4"`` all mean the same thing to every
console script, because cron lines are written by hand and both spellings are
in use. Parsing them in one place is what keeps that promise uniform: a
per-command copy is free to drift, and one that stops dropping empty tokens
turns ``-D "1, "`` into a silently different selection.
"""

import typer


def parse_csv(raw: str | list[str] | None) -> tuple[str, ...]:
    """Flatten repeated and/or comma-separated string options, dropping blanks.

    Parameters
    ----------
    raw:
        What typer hands over for the option: one occurrence as a string, every
        occurrence as a list, or ``None`` when the option was never given.

    Returns
    -------
    tuple of str
        Whitespace-stripped tokens in the order typed, empty when nothing was
        given. Occurrence order and comma order are not distinguished.
    """
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

    Parameters
    ----------
    raw:
        As for :func:`parse_csv`.

    Returns
    -------
    tuple of int
        The tokens parsed in the order typed, empty when nothing was given.

    Raises
    ------
    typer.BadParameter
        For a token that is not a whole number. A mistyped domain is a usage
        error, so it is reported as one instead of escaping as a bare
        ``ValueError`` traceback.
    """
    numbers: list[int] = []
    for token in parse_csv(raw):
        try:
            numbers.append(int(token))
        except ValueError:
            raise typer.BadParameter(f"expected a whole number, got {token!r}") from None
    return tuple(numbers)
