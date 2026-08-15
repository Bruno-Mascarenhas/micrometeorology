"""Setup every ``allsky`` command runs before it does any work.

Its own module rather than the package ``__init__``: that one imports each
command module to register it, so a command importing back from it would close
the cycle.
"""

import logging

__all__ = ["CLI_LOG_FORMAT", "configure_cli_logging"]

#: One format for every command, so a log line read out of a cron mail names the
#: logger that produced it and the moment it did.
CLI_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def configure_cli_logging() -> None:
    """Send INFO and above to stderr in :data:`CLI_LOG_FORMAT`.

    ``basicConfig`` is a no-op once the root logger has a handler, so calling
    this from a command that another command already called it from — or from a
    test that configured logging itself — changes nothing.
    """
    logging.basicConfig(level=logging.INFO, format=CLI_LOG_FORMAT)
