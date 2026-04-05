"""Runtime compatibility helpers for third-party dependencies."""

from __future__ import annotations

import sys
import warnings

_LANGCHAIN_PYDANTIC_WARNING = (
    r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\."
)


def suppress_known_warnings() -> None:
    """Silence known third-party warnings that are harmless for this package.

    LangChain 1.2.x currently imports Pydantic's V1 compatibility layer, which emits a
    Python 3.14 warning during import even though the repository's current test suite passes.
    Keep the filter narrow so unrelated warnings remain visible.
    """
    if sys.version_info < (3, 14):
        return

    warnings.filterwarnings(
        "ignore",
        message=_LANGCHAIN_PYDANTIC_WARNING,
        category=UserWarning,
    )
