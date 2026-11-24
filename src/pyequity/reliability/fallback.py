
"""Fallback orchestration for data-source operations."""

from dataclasses import dataclass
from typing import Callable

from .outcome import FetchOutcome, Status


@dataclass(frozen=True)
class FallbackResult:
    """Outcome and metadata describing which source supplied the result."""

    outcome: FetchOutcome
    source: str
    fallback_used: bool = False
    primary_reason: str = ""


def run_with_fallback(
    primary: Callable[[], FetchOutcome],
    fallback: Callable[[], FetchOutcome],
    *,
    primary_source: str = "NSE",
    fallback_source: str = "BSE",
) -> FallbackResult:
    """Run the primary operation and use the fallback only after failure.

    A valid empty result is returned as-is and does not trigger fallback.
    If both sources fail, the returned failure includes both reasons.
    """
    first = primary()

    if first.status is not Status.FAILED:
        return FallbackResult(
            outcome=first,
            source=primary_source,
        )

    second = fallback()

    if second.status is not Status.FAILED:
        return FallbackResult(
            outcome=second,
            source=fallback_source,
            fallback_used=True,
            primary_reason=first.reason,
        )

    combined = FetchOutcome(
        Status.FAILED,
        None,
        (
            f"{primary_source} failed: {first.reason}; "
            f"{fallback_source} failed: {second.reason}"
        ),
        first.attempts + second.attempts,
    )
    return FallbackResult(
        outcome=combined,
        source="",
        fallback_used=True,
        primary_reason=first.reason,
    )
