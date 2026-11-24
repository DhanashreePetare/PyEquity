
from pyequity.reliability.fallback import run_with_fallback
from pyequity.reliability.outcome import FetchOutcome, Status


def test_primary_success_does_not_call_fallback():
    fallback_called = False

    def fallback():
        nonlocal fallback_called
        fallback_called = True
        return FetchOutcome(Status.OK, "bse")

    result = run_with_fallback(
        lambda: FetchOutcome(Status.OK, "nse"),
        fallback,
    )

    assert result.outcome.data == "nse"
    assert result.source == "NSE"
    assert result.fallback_used is False
    assert fallback_called is False


def test_valid_empty_primary_does_not_call_fallback():
    fallback_called = False

    def fallback():
        nonlocal fallback_called
        fallback_called = True
        return FetchOutcome(Status.OK, "bse")

    result = run_with_fallback(
        lambda: FetchOutcome(Status.EMPTY, [], "no rows"),
        fallback,
    )

    assert result.outcome.status is Status.EMPTY
    assert result.source == "NSE"
    assert result.fallback_used is False
    assert fallback_called is False


def test_primary_failure_uses_fallback():
    result = run_with_fallback(
        lambda: FetchOutcome(Status.FAILED, None, "NSE unavailable", 2),
        lambda: FetchOutcome(Status.OK, "bse data", attempts=1),
    )

    assert result.outcome.status is Status.OK
    assert result.outcome.data == "bse data"
    assert result.source == "BSE"
    assert result.fallback_used is True
    assert result.primary_reason == "NSE unavailable"


def test_both_sources_failing_preserves_both_reasons():
    result = run_with_fallback(
        lambda: FetchOutcome(Status.FAILED, None, "NSE unavailable", 2),
        lambda: FetchOutcome(Status.FAILED, None, "BSE unavailable", 3),
    )

    assert result.outcome.status is Status.FAILED
    assert "NSE unavailable" in result.outcome.reason
    assert "BSE unavailable" in result.outcome.reason
    assert result.outcome.attempts == 5
