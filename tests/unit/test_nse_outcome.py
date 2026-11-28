"""Offline tests for the exception handlers of ``fetch_nse_historical_outcome``.

Both go through the public outcome function and replace only the chunk
helper, so a later cleanup of the private request code does not break them.
Nothing touches the network or sleeps.
"""
import pandas as pd
import pytest

from pyequity.connectors import nse
from pyequity.reliability.outcome import Status
from pyequity.schema import empty_frame


class QuietSession:
    """Stands in for the warmed-up NSE session; never used for a real request."""


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    nse._NSE_SESSIONS.close()
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(nse, "create_nse_session", lambda *a, **k: QuietSession())
    yield
    nse._NSE_SESSIONS.close()


def good_chunk(day="2026-09-15"):
    return pd.DataFrame({
        "date": pd.to_datetime([day]), "open": [1242.0], "high": [1252.8],
        "low": [1238.9], "close": [1248.0], "volume": [8352048], "trades": [147001]})


def test_unexpected_exception_while_fetching_is_a_failed_outcome_and_drops_the_session(
        monkeypatch):
    invalidated = []
    monkeypatch.setattr(nse._NSE_SESSIONS, "invalidate", invalidated.append)

    def boom(*args, **kwargs):
        raise RuntimeError("something odd")

    monkeypatch.setattr(nse, "_chunk_with_retry", boom)
    result = nse.fetch_nse_historical_outcome("reliance", "2026-09-15", "2026-09-23")
    assert result.status is Status.FAILED and result.data is None
    assert "unexpected fetch error" in result.reason and "something odd" in result.reason
    assert result.attempts == 0
    assert invalidated == ["RELIANCE"]            # a possibly broken session is not reused

    # the legacy wrapper turns the same failure into an empty frame, never an exception
    assert nse.fetch_nse_historical("reliance", "2026-09-15", "2026-09-23").equals(
        empty_frame("equity"))


def test_response_that_cannot_be_processed_is_a_failed_outcome_not_an_exception(monkeypatch):
    def lacks_a_column(session, symbol, fmt_from, fmt_to, series, *, return_attempts=False):
        frame = good_chunk().drop(columns=["trades"])   # reported "ok" but malformed
        return (frame, "ok", 1) if return_attempts else (frame, "ok")

    monkeypatch.setattr(nse, "_chunk_with_retry", lacks_a_column)
    result = nse.fetch_nse_historical_outcome("RELIANCE", "2026-09-15", "2026-09-23")
    assert result.status is Status.FAILED and result.failed
    assert result.data is None                    # never a half-built frame
    assert "response processing failed" in result.reason and "trades" in result.reason
    assert result.attempts == 1
