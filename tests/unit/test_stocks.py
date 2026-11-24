import pandas as pd
import pytest

from pyequity.api import stocks
from pyequity.api._errors import (
    DataUnavailableError,
    EmptyResultWarning,
    FallbackWarning,
    InvalidArgumentError,
)
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.reliability.cache import DiskCache

@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    cache = DiskCache(directory=tmp_path)
    monkeypatch.setattr(stocks, "get_default_cache", lambda: cache)


@pytest.fixture
def history_frame():
    return pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [110.0],
            "low": [95.0],
            "close": [105.0],
            "volume": [1000],
            "trades": [100],
            "symbol": ["RELIANCE"],
            "source": ["NSE"],
        }
    )


def success(frame):
    return FetchOutcome(Status.OK, frame)


def empty():
    return FetchOutcome(Status.EMPTY, pd.DataFrame())


def failure(reason="source unavailable"):
    return FetchOutcome(Status.FAILED, reason=reason)


def test_nse_success_does_not_call_bse(monkeypatch, history_frame):
    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: success(history_frame),
    )

    def unexpected_bse(*args):
        pytest.fail("BSE must not be called when NSE succeeds")

    monkeypatch.setattr(
        stocks._bse, "fetch_bse_historical_by_symbol_outcome", unexpected_bse
    )

    result = stocks.get_history("RELIANCE", "2025-01-01", "2025-01-01")

    assert len(result) == 1
    assert result.attrs["source"] == "NSE"
    assert result.attrs["fallback_used"] is False


def test_nse_failure_uses_bse(monkeypatch, history_frame):
    bse_frame = history_frame.copy()
    bse_frame["source"] = "BSE"

    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: failure("NSE unavailable"),
    )
    monkeypatch.setattr(
        stocks._bse,
        "fetch_bse_historical_by_symbol_outcome",
        lambda *args: success(bse_frame),
    )

    with pytest.warns(FallbackWarning):
        result = stocks.get_history(
            "RELIANCE", "2025-01-01", "2025-01-01"
        )

    assert len(result) == 1
    assert result["source"].iloc[0] == "BSE"
    assert result.attrs["source"] == "BSE"
    assert result.attrs["fallback_used"] is True
    assert "NSE unavailable" in result.attrs["fallback_reason"]


def test_valid_empty_nse_does_not_call_bse(monkeypatch):
    monkeypatch.setattr(
        stocks._nse, "fetch_nse_historical_outcome", lambda *args: empty()
    )

    def unexpected_bse(*args):
        pytest.fail("BSE must not be called after a valid empty NSE response")

    monkeypatch.setattr(
        stocks._bse, "fetch_bse_historical_by_symbol_outcome", unexpected_bse
    )

    with pytest.warns(EmptyResultWarning):
        result = stocks.get_history(
            "RELIANCE", "2025-01-01", "2025-01-01"
        )

    assert result.empty
    assert result.attrs["source"] == "NSE"
    assert result.attrs["fallback_used"] is False


def test_both_sources_failing_raise(monkeypatch):
    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: failure("NSE down"),
    )
    monkeypatch.setattr(
        stocks._bse,
        "fetch_bse_historical_by_symbol_outcome",
        lambda *args: failure("BSE down"),
    )

    with pytest.raises(DataUnavailableError) as exc:
        stocks.get_history("RELIANCE", "2025-01-01", "2025-01-01")

    assert "NSE down" in str(exc.value)
    assert "BSE down" in str(exc.value)


def test_bse_mapping_unavailable_raises(monkeypatch):
    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: failure("NSE down"),
    )
    monkeypatch.setattr(
        stocks._bse,
        "fetch_bse_historical_by_symbol_outcome",
        lambda *args: failure("BSE mapping unavailable"),
    )

    with pytest.raises(DataUnavailableError, match="BSE mapping unavailable"):
        stocks.get_history("UNKNOWN", "2025-01-01", "2025-01-01")


def test_invalid_connector_response_raises(monkeypatch):
    invalid_frame = pd.DataFrame({"unexpected": [1]})

    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: success(invalid_frame),
    )

    with pytest.raises(DataUnavailableError):
        stocks.get_history(
            "RELIANCE",
            "2025-01-01",
            "2025-01-01",
            source="nse",
        )


def test_explicit_bse_selection_does_not_call_nse(
    monkeypatch, history_frame
):
    bse_frame = history_frame.copy()
    bse_frame["source"] = "BSE"

    def unexpected_nse(*args):
        pytest.fail("NSE must not be called for source='bse'")

    monkeypatch.setattr(
        stocks._nse, "fetch_nse_historical_outcome", unexpected_nse
    )
    monkeypatch.setattr(
        stocks._bse,
        "fetch_bse_historical_by_symbol_outcome",
        lambda *args: success(bse_frame),
    )

    result = stocks.get_history(
        "RELIANCE",
        "2025-01-01",
        "2025-01-01",
        source="bse",
    )

    assert result.attrs["source"] == "BSE"
    assert result.attrs["fallback_used"] is False


def test_explicit_nse_failure_does_not_fallback(monkeypatch):
    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: failure("NSE down"),
    )

    def unexpected_bse(*args):
        pytest.fail("Explicit NSE selection must not use BSE")

    monkeypatch.setattr(
        stocks._bse, "fetch_bse_historical_by_symbol_outcome", unexpected_bse
    )

    with pytest.raises(DataUnavailableError, match="NSE down"):
        stocks.get_history(
            "RELIANCE",
            "2025-01-01",
            "2025-01-01",
            source="nse",
        )


def test_invalid_source_raises():
    with pytest.raises(InvalidArgumentError):
        stocks.get_history(
            "RELIANCE",
            "2025-01-01",
            "2025-01-01",
            source="unknown",
        )

def test_nse_failure_bse_empty_preserves_fallback_metadata(monkeypatch):
    monkeypatch.setattr(
        stocks._nse,
        "fetch_nse_historical_outcome",
        lambda *args: failure("NSE unavailable"),
    )
    monkeypatch.setattr(
        stocks._bse,
        "fetch_bse_historical_by_symbol_outcome",
        lambda *args: empty(),
    )

    with pytest.warns(Warning) as recorded:
        result = stocks.get_history(
            "RELIANCE", "2025-01-01", "2025-01-01"
        )

    warning_types = {item.category for item in recorded}

    assert FallbackWarning in warning_types
    assert EmptyResultWarning in warning_types

    assert result.empty
    assert result.attrs["source"] == "BSE"
    assert result.attrs["fallback_used"] is True
    assert result.attrs["fallback_reason"] == "NSE unavailable"

def test_nse_success_is_cached(monkeypatch, history_frame):
    calls = []

    def fetch(*args):
        calls.append(args)
        return success(history_frame)

    monkeypatch.setattr(stocks._nse, "fetch_nse_historical_outcome", fetch)

    first = stocks.get_history("RELIANCE", "2025-01-01", "2025-01-01")
    second = stocks.get_history("RELIANCE", "2025-01-01", "2025-01-01")

    assert len(first) == len(second) == 1
    assert len(calls) == 1


def test_nse_failed_fetch_is_not_cached(monkeypatch, history_frame):
    calls = []

    def fetch(*args):
        calls.append(args)
        if len(calls) == 1:
            return failure("temporary NSE failure")
        return success(history_frame)

    monkeypatch.setattr(stocks._nse, "fetch_nse_historical_outcome", fetch)

    with pytest.raises(DataUnavailableError, match="temporary NSE failure"):
        stocks.get_history(
            "RELIANCE", "2025-01-01", "2025-01-01", source="nse"
        )

    result = stocks.get_history(
        "RELIANCE", "2025-01-01", "2025-01-01", source="nse"
    )

    assert len(result) == 1
    assert len(calls) == 2


def test_nse_and_bse_cache_entries_are_separate(monkeypatch, history_frame):
    nse_calls = []
    bse_calls = []

    bse_frame = history_frame.copy()
    bse_frame["source"] = "BSE"

    def fetch_nse(*args):
        nse_calls.append(args)
        return success(history_frame)

    def fetch_bse(*args):
        bse_calls.append(args)
        return success(bse_frame)

    monkeypatch.setattr(
        stocks._nse, "fetch_nse_historical_outcome", fetch_nse
    )
    monkeypatch.setattr(
        stocks._bse, "fetch_bse_historical_by_symbol_outcome", fetch_bse
    )

    stocks.get_history(
        "RELIANCE", "2025-01-01", "2025-01-01", source="nse"
    )
    stocks.get_history(
        "RELIANCE", "2025-01-01", "2025-01-01", source="bse"
    )

    assert len(nse_calls) == 1
    assert len(bse_calls) == 1