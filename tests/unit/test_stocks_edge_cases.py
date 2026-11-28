"""Offline edge-case tests for ``pyequity.api.stocks.get_history``.

Complements test_stocks.py: argument validation, responses that must never be
cached, an exception inside a fetch, and the integrity warning. The connector
outcome functions are faked; the cache is a real one in a temporary folder.
Nothing touches the network.
"""
import pandas as pd
import pytest

from pyequity.api import stocks
from pyequity.api._errors import (
    DataQualityWarning, DataUnavailableError, FallbackWarning, InvalidArgumentError)
from pyequity.reliability.cache import DiskCache
from pyequity.reliability.outcome import FetchOutcome, Status

DAY = "2025-01-01"


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    shared = DiskCache(directory=tmp_path)
    monkeypatch.setattr(stocks, "get_default_cache", lambda: shared)
    return shared


def frame(high=110.0, low=95.0, source="NSE"):
    return pd.DataFrame({
        "date": [DAY], "open": [100.0], "high": [high], "low": [low], "close": [105.0],
        "volume": [1000], "trades": [100], "symbol": ["RELIANCE"], "source": [source]})


class Counter:
    """Fake connector outcome function that counts its calls."""

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def __call__(self, *args):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def install(monkeypatch, nse=None, bse=None):
    nse = nse if nse is not None else Counter(FetchOutcome(Status.FAILED, reason="NSE not set up"))
    bse = bse if bse is not None else Counter(FetchOutcome(Status.FAILED, reason="BSE not set up"))
    monkeypatch.setattr(stocks._nse, "fetch_nse_historical_outcome", nse)
    monkeypatch.setattr(stocks._bse, "fetch_bse_historical_by_symbol_outcome", bse)
    return nse, bse


# ---- 1. invalid symbol ---------------------------------------------------
@pytest.mark.parametrize("symbol", ["", "   ", None, 5, ["RELIANCE"]])
def test_invalid_symbol_raises_before_any_fetch(monkeypatch, symbol):
    nse, bse = install(monkeypatch)
    with pytest.raises(InvalidArgumentError, match="symbol"):
        stocks.get_history(symbol, DAY, DAY)
    assert nse.calls == 0 and bse.calls == 0


# ---- 2. invalid series ---------------------------------------------------
@pytest.mark.parametrize("series", ["", "   ", None, 5])
def test_invalid_series_raises_before_any_fetch(monkeypatch, series):
    nse, bse = install(monkeypatch)
    with pytest.raises(InvalidArgumentError, match="series"):
        stocks.get_history("RELIANCE", DAY, DAY, series=series)
    assert nse.calls == 0 and bse.calls == 0


# ---- 3. an "ok" answer without usable rows is a failure and is not cached --
@pytest.mark.parametrize("data", [pd.DataFrame(), None, "not a frame"],
                         ids=["empty-frame", "none", "not-a-frame"])
def test_ok_outcome_without_usable_rows_fails_and_is_not_cached(monkeypatch, cache, data):
    nse, _ = install(monkeypatch, nse=Counter(FetchOutcome(Status.OK, data)))
    for _ in range(2):
        with pytest.raises(DataUnavailableError, match="invalid or incomplete frame"):
            stocks.get_history("RELIANCE", DAY, DAY, source="nse")
    assert nse.calls == 2                                  # the bad answer was not remembered
    assert cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}


# ---- 4. an exception inside the fetch ------------------------------------
def test_exception_in_a_fetch_is_reported_not_raised_raw_and_not_cached(monkeypatch, cache):
    nse, _ = install(monkeypatch, nse=Counter(RuntimeError("parser blew up")))
    with pytest.raises(DataUnavailableError, match="cache/fetch operation failed.*parser blew up"):
        stocks.get_history("RELIANCE", DAY, DAY, source="nse")
    assert cache.stats() == {"rows": 0, "coverage": 0, "snapshots": 0}

    # with source="auto" the same exception counts as an NSE failure, so BSE is tried
    bse_frame = frame(source="BSE")
    nse, bse = install(monkeypatch, nse=Counter(RuntimeError("parser blew up")),
                       bse=Counter(FetchOutcome(Status.OK, bse_frame)))
    with pytest.warns(FallbackWarning):
        out = stocks.get_history("RELIANCE", DAY, DAY)
    assert out.attrs["source"] == "BSE" and out.attrs["fallback_used"] is True
    assert "parser blew up" in out.attrs["fallback_reason"]
    assert nse.calls == 1 and bse.calls == 1


# ---- 5. integrity problems are warned about, never hidden ----------------
def test_integrity_errors_are_warned_and_the_rows_are_kept(monkeypatch):
    install(monkeypatch, nse=Counter(FetchOutcome(Status.OK, frame(high=90.0, low=95.0))))
    with pytest.warns(DataQualityWarning, match="integrity"):
        out = stocks.get_history("RELIANCE", DAY, DAY, source="nse")
    assert len(out) == 1                                   # reported, never silently dropped
    assert out.loc[0, "high"] < out.loc[0, "low"]
    assert out.attrs["source"] == "NSE"
