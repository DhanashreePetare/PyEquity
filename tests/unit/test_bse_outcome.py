"""Offline tests for ``fetch_bse_historical_by_symbol_outcome`` (connectors/bse.py).

This is the BSE leg of the ``source="auto"`` fallback, so the difference
between FAILED (try something else) and EMPTY (a real answer) matters. Only
the HTTP session is faked: chunking, retry, de-duplication and the outcome
logic are the real code. Nothing touches the network or sleeps.
"""
import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal

from pyequity.connectors import bse
from pyequity.reliability.outcome import Status
from pyequity.schema import SCHEMAS, empty_frame

# 2026-01-01..2026-03-15 is split into three requests by the connector:
#   chunk 1: 2026-01-01..2026-01-31, chunk 2: 2026-02-01..2026-03-03,
#   chunk 3: 2026-03-04..2026-03-15
RANGE = ("2026-01-01", "2026-03-15")
CHUNK_1, CHUNK_2, CHUNK_3 = "MonthDate=01/01/2026", "MonthDate=01/02/2026", "MonthDate=04/03/2026"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    bse._BSE_SESSIONS.close()
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(bse._BSE_RATE_LIMITER, "wait", lambda *a, **k: None)
    yield
    bse._BSE_SESSIONS.close()


def record(day):
    return {"RealDate": f"{day}T00:00:00", "qe_open": "1,254.45", "qe_high": "1,259.50",
            "qe_low": "1,236.00", "qe_close": "1,240.00",
            "no_of_shrs": "11,57,037", "no_trades": "40,761"}


def payload(*days, scrip_id="500325"):
    return {"scrip_id": scrip_id, "StockData": [record(d) for d in days]}


EMPTY_PAYLOAD = {"StockData": None, "spnError": "no data"}


class Response:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


def serve(monkeypatch, by_chunk):
    """Fake BSE. ``by_chunk`` maps a chunk marker to a payload, an exception or a callable."""
    calls = []

    class Session:
        headers = {}

        def get(self, url, timeout=None):
            calls.append(url)
            for marker, answer in by_chunk.items():
                if marker in url:
                    if isinstance(answer, Exception):
                        raise answer
                    return Response(answer)
            raise AssertionError(f"unexpected request {url}")

        def close(self):
            pass

    session = Session()
    monkeypatch.setattr(bse, "_create_bse_session", lambda *a, **k: session)
    return calls


def outcome(symbol="RELIANCE", start=RANGE[0], end=RANGE[1]):
    return bse.fetch_bse_historical_by_symbol_outcome(symbol, start, end)


# ---- arguments are rejected before any request ------------------------
@pytest.mark.parametrize("args,reason", [
    (("NOTASYMBOL", *RANGE), "mapping unavailable"),
    (("RELIANCE", "01/01/2026", RANGE[1]), "Invalid dates"),
    (("RELIANCE", RANGE[0], "not-a-date"), "Invalid dates"),
    (("RELIANCE", None, RANGE[1]), "Invalid dates"),
    (("RELIANCE", RANGE[1], RANGE[0]), "after end date"),
], ids=["unknown-symbol", "bad-start", "bad-end", "none-start", "reversed-range"])
def test_bad_arguments_fail_without_a_request(monkeypatch, args, reason):
    calls = serve(monkeypatch, {})
    result = bse.fetch_bse_historical_by_symbol_outcome(*args)
    assert result.status is Status.FAILED and result.failed
    assert result.data is None
    assert reason in result.reason
    assert calls == []


# ---- failures are never partial ---------------------------------------
def test_a_failed_chunk_fails_the_whole_call_with_no_partial_data(monkeypatch):
    calls = serve(monkeypatch, {
        CHUNK_1: payload("2026-01-05"),
        CHUNK_2: requests.ConnectionError("timed out"),
        CHUNK_3: payload("2026-03-10"),
    })
    result = outcome()
    assert result.failed and result.data is None
    assert "2026-02-01..2026-03-03" in result.reason          # names the chunk that failed
    assert result.attempts == 1 + bse._ATTEMPTS               # chunk 1 once, chunk 2 retried
    assert len(calls) == 1 + bse._ATTEMPTS
    assert not any(CHUNK_3 in u for u in calls)               # stopped; later chunk never asked


def test_a_chunk_at_the_row_cap_fails_instead_of_trusting_truncated_data(monkeypatch):
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-01", periods=bse.BSE_ROW_CAP)]
    calls = serve(monkeypatch, {CHUNK_1: payload(*days)})
    result = outcome(end="2026-01-31")
    assert result.failed and result.data is None
    assert "row cap" in result.reason and "2026-01-01..2026-01-31" in result.reason
    assert len(calls) == 1


def test_an_unexpected_exception_becomes_a_failed_outcome(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("something odd")

    monkeypatch.setattr(bse, "_chunk_with_retry", boom)
    result = outcome()                                         # must not raise
    assert result.failed and result.data is None
    assert "unexpected fetch error" in result.reason and "something odd" in result.reason
    assert result.attempts == 0


# ---- empty vs ok --------------------------------------------------------
def test_every_chunk_empty_is_empty_not_failed(monkeypatch):
    calls = serve(monkeypatch, {CHUNK_1: EMPTY_PAYLOAD, CHUNK_2: EMPTY_PAYLOAD,
                                CHUNK_3: EMPTY_PAYLOAD})
    result = outcome()
    assert result.status is Status.EMPTY and result.empty and not result.failed
    assert_frame_equal(result.data, empty_frame("equity"))
    assert len(calls) == 3                                     # an empty answer is not retried
    assert result.attempts == 3


def test_a_mix_of_empty_and_ok_chunks_is_ok_with_only_the_rows_returned(monkeypatch):
    serve(monkeypatch, {CHUNK_1: payload("2026-01-05"), CHUNK_2: EMPTY_PAYLOAD,
                        CHUNK_3: payload("2026-03-10")})
    result = outcome()
    assert result.status is Status.OK and result.ok
    assert result.data["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-05", "2026-03-10"]
    assert result.attempts == 3


def test_ok_merge_removes_duplicates_sorts_and_labels_rows(monkeypatch):
    serve(monkeypatch, {
        CHUNK_1: payload("2026-01-05", "2026-01-02", scrip_id="500325"),   # unsorted
        CHUNK_2: payload("2026-01-05", "2026-02-03", scrip_id="500325"),   # repeats 2026-01-05
        CHUNK_3: payload("2026-03-10", scrip_id="500325"),
    })
    result = outcome(symbol="reliance")
    assert result.ok
    df = result.data
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02", "2026-01-05", "2026-02-03", "2026-03-10"]
    assert (df["symbol"] == "RELIANCE").all()                  # the NSE symbol, not the scripcode
    assert (df["source"] == "BSE").all()
    assert list(df.columns) == list(SCHEMAS["equity"])
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["equity"]
    assert result.attempts == 3                                # summed over the three chunks
    assert df.index.tolist() == [0, 1, 2, 3]
    assert (df["volume"] == 1157037).all() and (df["open"] == 1254.45).all()   # "11,57,037" cleaned
