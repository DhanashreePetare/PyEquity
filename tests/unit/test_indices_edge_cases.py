"""Offline edge-case tests for ``pyequity.connectors.indices.fetch_index_history``.

Failure paths that must give an empty frame (never partial data) and a clear
error log: request-budget exhaustion, a failed retry under the long alias
name, and a payload that breaks parsing. They reuse the fake NSE server of
test_indices.py; nothing touches the network.
"""
import json
import logging
from datetime import date

import pytest
import requests

from pyequity.connectors import indices
from pyequity.connectors.indices import fetch_index_history
from pyequity.schema import SCHEMAS

LOGGER = "pyequity.connectors.indices"
START, END = "2026-03-02", "2026-03-13"
HISTORY, NAMES = "historicalOR/indicesHistory", "index-names"
NAME_PAIRS = [["NIFTY 50", "NIFTY 50"], ["NIFTY FIN SERVICE", "NIFTY FINANCIAL SERVICES"]]


class FakeResponse:
    def __init__(self, body, status=200):
        self.content = json.dumps(body).encode()
        self.text = self.content.decode()
        self.status_code = status

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error")


class FakeNSE:
    """Answers the name list itself; every history request goes to ``history_override``."""

    def __init__(self):
        self.calls = []
        self.history_override = None

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if NAMES in url:
            return FakeResponse({"stn": NAME_PAIRS})
        assert HISTORY in url, url
        return self.history_override(params)

    def history_calls(self):
        return [p for u, p in self.calls if HISTORY in u]

    def names_calls(self):
        return sum(1 for u, _ in self.calls if NAMES in u)

    @staticmethod
    def record(eod_name, day):
        return {"EOD_INDEX_NAME": eod_name, "EOD_OPEN_INDEX_VAL": 1000.0,
                "EOD_HIGH_INDEX_VAL": 1005.0, "EOD_CLOSE_INDEX_VAL": 1001.0,
                "EOD_LOW_INDEX_VAL": 995.0, "HIT_TRADED_QTY": 1000,
                "EOD_TIMESTAMP": day.strftime("%d-%b-%Y").upper()}


def assert_standard_empty(df):
    assert df.empty
    assert list(df.columns) == list(SCHEMAS["index"])
    assert {c: str(t) for c, t in df.dtypes.items()} == SCHEMAS["index"]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to use the real network")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    indices._reset_index_name_cache()
    yield
    indices._reset_index_name_cache()


@pytest.fixture
def server(monkeypatch):
    fake = FakeNSE()
    monkeypatch.setattr(indices, "create_nse_session", lambda *a, **k: fake)
    return fake


def errors(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_request_budget_exhaustion_returns_no_data_and_stops_requesting(
        server, monkeypatch, caplog):
    monkeypatch.setattr(indices, "_max_requests", lambda total_days, first_chunk_days: 3)
    full = [FakeNSE.record("NIFTY 50", date(2026, 3, 2))] * indices._ROW_CAP   # always "capped"
    server.history_override = lambda params: FakeResponse({"data": full})

    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", START, END)

    assert_standard_empty(df)                                  # never partial history
    assert len(server.history_calls()) == 3                    # exactly the budget, no more
    assert any("History incomplete" in m and "request budget exhausted" in m
               for m in errors(caplog)), errors(caplog)


def test_failed_retry_under_the_long_name_returns_no_data(server, caplog):
    # The short name answers "no rows", so the connector looks up the long name
    # and retries; that retry is then rejected by NSE (HTTP 422).
    def answer(params):
        if params["indexType"].lower() == "nifty fin service":
            return FakeResponse({"data": []})
        return FakeResponse({"error": True, "message": "invalid params"}, status=422)

    server.history_override = answer
    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY FIN SERVICE", START, END)

    assert_standard_empty(df)
    asked = [p["indexType"] for p in server.history_calls()]
    assert asked[0] == "NIFTY FIN SERVICE"
    assert asked[-1] == "NIFTY FINANCIAL SERVICES"             # the alias was really tried
    assert server.names_calls() == 1
    assert any("History incomplete" in m and "NIFTY FINANCIAL SERVICES" in m and "422" in m
               for m in errors(caplog)), errors(caplog)


def test_a_payload_that_breaks_parsing_is_an_error_not_an_exception(server, caplog):
    broken = FakeNSE.record("NIFTY 50", date(2026, 3, 2))
    broken["EOD_OPEN_INDEX_VAL"] = 10 ** 400                  # valid JSON, cannot become a float
    assert json.loads(json.dumps(broken))["EOD_OPEN_INDEX_VAL"] == 10 ** 400
    server.history_override = lambda params: FakeResponse({"data": [broken]})

    with caplog.at_level(logging.INFO, logger=LOGGER):
        df = fetch_index_history("NIFTY 50", START, END)       # must not raise

    assert_standard_empty(df)
    assert any("could not parse records" in m and "OverflowError" in m
               for m in errors(caplog)), errors(caplog)
    assert len(server.history_calls()) == 1                    # an error chunk is not retried or split
