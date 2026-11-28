"""Offline edge-case tests for the cache layer (reliability/cache.py).

Covers paths the other cache test files leave out: per-platform default
folders, refusal of malformed snapshots, identity checks on numeric columns,
key stability, numpy/NaN encoding, and date/datetime arguments.
Nothing here touches the network or the real clock.
"""
import logging
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyequity.reliability import cache as c
from pyequity.schema import SCHEMAS, empty_frame

NOW0 = pd.Timestamp("2025-06-10 12:00", tz="Asia/Kolkata").timestamp()


def equity_frame(days, symbol="RELIANCE", source="NSE"):
    n = len(days)
    return pd.DataFrame({
        "date": pd.to_datetime(days), "open": [100.5 + i for i in range(n)],
        "high": [101.5 + i for i in range(n)], "low": [99.5 + i for i in range(n)],
        "close": [100.25 + i for i in range(n)],
        "volume": np.arange(n, dtype="int64") + 10,
        "trades": np.arange(n, dtype="int64") + 1,
        "symbol": symbol, "source": source,
    })[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])


def nav_frame(code=119551):
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-05-02"]), "scheme_code": [code],
        "scheme_name": ["Some Fund"], "nav": [12.5], "source": ["AMFI"],
    })[list(SCHEMAS["nav"])].astype(SCHEMAS["nav"])


# ---- 1. default folder per platform -----------------------------------
@pytest.mark.parametrize("platform,env,expected", [
    ("win32", {"LOCALAPPDATA": "C:/LocalData"},
     Path("C:/LocalData") / "pyequity" / "cache"),
    ("win32", {}, Path("HOME") / "AppData" / "Local" / "pyequity" / "cache"),
    ("darwin", {}, Path("HOME") / "Library" / "Caches" / "pyequity"),
    ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg") / "pyequity"),
    ("linux", {}, Path("HOME") / ".cache" / "pyequity"),
], ids=["windows-localappdata", "windows-fallback", "macos", "linux-xdg",
        "linux-default"])
def test_default_cache_dir_per_platform(monkeypatch, tmp_path, platform, env, expected):
    for name in (c.ENV_DIR, "LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(c.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # "HOME" in the expected path stands for the faked home folder
    if expected.parts[0] == "HOME":
        expected = tmp_path.joinpath(*expected.parts[1:])
    assert c.default_cache_dir() == expected


def test_cache_dir_env_override_beats_every_platform(monkeypatch, tmp_path):
    monkeypatch.setenv(c.ENV_DIR, str(tmp_path / "custom"))
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(c.sys, "platform", platform)
        assert c.default_cache_dir() == tmp_path / "custom"


# ---- 2. malformed snapshot is refused ---------------------------------
def test_malformed_snapshot_refused_logged_and_refetched(tmp_path, caplog):
    dc = c.DiskCache(tmp_path, clock=lambda: NOW0)
    calls = []

    def fetch():
        calls.append(1)
        return equity_frame(["2025-05-02"]).drop(columns=["trades"])

    ident = {"source": "NSE", "symbol": "RELIANCE"}
    with caplog.at_level(logging.WARNING, logger=c.logger.name):
        first = dc.get_snapshot("equity", ident, fetch)
    assert "trades" not in first.columns          # handed back unchanged
    assert any("does not match the schema" in r.getMessage() for r in caplog.records)
    assert dc.stats()["snapshots"] == 0    # nothing stored
    dc.get_snapshot("equity", ident, fetch)
    assert len(calls) == 2                        # the second call fetched again


def test_store_refuses_frame_missing_a_schema_column(tmp_path, caplog):
    store = c.CacheStore(tmp_path)
    bad = equity_frame(["2025-05-02"]).drop(columns=["close"])
    with caplog.at_level(logging.WARNING, logger=c.logger.name):
        stored = store.put_snapshot("equity", "k", bad, now=NOW0, ttl=60.0)
    assert stored is False
    assert any("not storing snapshot" in r.getMessage() for r in caplog.records)
    assert store.get_snapshot("equity", "k", NOW0) is None


# ---- 3. numeric identity filter with non-numeric value ----------------
def test_numeric_identity_with_non_numeric_value_is_rejected_without_crash():
    frame = nav_frame()
    assert c._identity_matches(frame, "nav", {"source": "AMFI", "scheme_code": 119551})
    assert c._identity_matches(frame, "nav", {"scheme_code": "not-a-number"}) is False


def test_snapshot_with_unparseable_numeric_identity_not_cached(tmp_path, caplog):
    dc = c.DiskCache(tmp_path, clock=lambda: NOW0)
    ident = {"source": "AMFI", "scheme_code": "abc"}
    with caplog.at_level(logging.WARNING, logger=c.logger.name):
        out = dc.get_snapshot("nav", ident, nav_frame)
    assert len(out) == 1                          # rows still returned
    assert any("do not match identity" in r.getMessage() for r in caplog.records)
    assert dc.stats()["snapshots"] == 0    # but not cached


# ---- 4. key stability --------------------------------------------------
def test_key_treats_nan_and_none_alike_and_ignores_case_and_padding():
    base = c.make_key({"source": "NSE", "symbol": "RELIANCE", "series": None})
    assert c.make_key({"source": "nse", "symbol": " reliance ", "series": float("nan")}) == base
    assert c.make_key({"source": "NSE", "symbol": "RELIANCE", "series": np.nan}) == base
    assert c.make_key({"series": None, "symbol": "RELIANCE", "source": "NSE"}) == base
    assert c.make_key({"source": "NSE", "symbol": "RELIANCE", "series": "EQ"}) != base


# ---- 5. numpy values and NaN survive a store/load round trip ----------
def test_numpy_values_and_nan_round_trip_through_the_store(tmp_path):
    frame = equity_frame(["2025-05-02", "2025-05-05", "2025-05-06"])
    frame.loc[1, "open"] = np.nan
    frame.loc[2, "close"] = np.float64("nan")
    assert isinstance(frame.loc[0, "volume"], (np.integer, int))
    store = c.CacheStore(tmp_path)
    assert store.put_snapshot("equity", "k", frame, now=NOW0, ttl=60.0) is True
    back = store.get_snapshot("equity", "k", NOW0)
    pd.testing.assert_frame_equal(back, frame, check_exact=True)
    assert math.isnan(back.loc[1, "open"]) and math.isnan(back.loc[2, "close"])
    assert back.dtypes.to_dict() == {k: np.dtype(v) for k, v in SCHEMAS["equity"].items()}


# ---- 6. date / datetime arguments equal string arguments --------------
def test_date_and_datetime_arguments_share_coverage_with_strings(tmp_path):
    dc = c.DiskCache(tmp_path, clock=lambda: NOW0)
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        days = pd.date_range(a, b, freq="B")
        return equity_frame(list(days)) if len(days) else empty_frame("equity")

    ident = {"source": "NSE", "symbol": "RELIANCE"}
    first = dc.get_series("equity", ident, date(2025, 5, 1), date(2025, 5, 9), fetch)
    assert calls == [("2025-05-01", "2025-05-09")]
    by_str = dc.get_series("equity", ident, "2025-05-01", "2025-05-09", fetch)
    by_dt = dc.get_series("equity", ident, datetime(2025, 5, 1, 15, 30),
                          datetime(2025, 5, 9, 9, 0), fetch)
    assert len(calls) == 1                        # both answered from the cache
    pd.testing.assert_frame_equal(first, by_str)
    pd.testing.assert_frame_equal(first, by_dt)
    assert not first.empty
