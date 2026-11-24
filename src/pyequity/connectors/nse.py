"""
NSE equity historical (OHLCV) connector.

NSE returns at most about 70 rows per request, so long ranges are fetched in
60-day chunks (about 43 trading days each) and stitched together.

Policy: if any chunk still fails after retries, or a chunk comes back at the
row cap, the result cannot be guaranteed complete. The function then returns
an empty frame and logs an error. It never returns partial data and never
raises.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

from pyequity.connectors._session_utils import create_nse_session
from pyequity.schema import SCHEMAS, empty_frame
from pyequity.reliability.outcome import FetchOutcome, Status

logger = logging.getLogger(__name__)

_URL = "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
NSE_ROW_CAP = 70
_CHUNK_DAYS = 60
_ATTEMPTS = 3
_RETRY_DELAY = 2.0   # seconds, multiplied by the attempt number
_CHUNK_DELAY = 1.0   # seconds between chunks, to be gentle on NSE

_RENAME = {
    "mtimestamp":       "date",
    "chOpeningPrice":   "open",
    "chTradeHighPrice": "high",
    "chTradeLowPrice":  "low",
    "chClosingPrice":   "close",
    "chTotTradedQty":   "volume",
    "chTotalTrades":    "trades",
}


def _request_chunk(session, symbol: str, from_date: str, to_date: str,
                   series: str) -> Tuple[pd.DataFrame, str]:
    """
    Internal: one request to NSE.

    Args:
        session: Warmed-up NSE session.
        symbol: NSE symbol.
        from_date: DD-MM-YYYY.
        to_date: DD-MM-YYYY.
        series: NSE series, e.g. "EQ".

    Returns:
        (DataFrame, status). status is "ok", "empty" (NSE answered with no
        rows) or "error" (the request or its parsing failed). Keeping "empty"
        and "error" apart lets callers retry real failures only.
    """
    params = {
        "functionName": "getHistoricalTradeData",
        "symbol": symbol,
        "series": series,
        "fromDate": from_date,
        "toDate": to_date,
    }
    try:
        response = session.get(_URL, params=params, timeout=15)

        if response.status_code != 200:
            logger.error(f"[NSE] HTTP {response.status_code} for {symbol} "
                         f"{from_date}..{to_date}")
            return empty_frame("equity"), "error"

        text = response.text.strip()
        if not text or text == "null" or text == "[]":
            return empty_frame("equity"), "empty"

        data = json.loads(text)
        if not data:
            return empty_frame("equity"), "empty"

        df = pd.DataFrame(data).rename(columns=_RENAME)
        df = df[["date", "open", "high", "low", "close", "volume", "trades"]].copy()

        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y")
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype("float64")
        for col in ("volume", "trades"):
            df[col] = df[col].astype("int64")

        return df, "ok"

    except Exception as e:
        logger.error(f"[NSE] Chunk fetch error for {symbol} {from_date}..{to_date}: {e}")
        return empty_frame("equity"), "error"


def _chunk_with_retry(
    session,
    symbol: str,
    from_date: str,
    to_date: str,
    series: str,
    *,
    return_attempts: bool = False,
) -> Tuple[pd.DataFrame, str] | Tuple[pd.DataFrame, str, int]:
    """Retry real failures; optionally return the actual attempt count."""
    attempts = 1
    df, status = _request_chunk(session, symbol, from_date, to_date, series)

    for attempt in range(1, _ATTEMPTS):
        if status != "error":
            break
        time.sleep(_RETRY_DELAY * attempt)
        df, status = _request_chunk(
            session, symbol, from_date, to_date, series
        )
        attempts += 1

    if return_attempts:
        return df, status, attempts

    return df, status


def _fetch_nse_chunk(session, symbol, from_date, to_date, series="EQ"):
    """
    Internal: fetch one chunk and return just the DataFrame.

    Kept for tools and tests that call it directly. from_date and to_date
    must be DD-MM-YYYY. Returns an empty frame on any failure.
    """
    return _request_chunk(session, symbol, from_date, to_date, series)[0]


def fetch_nse_historical(symbol: str, from_date: str, to_date: str,
                         series: str = "EQ") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from NSE.

    Long ranges are split into chunks that stay under NSE's row cap. Failed
    requests are retried. If a chunk still fails, or reaches the cap, an empty
    frame is returned and an error is logged, so the result is never silently
    incomplete.

    Args:
        symbol    : NSE stock symbol e.g. "RELIANCE", "TCS"
        from_date : Start date in YYYY-MM-DD format e.g. "2025-09-23"
        to_date   : End date   in YYYY-MM-DD format e.g. "2026-09-23"
        series    : NSE series, default "EQ" (regular equity)

    Returns:
        pandas DataFrame in the standard equity schema:
        date, open, high, low, close, volume, trades, symbol, source.
        Sorted oldest to newest. Empty frame on any failure.
    """
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.error("[NSE] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("equity")

    if start > end:
        logger.error("[NSE] from_date cannot be after to_date")
        return empty_frame("equity")

    symbol = str(symbol).strip() if symbol is not None else ""
    if not symbol:
        logger.error("[NSE] symbol is required")
        return empty_frame("equity")

    logger.info(f"[NSE] Fetching {symbol} from {from_date} to {to_date}...")
    try:
        session = create_nse_session(symbol)
    except Exception as e:
        logger.error(f"[NSE] Could not create NSE session: {e}")
        return empty_frame("equity")

    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
        fmt_from = chunk_start.strftime("%d-%m-%Y")
        fmt_to = chunk_end.strftime("%d-%m-%Y")

        logger.info(f"[NSE] Fetching chunk: {fmt_from} to {fmt_to}...")
        chunk_df, status = _chunk_with_retry(session, symbol, fmt_from, fmt_to, series)

        if status == "error":
            logger.error(f"[NSE] {symbol}: chunk {fmt_from} to {fmt_to} failed after "
                         f"{_ATTEMPTS} attempts; returning no data instead of a partial result.")
            return empty_frame("equity")

        if len(chunk_df) >= NSE_ROW_CAP:
            logger.error(f"[NSE] {symbol}: chunk {fmt_from} to {fmt_to} returned "
                         f"{len(chunk_df)} rows, at NSE's {NSE_ROW_CAP}-row cap, so it may "
                         f"be incomplete; returning no data.")
            return empty_frame("equity")

        if status == "ok":
            chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(_CHUNK_DELAY)

    if not chunks:
        logger.warning(f"[NSE] No data returned for {symbol}")
        return empty_frame("equity")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"])
    df["symbol"] = symbol.upper()
    df["source"] = "NSE"
    df = df.sort_values("date").reset_index(drop=True)
    df = df[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])

    logger.info(f"[NSE] ✅ {symbol} — {len(df)} rows fetched")
    return df

def fetch_nse_historical_outcome(
    symbol: str, from_date: str, to_date: str, series: str = "EQ"
) -> FetchOutcome:
    """Fetch NSE history while preserving EMPTY versus FAILED outcomes.

    Returns:
        FetchOutcome containing the equity DataFrame, status, reason, and
        actual request-attempt count.
    """

    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        return FetchOutcome(Status.FAILED, reason=f"Invalid dates: {exc}")

    symbol = str(symbol).strip().upper() if symbol is not None else ""
    if not symbol or start > end:
        return FetchOutcome(Status.FAILED, reason="Invalid symbol or date range")

    try:
        session = create_nse_session(symbol)
    except Exception as exc:
        return FetchOutcome(Status.FAILED, reason=f"Session creation failed: {exc}")

    chunks = []
    saw_empty = False
    attempts = 0
    chunk_start = start

    try:
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
            fmt_from = chunk_start.strftime("%d-%m-%Y")
            fmt_to = chunk_end.strftime("%d-%m-%Y")

            df, status, chunk_attempts = _chunk_with_retry(
                session, symbol, fmt_from, fmt_to, series,
                return_attempts=True,
            )
            attempts += chunk_attempts

            if status == "error":
                return FetchOutcome(
                    Status.FAILED, reason=f"NSE request failed for {fmt_from}..{fmt_to}",
                    attempts=attempts,
                )

            if len(df) >= NSE_ROW_CAP:
                return FetchOutcome(
                    Status.FAILED,
                    reason=f"NSE row cap reached for {fmt_from}..{fmt_to}",
                    attempts=attempts,
                )

            if status == "ok":
                chunks.append(df)
            else:
                saw_empty = True

            chunk_start = chunk_end + timedelta(days=1)
            if chunk_start <= end:
                time.sleep(_CHUNK_DELAY)

    except Exception as exc:
        return FetchOutcome(
            Status.FAILED,
            reason=f"NSE unexpected fetch error: {exc}",
            attempts=attempts,
        )

    if not chunks:
        return FetchOutcome(
            Status.EMPTY if saw_empty else Status.FAILED,
            empty_frame("equity"),
            "" if saw_empty else "NSE returned no usable chunk result",
            attempts,
        )

    try:
        result = pd.concat(chunks, ignore_index=True)
        result = result.drop_duplicates(subset=["date"])
        result["symbol"] = symbol
        result["source"] = "NSE"
        result = result.sort_values("date").reset_index(drop=True)
        result = result[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])
        return FetchOutcome(Status.OK, result, attempts=attempts)
    except Exception as exc:
        return FetchOutcome(
            Status.FAILED,
            reason=f"NSE response processing failed: {exc}",
            attempts=attempts,
        )
