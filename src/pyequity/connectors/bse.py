"""
BSE equity historical (OHLCV) connector.

BSE returns at most 31 rows per request, so long ranges are fetched in 30-day
chunks (about 22 trading days each) and stitched together.

Policy: if any chunk still fails after retries, or a chunk comes back at the
row cap, the result cannot be guaranteed complete. The function then returns
an empty frame and logs an error. It never returns partial data and never
raises. An empty answer from BSE (for example an unknown scripcode) is not
retried, because retrying cannot change it.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd
import requests

from pyequity.connectors.symbol_map import get_bse_code
from pyequity.schema import SCHEMAS, empty_frame
from pyequity.reliability.outcome import FetchOutcome, Status
from pyequity.reliability.rate_limiter import RateLimiter
from pyequity.reliability.retry import retry_call
from pyequity.reliability.session import SessionManager

logger = logging.getLogger(__name__)

BSE_ROW_CAP = 31
_CHUNK_DAYS = 30
_ATTEMPTS = 3
_RETRY_DELAY = 2.0   # seconds, multiplied by the attempt number
_CHUNK_DELAY = 1.0   # seconds between chunks, to be gentle on BSE

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Mobile Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin":          "https://www.bseindia.com",
    "Referer":         "https://www.bseindia.com/",
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-site",
}

def _create_bse_session(_key=None):
    """Create a reusable BSE HTTP session."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


_BSE_SESSIONS = SessionManager(
    lambda key: _create_bse_session(key),
    sleep=time.sleep,
)

_BSE_RATE_LIMITER = RateLimiter(
    min_interval=1.0,
    sleep=time.sleep,
)

def _clean_number(value):
    """BSE sends numbers as strings with thousands separators."""
    if isinstance(value, str):
        return value.replace(",", "")
    return value


def _request_chunk_once(
    session: requests.Session,
    scripcode: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Perform one BSE HTTP request. Exceptions are handled by retry_call."""
    fmt_from = datetime.strptime(from_date, "%Y-%m-%d").strftime("%d/%m/%Y")
    fmt_to = datetime.strptime(to_date, "%Y-%m-%d").strftime("%d/%m/%Y")

    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/StockpricesearchData/w"
        f"?MonthDate={fmt_from}"
        f"&YearDate={fmt_to}"
        f"&pageType=0"
        f"&Scode={scripcode}"
        f"&Seg=C"
        f"&rbType=D"
        f"&SortOrder=true"
    )

    response = session.get(url, timeout=30)

    if response.status_code != 200:
        raise requests.HTTPError(
            f"BSE returned HTTP {response.status_code} for {scripcode}",
            response=response,
        )

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("BSE returned an invalid JSON payload")

    records = data.get("StockData")
    if not records:
        logger.warning("[BSE] No records found for scripcode %s", scripcode)
        return empty_frame("equity")

    df = pd.DataFrame(records)
    df = df[
        [
            "RealDate", "qe_open", "qe_high", "qe_low", "qe_close",
            "no_of_shrs", "no_trades",
        ]
    ].copy()

    df = df.rename(
        columns={
            "RealDate": "date",
            "qe_open": "open",
            "qe_high": "high",
            "qe_low": "low",
            "qe_close": "close",
            "no_of_shrs": "volume",
            "no_trades": "trades",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].apply(_clean_number).astype("float64")

    for col in ("volume", "trades"):
        df[col] = (
            df[col]
            .apply(_clean_number)
            .astype("float64")
            .astype("int64")
        )

    df["symbol"] = data.get("scrip_id") or scripcode
    df["source"] = "BSE"

    return df.sort_values("date").reset_index(drop=True)


def _request_chunk(
    scripcode: str,
    from_date: str,
    to_date: str,
) -> Tuple[pd.DataFrame, str]:
    """Compatibility helper: make one request without retrying."""
    session_outcome = _BSE_SESSIONS.get("bse")

    if not session_outcome.ok:
        logger.error("[BSE] Session creation failed: %s", session_outcome.reason)
        return empty_frame("equity"), "error"

    try:
        _BSE_RATE_LIMITER.wait("api.bseindia.com")
        df = _request_chunk_once(
            session_outcome.data, scripcode, from_date, to_date
        )
        return df, "empty" if df.empty else "ok"
    except Exception as exc:
        logger.error("[BSE] Request failed for %s: %s", scripcode, exc)
        return empty_frame("equity"), "error"


def _chunk_with_retry(
    scripcode: str,
    from_date: str,
    to_date: str,
    *,
    return_attempts: bool = False,
) -> Tuple[pd.DataFrame, str] | Tuple[pd.DataFrame, str, int]:
    """Fetch one chunk with shared retry and session management."""
    session_outcome = _BSE_SESSIONS.get("bse")

    if not session_outcome.ok:
        result = (empty_frame("equity"), "error", session_outcome.attempts)
        return result if return_attempts else result[:2]

    session = session_outcome.data

    outcome = retry_call(
        lambda: (
            _BSE_RATE_LIMITER.wait("api.bseindia.com"),
            _request_chunk_once(session, scripcode, from_date, to_date),
        )[1],
        max_attempts=_ATTEMPTS,
        base_delay=_RETRY_DELAY,
        label=f"BSE {scripcode} {from_date}..{to_date}",
        is_empty=lambda df: df.empty,
        sleep=time.sleep,
    )

    if outcome.ok:
        result = (outcome.data, "ok", outcome.attempts)
    elif outcome.empty:
        result = (empty_frame("equity"), "empty", outcome.attempts)
    else:
        _BSE_SESSIONS.invalidate("bse")
        logger.error(
            "[BSE] Request failed for %s %s..%s: %s",
            scripcode, from_date, to_date, outcome.reason,
        )
        result = (empty_frame("equity"), "error", outcome.attempts)

    return result if return_attempts else result[:2]


def _fetch_bse_chunk(scripcode: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Internal: fetch one chunk and return just the DataFrame.

    Kept for tools and tests that call it directly. Returns an empty frame on
    any failure.
    """
    return _request_chunk(scripcode, from_date, to_date)[0]


def fetch_bse_historical(scripcode: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a stock from BSE.

    Long ranges are split into 30-day chunks that stay under BSE's 31-row cap.
    Failed requests are retried. If a chunk still fails, or reaches the cap,
    an empty frame is returned and an error is logged, so the result is never
    silently incomplete.

    Args:
        scripcode : BSE stock code e.g. "500325" for Reliance
        from_date : Start date in YYYY-MM-DD format
        to_date   : End date   in YYYY-MM-DD format

    Returns:
        pandas DataFrame in the standard equity schema:
        date, open, high, low, close, volume, trades, symbol, source.
        Sorted oldest to newest. Empty frame on any failure.
    """
    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        logger.error("[BSE] Invalid date format. Use YYYY-MM-DD")
        return empty_frame("equity")

    if start > end:
        logger.error("[BSE] from_date cannot be after to_date")
        return empty_frame("equity")

    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
        label = f"{chunk_start.date()} to {chunk_end.date()}"

        chunk_df, status = _chunk_with_retry(
            scripcode,
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        if status == "error":
            logger.error(f"[BSE] {scripcode}: chunk {label} failed after {_ATTEMPTS} "
                         f"attempts; returning no data instead of a partial result.")
            return empty_frame("equity")

        if len(chunk_df) >= BSE_ROW_CAP:
            logger.error(f"[BSE] {scripcode}: chunk {label} returned {len(chunk_df)} rows, "
                         f"at BSE's {BSE_ROW_CAP}-row cap, so it may be incomplete; "
                         f"returning no data.")
            return empty_frame("equity")

        if status == "ok":
            chunks.append(chunk_df)

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end:
            time.sleep(_CHUNK_DELAY)

    if not chunks:
        return empty_frame("equity")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])


def fetch_bse_historical_by_symbol(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch BSE historical data using an NSE symbol instead of a BSE scripcode.

    Args:
        symbol    : NSE stock symbol, e.g. "RELIANCE"
        from_date : Start date in YYYY-MM-DD format
        to_date   : End date   in YYYY-MM-DD format

    Returns:
        Same DataFrame as fetch_bse_historical(). If the symbol has no BSE
        mapping, an error is logged and an empty frame is returned; nothing
        is guessed and nothing raises.
    """
    try:
        scripcode = get_bse_code(symbol)
    except ValueError as e:
        logger.error(f"[BSE] {e}")
        return empty_frame("equity")
    return fetch_bse_historical(scripcode, from_date, to_date)


def fetch_bse_historical_by_symbol_outcome(
    symbol: str, from_date: str, to_date: str
) -> FetchOutcome:
    """Fetch BSE history by NSE symbol

    Returns:
        FetchOutcome containing the equity DataFrame, status, reason, and
        actual request-attempt count.
    """

    try:
        scripcode = get_bse_code(symbol)
    except (ValueError, TypeError) as exc:
        return FetchOutcome(Status.FAILED, reason=f"BSE mapping unavailable: {exc}")

    try:
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        return FetchOutcome(Status.FAILED, reason=f"Invalid dates: {exc}")

    if start > end:
        return FetchOutcome(Status.FAILED, reason="Start date is after end date")

    chunks = []
    saw_empty = False
    attempts = 0
    chunk_start = start

    try:
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
            label = f"{chunk_start.date()}..{chunk_end.date()}"

            df, status, chunk_attempts = _chunk_with_retry(
                scripcode,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                return_attempts=True,
            )
            attempts += chunk_attempts

            if status == "error":
                return FetchOutcome(
                    Status.FAILED, reason=f"BSE request failed for {label}",
                    attempts=attempts,
                )

            if len(df) >= BSE_ROW_CAP:
                return FetchOutcome(
                    Status.FAILED, reason=f"BSE row cap reached for {label}",
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
            reason=f"BSE unexpected fetch error: {exc}",
            attempts=attempts,
        )

    if not chunks:
        return FetchOutcome(
            Status.EMPTY if saw_empty else Status.FAILED,
            empty_frame("equity"),
            "" if saw_empty else "BSE returned no usable chunk result",
            attempts,
        )

    try:
        result = pd.concat(chunks, ignore_index=True)
        result = result.drop_duplicates(subset=["date"])
        result["symbol"] = str(symbol).strip().upper()
        result["source"] = "BSE"
        result = result.sort_values("date").reset_index(drop=True)
        result = result[list(SCHEMAS["equity"])].astype(SCHEMAS["equity"])
        return FetchOutcome(Status.OK, result, attempts=attempts)
    except Exception as exc:
        return FetchOutcome(
            Status.FAILED,
            reason=f"BSE response processing failed: {exc}",
            attempts=attempts,
        )
