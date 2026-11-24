
"""Public API for historical equity prices with NSE-to-BSE fallback."""

import warnings

import pandas as pd

from pyequity.api._args import DateLike, check_range
from pyequity.api._errors import (
    DataQualityWarning,
    DataUnavailableError,
    EmptyResultWarning,
    FallbackWarning,
    InvalidArgumentError,
)
from pyequity.connectors import nse as _nse
from pyequity.connectors import bse as _bse
from pyequity.processing.normalize import normalize, validate
from pyequity.reliability.fallback import run_with_fallback
from pyequity.schema import empty_frame
from pyequity.reliability.cache import get_default_cache
from pyequity.reliability.outcome import FetchOutcome, Status

class _CachedFetchFailure(Exception):
    """Preserve a failed connector outcome through the cache callback."""

    def __init__(self, outcome: FetchOutcome):
        self.outcome = outcome
        super().__init__(outcome.reason)


def _fetch_cached(
    source: str,
    symbol: str,
    start: str,
    end: str,
    fetch,
    *,
    series: str | None = None,
) -> FetchOutcome:
    """Fetch complete equity history through the source-specific range cache."""
    identity = {
        "source": source,
        "symbol": symbol,
    }
    if series is not None:
        identity["series"] = series

    def fetch_range(from_date: str, to_date: str) -> pd.DataFrame:
        outcome = fetch(from_date, to_date)

        if outcome.failed:
            raise _CachedFetchFailure(outcome)

        if outcome.empty:
            return empty_frame("equity")

        if not isinstance(outcome.data, pd.DataFrame) or outcome.data.empty:
            raise _CachedFetchFailure(
                FetchOutcome(
                    Status.FAILED,
                    reason=f"{source} returned an invalid or incomplete frame",
                    attempts=outcome.attempts,
                )
            )

        frame = outcome.data.copy()
        frame["symbol"] = symbol
        frame["source"] = source
        return frame

    try:
        result = get_default_cache().get_series(
            "equity",
            identity,
            start,
            end,
            fetch_range,
        )
    except _CachedFetchFailure as exc:
        return exc.outcome
    except Exception as exc:
        return FetchOutcome(
            Status.FAILED,
            reason=f"{source} cache/fetch operation failed: {exc}",
        )

    if result.empty:
        return FetchOutcome(Status.EMPTY, empty_frame("equity"))

    return FetchOutcome(Status.OK, result)

def get_history(
    symbol: str,
    start: DateLike,
    end: DateLike,
    *,
    source: str = "auto",
    series: str = "EQ",
) -> pd.DataFrame:
    """Get daily equity history from NSE, BSE, or automatic fallback.

    Args:
        symbol: NSE equity symbol, such as "RELIANCE".
        start: Start date as an ISO string, date, or datetime.
        end: End date as an ISO string, date, or datetime.
        source: Data source: "auto", "nse", or "bse". Auto tries NSE first
            and uses BSE only if NSE fails.
        series: NSE equity series, usually "EQ". Used only for NSE requests.

    Returns:
        A normalized DataFrame containing historical equity prices.
        The DataFrame attrs include source, fallback_used, and
        fallback_reason metadata.

    Raises:
        InvalidArgumentError: If the symbol, dates, source, or series are invalid.
        DataUnavailableError: If the requested data cannot be fetched or
            the response is invalid.

    A valid empty response does not trigger fallback.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise InvalidArgumentError("symbol must be a non-empty string")

    sym = symbol.strip().upper()
    start_s, end_s = check_range(start, end)

    if not isinstance(source, str) or source.strip().lower() not in {
        "auto", "nse", "bse"
    }:
        raise InvalidArgumentError("source must be 'auto', 'nse', or 'bse'")

    source = source.strip().lower()
    if not isinstance(series, str) or not series.strip():
        raise InvalidArgumentError("series must be a non-empty string")

    nse_series = series.strip().upper()

    def fetch_nse(from_date=start_s, to_date=end_s):
        return _fetch_cached(
            "NSE",
            sym,
            start_s,
            end_s,
            lambda a, b: _nse.fetch_nse_historical_outcome(
                sym, a, b, nse_series
            ),
            series=nse_series,
        )

    def fetch_bse(from_date=start_s, to_date=end_s):
        return _fetch_cached(
            "BSE",
            sym,
            start_s,
            end_s,
            lambda a, b: _bse.fetch_bse_historical_by_symbol_outcome(
                sym, a, b
            ),
        )

    if source == "nse":
        result = fetch_nse()
        used_source = "NSE"
        fallback_used = False
    elif source == "bse":
        result = fetch_bse()
        used_source = "BSE"
        fallback_used = False
    else:
        combined = run_with_fallback(fetch_nse, fetch_bse)
        result = combined.outcome
        used_source = combined.source
        fallback_used = combined.fallback_used

    if result.failed:
        raise DataUnavailableError(
            f"Could not fetch {sym} history for {start_s}..{end_s}: "
            f"{result.reason}"
        )

    if result.empty:
        warnings.warn(
            f"{used_source} returned no history for {sym} "
            f"between {start_s} and {end_s}.",
            EmptyResultWarning,
            stacklevel=2,
        )

        if fallback_used:
            warnings.warn(
                f"NSE failed for {sym}; BSE was tried but returned no history.",
                FallbackWarning,
                stacklevel=2,
            )

        out = empty_frame("equity")
        out.attrs["source"] = used_source
        out.attrs["fallback_used"] = fallback_used
        out.attrs["fallback_reason"] = (
            combined.primary_reason
            if source == "auto" and fallback_used
            else ""
        )
        return out

    df = result.data
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataUnavailableError(
            f"{used_source} returned an invalid or incomplete response for {sym}"
        )

    df = df.copy()
    df["symbol"] = sym
    df["source"] = used_source
    out = normalize(df, "equity")

    if out.empty:
        raise DataUnavailableError(
            f"{used_source} response for {sym} did not match the equity schema"
        )

    report = validate(out, "equity", start=start_s, end=end_s)
    if report.errors:
        warnings.warn(
            f"{used_source} data failed integrity checks: {report.summary()}",
            DataQualityWarning,
            stacklevel=2,
        )

    if fallback_used:
        warnings.warn(
            f"NSE failed for {sym}; history was supplied by BSE instead.",
            FallbackWarning,
            stacklevel=2,
        )

    out.attrs["source"] = used_source
    out.attrs["fallback_used"] = fallback_used
    out.attrs["fallback_reason"] = (
        combined.primary_reason if source == "auto" and fallback_used else ""
    )
    return out
