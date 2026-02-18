"""
Data module handles all external data ingestion for the project.

1) Download OHLCV data from Yahoo Finance (primary source)
2) If Yahoo fails, fallback to Stooq
3) Normalize raw data into a consistent OHLCV schema
4) Merge target asset with exogenous proxies into a single DataFrame

- Fail-fast (strict) when required data cannot be retrieved
- Optional fallback to secondary source (Stooq)
- Clear, explicit reporting of which source was used (no "mystery success")
- Keep ingestion separate from feature engineering
- yfinance may print warnings/errors internally even when we handle fallback.
"""

from __future__ import annotations

import io
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


# ==========================================================
# Diagnostics / errors
# ==========================================================

@dataclass
class TickerIngestionReport:
    """
    Tracks ingestion attempts for one ticker.

    We attach these reports to the final DataFrame via df.attrs["ingestion_report"].
    """
    ticker: str
    start: str
    end: str
    yahoo_ok: bool = False
    stooq_ok: bool = False
    source_used: str = "none"
    yahoo_error: str | None = None
    stooq_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "start": self.start,
            "end": self.end,
            "yahoo_ok": self.yahoo_ok,
            "stooq_ok": self.stooq_ok,
            "source_used": self.source_used,
            "yahoo_error": self.yahoo_error,
            "stooq_error": self.stooq_error,
        }


class DataDownloadError(RuntimeError):
    """Raised when data cannot be downloaded from all allowed sources."""


# ==========================================================
# Internal normalization helpers
# ==========================================================

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw OHLCV DataFrame to a consistent schema.

    Standard output:
    - DatetimeIndex
    - Columns: open, high, low, close, volume
    """
    df = df.copy()

    # Normalize column names
    df = df.rename(columns=lambda c: str(c).strip().lower())

    # If "date" is present as column, move to index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")

    # Ensure datetime index
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Drop rows with invalid timestamps
    df = df[~df.index.isna()].copy()

    # Guarantee required OHLCV columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure consistent order
    df = df[["open", "high", "low", "close", "volume"]].copy()

    return df


def _is_usable_ohlcv(df: pd.DataFrame) -> bool:
    """
    Basic sanity check:
    - not empty
    - has a non-null close series with at least some values
    """
    if df is None or df.empty:
        return False

    if "close" not in df.columns:
        return False

    close_non_null = df["close"].dropna()
    return len(close_non_null) >= 10


# ==========================================================
# Data source: Yahoo Finance
# ==========================================================

def download_yahoo(
    ticker: str,
    start: str,
    end: str,
    *,
    retries: int = 2,
    sleep_s: float = 1.0,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance using yfinance.
    """
    last_err: str | None = None

    for attempt in range(1, retries + 1):
        try:
            buf_out = io.StringIO()
            buf_err = io.StringIO()

            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                t = yf.Ticker(ticker)
                df = t.history(start=start, end=end, auto_adjust=False)

            yahoo_noise = (buf_out.getvalue() + "\n" + buf_err.getvalue()).strip()
            if yahoo_noise:
                logger.warning("yfinance noise for %s: %s", ticker, yahoo_noise[:800])

            if df is None or df.empty:
                last_err = "Ticker.history() returned empty dataframe"
            else:
                # history() returns columns like Open/High/Low/Close/Volume
                df = df.reset_index()

                # Some environments return Datetime in 'Date' column, some in 'Datetime'
                # _normalize_ohlcv will handle both if we rename it to 'date' when present
                if "Datetime" in df.columns and "Date" not in df.columns:
                    df = df.rename(columns={"Datetime": "Date"})

                df_norm = _normalize_ohlcv(df)
                if _is_usable_ohlcv(df_norm):
                    return df_norm

                last_err = "Ticker.history() returned dataframe but OHLCV looks unusable"

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < retries:
            time.sleep(sleep_s)

    logger.warning("Yahoo download failed for %s (%s -> %s): %s", ticker, start, end, last_err)
    return pd.DataFrame()


# ==========================================================
# Data source: Stooq (fallback)
# ==========================================================

def download_stooq_us(ticker: str) -> pd.DataFrame:
    """
    Download OHLCV data from Stooq as fallback.

    Stooq requires lowercase ticker with '.us' suffix.
    """
    t = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={t}&i=d"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        logger.warning("Stooq download failed for %s: %s", ticker, f"{type(e).__name__}: {e}")
        return pd.DataFrame()

    df_norm = _normalize_ohlcv(df)

    # Stooq typically returns oldest->newest, but we want newest->oldest 
    # for consistency with Yahoo
    return df_norm


# ==========================================================
# Source orchestrator
# ==========================================================

def download_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    allow_fallback: bool = True,
    strict: bool = True,
    yahoo_retries: int = 2,
    yahoo_sleep_s: float = 1.0,
) -> tuple[pd.DataFrame, TickerIngestionReport]:
    """
    Attempt to download OHLCV data using multiple sources.

    Strategy:
    1) Try Yahoo Finance (retries)
    2) If Yahoo fails and allow_fallback=True, fallback to Stooq
    3) Restrict Stooq data to requested date range

    Returns:
    - df: normalized OHLCV
    - report: ingestion diagnostics
    """
    report = TickerIngestionReport(ticker=ticker, start=start, end=end)

    # 1) Yahoo
    df_yahoo = download_yahoo(ticker, start, end, retries=yahoo_retries, sleep_s=yahoo_sleep_s)
    if _is_usable_ohlcv(df_yahoo):
        report.yahoo_ok = True
        report.source_used = "yahoo"
        return df_yahoo, report

    report.yahoo_ok = False
    report.yahoo_error = "Yahoo returned empty/unusable OHLCV (see logs above)"

    # 2) Optional fallback: Stooq
    if not allow_fallback:
        if strict:
            raise DataDownloadError(f"Yahoo failed for {ticker} and fallback disabled.")
        return pd.DataFrame(), report

    df_stooq = download_stooq_us(ticker)
    if _is_usable_ohlcv(df_stooq):
        report.stooq_ok = True
        report.source_used = "stooq"

        # Restrict to requested period
        df_stooq = df_stooq.sort_index()
        df_stooq = df_stooq.loc[
            (df_stooq.index >= pd.to_datetime(start)) &
            (df_stooq.index <= pd.to_datetime(end))
        ]

        if _is_usable_ohlcv(df_stooq):
            return df_stooq, report

        report.stooq_ok = False
        report.stooq_error = "Stooq returned data, but after date filtering it became unusable."
    else:
        report.stooq_ok = False
        report.stooq_error = "Stooq returned empty/unusable OHLCV."

    # 3) All failed
    if strict:
        raise DataDownloadError(
            f"Failed to download data for {ticker} from all allowed sources. "
            f"(yahoo_ok={report.yahoo_ok}, stooq_ok={report.stooq_ok})"
        )

    return pd.DataFrame(), report


# ==========================================================
# Final merged dataset builder
# ==========================================================

def build_merged_frame(
    symbol: str = "NVDA",
    exogenous: list[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    *,
    strict: bool = True,
    allow_fallback: bool = True,
    strict_exogenous: bool = True,
    yahoo_retries: int = 2,
    yahoo_sleep_s: float = 1.0,
) -> pd.DataFrame:
    """
    Build a merged dataset including:
    - Target asset (full OHLCV)
    - Exogenous assets (close + volume only)

    Parameters
    ----------
    strict:
        If True, raise if required downloads fail.
    allow_fallback:
        If True, allow fallback from Yahoo -> Stooq.
    strict_exogenous:
        If True, missing exogenous raises (production strict).
        If False, missing exogenous is skipped and recorded in attrs.
    """
    if exogenous is None:
        exogenous = ["SOXX", "MU", "QQQ"]

    if end is None:
        end = pd.Timestamp.today().date().isoformat()

    ingestion_reports: list[dict[str, Any]] = []
    missing_exogenous: list[str] = []

    # -------------------------
    # Download base asset
    # -------------------------
    base, rep = download_prices(
        symbol,
        start,
        end,
        allow_fallback=allow_fallback,
        strict=strict,
        yahoo_retries=yahoo_retries,
        yahoo_sleep_s=yahoo_sleep_s,
    )
    ingestion_reports.append(rep.as_dict())

    if base.empty:
        # If strict=False, base can be empty
        raise DataDownloadError(f"Target ticker {symbol} returned empty dataframe.")

    # Prefix base columns
    base = base.sort_index()
    base.columns = [f"{symbol.lower()}_{c}" for c in base.columns]

    # -------------------------
    # Download and merge exogenous
    # -------------------------
    for t in exogenous:
        ex, rep = download_prices(
            t,
            start,
            end,
            allow_fallback=allow_fallback,
            strict=(strict and strict_exogenous),
            yahoo_retries=yahoo_retries,
            yahoo_sleep_s=yahoo_sleep_s,
        )
        ingestion_reports.append(rep.as_dict())

        if ex.empty:
            if strict_exogenous:
                raise DataDownloadError(f"Exogenous ticker {t} returned empty dataframe.")
            missing_exogenous.append(t)
            continue

        ex = ex.sort_index()

        # Keep only close and volume for exogenous assets
        ex = ex[["close", "volume"]].copy()
        ex.columns = [f"{t.lower()}_close", f"{t.lower()}_volume"]

        # Left join on date index
        base = base.join(ex, how="left")

    # Forward fill missing values and drop remaining NaNs
    # mportant when exogenous has missing days - we want to keep the target rows 
    # and just have NaN for exogenous, which can be handled in feature engineering
    base = base.ffill().dropna()

    # Attach ingestion report for debugging/testing
    base.attrs["ingestion_report"] = ingestion_reports
    base.attrs["missing_exogenous"] = missing_exogenous
    base.attrs["allow_fallback"] = allow_fallback
    base.attrs["strict_exogenous"] = strict_exogenous

    return base
