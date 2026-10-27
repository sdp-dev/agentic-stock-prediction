"""Market data access with point-in-time slicing.

Every read goes through `get_history`, which cuts the series at `as_of` so
backtests can never see prices from after the simulated decision date.
"""

import warnings
from functools import lru_cache

import pandas as pd
import yfinance as yf

from .config import HISTORY_START

# yfinance still uses pandas APIs that pandas 3 deprecates; the warnings are noise here.
warnings.filterwarnings("ignore", module="yfinance")


@lru_cache(maxsize=64)
def _download(ticker: str) -> pd.DataFrame:
    """Download and cache full daily history for a ticker (split/dividend adjusted)."""
    df = yf.Ticker(ticker).history(start=HISTORY_START, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for '{ticker}'")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.DatetimeIndex(df.index.date, name="Date")
    return df


def get_history(ticker: str, as_of: str | None = None) -> pd.DataFrame:
    """Daily OHLCV for `ticker`, including only rows on or before `as_of` (YYYY-MM-DD)."""
    df = _download(ticker.upper().strip())
    if as_of:
        df = df.loc[: pd.Timestamp(as_of)]
    if df.empty:
        raise ValueError(f"No price data for '{ticker}' on or before {as_of}")
    return df


def get_info(ticker: str) -> dict:
    """Current company profile and valuation fields (not point-in-time)."""
    return yf.Ticker(ticker.upper().strip()).info or {}


def get_news(ticker: str, count: int = 8) -> list[dict]:
    """Current headlines for a ticker (not point-in-time)."""
    return yf.Ticker(ticker.upper().strip()).get_news(count=count) or []
