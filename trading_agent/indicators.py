"""Technical indicators computed from a daily close series."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's relative strength index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "histogram": line - signal_line})


def bollinger_percent_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Position within the Bollinger Bands: 0 = lower band, 1 = upper band."""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper, lower = mid + num_std * std, mid - num_std * std
    return (close - lower) / (upper - lower)


def trailing_return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - days] - 1)


def annualized_volatility(close: pd.Series, days: int = 63) -> float | None:
    returns = close.pct_change().dropna().tail(days)
    if len(returns) < 2:
        return None
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline over the series, as a negative fraction."""
    return float((close / close.cummax() - 1).min())
