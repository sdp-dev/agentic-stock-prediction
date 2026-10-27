"""Tools the analyst agents call.

The analysis date comes from session state (`as_of`), never from tool
arguments, so the model cannot ask for data from after the date it is
deciding on. In backtests, fundamentals and news are disabled because
yfinance only serves their current values, which would leak the future.
"""

from typing import Any

from google.adk.tools import ToolContext

from . import data, indicators
from .config import MARKET_INDEX, VOLATILITY_INDEX


def _as_of(tool_context: ToolContext | None) -> str | None:
    return tool_context.state.get("as_of") if tool_context else None


def _backtesting(tool_context: ToolContext | None) -> bool:
    return bool(tool_context and tool_context.state.get("backtest"))


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "error_message": message}


def get_price_summary(ticker: str, tool_context: ToolContext) -> dict[str, Any]:
    """Summarize a stock's recent price action: latest close, trailing returns,
    volatility, 52-week range, and volume trend.

    Args:
        ticker: Stock symbol, e.g. "AAPL".

    Returns:
        A dict with status "ok" and price statistics, or status "error".
    """
    try:
        df = data.get_history(ticker, _as_of(tool_context))
    except Exception as e:
        return _error(str(e))

    close, volume = df["Close"], df["Volume"]
    last_year = close.tail(252)
    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "as_of": str(df.index[-1].date()),
        "last_close": _round(close.iloc[-1], 2),
        "return_1w": _round(indicators.trailing_return(close, 5)),
        "return_1m": _round(indicators.trailing_return(close, 21)),
        "return_3m": _round(indicators.trailing_return(close, 63)),
        "return_1y": _round(indicators.trailing_return(close, 252)),
        "annualized_volatility_3m": _round(indicators.annualized_volatility(close, 63)),
        "high_52w": _round(last_year.max(), 2),
        "low_52w": _round(last_year.min(), 2),
        "pct_below_52w_high": _round(close.iloc[-1] / last_year.max() - 1),
        "max_drawdown_1y": _round(indicators.max_drawdown(last_year)),
        "volume_vs_3m_avg": _round(volume.tail(5).mean() / volume.tail(63).mean()),
    }


def get_technical_indicators(ticker: str, tool_context: ToolContext) -> dict[str, Any]:
    """Compute technical indicators for a stock: moving averages, RSI, MACD,
    and Bollinger Band position.

    Args:
        ticker: Stock symbol, e.g. "AAPL".

    Returns:
        A dict with status "ok" and indicator values, or status "error".
    """
    try:
        df = data.get_history(ticker, _as_of(tool_context))
    except Exception as e:
        return _error(str(e))

    close = df["Close"]
    if len(close) < 200:
        return _error(f"Need at least 200 trading days of history, found {len(close)}")

    price = close.iloc[-1]
    sma20, sma50, sma200 = (indicators.sma(close, w).iloc[-1] for w in (20, 50, 200))
    sma50_series, sma200_series = indicators.sma(close, 50), indicators.sma(close, 200)
    macd = indicators.macd(close).iloc[-1]
    macd_prev = indicators.macd(close).iloc[-2]

    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "as_of": str(df.index[-1].date()),
        "price": _round(price, 2),
        "sma_20": _round(sma20, 2),
        "sma_50": _round(sma50, 2),
        "sma_200": _round(sma200, 2),
        "price_vs_sma_50": _round(price / sma50 - 1),
        "price_vs_sma_200": _round(price / sma200 - 1),
        "golden_cross": bool(sma50 > sma200),
        "cross_in_last_20d": bool(
            ((sma50_series > sma200_series).tail(21).astype(int).diff().abs() > 0).any()
        ),
        "rsi_14": _round(indicators.rsi(close).iloc[-1], 1),
        "macd": _round(macd["macd"]),
        "macd_signal": _round(macd["signal"]),
        "macd_histogram": _round(macd["histogram"]),
        "macd_histogram_rising": bool(macd["histogram"] > macd_prev["histogram"]),
        "bollinger_percent_b": _round(indicators.bollinger_percent_b(close).iloc[-1], 3),
    }


def get_market_context(tool_context: ToolContext) -> dict[str, Any]:
    """Describe the broad market backdrop: S&P 500 (SPY) trend and the VIX
    volatility index.

    Returns:
        A dict with status "ok" and market statistics, or status "error".
    """
    as_of = _as_of(tool_context)
    try:
        spy = data.get_history(MARKET_INDEX, as_of)["Close"]
        vix = data.get_history(VOLATILITY_INDEX, as_of)["Close"]
    except Exception as e:
        return _error(str(e))

    return {
        "status": "ok",
        "as_of": str(spy.index[-1].date()),
        "spy_return_1m": _round(indicators.trailing_return(spy, 21)),
        "spy_return_3m": _round(indicators.trailing_return(spy, 63)),
        "spy_above_sma_200": bool(spy.iloc[-1] > indicators.sma(spy, 200).iloc[-1]),
        "spy_rsi_14": _round(indicators.rsi(spy).iloc[-1], 1),
        "vix": _round(vix.iloc[-1], 2),
        "vix_1m_change": _round(indicators.trailing_return(vix, 21)),
    }


def get_fundamentals(ticker: str, tool_context: ToolContext) -> dict[str, Any]:
    """Get a company's current fundamentals: sector, valuation multiples,
    margins, growth, and analyst consensus.

    Args:
        ticker: Stock symbol, e.g. "AAPL".

    Returns:
        A dict with status "ok" and fundamentals, "unavailable" during
        backtests, or "error".
    """
    if _backtesting(tool_context):
        return {
            "status": "unavailable",
            "reason": "Fundamentals are not point-in-time, so they are disabled in backtests.",
        }
    try:
        info = data.get_info(ticker)
    except Exception as e:
        return _error(str(e))

    fields = [
        "shortName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
        "pegRatio", "priceToBook", "profitMargins", "revenueGrowth", "earningsGrowth",
        "debtToEquity", "freeCashflow", "recommendationKey", "targetMeanPrice",
    ]
    return {"status": "ok", "ticker": ticker.upper(), **{f: info.get(f) for f in fields}}


def get_recent_news(ticker: str, tool_context: ToolContext) -> dict[str, Any]:
    """Get recent news headlines for a company.

    Args:
        ticker: Stock symbol, e.g. "AAPL".

    Returns:
        A dict with status "ok" and a list of headlines, "unavailable" during
        backtests, or "error".
    """
    if _backtesting(tool_context):
        return {
            "status": "unavailable",
            "reason": "News is not point-in-time, so it is disabled in backtests.",
        }
    try:
        items = data.get_news(ticker)
    except Exception as e:
        return _error(str(e))

    headlines = []
    for item in items:
        content = item.get("content", item)
        title = content.get("title")
        if title:
            headlines.append({
                "title": title,
                "summary": (content.get("summary") or "")[:300],
                "published": content.get("pubDate") or content.get("providerPublishTime"),
                "source": (content.get("provider") or {}).get("displayName")
                if isinstance(content.get("provider"), dict) else content.get("publisher"),
            })
    return {"status": "ok", "ticker": ticker.upper(), "headlines": headlines}
