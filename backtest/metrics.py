"""Scoring recommendations against what actually happened."""

import pandas as pd

from trading_agent import indicators

POSITION = {"BUY": 1, "HOLD": 0, "SELL": -1}

# A HOLD counts as correct when the stock moved less than this either way.
HOLD_BAND = 0.02


def forward_return(close: pd.Series, as_of: str, horizon_days: int) -> float | None:
    """Return from the last close on or before `as_of` to `horizon_days` trading days later."""
    past = close.loc[: pd.Timestamp(as_of)]
    if past.empty:
        return None
    start = close.index.get_loc(past.index[-1])
    end = start + horizon_days
    if end >= len(close):
        return None
    return float(close.iloc[end] / close.iloc[start] - 1)


def is_correct(action: str, fwd_return: float, hold_band: float = HOLD_BAND) -> bool:
    if action == "BUY":
        return fwd_return > 0
    if action == "SELL":
        return fwd_return < 0
    return abs(fwd_return) <= hold_band


def trend_baseline(close: pd.Series) -> str:
    """Rule-based comparison: BUY in a clean uptrend, SELL in a clean downtrend."""
    price = close.iloc[-1]
    sma50 = indicators.sma(close, 50).iloc[-1]
    sma200 = indicators.sma(close, 200).iloc[-1]
    if price > sma50 > sma200:
        return "BUY"
    if price < sma50 < sma200:
        return "SELL"
    return "HOLD"


def score(df: pd.DataFrame, action_col: str) -> dict:
    """Summary statistics for one set of calls (agent or baseline).

    `df` needs a `fwd_return` column and an action column of BUY/SELL/HOLD.
    Strategy return per call = position (+1/0/-1) x forward return, i.e. go
    long on BUY, short on SELL, stay flat on HOLD.
    """
    df = df.dropna(subset=["fwd_return"])
    if df.empty:
        return {}
    position = df[action_col].map(POSITION)
    strategy = position * df["fwd_return"]
    correct = [is_correct(a, r) for a, r in zip(df[action_col], df["fwd_return"])]
    counts = df[action_col].value_counts()
    buys = df.loc[df[action_col] == "BUY", "fwd_return"]
    sells = df.loc[df[action_col] == "SELL", "fwd_return"]
    return {
        "calls": len(df),
        "buy / hold / sell": f"{counts.get('BUY', 0)} / {counts.get('HOLD', 0)} / {counts.get('SELL', 0)}",
        "accuracy": sum(correct) / len(df),
        "avg_strategy_return": strategy.mean(),
        "total_strategy_return": strategy.sum(),
        "win_rate_on_trades": (strategy[position != 0] > 0).mean() if (position != 0).any() else None,
        "avg_return_after_buy": buys.mean() if len(buys) else None,
        "avg_return_after_sell": sells.mean() if len(sells) else None,
    }


def confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of the agent's calls grouped by stated confidence."""
    df = df.dropna(subset=["fwd_return", "confidence"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["correct"] = [is_correct(a, r) for a, r in zip(df["action"], df["fwd_return"])]
    df["bucket"] = pd.cut(df["confidence"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    return (
        df.groupby("bucket", observed=True)
        .agg(calls=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
