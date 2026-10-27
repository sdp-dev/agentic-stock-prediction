"""Replay past dates through the agent and score its calls.

For each (date, ticker) the agent sees only data up to that date. Its
action is then compared with the stock's actual return over the next
`horizon_days` trading days, alongside two baselines:
  - buy_and_hold: always BUY
  - trend_rule:   BUY above a rising 50/200-day stack, SELL below a falling one

Predictions are appended to a CSV as they arrive, so an interrupted run
(e.g. hitting API rate limits) can be resumed with the same --out path.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd

from trading_agent import data
from trading_agent.config import MARKET_INDEX
from . import metrics

COLUMNS = [
    "date", "ticker", "action", "confidence", "rationale",
    "fwd_return", "buy_and_hold", "trend_rule",
]


def decision_dates(start: str, end: str, every: int) -> list[str]:
    """Every `every`-th trading day between start and end (market calendar from SPY)."""
    calendar = data.get_history(MARKET_INDEX).loc[start:end].index
    return [str(d.date()) for d in calendar[::every]]


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLUMNS)


async def _analyze_with_retry(analyst, ticker: str, as_of: str, horizon: int, retries: int):
    for attempt in range(retries + 1):
        try:
            return await analyst.analyze(ticker, as_of=as_of, horizon_days=horizon, backtest=True)
        except Exception as e:
            if attempt == retries:
                raise
            wait = 15 * 2**attempt
            print(f"    {type(e).__name__}: {str(e)[:120]} (retrying in {wait}s)")
            await asyncio.sleep(wait)


async def run_backtest(
    tickers: list[str],
    start: str,
    end: str,
    every: int = 20,
    horizon: int = 20,
    out: Path = Path("results/predictions.csv"),
    delay: float = 0.0,
    retries: int = 3,
    baseline_only: bool = False,
) -> pd.DataFrame:
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _load_existing(out)
    done = set(zip(results["date"].astype(str), results["ticker"]))

    analyst = None
    if not baseline_only:
        from trading_agent.runner import StockAnalyst
        analyst = StockAnalyst()

    dates = decision_dates(start, end, every)
    total = len(dates) * len(tickers)
    print(f"Backtesting {len(tickers)} tickers x {len(dates)} dates = {total} decisions")

    for i, (as_of, ticker) in enumerate(((d, t) for d in dates for t in tickers), 1):
        if (as_of, ticker) in done:
            continue
        close = data.get_history(ticker)["Close"]
        row = {
            "date": as_of,
            "ticker": ticker,
            "fwd_return": metrics.forward_return(close, as_of, horizon),
            "buy_and_hold": "BUY",
            "trend_rule": metrics.trend_baseline(close.loc[: pd.Timestamp(as_of)]),
        }
        if analyst:
            started = time.monotonic()
            try:
                analysis = await _analyze_with_retry(analyst, ticker, as_of, horizon, retries)
            except Exception as e:
                print(f"[{i}/{total}] {as_of} {ticker}: failed ({e}); skipping")
                continue
            rec = analysis.recommendation
            row.update(action=rec.action, confidence=rec.confidence, rationale=rec.rationale)
            print(f"[{i}/{total}] {as_of} {ticker}: {rec.action} ({rec.confidence:.2f}) "
                  f"in {time.monotonic() - started:.0f}s")
            if delay:
                await asyncio.sleep(delay)

        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(out, index=False)

    return results


def build_report(results: pd.DataFrame, horizon: int) -> str:
    """Markdown summary comparing the agent with the baselines."""
    strategies = ["buy_and_hold", "trend_rule"]
    if results["action"].notna().any():
        strategies.insert(0, "action")
    names = {"action": "Agent", "buy_and_hold": "Buy & hold", "trend_rule": "Trend rule"}

    scored = {names[s]: metrics.score(results, s) for s in strategies}
    table = pd.DataFrame(scored)

    def fmt(value, key):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "n/a"
        if key in ("calls", "buy / hold / sell"):
            return str(value)
        return f"{value:.1%}"

    lines = [
        "# Backtest report",
        "",
        f"- Decisions scored: {int(results['fwd_return'].notna().sum())} "
        f"(of {len(results)}; the rest are too recent to score)",
        f"- Tickers: {', '.join(sorted(results['ticker'].unique()))}",
        f"- Dates: {results['date'].min()} to {results['date'].max()}",
        f"- Horizon: {horizon} trading days; HOLD counts as correct within ±{metrics.HOLD_BAND:.0%}",
        "",
        "| Metric | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
    ]
    for key in table.index:
        lines.append(f"| {key} | " + " | ".join(fmt(table.at[key, c], key) for c in table.columns) + " |")

    if "action" in strategies:
        calibration = metrics.confidence_calibration(results)
        if not calibration.empty:
            lines += ["", "## Agent confidence calibration", "", "| Confidence | Calls | Accuracy |", "|---|---|---|"]
            for _, r in calibration.iterrows():
                lines.append(f"| {r['bucket']} | {r['calls']} | {r['accuracy']:.0%} |")

    lines += [
        "",
        "Strategy return per call = position (BUY +1, HOLD 0, SELL -1) x forward return. "
        "Calls on overlapping windows are not independent, so treat totals as indicative.",
    ]
    return "\n".join(lines) + "\n"
