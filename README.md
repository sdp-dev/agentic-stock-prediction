# Agentic Stock Prediction

A multi-agent system built with [Google ADK](https://google.github.io/adk-docs/) that researches a stock the way a small investment team would. Three analyst agents call market-data and technical-indicator tools in parallel, then a portfolio-manager agent weighs their reports and issues a structured **BUY / SELL / HOLD** recommendation with a confidence score and rationale.

An evaluation harness replays past dates through the same pipeline and scores every call against what the stock actually did next, compared with simple baselines.

> For research and education only. Not financial advice.

## How it works

```
           ┌─▶ technical_analyst ───┐
    START ─┼─▶ market_analyst ──────┼─▶ gather_reports ─▶ portfolio_manager ─▶ Recommendation
           └─▶ fundamental_analyst ─┘
```

| Agent | Tools | Looks at |
|---|---|---|
| `technical_analyst` | `get_price_summary`, `get_technical_indicators` | Trailing returns, volatility, 52-week range, SMA 20/50/200, golden/death cross, RSI, MACD, Bollinger %B, volume |
| `market_analyst` | `get_market_context` | S&P 500 (SPY) trend and the VIX |
| `fundamental_analyst` | `get_fundamentals`, `get_recent_news` | Valuation, margins, growth, analyst consensus, headlines |
| `portfolio_manager` | none | The three reports; returns a `Recommendation` (Pydantic schema) |

The pipeline is an ADK `Workflow` graph: the analysts fan out from `START` and run concurrently, a `JoinNode` waits for all three, and the portfolio manager produces JSON validated against the schema in [`trading_agent/schemas.py`](trading_agent/schemas.py).

### No lookahead bias

Backtests are only meaningful if the agent cannot see the future. Two safeguards:

1. **The analysis date lives in session state, not in tool arguments.** Every tool reads `as_of` from the ADK `ToolContext` and slices price history to that date, so the model has no way to request later data.
2. **Non-point-in-time data is switched off in backtests.** yfinance only serves *current* fundamentals and news, so those tools return `unavailable` when `backtest` is set in state.

Both are covered by tests.

## Evaluation harness

`python main.py backtest` picks every Nth trading day in a date range, asks the agent for a call on each ticker using only data up to that day, and measures the stock's actual return over the following `--horizon` trading days.

Each call is scored as a long/flat/short position (BUY +1, HOLD 0, SELL −1) and compared with:

- **Buy & hold**: always BUY.
- **Trend rule**: BUY when price > SMA 50 > SMA 200, SELL when price < SMA 50 < SMA 200, otherwise HOLD.

The report includes accuracy (HOLD counts as correct if the stock moved less than ±2%), average and total strategy return, win rate on trades, average return after BUY vs. SELL calls, and a **confidence calibration** table showing whether higher-confidence calls are actually right more often. Predictions are appended to a CSV as they arrive, so a run interrupted by API rate limits picks up where it left off.

## Setup

Requires Python 3.10+ and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then paste your key into .env
```

## Usage

**Analyze a stock today:**

```bash
python main.py analyze NVDA            # add -v to print each analyst's report
python main.py analyze NVDA --as-of 2025-04-08 --horizon 10
```

**Chat with the pipeline in ADK's web UI** (shows each agent's tool calls and outputs):

```bash
adk web
```

Then open http://localhost:8000, pick `trading_agent`, and ask something like "Analyze MSFT".

**Run a backtest:**

```bash
python main.py backtest --tickers AAPL MSFT NVDA JPM XOM --start 2025-01-01 --end 2025-12-31 --every 20 --horizon 20
```

Results are written to `results/predictions.csv` and `results/report.md`. On Gemini's free tier, add `--delay 10` to stay under rate limits. To see the baselines without calling the LLM:

```bash
python main.py backtest --baseline-only
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `TRADING_AGENT_MODEL` | `gemini-3.5-flash` | Model used by every agent |

## Tests

```bash
python -m pytest
```

The tests use synthetic prices and a scripted stand-in for the LLM, so they run offline with no API key. They cover the indicators, the point-in-time guarantees, the scoring math, and a full run of the ADK workflow.

## Project structure

```
main.py                    CLI: analyze / backtest
trading_agent/
  agent.py                 Agents and the Workflow graph (root_agent)
  tools.py                 Tools the analysts call
  data.py                  yfinance access with as-of slicing and caching
  indicators.py            SMA, EMA, RSI, MACD, Bollinger %B, returns, drawdown
  schemas.py               Recommendation output schema
  runner.py                Programmatic runner (fresh session per analysis)
  config.py                Model and data settings
backtest/
  harness.py               Replay loop, resumable CSV output, markdown report
  metrics.py               Forward returns, scoring, baselines, calibration
tests/                     Offline tests
```

## Limitations

- Fundamentals and news aren't available historically, so backtests evaluate the technical and market-regime reasoning only.
- Decisions are scored on non-overlapping windows when `--every` ≥ `--horizon`; smaller gaps overlap and the totals become less independent.
- LLM output varies between runs; repeat a backtest to gauge stability.
- No transaction costs, slippage, or position sizing.
