"""Settings for the trading agent, overridable with environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

# Gemini model used by every agent in the pipeline.
MODEL = os.getenv("TRADING_AGENT_MODEL", "gemini-3.5-flash")

# Client-side cap on model calls per minute, shared by all agents. Gemini's free
# tier allows 5; raise it on a paid tier, or set 0 to disable.
REQUESTS_PER_MINUTE = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "5"))

# How far back price history is downloaded (indicators need ~200 trading days).
HISTORY_START = "2015-01-01"

# Benchmarks used for market context.
MARKET_INDEX = "SPY"
VOLATILITY_INDEX = "^VIX"

APP_NAME = "agentic_stock_prediction"
