"""Settings for the trading agent, overridable with environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

# Gemini model used by every agent in the pipeline.
MODEL = os.getenv("TRADING_AGENT_MODEL", "gemini-3.5-flash")

# How far back price history is downloaded (indicators need ~200 trading days).
HISTORY_START = "2015-01-01"

# Benchmarks used for market context.
MARKET_INDEX = "SPY"
VOLATILITY_INDEX = "^VIX"

APP_NAME = "agentic_stock_prediction"
