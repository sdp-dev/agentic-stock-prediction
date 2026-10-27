"""Multi-agent stock analysis pipeline built as a Google ADK Workflow graph.

           ┌─▶ technical_analyst ───┐
    START ─┼─▶ market_analyst ──────┼─▶ gather_reports ─▶ portfolio_manager ─▶ Recommendation
           └─▶ fundamental_analyst ─┘

The analysts run in parallel; the join waits for all three and hands their
reports to the portfolio manager. `root_agent` is what `adk web` loads.
"""

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.workflow import START, JoinNode, Workflow
from google.genai import types

from .config import MODEL
from .rate_limit import throttle
from .schemas import Recommendation
from .tools import (
    get_fundamentals,
    get_market_context,
    get_price_summary,
    get_recent_news,
    get_technical_indicators,
)

# Shared by every agent: retry rate-limit and transient errors with backoff.
GEMINI = Gemini(
    model=MODEL,
    retry_options=types.HttpRetryOptions(
        attempts=5, initial_delay=10, max_delay=65, exp_base=2, http_status_codes=[429, 500, 503]
    ),
)

REPORT_STYLE = """
Start your report with the ticker symbol. Keep it under 150 words, as short
bullet points that cite the actual numbers from your tools. End with one line:
"Leaning: BULLISH", "Leaning: BEARISH", or "Leaning: NEUTRAL".
If a tool returns an error or is unavailable, say so briefly and work with what you have.
Do not make up data that your tools did not return.
"""

technical_analyst = LlmAgent(
    name="technical_analyst",
    model=GEMINI,
    mode="single_turn",
    before_model_callback=throttle,
    description="Analyzes price action and technical indicators for one stock.",
    instruction=f"""You are a technical analyst. Identify the stock ticker in the user's
request, then call get_price_summary and get_technical_indicators for it.

Assess trend (moving averages, golden/death cross), momentum (RSI, MACD),
and stretch (Bollinger %B, distance from 52-week high). Note overbought
(RSI > 70) or oversold (RSI < 30) conditions and unusual volume.
{REPORT_STYLE}""",
    tools=[get_price_summary, get_technical_indicators],
    output_key="technical_report",
)

market_analyst = LlmAgent(
    name="market_analyst",
    model=GEMINI,
    mode="single_turn",
    before_model_callback=throttle,
    description="Assesses the broad market regime.",
    instruction=f"""You are a market strategist. Call get_market_context and describe the
market regime: is the S&P 500 in an uptrend (above its 200-day average)? Is
volatility (VIX) calm (< 15), normal (15-25), or stressed (> 25) and rising or
falling? Say whether the backdrop favors taking risk in the stock the user
asked about.
{REPORT_STYLE}""",
    tools=[get_market_context],
    output_key="market_report",
)

fundamental_analyst = LlmAgent(
    name="fundamental_analyst",
    model=GEMINI,
    mode="single_turn",
    before_model_callback=throttle,
    description="Reviews valuation, financial health, and news for one stock.",
    instruction=f"""You are a fundamental analyst. Identify the stock ticker in the user's
request, then call get_fundamentals and get_recent_news for it.

Comment on valuation (P/E vs. growth), profitability, balance sheet, analyst
consensus, and any headlines that could move the stock. If both tools are
unavailable (as in backtests), reply only with the ticker and
"Fundamentals and news unavailable for this date. Leaning: NEUTRAL".
{REPORT_STYLE}""",
    tools=[get_fundamentals, get_recent_news],
    output_key="fundamental_report",
)

portfolio_manager = LlmAgent(
    name="portfolio_manager",
    model=GEMINI,
    mode="single_turn",
    before_model_callback=throttle,
    description="Weighs the analyst reports and makes the final call.",
    instruction="""You are a disciplined portfolio manager deciding on one stock over the
next {horizon_days?} trading days (assume 20 if blank). The analysis date is
{as_of?} (if blank, today).

You will receive reports from your technical analyst, market analyst, and
fundamental analyst, keyed by analyst. Decide BUY, SELL, or HOLD.
- BUY only when trend and momentum agree and the market backdrop is not hostile.
- SELL when the trend is broken and momentum is negative, or the stock is
  extremely overbought into a weak market.
- HOLD when signals conflict or the edge is small. HOLD is a valid answer.
Set confidence to reflect how strongly the reports agree (0.5 = coin flip).
Base everything on the reports; do not invent facts.""",
    output_schema=Recommendation,
    output_key="recommendation",
)

root_agent = Workflow(
    name="stock_analysis_pipeline",
    description="Researches a stock with parallel analysts, then issues a recommendation.",
    edges=[(
        START,
        (technical_analyst, market_analyst, fundamental_analyst),
        JoinNode(name="gather_reports"),
        portfolio_manager,
    )],
)
