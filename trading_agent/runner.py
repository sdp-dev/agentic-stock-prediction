"""Run the pipeline programmatically (used by the CLI and the backtester)."""

import asyncio
import uuid
from dataclasses import dataclass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import root_agent
from .config import APP_NAME
from .schemas import Recommendation

USER_ID = "local"
REPORT_KEYS = ("technical_report", "market_report", "fundamental_report")


@dataclass
class Analysis:
    recommendation: Recommendation
    reports: dict[str, str]


class StockAnalyst:
    """Wraps an ADK Runner; each analysis gets a fresh session."""

    def __init__(self) -> None:
        self.sessions = InMemorySessionService()
        self.runner = Runner(node=root_agent, app_name=APP_NAME, session_service=self.sessions)

    async def analyze(
        self,
        ticker: str,
        as_of: str | None = None,
        horizon_days: int = 20,
        backtest: bool = False,
    ) -> Analysis:
        """Analyze `ticker` using only data available on `as_of` (default: today)."""
        session_id = str(uuid.uuid4())
        await self.sessions.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id,
            state={
                "ticker": ticker.upper(),
                "as_of": as_of or "",
                "horizon_days": horizon_days,
                "backtest": backtest,
            },
        )
        message = types.Content(role="user", parts=[types.Part(text=f"Analyze {ticker.upper()}.")])
        async for _ in self.runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            pass

        session = await self.sessions.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
        state = session.state
        if "recommendation" not in state:
            raise RuntimeError("The pipeline finished without producing a recommendation")
        return Analysis(
            recommendation=Recommendation.model_validate(state["recommendation"]),
            reports={key: state.get(key, "") for key in REPORT_KEYS},
        )


def analyze(ticker: str, as_of: str | None = None, horizon_days: int = 20) -> Analysis:
    """Synchronous convenience wrapper for a single live analysis."""
    return asyncio.run(StockAnalyst().analyze(ticker, as_of=as_of, horizon_days=horizon_days))
