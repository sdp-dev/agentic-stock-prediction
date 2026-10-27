"""Run the full ADK pipeline with a scripted model, so wiring is tested without an API key."""

import asyncio
import json

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.agents import LlmAgent
from google.adk.models import LlmCapabilities
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from trading_agent import agent
from trading_agent.runner import StockAnalyst

RECOMMENDATION = {
    "ticker": "AAPL",
    "action": "BUY",
    "confidence": 0.72,
    "rationale": "Uptrend with supportive market.",
    "key_factors": ["price above 200-day"],
    "risks": ["overbought"],
}


class ScriptedLlm(BaseLlm):
    """Calls every available tool once, then reports what the tools returned."""

    model: str = "scripted"
    requests: list = Field(default_factory=list)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        self.requests.append(llm_request)
        last = llm_request.contents[-1] if llm_request.contents else None
        tool_results = [p.function_response.response for p in (last.parts if last else []) if p.function_response]

        if llm_request.config.response_schema is not None:
            text = json.dumps(RECOMMENDATION)
        elif llm_request.tools_dict and not tool_results:
            calls = [
                types.Part(function_call=types.FunctionCall(
                    name=name, args={} if name == "get_market_context" else {"ticker": "AAPL"}
                ))
                for name in llm_request.tools_dict
            ]
            yield LlmResponse(content=types.Content(role="model", parts=calls))
            return
        else:
            text = "AAPL tool results: " + json.dumps(tool_results, default=str) + " Leaning: BULLISH"
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


@pytest.fixture
def scripted(monkeypatch):
    llm = ScriptedLlm()
    # The Workflow graph holds its own copies of the agents, so patch those.
    for node in agent.root_agent.graph.nodes:
        if isinstance(node, LlmAgent):
            monkeypatch.setattr(node, "model", llm)
    return llm


def test_pipeline_produces_recommendation_from_point_in_time_reports(fake_market, scripted):
    result = asyncio.run(StockAnalyst().analyze("aapl", as_of="2023-03-15", backtest=True))

    assert result.recommendation.action == "BUY"
    assert result.recommendation.confidence == pytest.approx(0.72)

    # Analysts' tools ran against the backtest date, not the latest data.
    assert '"as_of": "2023-03-15"' in result.reports["technical_report"]
    assert '"as_of": "2023-03-15"' in result.reports["market_report"]
    # Fundamentals and news were blocked in backtest mode.
    assert "unavailable" in result.reports["fundamental_report"]

    # The portfolio manager ran last, with the backtest date injected from state
    # and all three reports passed in by the join node.
    pm_request = scripted.requests[-1]
    assert "2023-03-15" in str(pm_request.config.system_instruction)
    pm_input = pm_request.contents[-1].parts[0].text
    for name in ("technical_analyst", "market_analyst", "fundamental_analyst"):
        assert name in pm_input
