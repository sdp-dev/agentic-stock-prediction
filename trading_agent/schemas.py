"""Structured output for the decision agent."""

from typing import Literal

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    ticker: str = Field(description="Stock symbol the recommendation is for.")
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="BUY if the stock should outperform over the horizon, SELL if it "
        "should fall, HOLD if there is no clear edge."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the action, 0 to 1.")
    rationale: str = Field(description="Two to four sentences explaining the decision.")
    key_factors: list[str] = Field(description="The signals that most drove the decision.")
    risks: list[str] = Field(description="What would make this call wrong.")
