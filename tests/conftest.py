"""Shared fixtures: synthetic prices so tests never hit the network."""

import numpy as np
import pandas as pd
import pytest

from trading_agent import data


def make_prices(start="2022-01-03", days=600, drift=0.001, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=days, name="Date")
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01, days))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
        index=index,
    )


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch):
    """The scripted model is free; don't make tests wait for API rate limits."""
    from trading_agent.rate_limit import limiter
    monkeypatch.setattr(limiter, "requests_per_minute", 0)


@pytest.fixture
def fake_market(monkeypatch):
    """Patch the downloader: every ticker gets the same synthetic uptrend."""
    prices = make_prices()

    def fake_download(ticker):
        if ticker == "BADTICKER":
            raise ValueError(f"No price data found for '{ticker}'")
        return prices

    monkeypatch.setattr(data, "_download", fake_download)
    monkeypatch.setattr(data, "get_info", lambda t: {"shortName": "Fake Co", "trailingPE": 20})
    monkeypatch.setattr(data, "get_news", lambda t, count=8: [{"content": {"title": "Fake Co beats"}}])
    return prices
