from types import SimpleNamespace

import pandas as pd

from trading_agent import tools


def ctx(**state):
    return SimpleNamespace(state=state)


def test_tools_never_see_data_after_as_of(fake_market):
    as_of = "2023-03-15"
    summary = tools.get_price_summary("AAPL", ctx(as_of=as_of))
    technicals = tools.get_technical_indicators("AAPL", ctx(as_of=as_of))

    expected_close = fake_market.loc[:as_of, "Close"].iloc[-1]
    assert summary["as_of"] == technicals["as_of"] == as_of
    assert summary["last_close"] == round(expected_close, 2)


def test_as_of_on_a_weekend_uses_the_prior_trading_day(fake_market):
    summary = tools.get_price_summary("AAPL", ctx(as_of="2023-03-19"))  # Sunday
    assert summary["as_of"] == "2023-03-17"


def test_live_mode_uses_latest_data(fake_market):
    summary = tools.get_price_summary("AAPL", ctx())
    assert summary["as_of"] == str(fake_market.index[-1].date())


def test_fundamentals_and_news_are_disabled_in_backtests(fake_market):
    backtest = ctx(as_of="2023-03-15", backtest=True)
    assert tools.get_fundamentals("AAPL", backtest)["status"] == "unavailable"
    assert tools.get_recent_news("AAPL", backtest)["status"] == "unavailable"

    assert tools.get_fundamentals("AAPL", ctx())["shortName"] == "Fake Co"
    assert tools.get_recent_news("AAPL", ctx())["headlines"][0]["title"] == "Fake Co beats"


def test_bad_ticker_returns_error_instead_of_raising(fake_market):
    result = tools.get_price_summary("BADTICKER", ctx())
    assert result["status"] == "error"


def test_technicals_need_200_days(fake_market):
    early = str(fake_market.index[150].date())
    assert tools.get_technical_indicators("AAPL", ctx(as_of=early))["status"] == "error"


def test_market_context(fake_market):
    result = tools.get_market_context(ctx(as_of="2023-06-01"))
    assert result["status"] == "ok"
    assert pd.Timestamp(result["as_of"]) <= pd.Timestamp("2023-06-01")
