import numpy as np
import pandas as pd
import pytest

from trading_agent import indicators


def test_sma_matches_manual_mean():
    close = pd.Series([1.0, 2, 3, 4, 5])
    assert indicators.sma(close, 3).iloc[-1] == pytest.approx(4.0)


def test_rsi_is_100_for_steady_gains_and_0_for_steady_losses():
    up = pd.Series(np.arange(1.0, 50.0))
    assert indicators.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert indicators.rsi(up[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0.0)


def test_bollinger_percent_b_is_high_after_a_spike():
    close = pd.Series([100.0] * 19 + [101, 110])
    assert indicators.bollinger_percent_b(close).iloc[-1] > 0.9


def test_trailing_return_and_short_history():
    close = pd.Series([100.0, 105, 110])
    assert indicators.trailing_return(close, 2) == pytest.approx(0.10)
    assert indicators.trailing_return(close, 5) is None


def test_max_drawdown():
    close = pd.Series([100.0, 120, 90, 130])
    assert indicators.max_drawdown(close) == pytest.approx(-0.25)


def test_macd_histogram_is_line_minus_signal():
    close = pd.Series(np.linspace(100, 150, 100))
    m = indicators.macd(close)
    assert (m["histogram"] - (m["macd"] - m["signal"])).abs().max() < 1e-9
