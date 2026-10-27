import pandas as pd
import pytest

from backtest import metrics


@pytest.fixture
def close():
    return pd.Series(
        [100.0, 102, 104, 103, 110],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )


def test_forward_return(close):
    assert metrics.forward_return(close, "2024-01-01", 2) == pytest.approx(0.04)
    assert metrics.forward_return(close, "2024-01-04", 2) is None  # not enough future data


def test_is_correct():
    assert metrics.is_correct("BUY", 0.03)
    assert not metrics.is_correct("BUY", -0.01)
    assert metrics.is_correct("SELL", -0.05)
    assert metrics.is_correct("HOLD", 0.01)
    assert not metrics.is_correct("HOLD", 0.08)


def test_trend_baseline():
    rising = pd.Series(range(1, 301), dtype=float)
    assert metrics.trend_baseline(rising) == "BUY"
    assert metrics.trend_baseline(rising[::-1].reset_index(drop=True)) == "SELL"


def test_score_long_short_returns():
    df = pd.DataFrame({
        "action": ["BUY", "SELL", "HOLD", "BUY"],
        "fwd_return": [0.05, -0.02, 0.10, None],
    })
    s = metrics.score(df, "action")
    assert s["calls"] == 3  # unscored row dropped
    assert s["total_strategy_return"] == pytest.approx(0.07)  # +0.05 long, +0.02 short, 0 flat
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["win_rate_on_trades"] == pytest.approx(1.0)


def test_confidence_calibration_buckets():
    df = pd.DataFrame({
        "action": ["BUY", "BUY", "SELL"],
        "confidence": [0.9, 0.85, 0.55],
        "fwd_return": [0.02, -0.01, -0.03],
    })
    table = metrics.confidence_calibration(df)
    assert table["calls"].sum() == 3
