import numpy as np, pandas as pd
from research.funding_backtest import funding_carry_backtest

def test_funding_sign_logic():
    idx = pd.date_range("2024-01-01", periods=4, freq="H")
    fr = pd.Series([0.001, 0.001, -0.001, -0.001], index=idx)  # +0.1% then -0.1%
    bt = funding_carry_backtest(fr, threshold=0.0, notional_usd=1000, taker_fee_bps=0.0)
    # First two hours: short perps earns + funding; last two: long perps earns +
    assert bt["funding_pnl"].iloc[0] > 0 and bt["funding_pnl"].iloc[2] > 0

def test_fee_on_flip():
    idx = pd.date_range("2024-01-01", periods=3, freq="H")
    fr = pd.Series([0.001, -0.001, 0.001], index=idx)
    bt = funding_carry_backtest(fr, notional_usd=1000, taker_fee_bps=10.0)
    # Two flips -> 2 fees charged
    flips = int((bt['position'].shift(1).fillna(0) != bt['position']).sum())
    expected_fee = - (10.0 / 10000.0) * 1000 * flips
    assert abs(bt["fee"].sum() - expected_fee) < 1e-6
