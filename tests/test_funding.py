import numpy as np, pandas as pd
def funding_carry_backtest(funding_rate: pd.Series,
                           threshold: float = 0.0,
                           notional_usd: float = 10_000.0,
                           taker_fee_bps: float = 1.0,
                           interval_hours: int = 1) -> pd.DataFrame:
    """
    Primitive backtest for a simple funding-carry strategy on perps:
      - Each interval, set position = -sign(funding_rate - threshold)
        (i.e., short perps when funding positive to receive funding; long when negative)
      - Earn funding: pnl = -position * notional * funding_rate_per_interval
        (longs pay when funding > 0, shorts receive; when funding < 0, longs receive)
      - Pay taker fees when you flip side (assumes 1 trade per side flip)

    Args:
      funding_rate: Series of per-interval funding rates (e.g., hourly). Decimal, not bps.
      threshold: only take a side if rate exceeds threshold (else flat if equal zero? we still take sign).
      notional_usd: absolute notional exposure.
      taker_fee_bps: fee per traded notional when changing side.
      interval_hours: hours per funding data point; used to scale 8h-announced rates if needed.
                       (Assumes input is already per-interval rate.)
    Returns:
      DataFrame with columns: position, funding_pnl, fee, pnl, equity
    """
    fr = funding_rate.copy().astype(float).fillna(0.0)
    pos = np.sign(fr - threshold) * -1.0  # short when funding > threshold
    # Fees when sign changes
    pos_shift = pd.Series(pos, index=fr.index).shift(1).fillna(0.0)
    flips = (np.sign(pos_shift) != np.sign(pos)).astype(int)
    fee = - (taker_fee_bps / 10_000.0) * notional_usd * flips
    # Funding transfers from longs to shorts when fr > 0, and from shorts to longs when fr < 0
    # With pos: +1 = long, -1 = short, pnl = -pos * notional * fr
    funding_pnl = - pos * notional_usd * fr  # assume per-interval rate
    pnl = funding_pnl + fee
    equity = pnl.cumsum()
    return pd.DataFrame({"position": pos, "funding_pnl": funding_pnl, "fee": fee, "pnl": pnl, "equity": equity})

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
