import numpy as np
import pandas as pd


def funding_carry_backtest(
    funding_rate: pd.Series,
    threshold: float = 0.0,
    notional_usd: float = 10_000.0,
    taker_fee_bps: float = 1.0,
    interval_hours: int = 1,
) -> pd.DataFrame:
    """
    Primitive backtest for a simple funding-carry strategy on perps:
      - Each interval, set position = -sign(funding_rate - threshold)
        (i.e., short perps when funding positive to receive funding; long when negative)
      - Earn funding: pnl = position * notional * funding_rate_per_interval
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
    fee = -(taker_fee_bps / 10_000.0) * notional_usd * flips
    funding_pnl = pos * notional_usd * fr  # assume per-interval rate
    pnl = funding_pnl + fee
    equity = pnl.cumsum()
    return pd.DataFrame(
        {
            "position": pos,
            "funding_pnl": funding_pnl,
            "fee": fee,
            "pnl": pnl,
            "equity": equity,
        }
    )


def normalize_8h_to_hourly(funding_8h: pd.Series) -> pd.Series:
    """
    Many exchanges quote funding rates every 8 hours. If you have an 8-hourly series
    but want hourly granularity, this helper distributes the 8h rate across 8 hours.
    Assumes compounding-neutral split (equal per-hour slices).
    """
    funding_8h = funding_8h.asfreq("8H")
    per_hour = (funding_8h / 8.0).resample("H").ffill()
    return per_hour
