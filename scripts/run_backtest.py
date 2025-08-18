import numpy as np
import pandas as pd

from research.funding_backtest import funding_carry_backtest

# Make a small synthetic funding rate series: hourly for 7 days
idx = pd.date_range("2024-01-01", periods=24 * 7, freq="H")
# Simulate a regime: positive funding during uptrends, negative during chop
rng = np.random.default_rng(42)
base = 0.00005 * np.sin(np.linspace(0, 10, len(idx)))  # oscillates around 0
noise = rng.normal(0, 0.00002, len(idx))
fr = pd.Series(base + noise, index=idx)  # ~ +/- 5 bps/day spread
bt = funding_carry_backtest(fr, threshold=0.0, notional_usd=10_000, taker_fee_bps=1.0)

print(bt.tail())
print("Total PnL (USD):", round(bt["equity"].iloc[-1], 2))
print(
    "Trades (flips):", int((bt["position"].shift(1).fillna(0) != bt["position"]).sum())
)
