import numpy as np
import pandas as pd

from strategies.smc_sweep import backtest_smc_sweep

# Create synthetic 1-min data with trending sections (to exercise HTF bias)
n = 2000
idx = pd.date_range("2024-01-01", periods=n, freq="T")
rng = np.random.default_rng(7)
# regime: up, down, chop
regime = np.concatenate([np.ones(600), -np.ones(600), np.zeros(800)])
noise = rng.normal(0, 0.05, n)
drift = np.cumsum(0.01 * regime + noise)
price = 100 + drift
high = price + rng.uniform(0.0, 0.15, n)
low = price - rng.uniform(0.0, 0.15, n)
open_ = price + rng.normal(0, 0.02, n)
close = price + rng.normal(0, 0.02, n)
vol = np.exp(rng.normal(9.5, 0.25, n))

df = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
)

# Inject some sweep-like spikes & volumes
for j in [300, 550, 900, 1200, 1500, 1700]:
    if j % 2 == 0:
        df.iloc[j, df.columns.get_loc("high")] += 1.2
    else:
        df.iloc[j, df.columns.get_loc("low")] -= 1.2
    df.iloc[j, df.columns.get_loc("volume")] *= 6.0

result = backtest_smc_sweep(
    df,
    equity_usd=10_000,
    rr_targets=(0.5, 1.0, 2.0),
    lookback=20,
    wick_ratio=0.5,
    vol_burst_z=1.25,
    htf_rule="4H",
    bias_method="combo",
    kelly_p=0.55,
    kelly_b=1.0,
    kelly_fraction=0.25,
)

print("Stats:", result["stats"])
print(result["trades"].head(10))
print(f"Total trades: {len(result['trades'])}")
