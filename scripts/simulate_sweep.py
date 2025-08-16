import numpy as np, pandas as pd
from research.detectors import find_sweeps

# generate synthetic OHLCV
n = 500
idx = pd.date_range("2024-01-01", periods=n, freq="T")
prices = np.cumsum(np.random.normal(0, 0.1, n)) + 100
high = prices + np.random.uniform(0.0, 0.2, n)
low = prices - np.random.uniform(0.0, 0.2, n)
open_ = prices + np.random.uniform(-0.05, 0.05, n)
close = prices + np.random.uniform(-0.05, 0.05, n)
vol = np.random.lognormal(mean=10, sigma=0.3, size=n)

df = pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":vol}, index=idx)

# Inject a few synthetic sweeps by spiking highs/lows and volume
for j in [120, 250, 380]:
    df.iloc[j, df.columns.get_loc("high")] += 1.0
    df.iloc[j, df.columns.get_loc("volume")] *= 5.0

for j in [160, 420]:
    df.iloc[j, df.columns.get_loc("low")] -= 1.0
    df.iloc[j, df.columns.get_loc("volume")] *= 5.0

events = find_sweeps(df, lookback=20, wick_ratio=0.5, vol_burst_z=1.5, require_reject=True)
print(events.head(10))
print(f"Detected {len(events)} sweep events")
