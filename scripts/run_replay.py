import pandas as pd, numpy as np
from replay.replayer import MarketReplay, ExecutionSimulator
from strategies.smc_sweep import backtest_smc_sweep
from controller import Controller

# Build synthetic trade stream from previous synthetic price series (reuse run_smc_backtest generator style)
n = 2000
idx = pd.date_range("2024-01-01", periods=n, freq="T")
rng = np.random.default_rng(7)
regime = np.concatenate([np.ones(600), -np.ones(600), np.zeros(800)])
noise = rng.normal(0, 0.05, n)
drift = np.cumsum(0.01 * regime + noise)
price = 100 + drift
# Create aggressive trade ticks: alternate buys/sells with sizes
sides = ["buy" if i%2==0 else "sell" for i in range(n)]
sizes = rng.exponential(1.0, n) * 5.0
trades = pd.DataFrame({"timestamp": idx, "price": price, "size": sizes, "side": sides}).set_index("timestamp")

# Create a naive depth snapshot: best_bid/best_ask with sizes (scaled)
best_bid = price - 0.02
best_ask = price + 0.02
bid_size = np.maximum(1.0, rng.exponential(10.0, n))
ask_size = np.maximum(1.0, rng.exponential(10.0, n))
depth = pd.DataFrame({"best_bid": best_bid, "bid_size": bid_size, "best_ask": best_ask, "ask_size": ask_size}, index=idx)

replay = MarketReplay(trades, depth)
sim = ExecutionSimulator(replay)

# Example: place a limit buy at slightly above best_bid at t0 for qty 2.0
t0 = idx[300]
order = {"type":"limit","timestamp":t0,"side":"long","price":float(depth.loc[t0,"best_bid"])+0.0,"qty":2.0,"id":"test1"}
res = sim.place_order(order)
print("Sim result:", res)

# Now run controller-level replay with a simple strategy_func that places a limit order when sweep detected
from research.detectors import find_sweeps
from research.fvg import find_fvgs, fvg_midpoints
from research.htf import htf_bias

# Build a simple strategy function capturing earlier backtest logic but emitting orders for the replay sim
def simple_strategy_emitter(bar, state):
    # bar is a pandas namedtuple row with open, high, low, close, volume; timestamp in bar.Index
    orders = []
    ts = bar.Index
    # naive: if bar has a big upper wick (proxy sweep) place a short limit at mid-price
    o,h,l,c = bar.open, bar.high, bar.low, bar.close
    tr = max(h-l,1e-12)
    if (h - max(o,c)) >= 0.5 * tr and state.get("last_order_ts",None) != ts:
        entry = (h + l + c)/3.0
        qty = 1.0
        orders.append({"type":"limit","timestamp":ts,"side":"short","price":entry,"qty":qty,"id":f"emit-{ts}"})
        state["last_order_ts"] = ts
    return orders

# Create controller with same bar df used for strategy generation
from strategies.smc_sweep import backtest_smc_sweep
# Recreate bar df
high = price + rng.uniform(0.0, 0.15, n)
low = price - rng.uniform(0.0, 0.15, n)
open_ = price + rng.normal(0, 0.02, n)
close = price + rng.normal(0, 0.02, n)
vol = np.exp(rng.normal(9.5, 0.25, n))
df = pd.DataFrame({"open":open_, "high":high, "low":low, "close":close, "volume":vol}, index=idx)

ctrl = Controller(df)
trades = ctrl.run_replay(replay, simple_strategy_emitter, equity_usd=10000)
print("Replay trades count:", len(trades))
if len(trades):
    print(trades[:5])
