"""
Final demo script: multi-venue L2 replay + router + latency + scoring.
  - Synthesizes 3 venue datasets with different liquidity profiles and latency.
  - Uses a simple emitter that detects sweeps and posts limit orders routed to best venue.
  - Collects fill records across venues and exports a final CSV and summary.
"""
import pandas as pd, numpy as np, os, json, math
from multi_venue import Venue, Router, LatencyModel
from replay.l2_replayer import L2Replay
from replay.execution_l2 import ExecutionSimulatorL2
from strategies.smc_sweep import backtest_smc_sweep
from research.detectors import find_sweeps

def synthesize_venue(name, n=2000, seed=1, liquidity_scale=1.0, latency_base=20.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    price = 100 + np.cumsum(rng.normal(0, 0.05, n))
    # L2 snapshots with 5 levels
    l2_rows = []
    trades_rows = []
    for i, t in enumerate(idx):
        center = float(price[i])
        for lvl in range(1,6):
            bp = round(center - 0.01*lvl, 2)
            ap = round(center + 0.01*lvl, 2)
            # size scales with liquidity_scale and some randomness
            bs = float(max(0.1, rng.exponential(10.0) * liquidity_scale))
            asz = float(max(0.1, rng.exponential(8.0) * liquidity_scale))
            l2_rows.append({"timestamp": t, "side":"bid", "price": bp, "size": bs, "update_type":"snapshot"})
            l2_rows.append({"timestamp": t, "side":"ask", "price": ap, "size": asz, "update_type":"snapshot"})
        # trades aggressive: more frequent on deeper venues
        ntr = rng.integers(0,4) if liquidity_scale<2.0 else rng.integers(1,6)
        for k in range(ntr):
            side = "buy" if rng.random() < 0.5 else "sell"
            px = round(center + rng.normal(0,0.02), 2)
            sz = float(max(0.01, rng.exponential(2.0) * liquidity_scale))
            trades_rows.append({"timestamp": t, "price": px, "size": sz, "side": side})
    l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
    trades_df = pd.DataFrame(trades_rows).set_index("timestamp").sort_index()
    return Venue(name=name, l2_diffs=l2_df, trades=trades_df), LatencyModel(base_ms=latency_base, jitter_ms=5.0)

def main():
    # create 3 venues
    v1, lat1 = synthesize_venue("alpha", n=2000, seed=11, liquidity_scale=1.0, latency_base=20.0)
    v2, lat2 = synthesize_venue("beta", n=2000, seed=22, liquidity_scale=2.5, latency_base=50.0)
    v3, lat3 = synthesize_venue("gamma", n=2000, seed=33, liquidity_scale=0.6, latency_base=10.0)
    venues = [v1, v2, v3]
    latency_models = {"alpha":lat1, "beta":lat2, "gamma":lat3}
    router = Router(venues, latency_models=latency_models)

    # Build LTF bar df used by sweep detector (use venue alpha price for bars)
    # We'll reuse alpha's trade-derived midpoints as OHLC-like bars for detection
    alpha_px = v1.trades["price"].resample("1T").last().ffill().fillna(method="bfill")
    # make bars from mid price small random walk
    df = pd.DataFrame(index=alpha_px.index)
    df["close"] = alpha_px
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open","close"]].max(axis=1) + 0.05
    df["low"] = df[["open","close"]].min(axis=1) - 0.05
    df["volume"] = 1.0

    # detect sweeps events using detectors.find_sweeps
    events = find_sweeps(df, lookback=20, wick_ratio=0.5, vol_burst_z=1.5)
    print("Detected sweep events:", len(events))
    # For each event, create an order emit: short for up-sweep, long for down-sweep
    orders = []
    for ts, ev in events.iterrows():
        side = "long" if ev["direction"]=="long" else "short"
        # use entry price proxy from detector (already in event entry_price)
        price = float(ev["entry_price"])
        qty = 1.0
        orders.append({"id":f"e-{ts}", "timestamp": ts, "side": side, "price": price, "qty": qty})

    # route & place orders
    fill_records = []
    for o in orders:
        ts = o["timestamp"]
        side = o["side"]
        price = o["price"]
        qty = o["qty"]
        # router chooses best venue and returns (venue_name, prob)
        venue_name, prob = router.choose(ts, side, price, qty, horizon_ms=60_000)
        venue = next(v for v in venues if v.name==venue_name)
        # simulate placing the order with latency: we sample latency and shift the timestamp forward
        lat_ms = latency_models[venue_name].sample_ms()
        ts_submit = pd.to_datetime(ts) + pd.Timedelta(milliseconds=lat_ms)
        order = {"id": o["id"], "timestamp": ts_submit, "side": side, "price": price, "qty": qty}
        res = venue.exec_sim.place_limit(order)
        rec = {"order_id": o["id"], "venue": venue_name, "orig_ts": o["timestamp"], "submit_ts": ts_submit, "side": side,
               "price": price, "qty": qty, "route_prob": prob, "filled": res.get("filled"), "fill_price": res.get("fill_price"),
               "filled_qty": res.get("filled_qty"), "queue_ahead": res.get("queue_ahead"), "agg_consumed": res.get("agg_consumed")}
        fill_records.append(rec)

    rep_df = pd.DataFrame(fill_records)
    out_csv = "final_fill_quality_report.csv"
    rep_df.to_csv(out_csv, index=False)
    print("Wrote final report:", out_csv)
    # simple summary
    print("Summary:")
    print(rep_df.groupby("venue")["filled"].agg(["sum","count","mean"]))
    return rep_df

if __name__ == "__main__":
    csv_path = run_final_replay()
    # Auto-generate dashboard
    try:
        from dashboard import build_dashboard
        build_dashboard(csv_path, "dashboard.html")
        print("Dashboard generated: dashboard.html")
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
