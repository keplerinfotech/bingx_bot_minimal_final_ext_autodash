import os
import threading

import numpy as np
import pandas as pd

from replay.execution_l2 import ExecutionSimulatorL2
from replay.l2_replayer import L2Replay

try:
    from scripts.telemetry_client import write_heartbeat, write_pnl
except Exception:
    try:
        from telemetry_client import write_heartbeat, write_pnl
    except Exception:
        write_heartbeat = None
        write_pnl = None


def synthesize_week_data(
    start_ts="2024-01-01", periods=24 * 7 * 4, freq="15T", seed=42
):
    """
    Create a synthetic but realistic-seeming L2 diff stream and aggressive trades for one-week window.
    periods: number of ticks (e.g., 15-minute ticks default); here default ~1 week of 15-min ticks.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_ts, periods=periods, freq=freq)
    # build base price path with small drift/noise
    px = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    # create L2 snapshots every tick for top-5 levels on each side
    l2_rows = []
    trades_rows = []
    for i, t in enumerate(idx):
        center = px[i]
        # create 5 levels
        for lvl in range(1, 6):
            bid_p = round(center - 0.01 * lvl, 2)
            ask_p = round(center + 0.01 * lvl, 2)
            bid_size = float(max(0.5, rng.exponential(10.0)))
            ask_size = float(max(0.5, rng.exponential(10.0)))
            l2_rows.append(
                {
                    "timestamp": t,
                    "side": "bid",
                    "price": bid_p,
                    "size": bid_size,
                    "update_type": "snapshot",
                }
            )
            l2_rows.append(
                {
                    "timestamp": t,
                    "side": "ask",
                    "price": ask_p,
                    "size": ask_size,
                    "update_type": "snapshot",
                }
            )
        # create 0-3 aggressive trades in this tick
        n_trades = rng.integers(0, 4)
        for k in range(n_trades):
            side = "buy" if rng.random() < 0.5 else "sell"
            trade_px = float(round(center + rng.normal(0, 0.02), 2))
            trade_sz = float(max(0.01, rng.exponential(2.0)))
            trades_rows.append(
                {"timestamp": t, "price": trade_px, "size": trade_sz, "side": side}
            )
    l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
    trades_df = pd.DataFrame(trades_rows).set_index("timestamp").sort_index()
    return l2_df, trades_df


def run_demo(out_csv="fill_report_week.csv"):
    l2_df, trades_df = synthesize_week_data()
    # persist to tmp parquet for L2Replay
    tmp = "tmp_l2data"
    os.makedirs(tmp, exist_ok=True)
    l2_df.to_parquet(tmp + "/l2_diffs.parquet")
    trades_df.to_parquet(tmp + "/trades.parquet")

    replay = L2Replay(l2_df, trades_df)
    exec_sim = ExecutionSimulatorL2(replay)

    # naive strategy: each hour place a small limit at best_bid to buy, and at best_ask to sell alternately
    orders = []
    idx_hours = sorted(list(set(l2_df.index.floor("H"))))
    for i, ts in enumerate(idx_hours):
        tob = replay.top_of_book_at(ts)
        if tob["best_bid"] is None:
            continue
        # alternate between posting buy and sell
        if i % 2 == 0:
            order = {
                "id": f"o{i}",
                "timestamp": ts,
                "side": "long",
                "price": float(tob["best_bid"]),
                "qty": 1.0,
            }
        else:
            order = {
                "id": f"o{i}",
                "timestamp": ts,
                "side": "short",
                "price": float(tob["best_ask"]),
                "qty": 1.0,
            }
        res = exec_sim.place_limit(order)
        orders.append(res)
    report = exec_sim.export_fill_report()
    report.to_csv(out_csv, index=False)
    print("Wrote fill-quality CSV:", out_csv)
    # start heartbeat thread and write final pnl (best-effort)
    try:
        hb_stop = threading.Event()

        def _hb_loop():
            while not hb_stop.is_set():
                try:
                    if write_heartbeat:
                        write_heartbeat()
                except Exception:
                    pass
                hb_stop.wait(10.0)

        if write_heartbeat:
            t = threading.Thread(target=_hb_loop, daemon=True)
            t.start()
    except Exception:
        pass

    try:
        if write_pnl:
            write_pnl(0.0, float(os.environ.get("DAY_START_EQUITY", "10000")))
    except Exception:
        pass
    return report


if __name__ == "__main__":
    run_demo("fill_report_week.csv")
