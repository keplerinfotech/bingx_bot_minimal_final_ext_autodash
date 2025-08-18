# python
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from multi_venue import LatencyModel, Router, Venue
from replay.execution_l2 import ExecutionSimulatorL2  # noqa: F401
from replay.l2_replayer import L2Replay  # noqa: F401
from research.detectors import find_sweeps

try:
    # optional runtime telemetry helpers (best-effort)
    from scripts.telemetry_client import write_heartbeat, write_pnl
except Exception:
    try:
        from telemetry_client import write_heartbeat, write_pnl
    except Exception:
        write_heartbeat = None
        write_pnl = None

# Try to import a dashboard builder (prefer dashboard_builder, fall back to dashboard)
build_dashboard = None
try:
    from dashboard_builder import build_dashboard  # type: ignore
except Exception:
    try:
        from dashboard import build_dashboard  # type: ignore
    except Exception:
        build_dashboard = None

# Optional YAML config loader for SMC sweep parameters
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def synthesize_venue(
    name: str,
    n: int = 2000,
    seed: int = 1,
    liquidity_scale: float = 1.0,
    latency_base: float = 20.0,
) -> Tuple[Venue, LatencyModel]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    price = 100 + np.cumsum(rng.normal(0, 0.05, n))

    l2_rows: List[Dict[str, Any]] = []
    trades_rows: List[Dict[str, Any]] = []
    for i, t in enumerate(idx):
        center = float(price[i])
        for lvl in range(1, 6):
            bp = round(center - 0.01 * lvl, 2)
            ap = round(center + 0.01 * lvl, 2)
            bs = float(max(0.1, rng.exponential(10.0) * liquidity_scale))
            asz = float(max(0.1, rng.exponential(8.0) * liquidity_scale))
            l2_rows.append(
                {
                    "timestamp": t,
                    "side": "bid",
                    "price": bp,
                    "size": bs,
                    "update_type": "snapshot",
                }
            )
            l2_rows.append(
                {
                    "timestamp": t,
                    "side": "ask",
                    "price": ap,
                    "size": asz,
                    "update_type": "snapshot",
                }
            )

        ntr = (
            int(rng.integers(0, 4))
            if liquidity_scale < 2.0
            else int(rng.integers(1, 6))
        )
        for _ in range(ntr):
            side = "buy" if rng.random() < 0.5 else "sell"
            px = round(center + rng.normal(0, 0.02), 2)
            sz = float(max(0.01, rng.exponential(2.0) * liquidity_scale))
            trades_rows.append({"timestamp": t, "price": px, "size": sz, "side": side})

    l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
    trades_df = pd.DataFrame(trades_rows).set_index("timestamp").sort_index()

    return Venue(name=name, l2_diffs=l2_df, trades=trades_df), LatencyModel(
        base_ms=latency_base, jitter_ms=5.0
    )


def run_final_replay(output_csv: str | None = None, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_csv = output_csv or os.path.join(output_dir, "final_fill_quality_report.csv")

    v1, lat1 = synthesize_venue(
        "alpha", n=2000, seed=11, liquidity_scale=1.0, latency_base=20.0
    )
    v2, lat2 = synthesize_venue(
        "beta", n=2000, seed=22, liquidity_scale=2.5, latency_base=50.0
    )
    v3, lat3 = synthesize_venue(
        "gamma", n=2000, seed=33, liquidity_scale=0.6, latency_base=10.0
    )
    venues = [v1, v2, v3]
    latency_models = {"alpha": lat1, "beta": lat2, "gamma": lat3}
    router = Router(venues, latency_models=latency_models)

    alpha_px = v1.trades["price"].resample("1T").last().ffill().bfill()

    df = pd.DataFrame(index=alpha_px.index)
    df["close"] = alpha_px
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0

    # Detect sweep events (df is defined above)
    # Local helper to load sweep detector params (settings.yaml or env overrides)
    def _load_smc_params() -> Dict[str, Any]:
        defaults = {"sweep_lookback": 20, "wick_ratio": 0.5, "vol_burst_z": 1.5}
        settings_path = os.environ.get("SMC_SETTINGS_PATH") or os.path.join(
            "config", "settings.yaml"
        )
        cfg_vals = defaults.copy()
        if yaml is not None:
            try:
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}
                    smc = (cfg or {}).get("smc", {})
                    cfg_vals.update(
                        {
                            "sweep_lookback": int(
                                smc.get("sweep_lookback", cfg_vals["sweep_lookback"])
                            ),
                            "wick_ratio": float(
                                smc.get("wick_ratio", cfg_vals["wick_ratio"])
                            ),
                            "vol_burst_z": float(
                                smc.get("vol_burst_z", cfg_vals["vol_burst_z"])
                            ),
                        }
                    )
            except Exception:
                # fall back to defaults if parsing fails
                pass
        # Env var overrides (set by CLI in __main__)
        env_lookback = os.environ.get("SMC_SWEEP_LOOKBACK")
        env_wick = os.environ.get("SMC_WICK_RATIO")
        env_z = os.environ.get("SMC_VOL_BURST_Z")
        if env_lookback:
            try:
                cfg_vals["sweep_lookback"] = int(env_lookback)
            except Exception:
                pass
        if env_wick:
            try:
                cfg_vals["wick_ratio"] = float(env_wick)
            except Exception:
                pass
        if env_z:
            try:
                cfg_vals["vol_burst_z"] = float(env_z)
            except Exception:
                pass
        return cfg_vals

    smc_params = _load_smc_params()
    events = find_sweeps(
        df,
        lookback=int(smc_params["sweep_lookback"]),
        wick_ratio=float(smc_params["wick_ratio"]),
        vol_burst_z=float(smc_params["vol_burst_z"]),
    )
    print(f"Detected sweep events: {len(events)}")

    # Build orders from events
    orders: List[Dict[str, Any]] = []
    for ts, ev in events.iterrows():
        direction = ev.get("direction", "long")
        side = "long" if direction == "long" else "short"
        price = float(ev.get("entry_price", df.loc[ts, "close"]))
        qty = float(ev.get("qty", 1.0))
        oid = f"e-{int(pd.Timestamp(ts).value // 1_000_000)}"
        orders.append(
            {
                "id": oid,
                "timestamp": pd.Timestamp(ts),
                "side": side,
                "price": price,
                "qty": qty,
            }
        )

    # Route & place orders (use per-venue exec_sim to avoid unsupported constructor args)
    fill_records: List[Dict[str, Any]] = []
    for o in orders:
        ts = pd.Timestamp(o["timestamp"])
        side = o["side"]
        price = float(o["price"])
        qty = float(o["qty"])

        venue_name, prob = router.choose(ts, side, price, qty, horizon_ms=60_000)
        venue = next(v for v in venues if v.name == venue_name)

        # Simulate latency on submission
        lat_ms = float(latency_models[venue_name].sample_ms())
        ts_submit = ts + pd.Timedelta(milliseconds=lat_ms)

        order = {
            "id": o["id"],
            "timestamp": ts_submit,
            "side": side,
            "price": price,
            "qty": qty,
        }
        res = venue.exec_sim.place_limit(order)

        rec = {
            "order_id": o["id"],
            "venue": venue_name,
            "orig_ts": ts,
            "submit_ts": ts_submit,
            "side": side,
            "price": price,
            "qty": qty,
            "route_prob": prob,
            "filled": res.get("filled"),
            "fill_price": res.get("fill_price"),
            "filled_qty": res.get("filled_qty"),
            "queue_ahead": res.get("queue_ahead"),
            "agg_consumed": res.get("agg_consumed"),
        }
        fill_records.append(rec)

    # Ensure a stable schema even if there are zero events
    csv_columns = [
        "order_id",
        "venue",
        "orig_ts",
        "submit_ts",
        "side",
        "price",
        "qty",
        "route_prob",
        "filled",
        "fill_price",
        "filled_qty",
        "queue_ahead",
        "agg_consumed",
    ]
    rep_df = pd.DataFrame(fill_records, columns=csv_columns)
    rep_df.to_csv(out_csv, index=False)
    print(f"Wrote final report: {out_csv}")

    # Summary
    try:
        summary = rep_df.groupby("venue")["filled"].agg(["sum", "count", "mean"])
        if summary.size > 0:
            print("Per-venue fill summary:")
            print(summary.to_string())
        else:
            print("Summary: no fills recorded.")
    except Exception:
        print("Summary: no fills recorded.")

    # Build dashboard HTML next to the CSV
    try:
        dash_path = os.path.join(output_dir, "dashboard.html")
        if build_dashboard is not None:
            build_dashboard(out_csv, dash_path)
            print(f"Dashboard generated: {dash_path}")
        else:
            # Minimal inline dashboard if builder is not available
            html = "<html><head><title>Final Fill Quality Report</title></head><body>"
            html += "<h2>Summary</h2>"
            try:
                html += summary.reset_index().to_html(index=False)  # type: ignore[name-defined]
            except Exception:
                html += "<p>No fills recorded.</p>"
            html += "<h2>Sample Fills (first 200)</h2>"
            html += rep_df.head(200).to_html(index=False)
            html += "</body></html>"
            with open(dash_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Dashboard generated: {dash_path}")
    except Exception as e:
        print(f"Dashboard generation failed: {e}")

    return out_csv


if __name__ == "__main__":
    # CLI to override sweep params and output dir at runtime
    import argparse

    parser = argparse.ArgumentParser(
        description="Run final multi-venue L2 replay + router."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for CSV and dashboard outputs.",
    )
    parser.add_argument("--lookback", type=int, help="Sweep detector lookback.")
    parser.add_argument("--wick-ratio", type=float, help="Sweep detector wick ratio.")
    parser.add_argument(
        "--vol-burst-z", type=float, help="Sweep detector volume burst Z-score."
    )
    args = parser.parse_args()

    # Apply CLI overrides via environment variables checked by _load_smc_params()
    if args.lookback is not None:
        os.environ["SMC_SWEEP_LOOKBACK"] = str(args.lookback)
    if args.wick_ratio is not None:
        os.environ["SMC_WICK_RATIO"] = str(args.wick_ratio)
    if args.vol_burst_z is not None:
        os.environ["SMC_VOL_BURST_Z"] = str(args.vol_burst_z)

    # Only run the simulation; dashboard is generated inside run_final_replay
    csv_path = run_final_replay(output_dir=args.output_dir)
    print(f"CSV path: {csv_path}")
