# ... existing code ...


def run_final_replay(output_csv: str | None = None, output_dir: str = "reports") -> str:
    """
    Runs the final multi-venue L2 replay + routing simulation and writes a CSV report.
    # ... existing code ...
    """
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    out_csv = output_csv or os.path.join(output_dir, "final_fill_quality_report.csv")

    # Helper: load sweep detector params from settings.yaml (env SMC_SETTINGS_PATH overrides)
    def _load_smc_params() -> Dict[str, Any]:
        defaults = {"sweep_lookback": 20, "wick_ratio": 0.5, "vol_burst_z": 1.5}
        settings_path = os.environ.get("SMC_SETTINGS_PATH") or os.path.join(
            "config", "settings.yaml"
        )
        cfg_vals = defaults.copy()
        # ... existing code ...
        return cfg_vals

    # Create 3 venues and router
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

    # Build LTF bar df used by sweep detector (use venue alpha price for bars)
    alpha_px = v1.trades["price"].resample("1min").last().ffill().bfill()

    df = pd.DataFrame(index=alpha_px.index)
    df["close"] = alpha_px
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0

    # Detect sweep events (df is defined above)
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

    # ... existing code ...
    print(f"Dashboard generated: {dash_path}")

    return out_csv


# ... existing code ...
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
