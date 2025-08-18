# ... existing code ...
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
