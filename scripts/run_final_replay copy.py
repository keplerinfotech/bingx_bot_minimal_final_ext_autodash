import pandas as pd
import numpy as np
import random
import time

# Simulated final replay
def run_final_replay():
    # Simulate 50 trades
    venues = ["venueA", "venueB", "venueC"]
    rows = []
    for i in range(50):
        venue = random.choice(venues)
        route_prob = random.uniform(0.4, 0.9)
        filled = random.choice([0, 1])
        filled_qty = random.uniform(0.0, 1.0) if filled else 0.0
        rows.append({
            "order_id": i,
            "venue": venue,
            "orig_ts": time.time(),
            "submit_ts": time.time() + random.uniform(0.01, 0.2),
            "side": random.choice(["buy", "sell"]),
            "price": random.uniform(100, 200),
            "qty": 1.0,
            "route_prob": route_prob,
            "filled": filled,
            "fill_price": random.uniform(100, 200) if filled else None,
            "filled_qty": filled_qty,
            "queue_ahead": random.randint(0, 500),
            "agg_consumed": random.uniform(0, 100),
            "latency_ms": random.uniform(1, 100)
        })
    df = pd.DataFrame(rows)
    csv_path = "final_fill_quality_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"Replay finished, wrote {csv_path}")
    return csv_path

if __name__ == "__main__":
    csv_path = run_final_replay()
    # Auto-generate dashboard
    try:
        from dashboard import build_dashboard
        build_dashboard(csv_path, "dashboard.html")
        print("Dashboard generated: dashboard.html")
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
# python
python scripts/cleanup_and_merge_duplicates.py
