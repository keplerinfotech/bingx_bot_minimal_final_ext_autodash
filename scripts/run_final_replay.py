"""Very small run_final_replay used by tests.

Produces an empty CSV with required columns and a simple dashboard HTML.
"""

from __future__ import annotations

import os
import threading
from typing import List

import pandas as pd

try:
    from scripts.telemetry_client import write_heartbeat, write_pnl
except Exception:
    try:
        from telemetry_client import write_heartbeat, write_pnl
    except Exception:
        write_heartbeat = None
        write_pnl = None


def run_final_replay(
    output_csv: str | None = None,
    output_dir: str = "reports",
    settings_path: str | None = None,
) -> str:
    """Create an empty report CSV with expected columns and a simple dashboard.

    This minimal implementation is intentionally small so unit tests can import
    and call it without requiring the full replay stack.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_csv = output_csv or os.path.join(output_dir, "final_fill_quality_report.csv")

    csv_columns: List[str] = [
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

    df = pd.DataFrame(columns=csv_columns)
    df.to_csv(out_csv, index=False)

    dash_path = os.path.join(output_dir, "dashboard.html")
    html = "<html><head><title>Final Fill Quality Report</title></head><body>"
    html += "<h2>Summary</h2>"
    html += "<p>No fills recorded.</p>"
    html += "<h2>Sample Fills</h2>"
    html += df.head(200).to_html(index=False)
    html += "</body></html>"
    with open(dash_path, "w", encoding="utf-8") as f:
        f.write(html)

    # start heartbeat thread for monitors (best-effort)
    hb_stop = threading.Event()

    def _hb_loop():
        while not hb_stop.is_set():
            try:
                if write_heartbeat:
                    write_heartbeat()
            except Exception:
                pass
            hb_stop.wait(10.0)

    hb_thread = None
    try:
        if write_heartbeat:
            hb_thread = threading.Thread(target=_hb_loop, daemon=True)
            hb_thread.start()
    except Exception:
        hb_thread = None

    # write a final PnL snapshot for monitors (best-effort)
    try:
        if write_pnl:
            write_pnl(0.0, float(os.environ.get("DAY_START_EQUITY", "10000")))
    except Exception:
        pass

    return out_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    print(run_final_replay(output_dir=args.output_dir))
