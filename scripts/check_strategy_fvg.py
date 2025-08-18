from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Optional YAML for settings (not strictly needed, kept for parity)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# FVG detector
try:
    from research.fvg import find_fvgs  # type: ignore
except Exception as _e:  # pragma: no cover
    find_fvgs = None  # type: ignore


def synth_bars(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    """
    Synthesize a simple OHLCV minute-bar DataFrame for testing detectors.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    open_ = base + rng.normal(0, 0.01, n)
    close = base + rng.normal(0, 0.01, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.08, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.08, n)
    vol = np.exp(rng.normal(9.5, 0.25, n))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    return df


def derive_direction_from_bar(df: pd.DataFrame, ts: pd.Timestamp) -> str:
    """
    Fallback direction from bar body: 'long' if close >= open else 'short'.
    """
    try:
        row = df.loc[ts]
        return "long" if float(row["close"]) >= float(row["open"]) else "short"
    except Exception:
        return "long"


def score_events_fvg(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window: int = 10,
    move_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Score FVG events using a simple forward-move rule:
      - For 'short': success if forward min <= close_t - move_threshold within window
      - For 'long':  success if forward max >= close_t + move_threshold within window
    Returns a copy of events with columns: direction, close_t, fut_max, fut_min, success
    """
    if events is None or len(events) == 0:
        return events

    rows: List[Dict[str, Any]] = []
    for ts, ev in events.iterrows():
        if ts not in df.index:
            continue

        # Try to resolve direction from event; fallback to bar body direction
        side = (
            ev.get("direction")
            or ev.get("side")
            or ev.get("dir")
            or ("long" if derive_direction_from_bar(df, ts) == "long" else "short")
        )
        side = "long" if str(side).lower() == "long" else "short"

        close_t = float(df.loc[ts, "close"])
        fut = df.loc[ts : ts + pd.Timedelta(minutes=window)]
        fut_max = float(fut["close"].max()) if len(fut) else close_t
        fut_min = float(fut["close"].min()) if len(fut) else close_t

        if side == "short":
            success = fut_min <= close_t - move_threshold
        else:
            success = fut_max >= close_t + move_threshold

        rec = ev.to_dict()
        rec.update(
            {
                "direction": side,
                "close_t": close_t,
                "fut_max": fut_max,
                "fut_min": fut_min,
                "success": bool(success),
            }
        )
        rows.append({"timestamp": ts, **rec})

    out = pd.DataFrame(rows).set_index("timestamp") if rows else events
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Check FVG-based strategy signals and basic forward scoring."
    )
    ap.add_argument(
        "--n", type=int, default=2000, help="Number of synthetic minutes to generate."
    )
    ap.add_argument("--seed", type=int, default=11, help="Base RNG seed for bars.")
    ap.add_argument(
        "--eval-window",
        type=int,
        default=10,
        help="Forward window (minutes) for success scoring.",
    )
    ap.add_argument(
        "--move-threshold",
        type=float,
        default=0.05,
        help="Price move threshold for success scoring.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Where to write outputs (CSV).",
    )
    ap.add_argument(
        "--save-csv",
        action="store_true",
        help="Save events with scoring to output-dir/strategy_events_fvg.csv",
    )
    args = ap.parse_args()

    if find_fvgs is None:
        raise SystemExit(
            "research.fvg.find_fvgs is not available. Please ensure the module is present."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Build bars and detect FVGs
    df = synth_bars(n=args.n, seed=args.seed)

    # Attempt detection with flexible call pattern
    try:
        events = find_fvgs(df)  # type: ignore[call-arg]
    except TypeError:
        # Fallback with a generic kw if required by implementation; safe defaults
        try:
            events = find_fvgs(df, min_gap=0.01)  # type: ignore[call-arg]
        except Exception as _:
            events = None

    ev_n = int(len(events)) if isinstance(events, pd.DataFrame) else 0
    print(f"Bars: {len(df)} | FVG events: {ev_n}")

    if ev_n == 0:
        print(
            "No FVG events detected. Consider increasing n or adjusting detector defaults."
        )
        return

    # Score
    scored = score_events_fvg(
        df, events, window=args.eval_window, move_threshold=args.move_threshold
    )
    hitrate = (
        float(scored["success"].mean())
        if "success" in scored.columns and len(scored)
        else float("nan")
    )
    print(
        f"Success window={args.eval_window}m, threshold={args.move_threshold:.4f} | "
        f"Hitrate: {hitrate:.2%} over {len(scored)} events"
    )

    # Save if requested
    if args.save_csv:
        out_csv = os.path.join(args.output_dir, "strategy_events_fvg.csv")
        scored.to_csv(out_csv, index=True)
        print(f"Wrote FVG strategy events CSV: {out_csv}")

    # Preview
    print("Sample FVG events:")
    print(scored.head(10).to_string() if len(scored) else events.head(10).to_string())


if __name__ == "__main__":
    main()
