"""
Debug helper: build the bar DataFrame used by the replay, inspect it,
and run the sweep detector with chosen params so we can see why no events
are detected.

Usage (venv active, project root):
    PYTHONPATH=. python scripts/debug_sweep_detector.py --lookback 20 --wick_ratio 0.5 --vol_burst_z 1.5

The script prints dataframe stats, sample rows, and detector output, and
writes a CSV to reports/debug_bars.csv for offline inspection.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# ensure project root is importable
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import pandas as pd  # noqa: E402

try:
    # import helpers from the replay script
    from scripts.run_final_replay import synthesize_venue  # type: ignore
except Exception:
    print("Failed to import synthesize_venue from scripts.run_final_replay.py")
    traceback.print_exc()
    raise

try:
    # import detector (the one used by replay)
    from research.detectors import find_sweeps  # type: ignore
except Exception:
    print("Failed to import find_sweeps from research.detectors")
    traceback.print_exc()
    raise


def build_bars_from_alpha(seed: int = 11, n: int = 2000) -> pd.DataFrame:
    """Create LTF bars the same way run_final_replay does using alpha venue trades."""
    v1, _ = synthesize_venue(
        "alpha", n=n, seed=seed, liquidity_scale=1.0, latency_base=20.0
    )
    # Ensure trades exist and have 'price'
    if "price" not in v1.trades.columns:
        raise RuntimeError("synthesized trades missing 'price' column")
    alpha_px = v1.trades["price"].resample("1T").last().ffill().bfill()
    df = pd.DataFrame(index=alpha_px.index)
    df["close"] = alpha_px
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0
    return df


def inspect_df(df: pd.DataFrame) -> None:
    """Print quick diagnostics about the bar df."""
    print("\n--- DataFrame info ---")
    print(df.info())
    print("\n--- Head (first 10) ---")
    print(df.head(10).to_string())
    print("\n--- Tail (last 10) ---")
    print(df.tail(10).to_string())
    print("\n--- Stats ---")
    print(df[["open", "high", "low", "close"]].describe().to_string())

    # compute some derived metrics helpful for sweep detection
    df2 = df.copy()
    df2["range"] = df2["high"] - df2["low"]
    df2["body"] = (df2["close"] - df2["open"]).abs()
    df2["wick_prop"] = (df2["range"] - df2["body"]) / df2["range"].replace(0, 1)
    print("\n--- Derived metrics ---")
    print(df2[["range", "body", "wick_prop"]].describe().to_string())

    # percent of bars with tiny range
    tiny = (df2["range"] < 0.0001).mean() * 100
    print(f"\nPercent tiny-range bars (<0.0001): {tiny:.2f}%")

    # save csv for offline inspection
    os.makedirs("reports", exist_ok=True)
    outp = "reports/debug_bars.csv"
    df.to_csv(outp)
    print(f"\nSaved bars CSV to: {outp}")


def run_detector_and_report(
    df: pd.DataFrame,
    lookback: int,
    wick_ratio: float,
    vol_burst_z: float,
    require_reject: bool,
) -> None:
    """Run the detector and print summary of results."""
    print(
        f"\nRunning find_sweeps with lookback={lookback}, wick_ratio={wick_ratio}, vol_burst_z={vol_burst_z}, require_reject={require_reject}"
    )
    try:
        events = find_sweeps(
            df,
            lookback=lookback,
            wick_ratio=wick_ratio,
            vol_burst_z=vol_burst_z,
            require_reject=require_reject,
        )
    except TypeError:
        # some detectors accept positional args; try fallback without require_reject
        events = find_sweeps(
            df, lookback=lookback, wick_ratio=wick_ratio, vol_burst_z=vol_burst_z
        )

    if events is None:
        print("Detector returned None")
        return

    print(f"Detected sweep events: {len(events)}")
    if len(events) > 0:
        print("\n--- Events head (first 10) ---")
        print(events.head(10).to_string())
    else:
        print("No events produced by detector.")


def parse_args():
    p = argparse.ArgumentParser(description="Debug sweep detector on synthesized bars")
    p.add_argument("--lookback", type=int, default=20)
    p.add_argument("--wick_ratio", type=float, default=0.5)
    p.add_argument("--vol_burst_z", type=float, default=1.5)
    p.add_argument(
        "--require_reject",
        type=lambda s: s.lower() in ("true", "1", "yes", "y"),
        default=True,
        help="set to false to disable reject requirement (use 'false')",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=11,
        help="seed passed to synthesize_venue for reproducibility",
    )
    p.add_argument(
        "--n", type=int, default=2000, help="number of minutes of synthetic data"
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = build_bars_from_alpha(seed=args.seed, n=args.n)
    inspect_df(df)
    run_detector_and_report(
        df, args.lookback, args.wick_ratio, args.vol_burst_z, args.require_reject
    )


if __name__ == "__main__":
    main()
