from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Optional YAML for settings
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from research.detectors import find_sweeps


def load_smc_params(settings_path: str | None = None) -> Dict[str, Any]:
    """
    Load SMC sweep params from settings.yaml if available, else provide defaults.
    """
    defaults = {"sweep_lookback": 20, "wick_ratio": 0.5, "vol_burst_z": 1.5}
    if settings_path is None:
        settings_path = os.environ.get("SMC_SETTINGS_PATH") or os.path.join(
            "config", "settings.yaml"
        )
    if yaml is None:
        return defaults
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            smc = (cfg or {}).get("smc", {})
            return {
                "sweep_lookback": int(
                    smc.get("sweep_lookback", defaults["sweep_lookback"])
                ),
                "wick_ratio": float(smc.get("wick_ratio", defaults["wick_ratio"])),
                "vol_burst_z": float(smc.get("vol_burst_z", defaults["vol_burst_z"])),
            }
    except Exception:
        pass
    return defaults


def synth_bars(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    """
    Synthesize a simple OHLCV minute-bar DataFrame for testing the detector.
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


def derive_direction(df: pd.DataFrame) -> pd.Series:
    """
    Derive a naive direction label from wick sizes at each bar:
      - If upper wick >= lower wick: 'short' else 'long'
    """
    wk_up = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0.0)
    wk_dn = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0.0)
    return np.where(wk_up >= wk_dn, "short", "long")


def score_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window: int = 10,
    move_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Simple forward-return scoring:
      - For 'short': success if forward min <= close_t - move_threshold within window
      - For 'long':  success if forward max >= close_t + move_threshold within window
    Returns a copy of events with columns: close_t, fut_max, fut_min, success
    """
    if events is None or len(events) == 0:
        return events

    # Derive fallback direction where missing
    fallback_dir = derive_direction(df)
    rows: List[Dict[str, Any]] = []
    for ts, ev in events.iterrows():
        if ts not in df.index:
            continue
        side = ev.get("direction")
        if side is None or (isinstance(side, float) and pd.isna(side)):
            side = fallback_dir.loc[ts]
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
        description="Check SMC sweep strategy signals and basic forward scoring."
    )
    ap.add_argument(
        "--n", type=int, default=2000, help="Number of synthetic minutes to generate."
    )
    ap.add_argument("--seed", type=int, default=11, help="Base RNG seed for bars.")
    ap.add_argument("--lookback", type=int, help="Sweep detector lookback override.")
    ap.add_argument(
        "--wick-ratio", type=float, help="Sweep detector wick ratio override."
    )
    ap.add_argument(
        "--vol-burst-z",
        type=float,
        help="Sweep detector volume burst Z-score override.",
    )
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
        help="Save events with scoring to output-dir/strategy_events.csv",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load params and apply overrides
    smc = load_smc_params()
    if args.lookback is not None:
        smc["sweep_lookback"] = int(args.lookback)
    if args.wick_ratio is not None:
        smc["wick_ratio"] = float(args.wick_ratio)
    if args.vol_burst_z is not None:
        smc["vol_burst_z"] = float(args.vol_burst_z)

    # Synthesize bars and detect events
    df = synth_bars(n=args.n, seed=args.seed)
    events = find_sweeps(
        df,
        lookback=int(smc["sweep_lookback"]),
        wick_ratio=float(smc["wick_ratio"]),
        vol_burst_z=float(smc["vol_burst_z"]),
    )

    total = int(len(df))
    ev_n = int(len(events)) if events is not None else 0
    print(
        f"Bars: {total} | Detected events: {ev_n} (lookback={smc['sweep_lookback']}, "
        f"wick_ratio={smc['wick_ratio']}, vol_burst_z={smc['vol_burst_z']})"
    )

    if ev_n:
        # Direction distribution
        dir_series = (
            events["direction"]
            if "direction" in events.columns
            else derive_direction(df).loc[events.index]
        )
        dir_counts = (
            pd.Series(dir_series)
            .astype(str)
            .str.lower()
            .map(lambda s: "long" if s == "long" else "short")
            .value_counts()
        )
        print("Direction counts:", dict(dir_counts))

        # Score events
        scored = score_events(
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

        # Save CSV if requested
        if args.save_csv:
            out_csv = os.path.join(args.output_dir, "strategy_events.csv")
            scored.to_csv(out_csv, index=True)
            print(f"Wrote strategy events CSV: {out_csv}")

        # Preview
        print("Sample events:")
        print(
            scored.head(10).to_string() if len(scored) else events.head(10).to_string()
        )
    else:
        print("No events detected. Consider loosening thresholds or increasing n/seed.")


if __name__ == "__main__":
    main()
