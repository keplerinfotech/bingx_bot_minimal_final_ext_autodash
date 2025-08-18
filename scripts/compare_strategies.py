from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

# Detectors
try:
    from research.detectors import find_sweeps  # type: ignore
except Exception:
    find_sweeps = None  # type: ignore

try:
    from research.fvg import find_fvgs  # type: ignore
except Exception:
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


def derive_direction_wick(df: pd.DataFrame, ts: pd.Timestamp) -> str:
    """
    Wick-based fallback direction: 'short' if upper wick >= lower wick else 'long'.
    """
    row = df.loc[ts]
    wk_up = float(row["high"] - max(row["open"], row["close"]))
    wk_dn = float(min(row["open"], row["close"]) - row["low"])
    return "short" if wk_up >= wk_dn else "long"


def derive_direction_body(df: pd.DataFrame, ts: pd.Timestamp) -> str:
    """
    Body-based fallback direction: 'long' if close >= open else 'short'.
    """
    row = df.loc[ts]
    return "long" if float(row["close"]) >= float(row["open"]) else "short"


def score_events_forward(
    df: pd.DataFrame,
    events: pd.DataFrame,
    window: int,
    move_threshold: float,
    event_direction_resolver: Callable[
        [pd.DataFrame, pd.Timestamp, Dict[str, Any]], str
    ],
) -> Tuple[pd.DataFrame, float]:
    """
    Attach close_t, fut_max, fut_min, success to events and compute hit rate.
    Success rule:
      - short: forward min <= close_t - move_threshold
      - long:  forward max >= close_t + move_threshold
    """
    if events is None or len(events) == 0:
        return events, float("nan")

    rows: List[Dict[str, Any]] = []
    for ts, ev in events.iterrows():
        if ts not in df.index:
            continue
        side = event_direction_resolver(df, ts, ev.to_dict())
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

    scored = pd.DataFrame(rows).set_index("timestamp") if rows else events
    hitrate = (
        float(scored["success"].mean())
        if "success" in scored.columns and len(scored)
        else float("nan")
    )
    return scored, hitrate


def tp_sl_expectancy(
    scored: pd.DataFrame, tp: float, sl: float
) -> Tuple[int, int, int, float]:
    """
    Compute baseline TP/SL expectancy using forward extremes:
      - wins: hit & not stop -> +tp
      - losses: stop & not hit -> -sl
    """
    if scored is None or len(scored) == 0:
        return 0, 0, 0, float("nan")

    side = scored["direction"].astype(str).str.lower()
    e = pd.to_numeric(scored["close_t"], errors="coerce")
    mx = pd.to_numeric(scored["fut_max"], errors="coerce")
    mn = pd.to_numeric(scored["fut_min"], errors="coerce")

    valid = e.notna() & mx.notna() & mn.notna()
    L = valid & (side == "long")
    S = valid & (side != "long")

    hit = (L & (mx >= e * (1.0 + tp))) | (S & (mn <= e * (1.0 - tp)))
    stop = (L & (mn <= e * (1.0 - sl))) | (S & (mx >= e * (1.0 + sl)))

    wins = (hit & ~stop).sum()
    losses = (stop & ~hit).sum()
    usable = int(wins + losses)
    expectancy = (
        ((int(wins) * tp) - (int(losses) * sl)) / usable if usable else float("nan")
    )
    return int(usable), int(wins), int(losses), float(expectancy)


def resolve_sweep_dir(df: pd.DataFrame, ts: pd.Timestamp, ev: Dict[str, Any]) -> str:
    """
    Resolve direction for sweep events. Prefer event-provided direction, fallback to wick-based.
    """
    if "direction" in ev and str(ev["direction"]).lower() in ("long", "short"):
        return str(ev["direction"]).lower()
    return derive_direction_wick(df, ts)


def resolve_fvg_dir(df: pd.DataFrame, ts: pd.Timestamp, ev: Dict[str, Any]) -> str:
    """
    Resolve direction for FVG events. Try common keys; fallback to body-based direction.
    """
    for k in ("direction", "side", "dir"):
        if k in ev and str(ev[k]).lower() in ("long", "short"):
            return str(ev[k]).lower()
    return derive_direction_body(df, ts)


def main():
    ap = argparse.ArgumentParser(
        description="Compare SMC sweep vs FVG strategies with identical data & scoring."
    )
    ap.add_argument(
        "--n", type=int, default=2000, help="Number of synthetic minutes to generate."
    )
    ap.add_argument("--seed", type=int, default=11, help="Base RNG seed for bars.")
    ap.add_argument("--lookback", type=int, default=12, help="Sweep detector lookback.")
    ap.add_argument(
        "--wick-ratio", type=float, default=0.25, help="Sweep detector wick ratio."
    )
    ap.add_argument(
        "--vol-burst-z",
        type=float,
        default=1.2,
        help="Sweep detector volume burst Z-score.",
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
        "--tp",
        type=float,
        default=0.0015,
        help="Baseline TP (fractional, 0.0015 = 0.15%).",
    )
    ap.add_argument(
        "--sl",
        type=float,
        default=0.0010,
        help="Baseline SL (fractional, 0.0010 = 0.10%).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional path to write the comparison CSV.",
    )
    args = ap.parse_args()

    if find_sweeps is None:
        raise SystemExit("Sweep detector not available.")
    if find_fvgs is None:
        raise SystemExit("FVG detector not available.")

    # Build shared dataset
    df = synth_bars(n=args.n, seed=args.seed)

    # Detect sweeps
    sweeps = find_sweeps(  # type: ignore[call-arg]
        df,
        lookback=int(args.lookback),
        wick_ratio=float(args.wick_ratio),
        vol_burst_z=float(args.vol_burst_z),
    )
    sweeps_n = int(len(sweeps)) if isinstance(sweeps, pd.DataFrame) else 0
    sweeps_scored, sweeps_hitrate = score_events_forward(
        df,
        sweeps if sweeps_n else pd.DataFrame(index=df.index[:0]),
        window=args.eval_window,
        move_threshold=args.move_threshold,
        event_direction_resolver=resolve_sweep_dir,
    )
    s_usable, s_wins, s_losses, s_exp = tp_sl_expectancy(
        sweeps_scored, args.tp, args.sl
    )

    # Detect FVGs (try default signature then fallback)
    try:
        fvgs = find_fvgs(df)  # type: ignore[call-arg]
    except TypeError:
        try:
            fvgs = find_fvgs(df, min_gap=0.01)  # type: ignore[call-arg]
        except Exception:
            fvgs = None
    fvgs_n = int(len(fvgs)) if isinstance(fvgs, pd.DataFrame) else 0
    fvgs_scored, fvgs_hitrate = score_events_forward(
        df,
        fvgs if fvgs_n else pd.DataFrame(index=df.index[:0]),
        window=args.eval_window,
        move_threshold=args.move_threshold,
        event_direction_resolver=resolve_fvg_dir,
    )
    f_usable, f_wins, f_losses, f_exp = tp_sl_expectancy(fvgs_scored, args.tp, args.sl)

    # Build summary
    rows = [
        {
            "strategy": "sweeps",
            "events": sweeps_n,
            "hitrate": sweeps_hitrate,
            "usable_trades": s_usable,
            "wins": s_wins,
            "losses": s_losses,
            "tp": args.tp,
            "sl": args.sl,
            "expectancy": s_exp,
        },
        {
            "strategy": "fvg",
            "events": fvgs_n,
            "hitrate": fvgs_hitrate,
            "usable_trades": f_usable,
            "wins": f_wins,
            "losses": f_losses,
            "tp": args.tp,
            "sl": args.sl,
            "expectancy": f_exp,
        },
    ]
    comp = pd.DataFrame(rows).set_index("strategy")
    # Pretty print
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nComparison (same seed/data, same scoring):")
        print(
            comp.to_string(
                formatters={"hitrate": "{:.2%}".format, "expectancy": "{:.5f}".format}
            )
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        comp.to_csv(args.out_csv)
        print(f"\nWrote comparison CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
