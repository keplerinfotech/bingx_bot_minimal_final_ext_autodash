from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_events(path: str) -> pd.DataFrame:
    """
    Load strategy events CSV produced by scripts/check_strategy.py --save-csv.
    Expects columns: direction, close_t, fut_max, fut_min (others are ignored).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Events CSV not found: {path}")
    df = pd.read_csv(path)
    # Basic sanity
    required = {"direction", "close_t", "fut_max", "fut_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def score_tp_sl(df: pd.DataFrame, tp: float, sl: float) -> Tuple[int, int, int, int, float, pd.Series]:
    """
    Score events with TP/SL using forward extremes.
    Returns: (total_events, usable, wins, losses, win_rate, returns_series)
    returns_series contains fractional returns for usable events (wins -> +tp, losses -> -sl).
    """
    side = df["direction"].astype(str).str.lower()
    e = pd.to_numeric(df["close_t"], errors="coerce")
    mx = pd.to_numeric(df["fut_max"], errors="coerce")
    mn = pd.to_numeric(df["fut_min"], errors="coerce")

    valid = e.notna() & mx.notna() & mn.notna()
    L = valid & (side == "long")
    S = valid & (side != "long")

    # Hit/stop logic using forward extremes
    hit = (L & (mx >= e * (1.0 + tp))) | (S & (mn <= e * (1.0 - tp)))
    stop = (L & (mn <= e * (1.0 - sl))) | (S & (mx >= e * (1.0 + sl)))

    wins = (hit & ~stop)
    losses = (stop & ~hit)

    n_wins = int(wins.sum())
    n_losses = int(losses.sum())
    usable = n_wins + n_losses
    total = int(len(df))
    win_rate = (n_wins / usable) if usable else float("nan")

    # Build returns series only for usable rows (wins -> +tp, losses -> -sl)
    usable_mask = wins | losses
    r = pd.Series(0.0, index=df.index)
    r.loc[wins] = tp
    r.loc[losses] = -sl
    r = r.loc[usable_mask]
    return total, usable, n_wins, n_losses, win_rate, r


def equity_and_mdd(r: pd.Series, equity0: float) -> Tuple[pd.Series, float]:
    """
    Compute equity curve and max drawdown from a series of per-trade returns r.
    """
    if r.empty:
        return pd.Series([equity0]), 0.0
    curve = equity0 * (1.0 + r).cumprod()
    peak = curve.cummax()
    mdd = float((1.0 - (curve / peak)).max())
    return curve, mdd


def main():
    ap = argparse.ArgumentParser(description="Baseline TP/SL scorer for strategy events.")
    ap.add_argument("--events", type=str, default="reports/strategy_events.csv", help="Path to events CSV.")
    ap.add_argument("--tp", type=float, default=0.0015, help="Take-profit as fractional return (default 0.0015 = 0.15%).")
    ap.add_argument("--sl", type=float, default=0.0010, help="Stop-loss as fractional return (default 0.0010 = 0.10%).")
    ap.add_argument("--equity0", type=float, default=10_000.0, help="Starting equity for equity curve.")
    ap.add_argument("--out-eq", type=str, default="reports/eq_baseline.csv", help="Output CSV for equity curve.")
    ap.add_argument("--no-save", action="store_true", help="Do not write equity curve CSV.")
    ap.add_argument("--summary-only", action="store_true", help="Only print summary; do not compute equity/MDD.")
    args = ap.parse_args()

    df = load_events(args.events)
    total, usable, n_wins, n_losses, win_rate, r = score_tp_sl(df, tp=args.tp, sl=args.sl)
    expectancy = float(r.mean()) if len(r) else float("nan")
    unresolved = total - usable

    print(f"events={total} usable={usable} wins={n_wins} losses={n_losses} unresolved={unresolved}")
    print(f"TP={args.tp:.4%} SL={args.sl:.4%} | win_rate={win_rate:.2%} expectancy={expectancy:.5f}")

    if not args.summary_only:
        curve, mdd = equity_and_mdd(r, args.equity0)
        final_eq = float(curve.iloc[-1])
        print(f"final_equity={final_eq:.2f} max_drawdown={mdd:.2%}")
        if not args.no_save:
            os.makedirs(os.path.dirname(args.out_eq), exist_ok=True)
            pd.DataFrame({"equity": curve.reset_index(drop=True)}).to_csv(args.out_eq, index=False)
            print(f"Wrote equity curve: {args.out_eq}")


if __name__ == "__main__":
    main()
