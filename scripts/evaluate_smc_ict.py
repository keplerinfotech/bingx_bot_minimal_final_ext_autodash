# scripts/evaluate_smc_ict.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import re

# Optional YAML for settings (not required but kept for parity)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Optional project detectors
try:
    from research.detectors import find_sweeps  # type: ignore
except Exception:
    find_sweeps = None  # type: ignore

try:
    from research.fvg import find_fvgs  # type: ignore
except Exception:
    find_fvgs = None  # type: ignore


@dataclass
class EntryConfig:
    sweep_lookback: int = 12
    wick_ratio: float = 0.25
    vol_burst_z: float = 1.2
    fvg_min_gap: float = 0.0  # set >0 to enforce minimum gap (price units)
    bos_lookback: int = 10
    pd_lookback: int = 20  # swing window for premium/discount calc
    htf_minutes: int = 60   # HTF sample window for bias
    use_killzones: bool = True
    sweep_window_minutes: int = 10  # sweep must occur within last X minutes
    forward_minutes: int = 60       # simulation horizon


@dataclass
class ExitConfig:
    partial_at_r: float = 1.0  # take partial profits at 1R
    partial_frac: float = 0.5  # fraction closed at partial
    max_horizon_minutes: int = 60
    # Liquidity targeting preference: "swing", "session", "fvg_fill"
    target: str = "swing"


def synth_bars(n: int = 2000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    base = 100 + np.cumsum(rng.normal(0, 0.05, n))
    open_ = base + rng.normal(0, 0.01, n)
    close = base + rng.normal(0, 0.01, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.08, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.08, n)
    vol = np.exp(rng.normal(9.5, 0.25, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def in_killzone(ts: pd.Timestamp) -> bool:
    # London 2–5, NY AM 7–10, NY PM 12–14 (assume naive tz for synthetic data)
    h = ts.hour
    return (2 <= h < 5) or (7 <= h < 10) or (12 <= h < 14)


def rolling_swing(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    swing_hi = df["high"].rolling(lookback, min_periods=1).max().shift(1)
    swing_lo = df["low"].rolling(lookback, min_periods=1).min().shift(1)
    return swing_hi, swing_lo


def detect_bos_choch(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    swing_hi, swing_lo = rolling_swing(df, lookback)
    bull_bos = df["close"] > swing_hi  # break above prior swing high
    bear_bos = df["close"] < swing_lo  # break below prior swing low
    return bull_bos.fillna(False), bear_bos.fillna(False)


def detect_ob(df: pd.DataFrame, ts_idx: int, side: str, lookback: int = 10) -> Tuple[float, float]:
    """
    Simple OB detection: for long, last bearish candle body (open>close) before ts within lookback;
    for short, last bullish (close>open). Returns (ob_low, ob_high).
    """
    start = max(0, ts_idx - lookback)
    window = df.iloc[start:ts_idx]
    if side == "long":
        candidates = window[window["open"] > window["close"]]
        if len(candidates):
            ob = candidates.iloc[-1]
            return float(min(ob["open"], ob["close"], ob["low"])), float(max(ob["open"], ob["close"], ob["high"]))
    else:
        candidates = window[window["close"] > window["open"]]
        if len(candidates):
            ob = candidates.iloc[-1]
            return float(min(ob["open"], ob["close"], ob["low"])), float(max(ob["open"], ob["close"], ob["high"]))
    # Fallback: tiny zone around prior close
    ref = df.iloc[ts_idx - 1]["close"] if ts_idx > 0 else df.iloc[0]["close"]
    return float(ref - 0.01), float(ref + 0.01)


def htf_bias(df: pd.DataFrame, minutes: int = 60) -> pd.Series:
    # Approximate HTF trend by SMA slope over given minutes window
    win = minutes
    sma = df["close"].rolling(win, min_periods=1).mean()
    slope = sma - sma.shift(1)
    return slope.fillna(0.0)  # >0 bullish, <0 bearish


def premium_discount(df: pd.DataFrame, lookback: int) -> pd.Series:
    hi = df["high"].rolling(lookback, min_periods=1).max()
    lo = df["low"].rolling(lookback, min_periods=1).min()
    mid = (hi + lo) / 2.0
    # discount if close < mid; premium if > mid
    return df["close"] - mid  # negative => discount, positive => premium


def recent_sweep_mask(df: pd.DataFrame, params: EntryConfig) -> pd.Series:
    if find_sweeps is None:
        # Fallback: large wick heuristic as proxy
        wk_up = (df["high"] - df[["open", "close"]].max(axis=1))
        wk_dn = (df[["open", "close"]].min(axis=1) - df["low"])
        tr = (df["high"] - df["low"]).replace(0, np.nan)
        big_wick = ((wk_up / tr >= params.wick_ratio) | (wk_dn / tr >= params.wick_ratio)).fillna(False)
        return big_wick.rolling(params.sweep_window_minutes, min_periods=1).max().astype(bool)
    try:
        ev = find_sweeps(df, lookback=params.sweep_lookback, wick_ratio=params.wick_ratio, vol_burst_z=params.vol_burst_z)  # type: ignore
    except Exception:
        return pd.Series(False, index=df.index)
    if ev is None or len(ev) == 0:
        return pd.Series(False, index=df.index)
    # Mark sweeps within window minutes back from each ts
    mask = pd.Series(False, index=df.index)
    sweep_idx = ev.index
    # For efficiency, mark with rolling window
    for ts in sweep_idx:
        win_end = ts + pd.Timedelta(minutes=params.sweep_window_minutes)
        mask.loc[ts:win_end] = True
    return mask


def fvg_at_bar(df: pd.DataFrame, ts: pd.Timestamp, min_gap: float = 0.0) -> Optional[Dict[str, Any]]:
    if find_fvgs is None:
        # Fallback: displacement candle: body > median of recent bodies and gaps on one side
        i = df.index.get_indexer([ts])[0]
        if i < 2:
            return None
        o1, c1 = df.iloc[i - 2][["open", "close"]]
        o2, c2 = df.iloc[i - 1][["open", "close"]]
        o3, c3 = df.iloc[i][["open", "close"]]
        # Simple check: bullish gap between prior high and current low or vice versa
        gap_up = (min(o2, c2) > max(o1, c1)) and (min(o3, c3) > max(o2, c2))
        gap_dn = (max(o2, c2) < min(o1, c1)) and (max(o3, c3) < min(o2, c2))
        if gap_up or gap_dn:
            return {"direction": "long" if gap_up else "short", "mid": float((df.iloc[i]["high"] + df.iloc[i]["low"]) / 2.0)}
        return None
    try:
        events = find_fvgs(df)  # type: ignore
    except TypeError:
        try:
            events = find_fvgs(df, min_gap=min_gap)  # type: ignore
        except Exception:
            events = None
    if events is None or len(events) == 0:
        return None
    if ts in events.index:
        ev = events.loc[ts].to_dict() if hasattr(events.loc[ts], "to_dict") else {}
        # Compute a midpoint if not present
        ev.setdefault("mid", float((df.loc[ts, "high"] + df.loc[ts, "low"]) / 2.0))
        return ev
    return None


def simulate_path(df: pd.DataFrame, start_idx: int, side: str, entry: float, sl: float, tp: float, horizon: int) -> Tuple[str, int, float]:
    """
    Iterate forward bar-by-bar up to horizon to see which is hit first: SL or TP.
    Returns (result: 'tp'/'sl'/'timeout', bars_elapsed, exit_price).
    """
    for k in range(1, horizon + 1):
        if start_idx + k >= len(df):
            break
        hi = float(df.iloc[start_idx + k]["high"])
        lo = float(df.iloc[start_idx + k]["low"])
        if side == "long":
            # First touch logic: SL if low <= sl; TP if high >= tp
            if lo <= sl:
                return "sl", k, sl
            if hi >= tp:
                return "tp", k, tp
        else:
            if hi >= sl:
                return "sl", k, sl
            if lo <= tp:
                return "tp", k, tp
    return "timeout", horizon, float(df.iloc[min(start_idx + horizon, len(df) - 1)]["close"])


def evaluate(df: pd.DataFrame, econf: EntryConfig, xconf: ExitConfig) -> pd.DataFrame:
    s_hi, s_lo = rolling_swing(df, econf.bos_lookback)
    bull_bos, bear_bos = detect_bos_choch(df, econf.bos_lookback)
    pd_offset = premium_discount(df, econf.pd_lookback)
    bias = htf_bias(df, econf.htf_minutes)
    sweep_recent = recent_sweep_mask(df, econf)

    trades: List[Dict[str, Any]] = []

    for i, ts in enumerate(df.index):
        # Killzone filter
        if econf.use_killzones and not in_killzone(ts):
            continue

        # HTF alignment filter
        trend = "long" if bias.iloc[i] > 0 else ("short" if bias.iloc[i] < 0 else "neutral")
        if trend == "neutral":
            continue

        # Premium/Discount filter
        in_discount = pd_offset.iloc[i] < 0.0
        in_premium = pd_offset.iloc[i] > 0.0

        # Require recent sweep
        if not sweep_recent.iloc[i]:
            continue

        # Require FVG at bar (direction may be supplied)
        fvg = fvg_at_bar(df, ts, min_gap=econf.fvg_min_gap)
        if fvg is None:
            continue

        # BOS/CHOCH agreement
        bull = bool(bull_bos.iloc[i])
        bear = bool(bear_bos.iloc[i])

        # Resolve side: intersection of HTF trend, FVG direction, BOS, PD
        fvg_dir = str(fvg.get("direction", trend)).lower()
        side: Optional[str] = None
        if trend == "long" and fvg_dir == "long" and bull and in_discount:
            side = "long"
        if trend == "short" and fvg_dir == "short" and bear and in_premium:
            side = "short"
        if side is None:
            continue

        # Entry/SL/TP determination
        entry = float(fvg.get("mid", (df.iloc[i]["high"] + df.iloc[i]["low"]) / 2.0))
        ob_low, ob_high = detect_ob(df, i, side, lookback=max(10, econf.bos_lookback))
        if side == "long":
            sl = min(ob_low, float(df.iloc[i]["low"]), float(s_lo.iloc[i]))
            tp = float(s_hi.iloc[i])  # target prior swing high (liquidity)
        else:
            sl = max(ob_high, float(df.iloc[i]["high"]), float(s_hi.iloc[i]))
            tp = float(s_lo.iloc[i])  # target prior swing low

        if not np.isfinite(sl) or not np.isfinite(tp) or not np.isfinite(entry):
            continue

        # Ensure R > 0 and realistic
        r_denom = (entry - sl) if side == "long" else (sl - entry)
        if r_denom <= 0:
            continue
        r_target = (tp - entry) / r_denom if side == "long" else (entry - tp) / r_denom
        if r_target <= 0.1:
            # skip too-close targets
            continue

        # Simulate
        result, bars, exit_price = simulate_path(
            df, i, side, entry=entry, sl=sl, tp=tp, horizon=min(econf.forward_minutes, xconf.max_horizon_minutes)
        )

        # Partial at 1R handling: if full TP hit -> partial applied automatically
        # For "timeout", compute partial outcome if 1R hit along the way
        # Re-simulate 1R touch
        one_r = entry + r_denom if side == "long" else entry - r_denom
        r_hit, _, _ = simulate_path(df, i, side, entry=entry, sl=sl, tp=one_r, horizon=min(econf.forward_minutes, xconf.max_horizon_minutes))
        took_partial = r_hit == "tp"

        # Compute realized R with partials
        if result == "tp":
            realized_r = xconf.partial_frac * 1.0 + (1.0 - xconf.partial_frac) * min(r_target, 3.0)  # cap runner R to 3R
        elif result == "sl":
            realized_r = -1.0
        else:
            # timeout: if partial taken, hold remainder to exit_price
            if took_partial:
                remainder_move = (exit_price - entry) / r_denom if side == "long" else (entry - exit_price) / r_denom
                realized_r = xconf.partial_frac * 1.0 + (1.0 - xconf.partial_frac) * remainder_move
            else:
                remainder_move = (exit_price - entry) / r_denom if side == "long" else (entry - exit_price) / r_denom
                realized_r = remainder_move

        trades.append(
            {
                "timestamp": ts,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "result": result,
                "bars": bars,
                "r": realized_r,
                "r_target": r_target,
                "htf_trend": trend,
                "pd_offset": float(pd_offset.iloc[i]),
                "killzone": in_killzone(ts),
            }
        )

    return pd.DataFrame(trades).set_index("timestamp") if trades else pd.DataFrame(columns=["timestamp"]).set_index("timestamp")


# -------------------------
# New wrapper: run_backtest
# -------------------------
def run_backtest(
    n: int = 2000,
    seed: int = 123,
    lookback: int = 12,
    wick_ratio: float = 0.25,
    vol_burst_z: float = 1.2,
    fvg_min_gap: float = 0.0,
    bos_lookback: int = 10,
    pd_lookback: int = 20,
    htf_minutes: int = 60,
    no_killzones: bool = False,
    sweep_window: int = 10,
    forward_minutes: int = 60,
    partial_at_r: float = 1.0,
    partial_frac: float = 0.5,
    target: str = "swing",
    tp_override: Optional[float] = None,   # decimal ratio, e.g. 0.0015
    sl_override: Optional[float] = None,   # decimal ratio
    out: str = "reports/smc_ict_trades.csv",
) -> Dict[str, Any]:
    """
    Runs the full pipeline and returns a results dictionary containing:
      - trades_df (DataFrame)
      - metrics: events, usable, wins, losses, win_rate, expectancy, final_equity, max_drawdown, profit_factor
    If tp_override/sl_override provided, recomputes realized R by resimulating with those price levels.
    """

    econf = EntryConfig(
        sweep_lookback=lookback,
        wick_ratio=wick_ratio,
        vol_burst_z=vol_burst_z,
        fvg_min_gap=fvg_min_gap,
        bos_lookback=bos_lookback,
        pd_lookback=pd_lookback,
        htf_minutes=htf_minutes,
        use_killzones=not no_killzones,
        sweep_window_minutes=sweep_window,
        forward_minutes=forward_minutes,
    )
    xconf = ExitConfig(
        partial_at_r=partial_at_r,
        partial_frac=partial_frac,
        max_horizon_minutes=forward_minutes,
        target=target,
    )

    df = synth_bars(n=n, seed=seed)
    trades_df = evaluate(df, econf, xconf)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # If no trades found - write empty CSV and return minimal metrics
    if trades_df.empty:
        # Write empty file with no trades
        trades_df.to_csv(out, index=True)
        return {
            "trades_df": trades_df,
            "events": 0,
            "usable": 0,
            "wins": 0,
            "losses": 0,
            "unresolved": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "final_equity": float("nan"),
            "max_drawdown": float("nan"),
            "profit_factor": None,
        }

    # If tp/sl override is provided, recompute r for each trade by re-simulating
    if tp_override is not None or sl_override is not None:
        # iterate through trades in chronological order
        recomputed_rs: List[float] = []
        recomputed_results: List[str] = []
        recomputed_exit_prices: List[float] = []
        for ts, row in trades_df.iterrows():
            try:
                start_idx = df.index.get_loc(ts)
            except KeyError:
                # If timestamp not found in synthetic df, skip recompute (shouldn't happen)
                recomputed_rs.append(row.get("r", 0.0))
                recomputed_results.append(row.get("result", "timeout"))
                recomputed_exit_prices.append(row.get("tp") if row.get("result") == "tp" else row.get("sl"))
                continue

            entry = float(row["entry"])
            side = str(row["side"])

            # Base sl/tp from evaluate
            base_sl = float(row["sl"])
            base_tp = float(row["tp"])

            # Compute overridden absolute prices if ratios provided
            if tp_override is not None:
                if side == "long":
                    tp_price = entry * (1.0 + tp_override)
                else:
                    tp_price = entry * (1.0 - tp_override)
            else:
                tp_price = base_tp

            if sl_override is not None:
                if side == "long":
                    sl_price = entry * (1.0 - sl_override)
                else:
                    sl_price = entry * (1.0 + sl_override)
            else:
                sl_price = base_sl

            # Ensure horizon
            horizon = min(econf.forward_minutes, xconf.max_horizon_minutes)

            # Re-simulate path with overridden levels
            result, bars, exit_price = simulate_path(df, start_idx, side, entry=entry, sl=sl_price, tp=tp_price, horizon=horizon)

            # compute realized r similar to earlier evaluate logic
            r_denom = (entry - sl_price) if side == "long" else (sl_price - entry)
            if r_denom <= 0 or not np.isfinite(r_denom):
                realized_r = 0.0
            else:
                r_target = (tp_price - entry) / r_denom if side == "long" else (entry - tp_price) / r_denom
                # re-simulate one-R
                one_r_price = entry + r_denom if side == "long" else entry - r_denom
                r_hit_one, _, _ = simulate_path(df, start_idx, side, entry=entry, sl=sl_price, tp=one_r_price, horizon=horizon)
                took_partial = r_hit_one == "tp"

                if result == "tp":
                    realized_r = xconf.partial_frac * 1.0 + (1.0 - xconf.partial_frac) * min(r_target, 3.0)
                elif result == "sl":
                    realized_r = -1.0
                else:
                    # timeout
                    remainder_move = (exit_price - entry) / r_denom if side == "long" else (entry - exit_price) / r_denom
                    if took_partial:
                        realized_r = xconf.partial_frac * 1.0 + (1.0 - xconf.partial_frac) * remainder_move
                    else:
                        realized_r = remainder_move

            recomputed_rs.append(float(realized_r))
            recomputed_results.append(result)
            recomputed_exit_prices.append(float(exit_price))

        # overwrite r/result/tp/sl in trades_df
        trades_df = trades_df.copy()
        trades_df["r"] = recomputed_rs
        trades_df["result"] = recomputed_results
        # Update tp/sl to overridden absolute prices for clarity
        # If an override was provided, set tp/sl columns accordingly per trade
        for idx, ts in enumerate(trades_df.index):
            side = trades_df.loc[ts, "side"]
            entry = float(trades_df.loc[ts, "entry"])
            if tp_override is not None:
                trades_df.at[ts, "tp"] = float(entry * (1.0 + tp_override)) if side == "long" else float(entry * (1.0 - tp_override))
            if sl_override is not None:
                trades_df.at[ts, "sl"] = float(entry * (1.0 - sl_override)) if side == "long" else float(entry * (1.0 + sl_override))

    # Compute metrics
    total_events = len(trades_df)
    wins = int((trades_df["r"] > 0).sum())
    losses = int((trades_df["r"] < 0).sum())
    usable = int((trades_df["r"].notna()).sum())  # all realized (we use all rows)
    unresolved = total_events - (wins + losses)
    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else float("nan")
    expectancy = float(trades_df["r"].mean()) if total_events > 0 else float("nan")

    # equity curve and drawdown (assume starting equity)
    initial_equity = 10000.0
    cum_r = trades_df["r"].fillna(0.0).cumsum()
    equity_curve = initial_equity * (1.0 + cum_r)
    final_equity = float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else initial_equity

    # max drawdown percent
    peak = equity_curve.cummax()
    drawdowns = (peak - equity_curve) / peak
    max_drawdown = float(drawdowns.max() * 100.0) if len(drawdowns) > 0 else 0.0

    # profit factor: sum(gains) / abs(sum(losses))
    positive = trades_df.loc[trades_df["r"] > 0, "r"].sum()
    negative = trades_df.loc[trades_df["r"] < 0, "r"].sum()
    profit_factor = None
    if negative == 0 and positive > 0:
        profit_factor = float("inf")
    elif negative == 0 and positive == 0:
        profit_factor = None
    else:
        profit_factor = float(positive / abs(negative)) if negative != 0 else None

    # Write trades CSV (preserve original behaviour)
    trades_df.to_csv(out, index=True)

    return {
        "trades_df": trades_df,
        "events": total_events,
        "usable": usable,
        "wins": wins,
        "losses": losses,
        "unresolved": unresolved,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "final_equity": final_equity,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "out_path": out,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate SMC/ICT combined strategy with full entry/exit checklist.")
    ap.add_argument("--n", type=int, default=5000, help="Number of synthetic minutes to generate.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    # Sweep detector settings
    ap.add_argument("--lookback", type=int, default=12, help="Sweep detector lookback.")
    ap.add_argument("--wick-ratio", type=float, default=0.25, help="Sweep detector wick ratio.")
    ap.add_argument("--vol-burst-z", type=float, default=1.2, help="Sweep detector volume burst Z.")
    # FVG settings
    ap.add_argument("--fvg-min-gap", type=float, default=0.0, help="Minimum FVG gap size (price units).")
    # Structure & context
    ap.add_argument("--bos-lookback", type=int, default=10, help="Lookback for BOS/CHOCH swing calc.")
    ap.add_argument("--pd-lookback", type=int, default=20, help="Lookback for premium/discount (swing mid).")
    ap.add_argument("--htf-minutes", type=int, default=60, help="HTF bias window (minutes).")
    ap.add_argument("--no-killzones", action="store_true", help="Disable killzone session filters.")
    ap.add_argument("--sweep-window", type=int, default=10, help="Require sweep within last X minutes.")
    ap.add_argument("--forward-minutes", type=int, default=60, help="Simulation horizon (minutes).")
    # Exit/management
    ap.add_argument("--partial-at-r", type=float, default=1.0, help="Partial take-profit at R.")
    ap.add_argument("--partial-frac", type=float, default=0.5, help="Fraction to close at partial TP.")
    ap.add_argument("--target", type=str, default="swing", choices=["swing", "fvg_fill", "session"], help="Target preference (currently uses swing).")
    ap.add_argument("--out", type=str, default="reports/smc_ict_trades.csv", help="Output CSV for trades.")
    # NEW flags for TP/SL override (decimal ratios)
    ap.add_argument("--tp", type=float, default=None, help="(Optional) Override TP as decimal ratio relative to entry (e.g., 0.0015 for 0.15%).")
    ap.add_argument("--sl", type=float, default=None, help="(Optional) Override SL as decimal ratio relative to entry (e.g., 0.0010 for 0.10%).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Run the wrapped backtest (will compute PF, equity, dd)
    results = run_backtest(
        n=args.n,
        seed=args.seed,
        lookback=args.lookback,
        wick_ratio=args.wick_ratio,
        vol_burst_z=args.vol_burst_z,
        fvg_min_gap=args.fvg_min_gap,
        bos_lookback=args.bos_lookback,
        pd_lookback=args.pd_lookback,
        htf_minutes=args.htf_minutes,
        no_killzones=args.no_killzones,
        sweep_window=args.sweep_window,
        forward_minutes=args.forward_minutes,
        partial_at_r=args.partial_at_r,
        partial_frac=args.partial_frac,
        target=args.target,
        tp_override=args.tp,
        sl_override=args.sl,
        out=args.out,
    )

    trades_df = results["trades_df"]

    # If no trades
    if trades_df.empty:
        print("No trades matched the full checklist. Try loosening parameters.")
        return

    # Print a parse-friendly summary (compatible with batch/scoring parsers)
    events = results["events"]
    usable = results["usable"]
    wins = results["wins"]
    losses = results["losses"]
    unresolved = results["unresolved"]
    win_rate_pct = results["win_rate"] * 100 if not math.isnan(results["win_rate"]) else float("nan")
    expectancy = results["expectancy"]
    final_equity = results["final_equity"]
    max_dd = results["max_drawdown"]
    pf = results["profit_factor"]

    print(f"events={events} usable={usable} wins={wins} losses={losses} unresolved={unresolved}")
    tp_display = f"{args.tp:.4%}" if args.tp is not None else "auto"
    sl_display = f"{args.sl:.4%}" if args.sl is not None else "auto"
    print(f"TP={tp_display} SL={sl_display} | win_rate={win_rate_pct:.2f}% expectancy={expectancy:.5f}")
    pf_display = f"{pf:.2f}" if pf is not None and not math.isinf(pf) else ("inf" if pf is not None and math.isinf(pf) else "N/A")
    print(f"final_equity={final_equity:.2f} max_drawdown={max_dd:.2f}% profit_factor={pf_display}")
    print(f"Wrote trades CSV: {args.out}")

    # Optional: preview first trades
    try:
        print(trades_df.head(10).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()
