import numpy as np
import pandas as pd

from research.detectors import find_sweeps
from research.fvg import find_fvgs, fvg_midpoints
from research.htf import htf_bias
from research.sizing import capped_kelly_size


def backtest_smc_sweep(
    df: pd.DataFrame,
    equity_usd: float = 10_000.0,
    risk_per_trade_r: float = 1.0,
    rr_targets=(0.5, 1.0, 2.0),
    lookback=20,
    wick_ratio=0.5,
    vol_burst_z=1.5,
    htf_rule="4H",
    bias_method="combo",
    kelly_p=0.55,
    kelly_b=1.0,
    kelly_fraction=0.25,
) -> dict:
    """
    Simple event-driven backtest:
      - Detect sweeps on LTF data.
      - Compute FVGs; use gap midpoints as entry anchors.
      - Allow longs only when HTF bias==+1; shorts only when bias==-1 (configurable).
      - Entry: on next bar, limit at nearest relevant FVG midpoint; stop at invalidation (swing extreme from detector).
      - Size: capped Kelly notional; convert to units by entry price.
      - Exits: take-profit ladder at multiples of R; time-stop after N bars (optional: not implemented here).
    """
    df = df.copy()
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    events = find_sweeps(
        df,
        lookback=lookback,
        wick_ratio=wick_ratio,
        vol_burst_z=vol_burst_z,
        require_reject=True,
    )
    if events.empty:
        return {"trades": pd.DataFrame(), "stats": {}}

    fvgs = find_fvgs(df, min_body_mult=1.0)
    mids = fvg_midpoints(fvgs)
    bias = htf_bias(df, htf_rule=htf_rule, method=bias_method)

    trades = []
    for ts, ev in events.iterrows():
        direction = ev["direction"]
        if direction == "long" and bias.loc[ts] != 1:
            continue
        if direction == "short" and bias.loc[ts] != -1:
            continue
        # choose entry midpoint
        if direction == "long":
            entry = (
                float(mids.loc[ts, "bull_mid"])
                if not np.isnan(mids.loc[ts, "bull_mid"])
                else float(df.loc[ts, "close"])
            )
            invalid = float(ev["invalidation"])  # lower than swing low
            if np.isnan(entry) or entry <= 0 or entry <= invalid:
                continue
        else:
            entry = (
                float(mids.loc[ts, "bear_mid"])
                if not np.isnan(mids.loc[ts, "bear_mid"])
                else float(df.loc[ts, "close"])
            )
            invalid = float(ev["invalidation"])  # above swing high
            if np.isnan(entry) or entry <= 0 or entry >= invalid:
                continue
        # compute R (risk per unit)
        if direction == "long":
            risk_per_unit = entry - invalid
        else:
            risk_per_unit = invalid - entry
        if risk_per_unit <= 0:
            continue

        # Kelly-capped sizing
        k = capped_kelly_size(
            equity_usd,
            p=kelly_p,
            b=kelly_b,
            fraction_of_kelly=kelly_fraction,
            price=entry,
        )
        units = max(0.0, k["units"])
        if units <= 0:
            continue

        # simulate forward from next bar until stop or max TP
        ix = df.index.get_loc(ts)
        hit_tp = None
        exit_px = None
        rr_hit = None
        for j in range(ix + 1, len(df)):
            hi = df["high"].iloc[j]
            lo = df["low"].iloc[j]
            if direction == "long":
                # stop
                if lo <= invalid:
                    exit_px = invalid
                    rr_hit = -1.0
                    break
                # check TPs in ascending order
                for rr in rr_targets:
                    tp = entry + rr * risk_per_unit
                    if hi >= tp:
                        hit_tp = rr
                        exit_px = tp
                        rr_hit = rr
                        break
                if hit_tp is not None:
                    break
            else:
                if hi >= invalid:
                    exit_px = invalid
                    rr_hit = -1.0
                    break
                for rr in rr_targets:
                    tp = entry - rr * risk_per_unit
                    if lo <= tp:
                        hit_tp = rr
                        exit_px = tp
                        rr_hit = rr
                        break
                if hit_tp is not None:
                    break
        if exit_px is None:
            # timeout at last bar
            exit_px = float(df["close"].iloc[-1])
            rr_hit = (
                (exit_px - entry) / risk_per_unit
                if direction == "long"
                else (entry - exit_px) / risk_per_unit
            )

        pnl_per_unit = (exit_px - entry) if direction == "long" else (entry - exit_px)
        pnl = pnl_per_unit * units
        trades.append(
            {
                "timestamp": ts,
                "dir": direction,
                "entry": entry,
                "stop": invalid,
                "units": units,
                "exit": exit_px,
                "rr": rr_hit,
                "pnl": pnl,
            }
        )

    trades_df = pd.DataFrame(trades).sort_values("timestamp")
    stats = {}
    if len(trades_df):
        stats = {
            "trades": int(len(trades_df)),
            "hit_rate": float((trades_df["rr"] > 0).mean()),
            "avg_rr": float(trades_df["rr"].mean()),
            "sum_pnl": float(trades_df["pnl"].sum()),
            "median_rr": float(trades_df["rr"].median()),
        }
    return {"trades": trades_df, "stats": stats}
