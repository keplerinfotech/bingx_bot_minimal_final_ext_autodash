import numpy as np
import pandas as pd

def find_sweeps(df: pd.DataFrame,
                lookback: int = 20,
                wick_ratio: float = 0.5,
                vol_burst_z: float = 1.5,
                require_reject: bool = True) -> pd.DataFrame:
    """
    Detects liquidity sweeps (ICT-style) on OHLCV data.
    Rules (configurable):
      - Up-sweep: current high > prior rolling max(high, lookback)
      - Long upper wick: (high - max(open, close)) >= wick_ratio * (high - low)
      - Optional rejection: close <= prior rolling max high (failed continuation)
      - Volume burst: z-score(volume) >= vol_burst_z
    Similarly for down-sweep (mirror conditions).
    Returns a DataFrame with columns: ['timestamp','direction','entry_price','invalidation','context']
    """
    req_cols = {"open","high","low","close","volume"}
    assert req_cols.issubset(set(df.columns)), "df must contain OHLCV columns"

    out = []
    vol = df["volume"].astype(float)
    vol_z = (vol - vol.rolling(100, min_periods=20).mean()) / (vol.rolling(100, min_periods=20).std(ddof=0) + 1e-12)

    roll_max_hi = df["high"].shift(1).rolling(lookback, min_periods=1).max()
    roll_min_lo = df["low"].shift(1).rolling(lookback, min_periods=1).min()

    for i in range(len(df)):
        o,h,l,c = df.iloc[i][["open","high","low","close"]]
        ts = df.index[i]
        # Skip first lookback bars
        if i < lookback:
            continue
        tr = max(h-l, 1e-12)
        # Up-sweep
        up_break = h > roll_max_hi.iloc[i]
        up_wick = (h - max(o,c)) >= wick_ratio * tr
        up_reject = c <= roll_max_hi.iloc[i]
        up_ok = up_break and up_wick and (vol_z.iloc[i] >= vol_burst_z) and (up_reject if require_reject else True)

        # Down-sweep
        dn_break = l < roll_min_lo.iloc[i]
        dn_wick = (min(o,c) - l) >= wick_ratio * tr
        dn_reject = c >= roll_min_lo.iloc[i]
        dn_ok = dn_break and dn_wick and (vol_z.iloc[i] >= vol_burst_z) and (dn_reject if require_reject else True)

        if up_ok:
            entry = (df.iloc[i]["high"] + df.iloc[i]["low"] + df.iloc[i]["close"])/3.0  # FVG mid proxy
            invalid = h + 1e-6  # swing high break invalidation
            out.append({"timestamp":ts, "direction":"short", "entry_price":entry, "invalidation":invalid,
                        "context":{"type":"up-sweep","roll_max_hi":float(roll_max_hi.iloc[i]),"vol_z":float(vol_z.iloc[i])}})
        elif dn_ok:
            entry = (df.iloc[i]["high"] + df.iloc[i]["low"] + df.iloc[i]["close"])/3.0
            invalid = l - 1e-6
            out.append({"timestamp":ts, "direction":"long", "entry_price":entry, "invalidation":invalid,
                        "context":{"type":"down-sweep","roll_min_lo":float(roll_min_lo.iloc[i]),"vol_z":float(vol_z.iloc[i])}})

    return pd.DataFrame(out).set_index("timestamp") if out else pd.DataFrame(columns=["timestamp","direction","entry_price","invalidation","context"]).set_index("timestamp")
