import pandas as pd
import numpy as np

def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df.get("volume", pd.Series(index=df.index, dtype=float)).resample(rule).sum()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    return out

def anchored_vwap(df: pd.DataFrame, anchor_ts: pd.Timestamp | None = None) -> pd.Series:
    """
    Simple rolling cumulative VWAP since anchor timestamp (or start).
    """
    px = df["close"]
    vol = df.get("volume", None)
    if vol is None:
        vol = pd.Series(1.0, index=px.index)  # fallback equal volume
    mask = px.index >= (anchor_ts or px.index[0])
    vol_cum = vol.where(mask).fillna(0).cumsum()
    pv_cum = (px * vol.where(mask).fillna(0)).cumsum()
    vwap = pv_cum / (vol_cum.replace(0, np.nan))
    return vwap.ffill()

def donchian_state(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hh = df["high"].rolling(window, min_periods=1).max()
    ll = df["low"].rolling(window, min_periods=1).min()
    mid = (hh + ll) / 2.0
    # state: +1 up if close>mid and making higher highs; -1 down if close<mid and making lower lows; else 0
    up = (df["close"] > mid) & (df["high"] >= hh.shift(1))
    dn = (df["close"] < mid) & (df["low"] <= ll.shift(1))
    state = pd.Series(0, index=df.index, dtype=int)
    state[up] = 1
    state[dn] = -1
    return state

def htf_bias(ltf_df: pd.DataFrame, htf_rule="4H", method="combo") -> pd.Series:
    """
    Compute higher timeframe bias aligned to ltf_df index.
    - method 'donchian': uses donchian_state on HTF
    - method 'vwap_slope': AVWAP slope sign (+1 up, -1 down)
    - method 'combo': require agreement; else 0
    """
    htf = resample_ohlc(ltf_df, htf_rule)
    d_state = donchian_state(htf, window=20)
    avwap = anchored_vwap(htf)
    slope = avwap.diff().rolling(3).mean()
    v_state = pd.Series(0, index=htf.index, dtype=int)
    v_state[slope > 0] = 1
    v_state[slope < 0] = -1
    if method == "donchian":
        bias = d_state
    elif method == "vwap_slope":
        bias = v_state
    else:
        agree = (d_state == v_state) & (d_state != 0)
        bias = pd.Series(0, index=htf.index, dtype=int)
        bias[agree] = d_state[agree]
    # align back to ltf index
    return bias.reindex(ltf_df.index, method="ffill").fillna(0).astype(int)
