import pandas as pd

def find_fvgs(df: pd.DataFrame, min_body_mult: float = 1.0) -> pd.DataFrame:
    """
    Detect 3-candle fair value gaps (FVG) with displacement.
    Bullish FVG at index i when: low[i+1] > high[i-1] and candle i body >= min_body_mult * median body
    Bearish FVG at index i when: high[i+1] < low[i-1] and candle i body >= threshold
    Returns a DataFrame indexed like df with columns:
      - bull_gap_low, bull_gap_high (NaN if none)
      - bear_gap_low, bear_gap_high
    """
    req = {"open","high","low","close"}
    assert req.issubset(df.columns), "OHLC required"
    body = (df["close"] - df["open"]).abs()
    med_body = body.rolling(100, min_periods=20).median()
    big = body >= (min_body_mult * (med_body.replace(0, med_body.mean())))

    bull = (df["low"].shift(-1) > df["high"].shift(1)) & big
    bear = (df["high"].shift(-1) < df["low"].shift(1)) & big

    bull_low = df["high"].shift(1).where(bull)
    bull_high = df["low"].shift(-1).where(bull)
    bear_low = df["high"].shift(-1).where(bear)
    bear_high = df["low"].shift(1).where(bear)

    out = pd.DataFrame({
        "bull_gap_low": bull_low,
        "bull_gap_high": bull_high,
        "bear_gap_low": bear_low,
        "bear_gap_high": bear_high
    }, index=df.index)
    return out

def fvg_midpoints(fvg_df: pd.DataFrame) -> pd.Series:
    bull_mid = (fvg_df["bull_gap_low"] + fvg_df["bull_gap_high"]) / 2.0
    bear_mid = (fvg_df["bear_gap_low"] + fvg_df["bear_gap_high"]) / 2.0
    # Prefer non-null mid depending on direction later; here return both combined for convenience
    return pd.DataFrame({"bull_mid": bull_mid, "bear_mid": bear_mid})
