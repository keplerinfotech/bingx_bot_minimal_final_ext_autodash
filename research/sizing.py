import numpy as np

def kelly_fraction(p: float, b: float) -> float:
    """
    Kelly fraction for a Bernoulli bet with win prob p and payoff ratio b (avg_win/avg_loss).
    f* = p - (1-p)/b
    """
    if b <= 0:
        return 0.0
    f = p - (1.0 - p) / b
    return max(0.0, f)

def capped_kelly_size(equity: float,
                      p: float,
                      b: float,
                      fraction_of_kelly: float = 0.25,
                      vol_cap_frac: float | None = None,
                      price: float | None = None,
                      contract_value: float | None = None) -> dict:
    """
    Returns dict with target notional and units, applying a cap (e.g., 1/4 Kelly).
    Optionally cap by volatility (not provided here; placeholder param).
    If price & contract_value given, compute units for a linear perp (units = notional / price).
    """
    f = kelly_fraction(p, b) * fraction_of_kelly
    f = max(0.0, min(f, 1.0))
    target_notional = equity * f
    units = None
    if price is not None:
        units = target_notional / max(price, 1e-12)
    if contract_value is not None and price is None:
        # for inverse/futures where contract_value specifies USD value per contract
        units = target_notional / contract_value
    return {"fraction": f, "target_notional": float(target_notional), "units": None if units is None else float(units)}

def kelly_from_returns(r: np.ndarray) -> float:
    """
    Approximate Kelly fraction from a sequence of trade returns (fractional, e.g., +0.01 = +1%).
    Uses mean/variance method for small returns: f* ≈ mean/var.
    Clipped to [0,1].
    """
    r = np.asarray(r, dtype=float)
    mu = r.mean()
    var = r.var() + 1e-12
    f = mu / var if var > 0 else 0.0
    return max(0.0, min(float(f), 1.0))
