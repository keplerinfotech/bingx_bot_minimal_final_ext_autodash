"""Utility to load sensitive keys from environment variables.

Provides a small helper to avoid storing secrets in files.
"""

from __future__ import annotations

import os
from typing import Tuple


def load_binance_keys() -> Tuple[str | None, str | None]:
    """Return (api_key, api_secret) from environment or None if missing."""
    return os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")


def ensure_binance_keys() -> None:
    k, s = load_binance_keys()
    if not k or not s:
        raise RuntimeError(
            "BINANCE_API_KEY and BINANCE_API_SECRET must be set in environment for Binance live runs"
        )


def load_kucoin_keys() -> Tuple[str | None, str | None, str | None]:
    """Return (api_key, api_secret, passphrase) for KuCoin or Nones if missing."""
    return (
        os.environ.get("KUCOIN_API_KEY"),
        os.environ.get("KUCOIN_API_SECRET"),
        os.environ.get("KUCOIN_API_PASSPHRASE"),
    )


def ensure_kucoin_keys() -> None:
    k, s, p = load_kucoin_keys()
    if not k or not s or not p:
        raise RuntimeError(
            "KUCOIN_API_KEY, KUCOIN_API_SECRET and KUCOIN_API_PASSPHRASE must be set in environment for KuCoin live runs"
        )


def ensure_any_exchange_keys() -> str:
    """Ensure keys for a supported exchange exist in env; return exchange name.

    Preference: KUCOIN if its env vars are present, otherwise BINANCE.
    Raises RuntimeError if neither has valid keys.
    """
    k, s, p = load_kucoin_keys()
    if k and s and p:
        return "KUCOIN"
    bk, bs = load_binance_keys()
    if bk and bs:
        return "BINANCE"
    raise RuntimeError(
        "No supported exchange keys found in env; set KUCOIN_* or BINANCE_*"
    )
