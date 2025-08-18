"""Preflight checks before enabling live trading.

Checks configuration files for dangerous live settings and enforces conservative
risk caps. Returns exit code 0 on PASS, non-zero on FAIL.

Usage:
    python3 scripts/preflight_live_check.py [--config-settings path] [--config-trading path]

This script is intentionally conservative and designed to be editable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_settings(settings: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    binance = settings.get("binance", {})
    # Ensure simulated/test flags are disabled for true live mode
    for flag in ("testnet", "paper_trade", "simulate_orders"):
        if binance.get(flag, False):
            errs.append(f"binance.{flag} is True (should be False for live mode)")

    return errs


def check_trading(trading: dict[str, Any]) -> list[str]:
    errs: list[str] = []

    general = trading.get("general", {})
    risk_per_trade = general.get("risk_per_trade")
    max_daily_risk = general.get("max_daily_risk")

    # Conservative caps for initial live run
    if risk_per_trade is None:
        errs.append("general.risk_per_trade missing")
    else:
        if float(risk_per_trade) > 0.001:
            errs.append(f"risk_per_trade={risk_per_trade} > 0.001 (conservative cap)")

    if max_daily_risk is None:
        errs.append("general.max_daily_risk missing")
    else:
        if float(max_daily_risk) > 0.01:
            errs.append(f"max_daily_risk={max_daily_risk} > 0.01 (conservative cap)")

    # Check settings.yaml's max_daily_loss_bps if present under top-level risk
    return errs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config-settings", default="config/settings.yaml")
    p.add_argument("--config-trading", default="config/trading_config.yaml")
    args = p.parse_args(argv)

    s_path = Path(args.config_settings)
    t_path = Path(args.config_trading)

    failures: list[str] = []

    if not s_path.exists():
        print(f"ERROR: settings file not found: {s_path}")
        return 2
    if not t_path.exists():
        print(f"ERROR: trading config not found: {t_path}")
        return 2

    settings = load_yaml(s_path)
    trading = load_yaml(t_path)

    failures += check_settings(settings)
    failures += check_trading(trading)

    # Check max_daily_loss_bps under settings.risk if present
    risk = settings.get("risk", {})
    max_daily_loss_bps = risk.get("max_daily_loss_bps")
    if max_daily_loss_bps is None:
        print(
            "WARNING: settings.risk.max_daily_loss_bps not set — consider setting a low cap for first live runs"
        )
    else:
        # convert basis points to fraction check (50 bps = 0.005)
        if float(max_daily_loss_bps) > 100:  # 1% = 100 bps
            failures.append(
                f"settings.risk.max_daily_loss_bps={max_daily_loss_bps} > 100 bps (conservative cap)"
            )

    if failures:
        print("PRECHECK FAIL — the following issues were found:")
        for f in failures:
            print(" - ", f)
        print("Fix the above and re-run the preflight. Exiting with code 3.")
        return 3

    print("PRECHECK PASS — configuration looks conservative for an initial live run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
