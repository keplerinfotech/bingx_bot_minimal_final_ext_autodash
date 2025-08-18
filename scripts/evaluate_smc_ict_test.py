#!/usr/bin/env python3
"""
Test evaluation script for SMC/ICT.
Guarantees non-empty trades for batch testing.
Calculates:
- trades
- expectancy
- profit_factor
- max_dd
"""

import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=float, required=True)
    parser.add_argument("--sl", type=float, required=True)
    parser.add_argument("--wick-ratio", type=float, default=0.0)
    parser.add_argument("--fvg-min-gap", type=float, default=0.0)
    parser.add_argument("--bos-lookback", type=int, default=3)
    parser.add_argument("--pd-lookback", type=int, default=5)
    parser.add_argument("--sweep-window", type=int, default=15)
    parser.add_argument("--forward-minutes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--no-killzones", action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # Fake trade generation for testing
    trades = max(1, int(args.n * 0.003))  # always at least 1 trade
    wins = max(1, int(trades * 0.5))
    losses = trades - wins

    gain_per_trade = args.tp
    loss_per_trade = args.sl

    total_gain = wins * gain_per_trade
    total_loss = losses * loss_per_trade
    expectancy = (total_gain - total_loss) / trades if trades else 0.0
    profit_factor = total_gain / total_loss if total_loss != 0 else 1.0
    max_dd = min(1.0, total_loss / (total_gain + 0.0001))

    result = {
        "tp": args.tp,
        "sl": args.sl,
        "wick_ratio": args.wick_ratio,
        "fvg_min_gap": args.fvg_min_gap,
        "bos_lookback": args.bos_lookback,
        "pd_lookback": args.pd_lookback,
        "sweep_window": args.sweep_window,
        "forward_minutes": args.forward_minutes,
        "trades": trades,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_dd": max_dd
    }

    print(result)

if __name__ == "__main__":
    main()
