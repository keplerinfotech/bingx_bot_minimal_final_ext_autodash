#!/usr/bin/env python3
"""
Full-featured batch evaluation for SMC/ICT strategy using test script.
Features:
- Parallel execution for speed
- Debug logging
- Results always contain: expectancy, profit_factor, max_dd, trades
- CSV + tabular output
- Top-3 configs by expectancy & risk-adjusted
- Heatmap output (if matplotlib installed)
- Automatically uses evaluate_smc_ict_test.py for non-empty results
"""

import subprocess
import itertools
import csv
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Directory for reports
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Parameter grid
TPS = [0.0015, 0.002, 0.003]
SLS = [0.001, 0.0015, 0.002]
WICK_RATIOS = [0.0, 0.15, 0.3]
FVG_GAPS = [0.0, 0.02]
BOS_LOOKBACKS = [3, 6, 10]
PD_LOOKBACKS = [5, 10, 20]
SWEEP_WINDOWS = [15, 30, 60]
FORWARD_MINUTES = [60, 90]

SEED = 123
N_TRADES = 2000

def run_backtest(tp, sl, params):
    """Run single backtest and return a result dictionary."""
    cmd = [
        sys.executable,
        "scripts/evaluate_smc_ict_test.py",
        f"--tp={tp}",
        f"--sl={sl}",
        f"--seed={SEED}",
        f"--n={N_TRADES}",
        "--no-killzones",
        f"--wick-ratio={params['wick_ratio']}",
        f"--fvg-min-gap={params['fvg_min_gap']}",
        f"--bos-lookback={params['bos_lookback']}",
        f"--pd-lookback={params['pd_lookback']}",
        f"--sweep-window={params['sweep_window']}",
        f"--forward-minutes={params['forward_minutes']}"
    ]
    logging.debug(f"Running: {' '.join(cmd)}")
    result = {"tp": tp, "sl": sl}
    result.update(params)
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = completed.stdout.strip()
        if output.startswith("{") and output.endswith("}"):
            import ast
            out_dict = ast.literal_eval(output)
            result.update({
                "trades": out_dict.get("trades", 0),
                "expectancy": out_dict.get("expectancy", 0.0),
                "profit_factor": out_dict.get("profit_factor", 0.0),
                "max_dd": out_dict.get("max_dd", 0.0)
            })
        else:
            logging.warning(f"Unexpected output: {output}")
            result.update({"trades": 0, "expectancy": 0.0, "profit_factor": 1.0, "max_dd": 0.0})
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        result.update({"trades": 0, "expectancy": 0.0, "profit_factor": 1.0, "max_dd": 0.0})
    return result

def save_results(results):
    """Save results to CSV and optionally print tabular summary."""
    csv_path = os.path.join(REPORT_DIR, "batch_eval_summary.csv")
    headers = ["tp", "sl", "wick_ratio", "fvg_min_gap", "bos_lookback", "pd_lookback", "sweep_window",
               "forward_minutes", "expectancy", "profit_factor", "max_dd", "trades"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, 0) for k in headers})
    logging.info(f"Saved results to {csv_path}")

    if tabulate:
        print("\nSimple results preview:\n")
        print(tabulate(results, headers=headers, floatfmt=".6f"))

def generate_heatmap(results):
    """Generate TP vs SL expectancy heatmap if matplotlib is available."""
    if plt is None or not results:
        logging.warning("Matplotlib not installed or no results; skipping heatmap.")
        return

    import numpy as np
    tp_vals = sorted(set(r['tp'] for r in results))
    sl_vals = sorted(set(r['sl'] for r in results))
    heatmap = np.zeros((len(sl_vals), len(tp_vals)))

    for r in results:
        i = sl_vals.index(r['sl'])
        j = tp_vals.index(r['tp'])
        heatmap[i, j] = r['expectancy']

    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, origin='lower', cmap='RdYlGn', aspect='auto')
    plt.xticks(range(len(tp_vals)), [f"{v:.4f}" for v in tp_vals])
    plt.yticks(range(len(sl_vals)), [f"{v:.4f}" for v in sl_vals])
    plt.colorbar(label="Expectancy")
    plt.xlabel("TP")
    plt.ylabel("SL")
    plt.title("TP vs SL Expectancy Heatmap")
    path = os.path.join(REPORT_DIR, "heatmap_tp_sl_expectancy.png")
    plt.savefig(path)
    plt.close()
    logging.info(f"Saved heatmap: {path}")

def main():
    param_grid = list(itertools.product(WICK_RATIOS, FVG_GAPS, BOS_LOOKBACKS, PD_LOOKBACKS, SWEEP_WINDOWS, FORWARD_MINUTES))
    param_dicts = [
        {
            "wick_ratio": w, "fvg_min_gap": fvg, "bos_lookback": bos, "pd_lookback": pd,
            "sweep_window": sweep, "forward_minutes": fwd
        } for (w, fvg, bos, pd, sweep, fwd) in param_grid
    ]

    logging.info(f"Submitting {len(param_dicts)} jobs with up to 6 workers...")
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_params = {executor.submit(run_backtest, tp, sl, p): (tp, sl, p)
                            for tp in TPS for sl in SLS for p in param_dicts}
        for idx, future in enumerate(as_completed(future_to_params), 1):
            tp, sl, p = future_to_params[future]
            try:
                res = future.result()
            except Exception as e:
                logging.error(f"Job failed: {e}")
                res = {"tp": tp, "sl": sl, **p, "trades": 0, "expectancy": 0.0, "profit_factor": 1.0, "max_dd": 0.0}
            results.append(res)
            logging.info(f"[PROGRESS] {idx}/{len(future_to_params)} done | tp={tp} sl={sl} dur=~ err={res['trades']==0}")

    save_results(results)
    generate_heatmap(results)

    # Top 3 by expectancy
    top_exp = sorted(results, key=lambda x: x['expectancy'], reverse=True)[:3]
    top_risk_adj = sorted(results, key=lambda x: (x['expectancy']/max(x['max_dd'], 0.0001)), reverse=True)[:3]

    print("\n--- Top by Expectancy ---")
    for i, r in enumerate(top_exp, 1):
        print(f"[{i}] exp={r['expectancy']:.6f} PF={r['profit_factor']:.3f} trades={r['trades']} TP={r['tp']} SL={r['sl']} params={r}")

    print("\n--- Top by Risk-Adjusted (expectancy / max_dd) ---")
    for i, r in enumerate(top_risk_adj, 1):
        print(f"[{i}] ratio={(r['expectancy']/max(r['max_dd'],0.0001)):.2f} equity=?? max_dd={r['max_dd']:.3f} exp={r['expectancy']:.6f} PF={r['profit_factor']:.3f} TP={r['tp']} SL={r['sl']} params={r}")

if __name__ == "__main__":
    main()
