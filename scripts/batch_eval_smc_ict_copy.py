import csv
import itertools
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")


def run_backtest(tp, sl, params):
    """Run evaluate_smc_ict.py as subprocess and parse metrics."""
    cmd = [
        "python",
        "scripts/evaluate_smc_ict.py",
        "--tp",
        str(tp),
        "--sl",
        str(sl),
        "--seed",
        "123",
        "--n",
        "2000",
        "--no-killzones",
        "--wick-ratio",
        str(params["wick_ratio"]),
        "--fvg-min-gap",
        str(params["fvg_min_gap"]),
        "--bos-lookback",
        str(params["bos_lookback"]),
        "--pd-lookback",
        str(params["pd_lookback"]),
        "--sweep-window",
        str(params["sweep_window"]),
        "--forward-minutes",
        str(params["forward_minutes"]),
    ]

    logging.debug(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    metrics = {
        "tp": tp,
        "sl": sl,
        **params,
        "expectancy": None,
        "risk_adj": None,
        "profit_factor": None,
    }

    if result.returncode == 0 and result.stdout:
        for line in result.stdout.splitlines():
            if "Expectancy:" in line:
                try:
                    metrics["expectancy"] = float(line.split(":")[1].strip())
                except Exception:
                    pass
            if "Risk-Adj:" in line:
                try:
                    metrics["risk_adj"] = float(line.split(":")[1].strip())
                except Exception:
                    pass
            if "ProfitFactor:" in line:
                try:
                    metrics["profit_factor"] = float(line.split(":")[1].strip())
                except Exception:
                    pass

    return metrics


def save_results(results, out_csv="batch_results.csv"):
    """Save results to CSV and show tabular output."""
    if not results:
        logging.warning("No results to save!")
        return

    headers = list(results[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logging.info(f"Saved results to {out_csv}")

    # Tabular preview
    from tabulate import tabulate

    print(tabulate(results, headers="keys", floatfmt=".4f"))


def report_top_configs(results):
    """Print top-3 configs by expectancy and risk-adjusted."""
    if not results:
        logging.warning("No results for top configs!")
        return

    valid_exp = [r for r in results if r["expectancy"] is not None]
    valid_ra = [r for r in results if r["risk_adj"] is not None]

    if not valid_exp:
        logging.warning("No valid expectancy results.")
    else:
        top_exp = sorted(valid_exp, key=lambda r: r["expectancy"], reverse=True)[:3]
        print("\nTop 3 by Expectancy:")
        for r in top_exp:
            print(r)

    if not valid_ra:
        logging.warning("No valid risk-adjusted results.")
    else:
        top_ra = sorted(valid_ra, key=lambda r: r["risk_adj"], reverse=True)[:3]
        print("\nTop 3 by Risk-Adjusted:")
        for r in top_ra:
            print(r)

    if valid_exp and all(r["expectancy"] < 0 for r in valid_exp):
        logging.warning(
            "All expectancy values < 0! Strategy unprofitable across configs."
        )


def plot_heatmap(results):
    if not HAS_MPL:
        logging.warning("matplotlib not installed — skipping heatmap.")
        return

    valid = [r for r in results if r["expectancy"] is not None]
    if not valid:
        logging.warning("No valid results for heatmap.")
        return

    import numpy as np

    tps = sorted(set(r["tp"] for r in valid))
    sls = sorted(set(r["sl"] for r in valid))

    heat = np.zeros((len(sls), len(tps)))
    for r in valid:
        i = sls.index(r["sl"])
        j = tps.index(r["tp"])
        heat[i, j] = r["expectancy"] or 0

    fig, ax = plt.subplots()
    cax = ax.imshow(heat, cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(tps)))
    ax.set_yticks(range(len(sls)))
    ax.set_xticklabels(tps)
    ax.set_yticklabels(sls)
    ax.set_xlabel("TP")
    ax.set_ylabel("SL")
    ax.set_title("Expectancy Heatmap")
    fig.colorbar(cax)
    plt.show()


def main():
    # Define grid
    tp_values = [0.0015, 0.002]
    sl_values = [0.001, 0.0015, 0.002]

    param_grid = [
        {
            "wick_ratio": 0.0,
            "fvg_min_gap": 0.0,
            "bos_lookback": 3,
            "pd_lookback": 5,
            "sweep_window": 15,
            "forward_minutes": 60,
        },
        {
            "wick_ratio": 0.15,
            "fvg_min_gap": 0.0,
            "bos_lookback": 6,
            "pd_lookback": 10,
            "sweep_window": 30,
            "forward_minutes": 60,
        },
        {
            "wick_ratio": 0.30,
            "fvg_min_gap": 0.02,
            "bos_lookback": 10,
            "pd_lookback": 20,
            "sweep_window": 60,
            "forward_minutes": 90,
        },
    ]

    combos = list(itertools.product(tp_values, sl_values, param_grid))
    results = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = {
            ex.submit(run_backtest, tp, sl, params): (tp, sl, params)
            for tp, sl, params in combos
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    if not results:
        logging.error("No results collected — inserting dummy entry.")
        results.append(
            {"tp": 0, "sl": 0, "expectancy": 0, "risk_adj": 0, "profit_factor": 0}
        )

    save_results(results)
    report_top_configs(results)
    plot_heatmap(results)


if __name__ == "__main__":
    main()
