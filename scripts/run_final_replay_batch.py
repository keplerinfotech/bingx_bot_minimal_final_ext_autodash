"""
Batch runner for the final multi-venue L2 replay.

This script runs the project's run_final_replay() multiple times while overriding
the sweep detector parameters (lookback, wick_ratio, vol_burst_z) on a per-run basis.
It writes each run's CSV and produces a summary CSV (batch_summary.csv) in the output dir.

Usage (from project root, with venv active):
    PYTHONPATH=. python scripts/run_final_replay_batch.py --output-dir reports/batch1

Notes:
- The script monkeypatches scripts.run_final_replay.find_sweeps at runtime so the
  existing replay function can be exercised without changing its source.
- Keep runs small when experimenting; increase grid later for thorough testing.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import os
import sys
import time
import traceback
from typing import Callable, List, Tuple


def _ensure_project_root_on_path() -> None:
    """Ensure project root is importable when running the script directly.

    This helper is executed early in main paths to avoid performing runtime
    sys.path mutations in the module import area (which can trigger E402).
    """
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)


_ensure_project_root_on_path()

import pandas as pd


def run_batch(
    output_dir: str,
    grid: List[Tuple[int, float, float]],
    prefix: str = "final_fill_quality",
) -> str:
    """
    Run replay for each params tuple in grid and collect summary.

    Args:
        output_dir: base directory for outputs.
        grid: list of tuples (lookback, wick_ratio, vol_burst_z).
        prefix: filename prefix for each run CSV.

    Returns:
        Path to the batch summary CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    # Import replay module and capture original detector
    replay_mod = importlib.import_module("scripts.run_final_replay")
    original_find = getattr(replay_mod, "find_sweeps", None)

    if original_find is None:
        raise RuntimeError("scripts.run_final_replay.find_sweeps not found in module")

    for i, (lookback, wick_ratio, vol_burst_z) in enumerate(grid):
        run_tag = f"run{i+1:02d}_lb{lookback}_wr{str(wick_ratio).replace('.','p')}_vz{str(vol_burst_z).replace('.','p')}"
        out_csv = os.path.join(output_dir, f"{prefix}_{run_tag}.csv")
        print(f"[{time.strftime('%H:%M:%S')}] Starting {run_tag} -> {out_csv}")

        # Build override wrapper that forces the detector params
        def make_override(orig: Callable, lb: int, wr: float, vz: float) -> Callable:
            def override(df, *args, **kwargs):
                kw = dict(kwargs)
                kw.update({"lookback": lb, "wick_ratio": wr, "vol_burst_z": vz})
                return orig(df, *args, **kw)

            return override

        # Monkeypatch the module-level find_sweeps
        try:
            replay_mod.find_sweeps = make_override(
                original_find, lookback, wick_ratio, vol_burst_z
            )
            # Run the replay (it will write out_csv to the given output_dir)
            try:
                replay_mod.run_final_replay(output_csv=out_csv, output_dir=output_dir)
            except Exception:
                print(f"Run {run_tag} failed during replay execution:")
                traceback.print_exc()
                summary_rows.append(
                    {
                        "run_tag": run_tag,
                        "lookback": lookback,
                        "wick_ratio": wick_ratio,
                        "vol_burst_z": vol_burst_z,
                        "status": "error",
                        "orders": None,
                        "filled": None,
                    }
                )
                continue

            # Read produced CSV and summarize
            if os.path.exists(out_csv):
                try:
                    df = pd.read_csv(out_csv)
                    orders = len(df)
                    filled = 0
                    if "filled" in df.columns:
                        # handle strings/booleans
                        filled = int(
                            pd.to_numeric(df["filled"], errors="coerce")
                            .fillna(0)
                            .astype(bool)
                            .sum()
                        )
                    summary_rows.append(
                        {
                            "run_tag": run_tag,
                            "lookback": lookback,
                            "wick_ratio": wick_ratio,
                            "vol_burst_z": vol_burst_z,
                            "status": "ok",
                            "orders": orders,
                            "filled": filled,
                        }
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Completed {run_tag}: orders={orders}, filled={filled}"
                    )
                except Exception:
                    print(f"Failed to parse output CSV for {run_tag}: {out_csv}")
                    traceback.print_exc()
                    summary_rows.append(
                        {
                            "run_tag": run_tag,
                            "lookback": lookback,
                            "wick_ratio": wick_ratio,
                            "vol_burst_z": vol_burst_z,
                            "status": "csv_error",
                            "orders": None,
                            "filled": None,
                        }
                    )
            else:
                print(f"No CSV produced for {run_tag} at expected path: {out_csv}")
                summary_rows.append(
                    {
                        "run_tag": run_tag,
                        "lookback": lookback,
                        "wick_ratio": wick_ratio,
                        "vol_burst_z": vol_burst_z,
                        "status": "no_csv",
                        "orders": None,
                        "filled": None,
                    }
                )
        finally:
            # restore original detector to avoid side-effects between runs
            replay_mod.find_sweeps = original_find

    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "batch_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Batch summary written: {summary_csv}")
    return summary_csv


def parse_grid(arg: str) -> List[Tuple[int, float, float]]:
    """
    Parse a compact grid specification from CLI, or return default grid if arg is empty.

    Example formats:
      "10,20:0.3,0.5:1.0,1.5"  -> lookbacks (10,20), wick (0.3,0.5), vol (1.0,1.5)
    """
    if not arg:
        # default small grid for quick dry-run
        return [(10, 0.3, 1.0), (20, 0.5, 1.5), (40, 0.5, 2.0)]

    try:
        parts = arg.split(":")
        lbs = [int(x) for x in parts[0].split(",") if x]
        wrs = [float(x) for x in parts[1].split(",") if x]
        vzs = [float(x) for x in parts[2].split(",") if x]
        grid = list(itertools.product(lbs, wrs, vzs))
        return grid
    except Exception:
        raise argparse.ArgumentTypeError(
            "Invalid grid format. Use lookback_comma:wick_comma:vol_comma"
        )


def main():
    p = argparse.ArgumentParser(
        description="Batch run final replay with different detector params."
    )
    p.add_argument(
        "--output-dir", "-o", default="reports/batch", help="Directory for outputs"
    )
    p.add_argument(
        "--grid",
        "-g",
        default="",
        help="Grid spec e.g. '10,20:0.3,0.5:1.0,1.5' or empty for default",
    )
    args = p.parse_args()

    grid = parse_grid(args.grid)
    print(f"Running batch with {len(grid)} runs. Output dir: {args.output_dir}")
    run_batch(args.output_dir, grid)


if __name__ == "__main__":
    main()
