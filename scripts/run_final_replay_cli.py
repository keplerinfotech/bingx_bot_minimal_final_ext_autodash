"""
CLI wrapper to run the project's final multi-venue L2 replay function reliably.

Usage:
    PYTHONPATH=. python scripts/run_final_replay_cli.py --output-dir reports

The wrapper ensures the project root is on sys.path so local imports resolve.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Ensure project root is on sys.path when run directly
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

try:
    # import the replay function from the project's script
    from scripts.run_final_replay import run_final_replay  # type: ignore
except Exception:  # pragma: no cover - helps debugging import issues
    print("Failed to import run_final_replay from scripts/run_final_replay.py")
    traceback.print_exc()
    raise


def main() -> None:
    p = argparse.ArgumentParser(description="Run final multi-venue L2 replay (wrapper)")
    p.add_argument(
        "--output-dir", "-o", default="reports", help="Directory for outputs"
    )
    p.add_argument("--out-csv", default=None, help="Optional CSV path")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        csv_path = run_final_replay(output_csv=args.out_csv, output_dir=args.output_dir)
        print(f"Replay finished, csv: {csv_path}")
    except Exception:  # pragma: no cover - runtime errors should be visible to user
        print("Replay failed with an exception:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
