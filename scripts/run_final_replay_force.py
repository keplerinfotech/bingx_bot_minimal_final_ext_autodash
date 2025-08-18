"""
Force-run the final replay with overridden detector params.

Usage:
    PYTHONPATH=. python scripts/run_final_replay_force.py --output-dir reports/force \
        --lookback 5 --wick_ratio 0.1 --vol_burst_z 0 --require_reject false
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root on sys.path
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import importlib


def setup_logging(level: str | int) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Force-run the final replay with overridden detector params."
    )
    p.add_argument("--output-dir", "-o", default="reports/force")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--wick_ratio", type=float, default=0.1)
    p.add_argument("--vol_burst_z", type=float, default=0.0)
    p.add_argument(
        "--require_reject",
        type=lambda s: s.lower() in ("true", "1", "yes", "y"),
        default=False,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions but don't execute the replay",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--config",
        "-c",
        dest="config",
        help="Path to YAML settings file to use (overrides env).",
    )
    args = p.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("run_final_replay_force")

    os.makedirs(args.output_dir, exist_ok=True)

    # Import the replay module and original detector
    try:
        replay_mod = importlib.import_module("scripts.run_final_replay")
        original_find = getattr(replay_mod, "find_sweeps", None)
    except Exception:
        log.debug(
            "Could not import scripts.run_final_replay, will try research.detectors as fallback"
        )
        replay_mod = importlib.import_module("research.detectors")
        original_find = getattr(replay_mod, "find_sweeps", None)

    # Fallback: try research.detectors directly if replay module didn't expose it
    if original_find is None:
        try:
            det_mod = importlib.import_module("research.detectors")
            original_find = getattr(det_mod, "find_sweeps", None)
        except Exception:
            original_find = None

    if original_find is None:
        log.error("Could not find find_sweeps implementation to override")
        return 2

    def make_override(orig, lb, wr, vz, req):
        def override(df, *a, **kw):
            k = dict(kw)
            # enforce detector kwargs
            k.update(
                {
                    "lookback": lb,
                    "wick_ratio": wr,
                    "vol_burst_z": vz,
                    "require_reject": req,
                }
            )
            return orig(df, *a, **k)

        return override

    override_fn = make_override(
        original_find,
        args.lookback,
        args.wick_ratio,
        args.vol_burst_z,
        args.require_reject,
    )

    # Monkeypatch on the replay module (where run_final_replay imports it)
    try:
        setattr(replay_mod, "find_sweeps", override_fn)
    except Exception:
        log.exception("Failed to monkeypatch find_sweeps on replay module")
        return 3

    if args.dry_run:
        log.info(
            "Dry run: would run forced replay with output_dir=%s out_csv=%s lookback=%s wick_ratio=%s vol_burst_z=%s require_reject=%s",
            args.output_dir,
            args.out_csv,
            args.lookback,
            args.wick_ratio,
            args.vol_burst_z,
            args.require_reject,
        )
        # restore original before exit
        try:
            setattr(replay_mod, "find_sweeps", original_find)
        except Exception:
            pass
        return 0

    try:
        log.info("Running forced replay -> %s", args.output_dir)
        csv_path = replay_mod.run_final_replay(
            output_csv=args.out_csv,
            output_dir=args.output_dir,
            settings_path=(args.config if getattr(args, "config", None) else None),
        )
        log.info("Replay finished, csv: %s", csv_path)
        return 0
    except Exception:
        log.exception("Replay failed with exception")
        return 1
    finally:
        # restore original
        try:
            setattr(replay_mod, "find_sweeps", original_find)
        except Exception:
            log.debug("Failed to restore original find_sweeps")


if __name__ == "__main__":
    raise SystemExit(main())
