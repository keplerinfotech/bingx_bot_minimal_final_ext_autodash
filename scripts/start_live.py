"""Supervised live starter.

Usage: set BINANCE_API_KEY and BINANCE_API_SECRET in environment, then run:
    python3 scripts/start_live.py

The script will:
 - Run preflight against config/settings.live.yaml and trading_config.live.yaml
 - Require an interactive confirmation (type YES)
 - Start the chosen runner (multi-venue router) in a subprocess
 - Write a PID file `.live_bot.pid` so `kill_switch.py` can target it
 - Launch the kill-switch monitor in a background thread

This is intentionally minimal to keep control in the operator's hands.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from scripts.env_loader import ensure_any_exchange_keys

PRELIGHT = "config/settings.live.yaml"
TRADE_CFG = "config/trading_config.live.yaml"


def run_preflight() -> int:
    import importlib

    preflight = importlib.import_module("scripts.preflight_live_check")
    return preflight.main(
        ["--config-settings", PRELIGHT, "--config-trading", TRADE_CFG]
    )


def start_kill_monitor_thread():
    import scripts.kill_switch as ks

    t = threading.Thread(target=ks.monitor, daemon=True)
    t.start()
    return t


def main() -> int:
    print("Running preflight checks against live configs...")
    rc = run_preflight()
    if rc != 0:
        print("Preflight failed. Aborting start.")
        return rc

    # ensure env keys exist for a supported exchange
    try:
        exchange = ensure_any_exchange_keys()
        print(f"Detected API keys for exchange: {exchange}")
    except RuntimeError as e:
        print("Missing env keys:", e)
        return 4

    print("Preflight PASS. To proceed with a supervised live run type YES (all caps).")
    ans = input("Type YES to continue: ")
    if ans.strip() != "YES":
        print("Aborting — confirmation not provided.")
        return 5

    # Try to start the runner in-process by importing the script module and
    # calling its main function when available. This preserves local package
    # imports. If that fails, fall back to starting a subprocess using
    # -m to run as a module or running the script with CWD set to project root.
    p = None
    try:
        # attempt in-process import and call
        import importlib

        mod = importlib.import_module("scripts.run_multi_venue_router")
        # call function directly in a background process (use subprocess-like child)
        # We still spawn a subprocess to keep operator control and to isolate runtime.
        runner = [
            sys.executable,
            "-u",
            "-m",
            "scripts.run_multi_venue_router",
            "--output-dir",
            "reports_live",
        ]
        print("Starting live runner (preferred - using -m):", runner)
        # ensure logs directory
        try:
            os.makedirs("logs", exist_ok=True)
            logf = open(
                os.path.join("logs", "bingx_bot_live_runner.log"), "a", buffering=1
            )
        except Exception:
            logf = None
        p = subprocess.Popen(runner, cwd=os.getcwd(), stdout=logf, stderr=logf)
    except Exception:
        # fallback: run the script file directly but set cwd to repository root so
        # relative imports resolve correctly
        runner = [
            sys.executable,
            "-u",
            "scripts/run_multi_venue_router.py",
            "--output-dir",
            "reports_live",
        ]
        print("Starting live runner (fallback):", runner)
        try:
            os.makedirs("logs", exist_ok=True)
            logf = open(
                os.path.join("logs", "bingx_bot_live_runner.log"), "a", buffering=1
            )
        except Exception:
            logf = None
        p = subprocess.Popen(runner, cwd=os.getcwd(), stdout=logf, stderr=logf)

    # write pid for kill_switch
    pid_file = Path(".live_bot.pid")
    pid_file.write_text(str(p.pid))
    print(f"Live runner started with pid {p.pid}. PID file written to {pid_file}")

    # start kill monitor
    t = start_kill_monitor_thread()
    print(
        "Kill monitor thread started. To trigger kill, create .live_bot_kill file in project root."
    )

    try:
        while True:
            ret = p.poll()
            if ret is not None:
                print("Live runner exited with code", ret)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, terminating child process")
        p.terminate()
        p.wait(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
