"""Basic kill-switch monitor for supervised/live runs.

Supports two triggers:
- file trigger: create `.live_bot_kill` to terminate the PID in `.live_bot.pid`
- PnL-based trigger: monitor `.live_pnl.json` (written by the trading process)
  and terminate when drawdown (bps) exceeds configured threshold.

The script is conservative: it handles missing PID/process and exits after
triggering once. Designed for supervised runs; for production use a proper
supervisor and alerting stack.
"""

from __future__ import annotations

import json
import os
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# yaml is imported lazily inside _load_threshold_from_config when available


PID_FILE = Path(".live_bot.pid")
KILL_FLAG = Path(".live_bot_kill")
PNL_FILE = Path(".live_pnl.json")
HEARTBEAT_FILE = Path(".live_heartbeat")
TELEMETRY_STATUS_URL = "http://127.0.0.1:9001/status"
HEARTBEAT_TIMEOUT = 300.0  # seconds; if no heartbeat within this, consider hung


def write_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid))


def read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def _read_pnl_file(p: Path) -> Optional[dict]:
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception:
        return None


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_threshold_from_config(
    settings_path: str = "config/settings.live.yaml",
) -> float:
    # Return threshold in basis points (bps). Default conservative: 50 bps.
    default_bps = 50.0
    try:
        # import yaml locally to avoid requiring it at module import time
        try:
            import yaml as _yaml
        except Exception:
            _yaml = None
        if _yaml is None:
            return default_bps
        if not os.path.exists(settings_path):
            return default_bps
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        risk = cfg.get("risk", {})
        v = risk.get("max_daily_loss_bps")
        if v is None:
            return default_bps
        return float(v)
    except Exception:
        return default_bps


def _load_heartbeat_timeout(settings_path: str = "config/settings.live.yaml") -> float:
    default_timeout = HEARTBEAT_TIMEOUT
    try:
        try:
            import yaml as _yaml
        except Exception:
            _yaml = None
        if _yaml is None:
            return default_timeout
        if not os.path.exists(settings_path):
            return default_timeout
        with open(settings_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        v = cfg.get("heartbeat_timeout_seconds")
        if v is None:
            # also check top-level key 'heartbeat_timeout_seconds'
            return default_timeout
        return float(v)
    except Exception:
        return default_timeout


def _signal_pid(pid: int) -> None:
    try:
        print(f"Sending SIGTERM to pid {pid}")
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print("Target process not found; nothing to kill.")
    except PermissionError:
        print("Permission denied sending signal to pid", pid)


def _notify(message: str) -> None:
    # Best-effort notification using scripts.notifier or local notifier module.
    try:
        from scripts.notifier import notify_on_kill
    except Exception:
        try:
            from notifier import notify_on_kill
        except Exception:
            return
    try:
        notify_on_kill(message)
    except Exception:
        pass


def monitor(
    interval: float = 5.0, settings_path: str = "config/settings.live.yaml"
) -> None:
    """Main monitor loop: watch for file-kill or PnL breach.

    PnL file format expected (JSON):
    {
      "realized_pnl": -123.45,    # negative = loss
      "day_start_equity": 10000.0
    }
    """
    threshold_bps = _load_threshold_from_config(settings_path)
    heartbeat_timeout = _load_heartbeat_timeout(settings_path)
    print("Kill-switch monitor started. file-kill=", KILL_FLAG, "pnL-file=", PNL_FILE)
    print(f"Using drawdown threshold (bps) = {threshold_bps}")
    print(f"Heartbeat timeout (s) = {heartbeat_timeout}")

    while True:
        # File-based manual kill
        if KILL_FLAG.exists():
            pid = read_pid()
            if pid:
                print("Manual kill file detected — terminating target process")
                _signal_pid(pid)
            else:
                print("Manual kill file detected but no pid found.")
            try:
                KILL_FLAG.unlink()
            except Exception:
                pass
            break

        # Prefer telemetry HTTP endpoint for PnL/heartbeat when available
        telemetry = None
        try:
            req = urllib.request.Request(TELEMETRY_STATUS_URL, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                raw = resp.read()
                data = json.loads(raw.decode("utf-8")) if raw else {}
                telemetry = data.get("telemetry") if isinstance(data, dict) else None
        except Exception:
            telemetry = None

        # Check heartbeat (telemetry preferred, fallback to heartbeat file)
        hb_ts = None
        if telemetry and isinstance(telemetry, dict):
            hb_val = telemetry.get("heartbeat")
            try:
                hb_ts = float(hb_val) if hb_val is not None else None
            except Exception:
                hb_ts = None
        if hb_ts is None and HEARTBEAT_FILE.exists():
            try:
                hb_ts = float(HEARTBEAT_FILE.read_text().strip())
            except Exception:
                hb_ts = None

        if hb_ts is not None:
            age = time.time() - hb_ts
            if age > heartbeat_timeout:
                msg = f"Heartbeat stale: last={hb_ts} age_s={age:.1f} > {heartbeat_timeout} -> killing"
                print(msg)
                pid = read_pid()
                if pid:
                    _signal_pid(pid)
                _notify(msg)
                break

        # PnL-based auto kill: telemetry preferred, fallback to file
        pnl = None
        if telemetry and isinstance(telemetry, dict):
            # telemetry may contain pnl keys at top-level
            if "realized_pnl" in telemetry and "day_start_equity" in telemetry:
                pnl = {
                    "realized_pnl": telemetry.get("realized_pnl"),
                    "day_start_equity": telemetry.get("day_start_equity"),
                }
        if pnl is None:
            pnl = _read_pnl_file(PNL_FILE)

        if pnl:
            try:
                realized = _safe_float(pnl.get("realized_pnl"), 0.0)
                day_eq = _safe_float(pnl.get("day_start_equity"), 0.0)
                if day_eq > 0:
                    dd_bps = 10000.0 * (max(0.0, -realized) / day_eq)
                    if dd_bps >= threshold_bps:
                        msg = f"Drawdown breach: realized_pnl={realized} day_start_equity={day_eq} dd_bps={dd_bps} >= {threshold_bps}"
                        print(msg)
                        pid = read_pid()
                        if pid:
                            _signal_pid(pid)
                        else:
                            print(
                                "No pid to signal; write .live_bot.pid to target the process."
                            )
                        _notify(msg)
                        break
            except Exception as e:
                print("Error parsing pnl:", e)

        time.sleep(interval)


if __name__ == "__main__":
    monitor()
