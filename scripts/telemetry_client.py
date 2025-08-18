"""Helper to post telemetry and write heartbeat files for supervised runs."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

TELEM_URL = "http://127.0.0.1:9001/telemetry"
HEARTBEAT = Path(".live_heartbeat")
PNL_FILE = Path(".live_pnl.json")


def send_telemetry(payload: dict[str, Any], url: str | None = None) -> bool:
    url = url or TELEM_URL
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            # urllib responses vary; prefer getcode()
            code = getattr(resp, "getcode", None)
            if callable(code):
                return code() == 200
            return getattr(resp, "status", 200) == 200
    except Exception:
        return False


def write_heartbeat():
    ts = str(time.time())
    try:
        HEARTBEAT.write_text(ts)
    except Exception:
        pass
    # best-effort POST to telemetry endpoint so monitors can read via HTTP
    try:
        send_telemetry({"heartbeat": ts})
    except Exception:
        pass


def write_pnl(realized_pnl: float, day_start_equity: float):
    payload = {"realized_pnl": realized_pnl, "day_start_equity": day_start_equity}
    try:
        PNL_FILE.write_text(json.dumps(payload))
    except Exception:
        pass
    try:
        send_telemetry(payload)
    except Exception:
        pass
