import subprocess
import sys
import threading
import time

from scripts.kill_switch import PID_FILE, monitor
from scripts.telemetry_service import run_server


def _start_telemetry_server():
    t = threading.Thread(target=lambda: run_server(9001), daemon=True)
    t.start()
    time.sleep(0.1)
    return t


def test_integration_flow(monkeypatch, tmp_path):
    # start telemetry server
    _start_telemetry_server()

    # create fake runner that writes heartbeat and a losing pnl after a short delay
    fake_runner = tmp_path / "fake_runner.py"
    fake_runner.write_text(
        """
import time, json
from pathlib import Path
# write pid file
Path('.live_bot.pid').write_text(str(999999))
# initial heartbeat
Path('.live_heartbeat').write_text(str(time.time()))
# sleep then write losing pnl
time.sleep(0.5)
Path('.live_pnl.json').write_text(json.dumps({'realized_pnl': -500.0, 'day_start_equity': 10000.0}))
"""
    )
    # run fake runner in background
    p = subprocess.Popen(
        [sys.executable, str(fake_runner)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    called = {}

    def fake_notify(msg):
        called["msg"] = msg

    monkeypatch.setattr("scripts.kill_switch._notify", fake_notify)

    # run monitor (it should detect the pnl breach and call _notify)
    thr = threading.Thread(target=lambda: monitor(interval=0.2), daemon=True)
    thr.start()
    time.sleep(1.5)
    p.terminate()
    assert "msg" in called
    try:
        PID_FILE.unlink()
    except Exception:
        pass
