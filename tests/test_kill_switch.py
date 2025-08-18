import http.server
import json
import threading
import time

from scripts.kill_switch import PID_FILE, monitor


class _FakeTelemetryHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/status":
            self.send_response(404)
            self.end_headers()
            return
        # telemetry with large negative realized_pnl to trigger breach
        payload = {
            "telemetry": {
                "realized_pnl": -500.0,
                "day_start_equity": 10000.0,
                "heartbeat": time.time(),
            }
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args, **kwargs):
        pass


def _start_server():
    # bind to an ephemeral port to avoid conflicts on CI or when a service is
    # already using 9001; update the kill_switch module to use this port
    srv = http.server.HTTPServer(("127.0.0.1", 0), _FakeTelemetryHandler)
    port = srv.server_port
    # Ensure the monitor points to this test server
    import scripts.kill_switch as ks

    ks.TELEMETRY_STATUS_URL = f"http://127.0.0.1:{port}/status"
    t = threading.Thread(target=srv.handle_request, daemon=True)
    t.start()
    return srv, t


def test_monitor_triggers(tmp_path, monkeypatch):
    # ensure pid file exists with a dummy pid (current PID not killed because we patch _signal_pid)
    PID_FILE.write_text("999999")
    called = {}

    def fake_signal(pid):
        called["pid"] = pid

    monkeypatch.setattr("scripts.kill_switch._signal_pid", fake_signal)

    srv, t = _start_server()
    # run monitor in a thread and let it poll once
    thr = threading.Thread(target=lambda: monitor(interval=0.5), daemon=True)
    thr.start()
    time.sleep(1.0)
    assert "pid" in called
    # cleanup
    try:
        PID_FILE.unlink()
    except Exception:
        pass
