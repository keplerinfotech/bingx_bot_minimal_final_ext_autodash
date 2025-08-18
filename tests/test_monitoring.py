import http.server
import json
import threading
import time

from scripts.notifier import notify_on_kill, send_email, send_slack
from scripts.telemetry_client import send_telemetry, write_heartbeat, write_pnl


def _start_test_server(handler_class, port_holder):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    port = server.server_port
    port_holder.append(port)

    def _serve():
        server.handle_request()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return server, t


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = None
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args, **kwargs):
        pass


def test_send_telemetry_posts():
    ports = []
    srv, t = _start_test_server(_EchoHandler, ports)
    port = ports[0]
    url = f"http://127.0.0.1:{port}/telemetry"
    ok = send_telemetry({"heartbeat": time.time()}, url=url)
    assert ok is True


def test_write_heartbeat_and_pnl_files(tmp_path):
    # ensure functions do not raise
    write_heartbeat()
    write_pnl(-1.23, 10000.0)


# Not a full SMTP/Slack integration test; just verify functions return bools


def test_notifier_sanity():
    # Slack: invalid URL should return False
    assert send_slack("http://127.0.0.1:1/nomatch", "test") is False
    # Email: using invalid host should return False
    assert send_email("127.0.0.1", 1025, "u", "p", "to@example.com", "s", "b") is False


def test_notify_on_kill_called(monkeypatch):
    called = {}

    def fake_slack(url, text):
        called["slack"] = (url, text)
        return True

    def fake_email(host, port, user, pw, to, subj, body):
        called["email"] = (host, port, to, subj)
        return True

    monkeypatch.setattr("scripts.notifier.send_slack", fake_slack)
    monkeypatch.setattr("scripts.notifier.send_email", fake_email)

    # Ensure notify_on_kill attempts to send by setting at least one alert env var
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "http://127.0.0.1/dummy")

    notify_on_kill("test-kill")
    assert "slack" in called or "email" in called
