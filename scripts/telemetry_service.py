"""Simple telemetry HTTP service for live runs.

POST /telemetry  - accept JSON payload and store in-memory
GET  /status     - return last telemetry JSON

Runs on localhost and is intended for local supervised runs only.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

_last: Optional[dict] = None


class TelemetryHandler(BaseHTTPRequestHandler):
    def _set_json(self, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        global _last
        parsed = urlparse(self.path)
        if parsed.path != "/telemetry":
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            _last = payload
            self._set_json(200)
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        except Exception:
            self._set_json(400)
            self.wfile.write(json.dumps({"error": "bad json"}).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/status":
            self._set_json(404)
            self.wfile.write(json.dumps({"error": "not found"}).encode())
            return
        self._set_json(200)
        self.wfile.write(json.dumps({"telemetry": _last}).encode())

    def log_message(self, format, *args):
        # keep logs minimal
        pass


def run_server(port: int = 9001):
    srv = ThreadingHTTPServer(("127.0.0.1", port), TelemetryHandler)
    print(f"Telemetry server listening on http://127.0.0.1:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    run_server()
