"""Stdlib HTTP server for the operator console. Loopback only."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from compiler.llm.provider import provider_catalog
from compiler.ui.jobs import JobBusy, JobRunner
from compiler.ui.view import build_dashboard, report_rows, save_upload, scrub

STATIC = Path(__file__).resolve().parent / "static" / "index.html"
_LOOPBACK = {"127.0.0.1", "localhost"}
_MAX_UPLOAD = 200 * 1024 * 1024


def require_loopback(host: str) -> str:
    """Refuse any bind that is not this machine's loopback interface."""

    if host not in _LOOPBACK:
        raise ValueError("operator UI binds to 127.0.0.1 only")
    return "127.0.0.1"


class OperatorUI:
    def __init__(self, results: Path, uploads: Path | None = None) -> None:
        self.results = results
        self.uploads = uploads or (results.parent / "uploads")
        self.runner = JobRunner(results)

    def close(self) -> None:
        self.runner.close()


def _handler(app: OperatorUI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._bytes(200, STATIC.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/dashboard":
                self._json(200, build_dashboard(app.results, uploads=app.uploads))
                return
            if path == "/api/report":
                self._json(200, report_rows(app.results))
                return
            if path == "/api/providers":
                self._json(200, {"providers": provider_catalog(), "fps_claimed": False})
                return
            if path == "/api/jobs":
                self._json(200, {"jobs": app.runner.list_jobs()})
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/")
                try:
                    self._json(200, app.runner.get(job_id))
                except KeyError:
                    self._json(404, {"error": "unknown job"})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/upload":
                self._upload()
                return
            if path != "/api/jobs":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length > 65536:
                self._json(400, {"error": "body too large"})
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json(400, {"error": "body must be JSON"})
                return
            if not isinstance(payload, dict):
                self._json(400, {"error": "body must be a JSON object"})
                return
            action = str(payload.get("action") or "")
            params = {key: value for key, value in payload.items() if key != "action"}
            try:
                job = app.runner.submit(action, params)
            except JobBusy as exc:
                self._json(409, {"error": str(exc)})
                return
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(202, job)

        def _upload(self) -> None:
            query = urlparse(self.path).query
            name = ""
            for part in query.split("&"):
                key, _, value = part.partition("=")
                if key == "name":
                    name = unquote(value)
            length = int(self.headers.get("Content-Length") or "0")
            if length > _MAX_UPLOAD:
                self._json(400, {"error": "upload is over 200 MB"})
                return
            data = self.rfile.read(length) if length else b""
            try:
                dest = save_upload(app.uploads, name, data)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(
                201,
                {
                    "path": str(dest),
                    "kind": f"upload_{dest.stem}",
                    "fps_claimed": False,
                },
            )

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(scrub(payload)).encode("utf-8")
            self._bytes(status, body, "application/json")

        def _bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(host: str, port: int, results: Path) -> None:
    """Block and serve the console. Ctrl-C stops it."""

    bound = require_loopback(host)
    app = OperatorUI(results)
    server = ThreadingHTTPServer((bound, port), _handler(app))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()


def start_background(host: str, port: int, results: Path) -> tuple[ThreadingHTTPServer, OperatorUI]:
    """Bind and serve on a daemon thread. Used by tests."""

    import threading

    bound = require_loopback(host)
    app = OperatorUI(results)
    server = ThreadingHTTPServer((bound, port), _handler(app))
    thread = threading.Thread(target=server.serve_forever, name="nnc-ui-http", daemon=True)
    thread.start()
    return server, app
