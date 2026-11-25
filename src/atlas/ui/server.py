"""Loopback-only HTTP server that streams a run's events to the browser dashboard.

Read-only by design: GET routes only, no CORS headers, Host/Origin checked against the
loopback address it is bound to (DNS-rebinding guard), and artifacts limited to an allowlist
inside the run folder.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from urllib.parse import parse_qs, unquote, urlsplit

from atlas.events import (
    RunEventBus,
    UILogHandler,
    attach,
    detach,
    load_run_events_with_offset,
    redact,
)

logger = logging.getLogger("atlas.ui")

ARTIFACTS: dict[str, str] = {
    "report.md": "text/markdown; charset=utf-8",
    "plan.md": "text/markdown; charset=utf-8",
    "spec.md": "text/markdown; charset=utf-8",
    "manifest.json": "application/json",
    "spec.json": "application/json",
    "research.json": "application/json",
    "clarifications.json": "application/json",
    "plan.json": "application/json",
    "e2e_tests.json": "application/json",
    "build.json": "application/json",
    "verification.json": "application/json",
    "supervisor_review.json": "application/json",
    "tokens.json": "application/json",
}
MAX_ARTIFACT_BYTES = 2_000_000
_WORKSPACE_FILE_CAP = 400
_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def ui_enabled(
    flag: bool | None,
    *,
    env: dict[str, str] | os._Environ[str] | None = None,
    stream: TextIO | None = None,
) -> bool:
    """`--ui` / `--no-ui` win; otherwise on only in an interactive terminal outside CI."""
    if flag is not None:
        return flag
    environ = os.environ if env is None else env
    for key in ("CI", "GITHUB_ACTIONS"):
        if str(environ.get(key, "")).strip().lower() not in ("", "0", "false", "no"):
            return False
    target = sys.stderr if stream is None else stream
    isatty = getattr(target, "isatty", None)
    try:
        return bool(isatty()) if callable(isatty) else False
    except (ValueError, OSError):
        return False


def _index_html() -> bytes:
    return resources.files("atlas.ui").joinpath("static/index.html").read_bytes()


def _announce(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] [ui] {message}", file=sys.stderr, flush=True)


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class WorkflowUIServer:
    """Serves the dashboard and the event stream for one run on 127.0.0.1."""

    def __init__(
        self,
        bus: RunEventBus,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        mode: str = "live",
        poll_seconds: float = 2.0,
        events_path: Path | None = None,
        events_offset: int | None = None,
    ) -> None:
        self.bus = bus
        self.mode = mode
        self.poll_seconds = poll_seconds
        self.events_path = events_path
        self.events_offset = events_offset
        self._stopping = threading.Event()
        self._clients = 0
        self._clients_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self.httpd = _HTTPServer((host, port), _make_handler(self))
        address = self.httpd.server_address
        self.host = str(address[0])
        self.port = int(address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def clients(self) -> int:
        with self._clients_lock:
            return self._clients

    def _client_delta(self, delta: int) -> None:
        with self._clients_lock:
            self._clients += delta

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def start(self) -> None:
        serve = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="atlas-ui-http",
            daemon=True,
        )
        serve.start()
        self._threads.append(serve)
        poll_target = self._watch_workspace if self.mode == "live" else self._tail_events
        poller = threading.Thread(target=poll_target, name="atlas-ui-poll", daemon=True)
        poller.start()
        self._threads.append(poller)

    def stop(self, grace_seconds: float = 0.0) -> None:
        """Let connected pages receive the final events (bounded), then shut down."""
        deadline = time.monotonic() + max(0.0, grace_seconds)
        while self.clients > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stopping.set()
        with self.bus._cond:  # wake streams waiting for events
            self.bus._cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        for thread in self._threads:
            thread.join(timeout=2.0)

    # -- background pollers ----------------------------------------------------------

    def _watch_workspace(self) -> None:
        """Report the product folder's files when they change (file list + stall timer)."""
        from atlas.harnesses.base import workspace_signature

        last: tuple[tuple[str, int, int], ...] | None = None
        while not self._stopping.wait(self.poll_seconds):
            app_dir = self.bus.app_dir
            if app_dir is None or self.bus.finished:
                continue
            try:
                signature = workspace_signature(app_dir)
            except OSError:
                continue
            if signature == last:
                continue
            if last is None:
                changed: list[str] = []
            else:
                prev_map = {p: (sz, mt) for p, sz, mt in last}
                changed = sorted(
                    p for p, sz, mt in signature if prev_map.get(p) != (sz, mt)
                )[:30]
            last = signature
            files = [[path, size] for path, size, _ in signature[:_WORKSPACE_FILE_CAP]]
            self.bus.add(
                "workspace",
                "build",
                None,
                "",
                {
                    "files": files,
                    "changed": changed,
                    "count": len(signature),
                    "truncated": len(signature) > _WORKSPACE_FILE_CAP,
                },
            )

    def _tail_events(self) -> None:
        """Replay mode: pick up new lines if the run is still writing `events.jsonl`."""
        path = self.events_path
        if path is None or not path.is_file():
            return
        offset = self.events_offset if self.events_offset is not None else path.stat().st_size
        while not self._stopping.wait(1.0):
            if self.bus.finished:
                return
            try:
                size = path.stat().st_size
                if size <= offset:
                    continue
                with path.open("rb") as handle:
                    handle.seek(offset)
                    raw_chunk = handle.read()
            except OSError:
                continue
            newline_pos = raw_chunk.rfind(b"\n")
            if newline_pos < 0:
                continue
            complete_bytes = raw_chunk[: newline_pos + 1]
            offset += len(complete_bytes)
            complete = complete_bytes.decode("utf-8", errors="replace")
            fresh: list[dict[str, Any]] = []
            last_seq = self.bus.last_seq
            for line in complete.splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict) and int(item.get("seq") or 0) > last_seq:
                    fresh.append(item)
            if fresh:
                self.bus.extend(fresh)


def _make_handler(ui: WorkflowUIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AtlasUI/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _allowed(self) -> bool:
            hosts = {f"127.0.0.1:{ui.port}", f"localhost:{ui.port}"}
            if self.headers.get("Host", "") not in hosts:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin in {f"http://{h}" for h in hosts}

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if content_type.startswith("text/html"):
                self.send_header("Content-Security-Policy", _CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if not self._allowed():
                self._send(403, b"Forbidden", "text/plain; charset=utf-8")
                return
            parts = urlsplit(self.path)
            path = parts.path
            if path in ("/", "/index.html"):
                self._send(200, _index_html(), "text/html; charset=utf-8")
            elif path == "/api/info":
                self._json(
                    {
                        "mode": ui.mode,
                        "run_id": ui.bus.run_id,
                        "out_dir": str(ui.bus.out_dir) if ui.bus.out_dir else None,
                        "finished": ui.bus.finished,
                    }
                )
            elif path == "/api/events":
                self._stream(parse_qs(parts.query))
            elif path.startswith("/api/artifact/"):
                self._artifact(unquote(path[len("/api/artifact/") :]))
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")

        def _artifact(self, name: str) -> None:
            out_dir = ui.bus.out_dir
            if name not in ARTIFACTS or out_dir is None:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            target = out_dir / name
            try:
                resolved = target.resolve(strict=True)
                if resolved.parent != out_dir.resolve() or not resolved.is_file():
                    raise FileNotFoundError(name)
                if resolved.stat().st_size > MAX_ARTIFACT_BYTES:
                    self._send(413, b"Artifact too large to show", "text/plain; charset=utf-8")
                    return
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            safe_body = redact(text, limit=MAX_ARTIFACT_BYTES).encode("utf-8")
            self._send(200, safe_body, ARTIFACTS[name])

        def _stream(self, query: dict[str, list[str]]) -> None:
            raw_after = self.headers.get("Last-Event-ID") or (query.get("after") or ["0"])[0]
            try:
                after = max(0, int(raw_after))
            except ValueError:
                after = 0
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            ui._client_delta(1)
            last_write = time.monotonic()
            try:
                self.wfile.write(b"retry: 2000\n\n")
                while not ui.stopping:
                    events = ui.bus.events_after(after, timeout=1.0)
                    if events:
                        for event in events:
                            line = json.dumps(event, ensure_ascii=False)
                            self.wfile.write(f"id: {event['seq']}\ndata: {line}\n\n".encode())
                            after = int(event["seq"])
                        self.wfile.flush()
                        last_write = time.monotonic()
                        continue
                    if ui.bus.finished:
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        break
                    if time.monotonic() - last_write >= 15:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_write = time.monotonic()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                ui._client_delta(-1)

    return Handler


class LiveWorkflowUI:
    """Context manager the CLI wraps around `run_build` to show the run live.

    Never stops a build: if the server cannot start, it warns and the run goes on headless.
    """

    def __init__(
        self,
        *,
        port: int = 0,
        open_browser: bool = True,
        keep_open: bool = False,
    ) -> None:
        self.port = port
        self.open_browser = open_browser
        self.keep_open = keep_open
        self.bus = RunEventBus()
        self.server: WorkflowUIServer | None = None
        self._handler: UILogHandler | None = None

    def __enter__(self) -> LiveWorkflowUI:
        try:
            self.server = WorkflowUIServer(self.bus, port=self.port, mode="live")
            self.server.start()
        except OSError as exc:
            _announce(f"Could not start the workflow UI ({exc}); continuing without it.")
            self.server = None
            return self
        attach(self.bus)
        self._handler = UILogHandler(self.bus)
        logging.getLogger("atlas").addHandler(self._handler)
        _announce(f"Live workflow UI: {self.server.url}")
        if self.open_browser:
            try:
                opened = webbrowser.open(self.server.url, new=2)
            except Exception:
                opened = False
            if not opened:
                _announce(f"Could not open a browser; open {self.server.url} yourself.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.server is None:
            return
        if exc is not None and not self.bus.finished:
            status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.bus.add(
                "run",
                "run",
                status,
                "",
                {"result": status, "error": f"{type(exc).__name__}: {exc}"},
            )
        if self.keep_open and exc_type is not KeyboardInterrupt:
            _announce(
                f"Run finished; keeping workflow UI open at {self.server.url} (Ctrl+C to stop)"
            )
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
        if self._handler is not None:
            logging.getLogger("atlas").removeHandler(self._handler)
        detach(self.bus)
        self.server.stop(grace_seconds=0.0 if self.keep_open else 2.0)
        self.bus.close()
        if self.bus.out_dir is not None:
            _announce(f"Replay this run any time: atlas ui {self.bus.out_dir}")


def latest_run_dir(runs_root: Path) -> Path:
    """Newest run folder under `runs_root` (by name, which starts with a UTC timestamp)."""
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    candidates = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    return candidates[-1]


def serve_replay(
    run_dir: Path,
    *,
    port: int = 0,
    open_browser: bool = True,
    wait: bool = True,
) -> WorkflowUIServer:
    """Serve a past (or still running) run until Ctrl+C. Returns the server when `wait=False`."""
    run_dir = run_dir.resolve()
    events, offset = load_run_events_with_offset(run_dir)
    bus = RunEventBus(persist=False)
    bus.extend(events)
    bus.out_dir = run_dir
    events_path = run_dir / "events.jsonl"
    server = WorkflowUIServer(
        bus,
        port=port,
        mode="replay",
        events_path=events_path if events_path.is_file() else None,
        events_offset=offset,
    )
    server.start()
    _announce(f"Replaying {run_dir.name}: {server.url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            webbrowser.open(server.url, new=2)
        except Exception:
            pass
    if not wait:
        return server
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return server
