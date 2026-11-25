"""The local workflow UI server: page, event stream, guards, workspace watcher, lifecycle."""

from __future__ import annotations

import http.client
import io
import json
import logging
import time

import pytest

from atlas.events import RunEventBus, detach, publish, publishing
from atlas.ui import LiveWorkflowUI, WorkflowUIServer, serve_replay, ui_enabled
from atlas.ui.server import latest_run_dir


def _get(server: WorkflowUIServer, path: str, headers: dict | None = None, timeout: float = 5):
    conn = http.client.HTTPConnection(server.host, server.port, timeout=timeout)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def _sse_events(body: bytes) -> list[dict]:
    out = []
    for line in body.decode().splitlines():
        if line.startswith("data: {") and line != "data: {}":
            out.append(json.loads(line[6:]))
    return out


@pytest.fixture
def finished_bus(tmp_path):
    bus = RunEventBus()
    out = tmp_path / "runs" / "r1"
    out.mkdir(parents=True)
    (out / "report.md").write_text("# Report\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    bus.add("run", "run", "running", "", {"run_id": "r1", "out_dir": str(out)})
    bus.add("stage", "parse", "passed", "", {"requirements": 2})
    bus.add("run", "run", "passed", "", {"result": "verified"})
    yield bus
    bus.close()


@pytest.fixture
def server(finished_bus):
    srv = WorkflowUIServer(finished_bus, mode="replay")
    srv.start()
    yield srv
    srv.stop()


def test_page_is_served_with_a_strict_content_security_policy(server):
    resp, body = _get(server, "/")
    assert resp.status == 200
    assert b"Atlas" in body and b"EventSource" in body
    csp = resp.getheader("Content-Security-Policy")
    assert "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert resp.getheader("Access-Control-Allow-Origin") is None


def test_event_stream_replays_everything_then_ends(server):
    resp, body = _get(server, "/api/events")
    assert resp.getheader("Content-Type").startswith("text/event-stream")
    assert [e["seq"] for e in _sse_events(body)] == [1, 2, 3]
    assert b"event: end" in body


def test_event_stream_resumes_after_last_event_id(server):
    _, body = _get(server, "/api/events", {"Last-Event-ID": "2"})
    assert [e["seq"] for e in _sse_events(body)] == [3]
    _, body = _get(server, "/api/events?after=1")
    assert [e["seq"] for e in _sse_events(body)] == [2, 3]


@pytest.mark.parametrize(
    "headers",
    [{"Host": "evil.example"}, {"Origin": "http://evil.example"}],
)
def test_other_hosts_and_origins_are_refused(server, headers):
    resp, _ = _get(server, "/api/events", headers)
    assert resp.status == 403


def test_only_allowlisted_run_artifacts_are_served_and_redacted(server, finished_bus):
    out = finished_bus.out_dir
    assert out is not None
    (out / "report.md").write_text(
        "# Report\nKey: sk-or-v1-abcdefghijklmno123456\n", encoding="utf-8"
    )
    resp, body = _get(server, "/api/artifact/report.md")
    assert resp.status == 200
    assert b"[redacted]" in body and b"sk-or-" not in body

    resp, info_body = _get(server, "/api/info")
    assert resp.status == 200
    info = json.loads(info_body)
    assert info["mode"] == "replay" and info["run_id"] == "r1" and info["finished"] is True

    for path in ("/api/artifact/.env", "/api/artifact/..%2F..%2F.env",
                 "/api/artifact/plan.md", "/etc/passwd", "/api/artifact/manifest.json/../x"):
        resp, _ = _get(server, path)
        assert resp.status == 404, path


def test_live_stream_delivers_new_events_and_ends_when_the_run_does(tmp_path):
    bus = RunEventBus()
    bus.add("run", "run", "running", "", {"run_id": "r2"})
    srv = WorkflowUIServer(bus, mode="live")
    srv.start()
    try:
        conn = http.client.HTTPConnection(srv.host, srv.port, timeout=10)
        conn.request("GET", "/api/events")
        resp = conn.getresponse()
        bus.add("stage", "parse", "running")
        bus.add("run", "run", "failed", "", {"error": "x"})
        body = resp.read()
        conn.close()
    finally:
        srv.stop()
        bus.close()
    assert [e["kind"] for e in _sse_events(body)] == ["run", "stage", "run"]


def test_workspace_watcher_reports_files_in_the_product_folder(tmp_path):
    app = tmp_path / "app"
    (app / "node_modules").mkdir(parents=True)
    (app / "node_modules" / "ignored.js").write_text("x")
    (app / "PLAN.md").write_text("# Seeded\n")
    bus = RunEventBus(persist=False)
    bus.add("run", "run", "running", "", {"run_id": "r", "app_dir": str(app)})
    srv = WorkflowUIServer(bus, mode="live", poll_seconds=0.05)
    srv.start()
    try:
        deadline = time.monotonic() + 3
        initial_ws: list = []
        while time.monotonic() < deadline:
            initial_ws = [e for e in bus.events_after(0) if e["kind"] == "workspace"]
            if initial_ws:
                break
            time.sleep(0.05)
        assert initial_ws and initial_ws[0]["data"]["changed"] == []

        (app / "index.html").write_text("<html>")
        files: list = []
        ws: list = []
        while time.monotonic() < deadline:
            ws = [e for e in bus.events_after(0) if e["kind"] == "workspace"]
            if ws and ws[-1]["data"]["count"] == 2:
                files = ws[-1]["data"]["files"]
                break
            time.sleep(0.05)
    finally:
        srv.stop()
    assert [f[0] for f in files] == ["PLAN.md", "index.html"]
    assert ws[-1]["data"]["changed"] == ["index.html"]


def test_replay_tails_from_exact_offset_and_serves_valid_page(tmp_path, monkeypatch):
    from atlas.ui.server import _index_html

    monkeypatch.setattr("sys.stderr", io.StringIO())
    out = tmp_path / "r_tail"
    out.mkdir()
    e1 = json.dumps({
        "v": 1, "seq": 1, "ts": 1.0, "run_id": "r_tail", "kind": "run",
        "stage": "run", "status": "running", "message": "", "data": {"run_id": "r_tail"},
    })
    (out / "events.jsonl").write_text(e1 + "\n", encoding="utf-8")
    srv = serve_replay(out, open_browser=False, wait=False)
    try:
        # Append seq 2 after replay starts; offset ensures seq 2 is picked up once.
        e2 = json.dumps({
            "v": 1, "seq": 2, "ts": 2.0, "run_id": "r_tail", "kind": "run",
            "stage": "run", "status": "passed", "message": "", "data": {"result": "verified"},
        })
        with (out / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(e2 + "\n")
        _, body = _get(srv, "/api/events")
        seqs = [e["seq"] for e in _sse_events(body)]
        assert seqs == [1, 2]
    finally:
        srv.stop()
    html = _index_html().decode("utf-8")
    assert "loadInfo" in html and "EventSource" in html


def test_live_ui_attaches_for_the_run_and_records_a_crash(monkeypatch, tmp_path):
    opened: list[str] = []
    monkeypatch.setattr("atlas.ui.server.webbrowser.open", lambda url, new=0: opened.append(url))
    monkeypatch.setattr("sys.stderr", io.StringIO())
    ui = LiveWorkflowUI(open_browser=True)
    with pytest.raises(RuntimeError, match="boom"), ui:
        assert publishing() is True
        assert opened == [ui.server.url]
        logging.getLogger("atlas.runner").warning("[build] something")
        raise RuntimeError("boom")
    assert publishing() is False
    kinds = [(e["kind"], e["status"]) for e in ui.bus.events_after(0)]
    assert ("log", None) in kinds
    assert kinds[-1] == ("run", "failed")
    assert all(not isinstance(h, type(ui._handler)) for h in logging.getLogger("atlas").handlers)


def test_live_ui_never_blocks_a_build_when_the_port_is_taken(monkeypatch):
    monkeypatch.setattr("sys.stderr", io.StringIO())
    first = LiveWorkflowUI(open_browser=False)
    with first:
        assert first.server is not None
        second = LiveWorkflowUI(port=first.server.port, open_browser=False)
        with second:
            assert second.server is None  # warned, carried on without a UI
    for b in (first.bus, second.bus):
        detach(b)


def test_replay_serves_an_old_run(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stderr", io.StringIO())
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "old", "status": "failed", "stages": {"parse": "completed"},
                    "error": "HarnessError: timed out", "mode": "build"}),
        encoding="utf-8",
    )
    srv = serve_replay(tmp_path, open_browser=False, wait=False)
    try:
        _, body = _get(srv, "/api/events")
    finally:
        srv.stop()
    events = _sse_events(body)
    assert events[0]["data"]["run_id"] == "old"
    assert events[-1]["kind"] == "run" and events[-1]["status"] == "failed"


class _Stream(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.parametrize(
    ("flag", "env", "tty", "expected"),
    [
        (None, {}, True, True),
        (None, {}, False, False),
        (None, {"CI": "true"}, True, False),
        (None, {"GITHUB_ACTIONS": "1"}, True, False),
        (None, {"CI": "0"}, True, True),
        (True, {"CI": "true"}, False, True),
        (False, {}, True, False),
    ],
)
def test_ui_turns_on_only_in_an_interactive_terminal(flag, env, tty, expected):
    assert ui_enabled(flag, env=env, stream=_Stream(tty)) is expected


def test_publish_reaches_an_attached_live_ui(monkeypatch):
    monkeypatch.setattr("sys.stderr", io.StringIO())
    ui = LiveWorkflowUI(open_browser=False)
    with ui:
        publish("stage", "parse", "running", target="m")
    assert ui.bus.events_after(0)[0]["stage"] == "parse"


def test_latest_run_dir_reports_missing_runs_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Runs directory not found"):
        latest_run_dir(tmp_path / "missing")
