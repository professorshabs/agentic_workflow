"""Tests for the native Python LocalOpenAIAdapter workspace builder."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import socketserver
import threading
from typing import Any

import pytest
import yaml

from atlas.harnesses import HarnessError, LocalOpenAIAdapter, get_local_harness_adapter
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.pipeline.build import build_app_workspace
from atlas.router import ModelRouter
from atlas.schemas import ParsedSpec, Requirement


def _candidate(base_url: str, name: str = "google/gemma-4-12b-qat") -> LocalModelCandidate:
    return LocalModelCandidate(
        provider="lm_studio",
        name=name,
        base_url=base_url,
        parameter_size_b=12.0,
        supports_tools=True,
    )


def _start_mock_openai_server(
    handler_fn: Any,
) -> tuple[str, Any, threading.Thread]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            status, ctype, chunks = handler_fn(self.path, payload)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
                self.wfile.flush()

    class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = _Srv(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1", srv, thread


def _sse_tool_turn(
    tool_id: str,
    name: str,
    args_str: str,
    *,
    usage: dict[str, Any],
) -> list[str]:
    mid = max(1, len(args_str) // 2)
    c1 = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args_str[:mid]},
                        }
                    ],
                },
            }
        ]
    }
    c2 = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": args_str[mid:]},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": usage,
    }
    return [f"data: {json.dumps(c1)}\n\n", f"data: {json.dumps(c2)}\n\n", "data: [DONE]\n\n"]


def _sse_text_turn(text: str, *, finish_reason: str = "stop", usage: dict[str, Any]) -> list[str]:
    c1 = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}
    c2 = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage}
    return [f"data: {json.dumps(c1)}\n\n", f"data: {json.dumps(c2)}\n\n", "data: [DONE]\n\n"]


async def test_local_openai_adapter_multiturn_sse_tool_loop(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "BUILD_BRIEF.md").write_text("Build app.py printing hello.\n", encoding="utf-8")

    requests_seen: list[dict[str, Any]] = []

    def handler(path: str, payload: dict[str, Any]):
        assert path == "/v1/chat/completions"
        requests_seen.append(payload)
        turn = len(requests_seen)
        u = {
            "prompt_tokens": 100 * turn,
            "completion_tokens": 20,
            "total_tokens": 100 * turn + 20,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
        if turn == 1:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn("call_1", "list_files", json.dumps({"path": "."}), usage=u),
            )
        if turn == 2:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "call_2", "read_file", json.dumps({"path": "BUILD_BRIEF.md"}), usage=u
                ),
            )
        if turn == 3:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "call_3",
                    "write_file",
                    json.dumps({"path": "app.py", "content": "print('hi')\n"}),
                    usage=u,
                ),
            )
        if turn == 4:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "call_4",
                    "edit_file",
                    json.dumps({"path": "app.py", "old_text": "hi", "new_text": "hello"}),
                    usage=u,
                ),
            )
        return (
            200,
            "text/event-stream",
            _sse_text_turn("Created app.py.\n\nSkills used: none", usage=u),
        )

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        adapter = LocalOpenAIAdapter()
        res, inv = await adapter.run_workspace_task(
            candidate=_candidate(base_url),
            prompt_text="Read BUILD_BRIEF.md and write app.py.",
            app_dir=app_dir,
            reasoning_effort="none",
            max_completion_tokens=4096,
            stall_seconds=None,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert (app_dir / "app.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert res.harness == "local_openai"
    assert res.created_files == ["app.py"]
    assert len(requests_seen) == 5
    # `run_command` omitted when allow_shell=False
    tool_names = [t["function"]["name"] for t in requests_seen[0]["tools"]]
    assert tool_names == ["list_files", "read_file", "write_file", "edit_file"]
    assert requests_seen[0]["reasoning_effort"] == "none"
    assert requests_seen[0]["max_completion_tokens"] == 4096
    assert inv.usage_reported is True
    assert inv.prompt_tokens == 1500
    assert inv.completion_tokens == 100
    assert inv.reasoning_tokens == 25
    assert inv.total_tokens == 1600


async def test_local_openai_adapter_workspace_boundaries_and_skill_symlinks(tmp_path):
    from atlas.harnesses.local_openai import _read_file_tool, _write_file_tool

    app_dir = tmp_path / "app"
    skill_src = tmp_path / "external_skills" / "demo"
    secret = tmp_path / "secret.txt"
    app_dir.mkdir()
    skill_src.mkdir(parents=True)
    secret.write_text("TOP_SECRET", encoding="utf-8")
    (skill_src / "SKILL.md").write_text("# Demo Skill\nUse clean code.", encoding="utf-8")
    (app_dir / ".agents" / "skills").mkdir(parents=True)
    os.symlink(skill_src, app_dir / ".agents" / "skills" / "demo")

    # Direct boundary check for escaping symlink without tripping snapshot_file_hashes.
    escape_dir = tmp_path / "app_symlink_check"
    escape_dir.mkdir()
    os.symlink(secret, escape_dir / "escape_link.txt")
    with pytest.raises(ValueError, match="resolves outside workspace directory"):
        _read_file_tool(escape_dir.resolve(), "escape_link.txt")
    with pytest.raises(ValueError, match="escapes workspace directory"):
        _write_file_tool(escape_dir.resolve(), "../pwned.txt", "bad")

    tool_outputs: list[str] = []
    step = 0

    def handler(path: str, payload: dict[str, Any]):
        nonlocal step
        step += 1
        if step > 1:
            tool_outputs.append(payload["messages"][-1]["content"])
        u = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if step == 1:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c1", "read_file", json.dumps({"path": "../secret.txt"}), usage=u
                ),
            )
        if step == 2:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c2",
                    "write_file",
                    json.dumps({"path": "../pwned.txt", "content": "bad"}),
                    usage=u,
                ),
            )
        if step == 3:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c3",
                    "write_file",
                    json.dumps({"path": ".agents/skills/demo/SKILL.md", "content": "bad"}),
                    usage=u,
                ),
            )
        if step == 4:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c4",
                    "read_file",
                    json.dumps({"path": ".agents/skills/demo/SKILL.md"}),
                    usage=u,
                ),
            )
        if step == 5:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c5", "run_command", json.dumps({"command": "echo hi"}), usage=u
                ),
            )
        return 200, "text/event-stream", _sse_text_turn("Done.", usage=u)

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        await LocalOpenAIAdapter().run_workspace_task(
            candidate=_candidate(base_url),
            prompt_text="Test boundaries",
            app_dir=app_dir,
            allow_shell=False,
            stall_seconds=None,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert not (tmp_path / "pwned.txt").exists()
    assert "escapes workspace directory" in tool_outputs[0]
    assert "escapes workspace directory" in tool_outputs[1]
    assert "Modifying files inside '.agents/' is not permitted" in tool_outputs[2]
    assert "# Demo Skill" in tool_outputs[3]
    assert "run_command is disabled" in tool_outputs[4]


async def test_local_openai_adapter_allows_shell_with_scrubbed_env_and_recovers_truncated_json(
    tmp_path, monkeypatch
):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    turn = 0

    observed_payloads: list[dict[str, Any]] = []

    def handler(path: str, payload: dict[str, Any]):
        nonlocal turn
        turn += 1
        observed_payloads.append(payload)
        u = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if turn == 1:
            # Truncated JSON arguments must NOT be replayed in Turn 2 tool_calls.
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c1", "write_file", '{"path": "ok.py", "content": "unterminated', usage=u
                ),
            )
        if turn == 2:
            # Verify every historical tool_call sent back to the server has valid JSON args.
            for msg in payload["messages"]:
                for tc in msg.get("tool_calls") or []:
                    json.loads(tc["function"]["arguments"])
            # Truncated text turn nudges the model to call write_file.
            return (
                200,
                "text/event-stream",
                _sse_text_turn("Thinking out loud...", finish_reason="length", usage=u),
            )
        if turn == 3:
            cmd = (
                'python3 -c "import os; print(os.getenv(\'OPENAI_API_KEY\', \'NONE\'))" '
                "> env.txt"
            )
            return (
                200,
                "text/event-stream",
                _sse_tool_turn("c3", "run_command", json.dumps({"command": cmd}), usage=u),
            )
        return 200, "text/event-stream", _sse_text_turn("Done.", usage=u)

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        res, _ = await LocalOpenAIAdapter().run_workspace_task(
            candidate=_candidate(base_url),
            prompt_text="Check shell",
            app_dir=app_dir,
            allow_shell=True,
            stall_seconds=None,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert "env.txt" in res.created_files
    assert (app_dir / "env.txt").read_text(encoding="utf-8").strip() == "NONE"
    assert not any(msg.get("tool_calls") for msg in observed_payloads[1]["messages"])


async def test_local_openai_adapter_breaks_repetition_loop_and_recovers(tmp_path):
    from atlas.harnesses.local_openai import _has_repetition_loop

    assert _has_repetition_loop("prefix " + "6c5JPQ0aHBwgGB" * 10) is True
    assert _has_repetition_loop("/" * 120) is False
    assert _has_repetition_loop("<div><p>Hello Claremont home</p></div>" * 3) is False

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    observed_payloads: list[dict[str, Any]] = []
    turn = 0

    def handler(path: str, payload: dict[str, Any]):
        nonlocal turn
        turn += 1
        observed_payloads.append(payload)
        u = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        if turn == 1:
            repeated = "6c5JPQ0aHBwgGB" * 30
            args_str = '{"path": "index.html", "content": "<img src=\\"' + repeated
            # Stream chunks in >=300-char segments so the loop detector fires mid-stream.
            d1 = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "loop_1",
                                    "type": "function",
                                    "function": {"name": "write_file", "arguments": args_str[:60]},
                                }
                            ],
                        },
                    }
                ]
            }
            d2 = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": args_str[60:]}}]
                        },
                    }
                ]
            }
            chunks = [
                f"data: {json.dumps(d1)}\n\n",
                f"data: {json.dumps(d2)}\n\n",
                "data: [DONE]\n\n",
            ]
            return 200, "text/event-stream", chunks
        if turn == 2:
            # Recovery message asks to avoid repeating base64 blobs, and no invalid tool_calls.
            assert not any(msg.get("tool_calls") for msg in payload["messages"])
            assert "repeating output loop" in payload["messages"][-1]["content"]
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "ok_1",
                    "write_file",
                    json.dumps({"path": "index.html", "content": "<h1>Clean</h1>\n"}),
                    usage=u,
                ),
            )
        return 200, "text/event-stream", _sse_text_turn("Finished index.html.", usage=u)

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        res, _ = await LocalOpenAIAdapter().run_workspace_task(
            candidate=_candidate(base_url),
            prompt_text="Build page.",
            app_dir=app_dir,
            stall_seconds=None,
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert res.created_files == ["index.html"]
    assert (app_dir / "index.html").read_text(encoding="utf-8") == "<h1>Clean</h1>\n"


async def test_local_openai_adapter_stall_preserves_partial_usage(tmp_path):
    import time

    app_dir = tmp_path / "app"
    app_dir.mkdir()

    async def stall_immediately(app_dir, stall_seconds, poll_seconds):
        await asyncio.sleep(0.05)
        return stall_seconds

    import atlas.harnesses.local_openai as lo_mod

    orig_wait = lo_mod._wait_for_stall
    lo_mod._wait_for_stall = stall_immediately  # type: ignore[assignment]
    turn = 0

    def handler(path: str, payload: dict[str, Any]):
        nonlocal turn
        turn += 1
        u = {"prompt_tokens": 250, "completion_tokens": 50, "total_tokens": 300}
        if turn >= 2:
            time.sleep(0.3)
        return (
            200,
            "text/event-stream",
            _sse_tool_turn("c1", "list_files", json.dumps({"path": "."}), usage=u),
        )

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        with pytest.raises(HarnessError, match="stalled") as info:
            await LocalOpenAIAdapter().run_workspace_task(
                candidate=_candidate(base_url),
                prompt_text="Stall test",
                app_dir=app_dir,
                stall_seconds=0.05,
                timeout_seconds=10.0,
            )
    finally:
        lo_mod._wait_for_stall = orig_wait  # type: ignore[assignment]
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert info.value.usage["prompt_tokens"] >= 250
    assert info.value.usage["total_tokens"] >= 300


async def test_build_app_workspace_dispatches_to_local_openai_and_forwards_max_turns(
    monkeypatch, tmp_path
):
    cand = LocalModelCandidate(
        provider="lm_studio",
        name="google/gemma-4-12b-qat",
        base_url="http://localhost:1234/v1",
        parameter_size_b=12.0,
        supports_tools=True,
    )
    seen: dict[str, Any] = {}

    async def ready(**kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand])

    async def fake_local_openai(self, **kwargs):
        seen.update(kwargs)
        d = kwargs["app_dir"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text("<h1>Hi</h1>\n", encoding="utf-8")
        from atlas.harnesses.base import HarnessInvocationResult
        from atlas.schemas import WorkspaceBuildResult

        return (
            WorkspaceBuildResult(
                summary="Built index.html\n\nSkills used: none",
                harness="local_openai",
                model="lm_studio/google/gemma-4-12b-qat",
                app_dir=str(d),
                created_files=["index.html"],
                files_snapshot=["index.html"],
            ),
            HarnessInvocationResult(
                text="done",
                used_model="lm_studio/google/gemma-4-12b-qat",
                harness="local_openai",
                auth_mode="local",
                cost_source="local_free",
                usage_reported=True,
                prompt_tokens=200,
                completion_tokens=40,
                total_tokens=240,
            ),
        )

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr(
        "atlas.pipeline.build.LocalOpenAIAdapter.run_workspace_task", fake_local_openai
    )

    cfg = {
        "roles": {"builder": ["local-auto"]},
        "backends": {
            "builder": {
                "type": "local_harness",
                "local_harness": "local_openai",
                "local_providers": {"lm_studio": "http://localhost:1234/v1"},
                "reasoning_effort": "none",
                "max_completion_tokens": 8192,
                "max_turns": 16,
                "offload": [],
            }
        },
        "limits": {
            "max_parallel_calls": 2,
            "max_tokens": 2000,
            "timeout_seconds": 30,
            "attempts_per_model": 1,
            "repair_attempts": 0,
        },
    }
    cfg_path = tmp_path / "local_openai.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    router = ModelRouter(cfg_path)
    spec = ParsedSpec(
        summary="Page",
        requirements=[Requirement(id="R1", title="Page", description="Render page")],
    )
    from atlas.schemas import E2ETestSuite, GeneratedFile

    e2e = E2ETestSuite(
        verifiable=True,
        summary="Page E2E checks",
        product_interface="index.html at repo root",
        files=[GeneratedFile(path="tests/test_e2e_product.py", content="def test_ok(): pass\n")],
    )

    res = await build_app_workspace(router, spec, tmp_path / "app" / "x", e2e_suite=e2e)
    assert res.harness == "local_openai"
    assert seen["reasoning_effort"] == "none"
    assert seen["max_completion_tokens"] == 8192
    assert seen["max_turns"] == 16
    assert "tests/test_e2e_product.py" in (seen.get("protected_files") or [])
    assert "Always pass full absolute paths" not in seen["prompt_text"]
    assert isinstance(get_local_harness_adapter("local_openai"), LocalOpenAIAdapter)


async def test_local_openai_adapter_blocks_modification_of_preseeded_and_protected_files(tmp_path):
    app_dir = tmp_path / "app"
    (app_dir / "tests").mkdir(parents=True)
    orig_e2e = "def test_contract():\n    assert True\n"
    orig_brief = "# Build brief\nKeep intact.\n"
    orig_extra = "LOCKED\n"
    (app_dir / "tests" / "test_e2e_product.py").write_text(orig_e2e, encoding="utf-8")
    (app_dir / "BUILD_BRIEF.md").write_text(orig_brief, encoding="utf-8")
    (app_dir / "contract.txt").write_text(orig_extra, encoding="utf-8")

    tool_outputs: list[str] = []
    step = 0

    def handler(path: str, payload: dict[str, Any]):
        nonlocal step
        step += 1
        if step > 1:
            tool_outputs.append(payload["messages"][-1]["content"])
        u = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        if step == 1:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c1",
                    "write_file",
                    json.dumps({"path": "tests/test_e2e_product.py", "content": "overwritten"}),
                    usage=u,
                ),
            )
        if step == 2:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c2",
                    "edit_file",
                    json.dumps(
                        {
                            "path": "Tests/Test_E2E_Product.py",
                            "old_text": "True",
                            "new_text": "False",
                        }
                    ),
                    usage=u,
                ),
            )
        if step == 3:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c3",
                    "write_file",
                    json.dumps({"path": "BUILD_BRIEF.md", "content": "wiped"}),
                    usage=u,
                ),
            )
        if step == 4:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c4",
                    "write_file",
                    json.dumps({"path": "contract.txt", "content": "hacked"}),
                    usage=u,
                ),
            )
        if step == 5:
            return (
                200,
                "text/event-stream",
                _sse_tool_turn(
                    "c5",
                    "write_file",
                    json.dumps({"path": "index.html", "content": "<h1>Real App</h1>\n"}),
                    usage=u,
                ),
            )
        return 200, "text/event-stream", _sse_text_turn("Built index.html.", usage=u)

    base_url, srv, thread = _start_mock_openai_server(handler)
    try:
        res, _ = await LocalOpenAIAdapter().run_workspace_task(
            candidate=_candidate(base_url),
            prompt_text="Build app without touching tests.",
            app_dir=app_dir,
            stall_seconds=None,
            protected_files=["contract.txt"],
        )
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2.0)

    assert (app_dir / "tests" / "test_e2e_product.py").read_text(encoding="utf-8") == orig_e2e
    assert (app_dir / "BUILD_BRIEF.md").read_text(encoding="utf-8") == orig_brief
    assert (app_dir / "contract.txt").read_text(encoding="utf-8") == orig_extra
    assert res.created_files == ["index.html"]
    assert res.modified_files == []
    assert all("pre-seeded read-only file" in out for out in tool_outputs[:4])
