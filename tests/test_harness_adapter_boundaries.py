"""Focused contract tests for the real CLI/SDK adapter boundaries."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from atlas.harnesses import (
    AntigravityCLIAdapter,
    AntigravitySDKAdapter,
    ClaudeCodeAdapter,
    HarnessError,
)
from atlas.local_models import LocalModelCandidate, run_tool_smoke_check


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Atlas tests"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "atlas-tests@example.invalid"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# Fixture repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial fixture"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path


def _fake_cli(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


async def test_agy_adapter_uses_real_subprocess_and_parses_structured_json(monkeypatch, tmp_path):
    args_file = tmp_path / "agy-args.json"
    schema_file_seen = tmp_path / "agy-schema-seen.txt"
    fake = _fake_cli(
        tmp_path / "agy-fake",
        """
import json
import pathlib
import sys
if sys.argv[1:] == ["--version"]:
    print("agy test 1.0")
else:
    args = sys.argv[1:]
    pathlib.Path("ARGS_FILE").write_text(json.dumps(args), encoding="utf-8")
    schema_idx = args.index("--json-schema") + 1
    pathlib.Path("SCHEMA_FILE").write_text(str(pathlib.Path(args[schema_idx]).is_file()))
    print(json.dumps({
        "status": "SUCCESS",
        "structured_output": {"summary": "parsed"},
        "usage": {"input": 9, "output": 4, "total": 13},
    }))
""".replace("ARGS_FILE", str(args_file)).replace("SCHEMA_FILE", str(schema_file_seen)),
    )
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(fake))

    result = await AntigravityCLIAdapter().run_structured(
        role="parser",
        model="gemini-test",
        system="Parse this.",
        user="Build a greeting app.",
        json_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        timeout_seconds=15,
    )

    args = json.loads(args_file.read_text(encoding="utf-8"))
    assert "--sandbox" in args
    assert args[args.index("--model") + 1] == "gemini-test"
    assert schema_file_seen.read_text(encoding="utf-8") == "True"
    assert json.loads(result.text) == {"summary": "parsed"}
    assert result.harness_version == "agy test 1.0"
    assert result.total_tokens == 13


async def test_claude_adapter_uses_real_subprocess_with_read_only_tools(monkeypatch, tmp_path):
    args_file = tmp_path / "claude-args.json"
    fake = _fake_cli(
        tmp_path / "claude-fake",
        """
import json
import pathlib
import sys
args = sys.argv[1:]
if args == ["--version"]:
    print("claude test 1.0")
elif args == ["auth", "status"]:
    print('{"loggedIn":true}')
else:
    pathlib.Path("ARGS_FILE").write_text(json.dumps(args), encoding="utf-8")
    print(json.dumps({
        "structured_output": {"approved": True},
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "total_cost_usd": 0.125,
    }))
""".replace("ARGS_FILE", str(args_file)),
    )
    monkeypatch.setattr("atlas.harnesses.claude_code.shutil.which", lambda _: str(fake))

    status = await ClaudeCodeAdapter().check_auth(auth="subscription")
    assert status.available is True
    assert status.authenticated is True
    assert status.version == "claude test 1.0"

    result = await ClaudeCodeAdapter().run_structured(
        role="judge",
        model="opus-test",
        system="Judge read-only.",
        user="Return a decision.",
        json_schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        tools=[],
        max_budget_usd=0.5,
        cwd=tmp_path,
    )

    args = json.loads(args_file.read_text(encoding="utf-8"))
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--model") + 1] == "opus-test"
    assert args[args.index("--max-budget-usd") + 1] == "0.5"
    assert "--dangerously-skip-permissions" not in args
    assert json.loads(result.text) == {"approved": True}
    assert result.cost_usd is None
    assert result.cost_source == "subscription"
    assert result.harness_version == "claude test 1.0"


async def test_agy_api_key_is_not_authenticated_when_models_check_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "configured-but-unverified")
    adapter = AntigravityCLIAdapter()
    monkeypatch.setattr(adapter, "_find_binary", lambda: "/bin/agy")
    monkeypatch.setattr(adapter, "_version", lambda binary: asyncio.sleep(0, result="1.5.15"))

    async def fake_run(args, *, env, timeout_seconds, cwd=None):
        assert args == ["/bin/agy", "models"]
        return 1, "", "CLI failed before it could verify credentials"

    monkeypatch.setattr("atlas.harnesses.antigravity._run_agy", fake_run)

    status = await adapter.check_auth(auth="api")

    assert status.available is True
    assert status.authenticated is False
    assert status.auth_mode == "api_key"
    assert "could verify credentials" in status.details


async def test_agy_workspace_task_captures_workspace_diff_and_usage(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    app_dir = repo / "app" / "demo"
    adapter = AntigravityCLIAdapter()
    monkeypatch.setattr(adapter, "_find_binary", lambda: "/bin/agy")
    monkeypatch.setattr(adapter, "_version", lambda binary: asyncio.sleep(0, result="1.5.15"))
    seen: dict[str, object] = {}

    async def fake_run(args, *, env, timeout_seconds, cwd=None):
        seen["args"] = args
        seen["cwd"] = cwd
        assert cwd == app_dir.resolve()
        Path(cwd, "main.py").write_text("print('hello')\n", encoding="utf-8")
        return (
            0,
            json.dumps(
                {
                    "status": "SUCCESS",
                    "response": "Created the demo app.",
                    "usage": {"input": 12, "output": 8, "thinking": 2, "total": 20},
                }
            ),
            "",
        )

    monkeypatch.setattr("atlas.harnesses.antigravity._run_agy", fake_run)

    build, call = await adapter.run_workspace_task(
        role="builder",
        model="gemini-test",
        prompt_text="Build a tiny app.",
        app_dir=app_dir,
        timeout_seconds=45,
        sandbox=True,
    )

    args = seen["args"]
    assert "--sandbox" in args
    assert args[args.index("--print-timeout") + 1] == "45s"
    assert build.created_files == ["main.py"]
    assert build.files_snapshot == ["main.py"]
    assert build.stack == "python"
    assert call.text == "Created the demo app."
    assert call.prompt_tokens == 12
    assert call.completion_tokens == 8
    assert call.reasoning_tokens == 2
    assert call.total_tokens == 20


@pytest.mark.parametrize(
    ("tool_calls", "content", "expected"),
    [
        (
            [{"type": "function", "function": {"name": "ping", "arguments": "{}"}}],
            None,
            True,
        ),
        ([], "OK", False),
        (
            [{"type": "function", "function": {"name": "other", "arguments": "{}"}}],
            None,
            False,
        ),
        (
            [{"type": "function", "function": {"name": "ping", "arguments": "[]"}}],
            None,
            False,
        ),
    ],
)
async def test_local_tool_smoke_requires_a_valid_ping_tool_call(
    monkeypatch, tool_calls, content, expected
):
    candidate = LocalModelCandidate(
        provider="ollama",
        name="model:latest",
        base_url="http://localhost:11434/v1",
        supports_tools=True,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tool_choice"] == "required"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"tool_calls": tool_calls, "content": content}}]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "atlas.local_models.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )

    passed, detail = await run_tool_smoke_check(candidate)

    assert passed is expected
    assert (detail == "ok") is expected


@pytest.mark.parametrize("escape_workspace", [False, True])
async def test_sdk_builder_scopes_workspace_and_restores_process_cwd(
    monkeypatch, tmp_path, escape_workspace
):
    repo = _git_repo(tmp_path / "repo")
    app_dir = repo / "app" / "demo"
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    candidate = LocalModelCandidate(
        provider="ollama",
        name="model:latest",
        base_url="http://localhost:11434/v1",
        supports_tools=True,
    )
    config_args: dict[str, object] = {}
    policies: list[str] = []

    class FakeConfig:
        def __init__(self, **kwargs):
            config_args.update(kwargs)

        def lightweight(self):
            config_args["lightweight"] = True
            return self

    class FakeResponse:
        usage_metadata = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            cached_content_token_count=2,
            thoughts_token_count=1,
            total_token_count=15,
        )

        async def text(self):
            return "Local build complete"

    class FakeAgent:
        def __init__(self, cfg):
            assert cfg is not None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def chat(self, prompt_text):
            assert prompt_text == "Build in this folder."
            assert Path.cwd() == app_dir.resolve()
            (app_dir / "main.py").write_text("print('local')\n", encoding="utf-8")
            if escape_workspace:
                (repo / "escape.txt").write_text("outside workspace\n", encoding="utf-8")
            return FakeResponse()

    fake_sdk = SimpleNamespace(Agent=FakeAgent, LocalOpenAIAgentConfig=FakeConfig)
    fake_hooks = SimpleNamespace(
        policy=SimpleNamespace(
            workspace_only=lambda paths: policies.append(f"workspace:{paths[0]}") or ["workspace"],
            auto=lambda: policies.append("auto") or "auto-policy",
        )
    )
    original_import = __import__("importlib").import_module

    def fake_import(name):
        if name == "google.antigravity":
            return fake_sdk
        if name == "google.antigravity.hooks":
            return fake_hooks
        return original_import(name)

    monkeypatch.setattr("atlas.harnesses.antigravity.importlib.import_module", fake_import)
    monkeypatch.setattr(AntigravitySDKAdapter, "package_version", lambda self: "0.1.24")

    previous_cwd = Path.cwd()
    adapter = AntigravitySDKAdapter()
    if escape_workspace:
        with pytest.raises(HarnessError, match="modified files outside"):
            await adapter.run_workspace_task(
                candidate=candidate,
                prompt_text="Build in this folder.",
                app_dir=app_dir,
                skill_dirs=[skill_dir],
            )
    else:
        build, call = await adapter.run_workspace_task(
            candidate=candidate,
            prompt_text="Build in this folder.",
            app_dir=app_dir,
            skill_dirs=[skill_dir],
        )
        assert build.created_files == ["main.py"]
        assert build.harness == "antigravity_sdk"
        assert call.used_model == "ollama/model:latest"
        assert call.prompt_tokens == 10
        assert call.completion_tokens == 5
        assert call.usage_reported is True
        assert call.cost_source == "local_free"

    assert Path.cwd() == previous_cwd
    assert config_args["model"] == "model:latest"
    assert config_args["base_url"] == "http://localhost:11434/v1"
    assert config_args["workspaces"] == [str(app_dir.resolve())]
    assert config_args["skills_paths"] == [str(skill_dir.resolve())]
    assert config_args["lightweight"] is True
    assert policies == [f"workspace:{app_dir.resolve()}", "auto"]


async def test_sdk_builder_timeout_restores_process_cwd(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    candidate = LocalModelCandidate(
        provider="lm_studio",
        name="model",
        base_url="http://localhost:1234/v1",
    )

    class FakeConfig:
        def __init__(self, **kwargs):
            pass

        def lightweight(self):
            return self

    class FakeAgent:
        def __init__(self, cfg):
            assert cfg is not None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def chat(self, prompt_text):
            await asyncio.sleep(0.05)

    fake_sdk = SimpleNamespace(Agent=FakeAgent, LocalOpenAIAgentConfig=FakeConfig)
    fake_hooks = SimpleNamespace(
        policy=SimpleNamespace(workspace_only=lambda _: [], auto=lambda: None)
    )
    original_import = __import__("importlib").import_module

    def fake_import(name):
        if name == "google.antigravity":
            return fake_sdk
        if name == "google.antigravity.hooks":
            return fake_hooks
        return original_import(name)

    monkeypatch.setattr("atlas.harnesses.antigravity.importlib.import_module", fake_import)
    monkeypatch.setattr(AntigravitySDKAdapter, "package_version", lambda self: "0.1.24")
    previous_cwd = Path.cwd()

    with pytest.raises(HarnessError, match="timed out"):
        await AntigravitySDKAdapter().run_workspace_task(
            candidate=candidate,
            prompt_text="This should time out.",
            app_dir=app_dir,
            timeout_seconds=0.001,
        )

    assert Path.cwd() == previous_cwd
