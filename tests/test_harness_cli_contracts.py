"""CLI contract tests: drive the adapters through real subprocesses.

The fake `claude` / `agy` executables below replay the JSON envelopes the real CLIs emit:
Claude Code's were captured from `claude` 2.1.331 run against a local mock of the Anthropic
API (see ``test_real_claude_cli_against_mock_api``); agy's follow the documented headless
envelope (https://antigravity.google/docs/cli/headless). No test here calls a paid API.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from atlas.harnesses import (
    AntigravityCLIAdapter,
    ClaudeCodeAdapter,
    HarnessError,
    HarnessUnavailableError,
)
from atlas.harnesses.antigravity import _run_agy
from atlas.harnesses.base import HarnessAuthError
from atlas.harnesses.claude_code import _run_cli
from atlas.router import ModelRouter

# --- Envelopes ----------------------------------------------------------------------------

# Claude Code 2.1.331, success (trimmed of fields Atlas does not read).
CLAUDE_SUCCESS = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": '{"verdict":"pass","score":9}',
    "structured_output": {"verdict": "pass", "score": 9},
    "num_turns": 2,
    "session_id": "5a98a28f-7bb0-454b-aeb3-244d3794c95e",
    "total_cost_usd": 0.00176,
    "usage": {
        "input_tokens": 5,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 2000,
        "output_tokens": 42,
        "output_tokens_details": {"thinking_tokens": 0},
    },
    "modelUsage": {
        "claude-opus-5-5": {
            "inputTokens": 5,
            "outputTokens": 42,
            "cacheReadInputTokens": 2000,
            "cacheCreationInputTokens": 100,
            "thinkingTokens": 0,
            "costUSD": 0.00176,
        }
    },
}

# 401 from the API: exit 1, empty stderr, error text in `result`.
CLAUDE_AUTH_FAILURE = {
    "type": "result",
    "subtype": "success",
    "is_error": True,
    "api_error_status": 401,
    "api_error_type": "authentication_error",
    "result": "Failed to authenticate. API Error: 401 invalid x-api-key",
    "session_id": "075c32e4-34cf-4374-9438-544b8b5796fa",
    "total_cost_usd": 0,
    "usage": {"input_tokens": 0, "output_tokens": 0},
    "modelUsage": {},
}

# --max-budget-usd hit: exit 1, top-level usage zeroed, the billed tokens only in modelUsage.
# The session id deliberately contains "401" to prove errors are not classified by substring.
CLAUDE_BUDGET_EXHAUSTED = {
    "type": "result",
    "subtype": "error_max_budget_usd",
    "is_error": True,
    "errors": ["Reached maximum budget ($0.0001)"],
    "session_id": "4da70656-7116-4401-9e21-d68e7432bfa4",
    "total_cost_usd": 0.00176,
    "usage": {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    },
    "modelUsage": CLAUDE_SUCCESS["modelUsage"],
}

# Documented agy headless envelope.
AGY_SUCCESS = {
    "conversation_id": "c-123",
    "status": "SUCCESS",
    "response": "{\"major\": 2, \"minor\": 14, \"patch\": 3}",
    "structured_output": {"major": 2, "minor": 14, "patch": 3},
    "duration_seconds": 4.2,
    "num_turns": 1,
    "usage": {
        "input_tokens": 10415,
        "output_tokens": 657,
        "thinking_tokens": 616,
        "cache_read_tokens": 8113,
        "total_tokens": 11072,
    },
}


# --- Helpers ------------------------------------------------------------------------------


def _write_cli(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_claude(
    tmp_path: Path,
    *,
    envelope: dict[str, Any] | None = None,
    rc: int = 0,
    auth_status: dict[str, Any] | None = None,
    auth_rc: int = 0,
) -> tuple[Path, Path]:
    """A `claude` stand-in that records argv, stdin and env, then replays an envelope."""
    record = tmp_path / "claude-record.json"
    script = f"""
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("2.1.331 (Claude Code)")
    sys.exit(0)
if args[:2] == ["auth", "status"]:
    print({json.dumps(json.dumps(auth_status or {"loggedIn": True, "authMethod": "claude.ai"}))})
    sys.exit({auth_rc})
stdin = sys.stdin.read()
with open({str(record)!r}, "w", encoding="utf-8") as fh:
    json.dump({{"args": args, "stdin": stdin, "env": dict(os.environ), "cwd": os.getcwd()}}, fh)
print({json.dumps(json.dumps(envelope or CLAUDE_SUCCESS))})
sys.exit({rc})
"""
    return _write_cli(tmp_path / "claude", script), record


def _fake_agy(
    tmp_path: Path,
    *,
    envelope: dict[str, Any] | None = None,
    rc: int = 0,
    stderr: str = "",
    models_rc: int = 0,
) -> tuple[Path, Path]:
    record = tmp_path / "agy-record.json"
    script = f"""
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("1.5.15")
    sys.exit(0)
if args == ["models"]:
    print("gemini-4-hydrogen-high\\ngemini-4-neon-high" if {models_rc} == 0 else "")
    sys.stderr.write("" if {models_rc} == 0 else "authentication required")
    sys.exit({models_rc})
schema = None
if "--json-schema" in args:
    with open(args[args.index("--json-schema") + 1], encoding="utf-8") as fh:
        schema = json.load(fh)
with open({str(record)!r}, "w", encoding="utf-8") as fh:
    json.dump({{"args": args, "env": dict(os.environ), "cwd": os.getcwd(), "schema": schema}}, fh)
print({json.dumps(json.dumps(envelope or AGY_SUCCESS))})
sys.stderr.write({stderr!r})
sys.exit({rc})
"""
    return _write_cli(tmp_path / "agy", script), record


def _use_binary(monkeypatch: pytest.MonkeyPatch, module: str, binary: Path | None) -> None:
    # Patch the adapter, not `shutil.which`: that is one shared module attribute, so patching
    # it for one harness would silently swap the binary of the other.
    adapter = {"antigravity": AntigravityCLIAdapter, "claude_code": ClaudeCodeAdapter}[module]
    found = str(binary) if binary is not None else None
    monkeypatch.setattr(adapter, "_find_binary", lambda self: found)


def _judge_router(tmp_path: Path) -> ModelRouter:
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(
        """
roles:
  judge: [openrouter/anthropic/claude-opus-5.5]
backends:
  judge:
    type: harness
    provider: claude_code
    model: opus
    read_only: true
    auth: subscription
    max_budget_usd: 1.50
limits:
  max_parallel_calls: 2
  max_tokens: 2000
  timeout_seconds: 30
  attempts_per_model: 1
""",
        encoding="utf-8",
    )
    return ModelRouter(cfg)


# --- Claude Code --------------------------------------------------------------------------


async def test_claude_tokens_include_cache_reads_and_writes(monkeypatch, tmp_path):
    binary, record = _fake_claude(tmp_path)
    _use_binary(monkeypatch, "claude_code", binary)

    result = await ClaudeCodeAdapter().run_structured(
        role="judge", model="opus", system="Judge.", user="Score it.", auth="subscription"
    )

    assert json.loads(result.text) == {"verdict": "pass", "score": 9}
    # input_tokens excludes cache traffic: 5 fresh + 2000 read + 100 written were all sent.
    assert result.prompt_tokens == 2105
    assert result.cached_tokens == 2000
    assert result.completion_tokens == 42
    assert result.total_tokens == 2147
    assert result.usage_reported is True
    assert result.cost_usd is None and result.cost_source == "subscription"
    assert result.harness_version == "2.1.331 (Claude Code)"
    assert json.loads(record.read_text())["stdin"].endswith("USER REQUEST:\nScore it.")


async def test_claude_effort_flag_and_validation(monkeypatch, tmp_path):
    binary, record = _fake_claude(tmp_path)
    _use_binary(monkeypatch, "claude_code", binary)

    await ClaudeCodeAdapter().run_structured(
        role="e2e",
        model="sonnet",
        effort="high",
        system="Spec.",
        user="Generate tests.",
        auth="subscription",
    )
    args = json.loads(record.read_text())["args"]
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--effort") + 1] == "high"

    with pytest.raises(ValueError, match="Unsupported Claude Code effort level"):
        await ClaudeCodeAdapter().run_structured(
            role="e2e", model="sonnet", effort="turbo", system="s", user="u"
        )


async def test_claude_sums_every_billed_model(monkeypatch, tmp_path):
    envelope = json.loads(json.dumps(CLAUDE_SUCCESS))
    envelope["modelUsage"]["claude-haiku-5-5"] = {
        "inputTokens": 300,
        "outputTokens": 12,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
        "thinkingTokens": 0,
    }
    envelope["modelUsage"]["claude-opus-5-5"]["thinkingTokens"] = 30
    binary, _ = _fake_claude(tmp_path, envelope=envelope)
    _use_binary(monkeypatch, "claude_code", binary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    result = await ClaudeCodeAdapter().run_structured(
        role="judge", model="opus", system="s", user="u", auth="api"
    )

    # Top-level `usage` only covers the main loop; the background Haiku call is billed too.
    assert result.prompt_tokens == 2105 + 300
    assert result.completion_tokens == 42 + 12
    assert result.reasoning_tokens == 30
    assert result.total_tokens == 2405 + 54
    assert result.cost_usd == 0.00176 and result.cost_source == "reported"
    assert result.auth_mode == "api_key"


async def test_claude_prompt_goes_on_stdin_so_large_prompts_work(monkeypatch, tmp_path):
    binary, record = _fake_claude(tmp_path)
    _use_binary(monkeypatch, "claude_code", binary)
    big_user = "x" * 400_000  # > 128 KiB: would fail with E2BIG as one argv entry on Linux

    await ClaudeCodeAdapter().run_structured(
        role="supervisor",
        model="opus",
        system="Review.",
        user=big_user,
        tools=["Read", "Grep", "Glob"],
        json_schema={"type": "object"},
        max_budget_usd=1.5,
        cwd=tmp_path,
    )

    seen = json.loads(record.read_text())
    args = seen["args"]
    assert args[0] == "-p" and big_user not in " ".join(args)
    assert seen["stdin"].endswith(big_user)
    assert args[args.index("--tools") + 1] == "Read,Grep,Glob"
    assert json.loads(args[args.index("--json-schema") + 1]) == {"type": "object"}
    assert args[args.index("--max-budget-usd") + 1] == "1.5"
    assert Path(seen["cwd"]).resolve() == tmp_path.resolve()
    assert "--dangerously-skip-permissions" not in args


async def test_claude_env_is_scoped_and_api_key_stripped_on_subscription(monkeypatch, tmp_path):
    binary, record = _fake_claude(tmp_path)
    _use_binary(monkeypatch, "claude_code", binary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-would-bill-the-api")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")

    result = await ClaudeCodeAdapter().run_structured(
        role="judge", model="opus", system="s", user="u", auth="subscription"
    )

    env = json.loads(record.read_text())["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env and "GEMINI_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
    assert result.auth_mode == "oauth_token"


async def test_claude_auth_failure_is_classified_from_structured_fields(monkeypatch, tmp_path):
    binary, _ = _fake_claude(tmp_path, envelope=CLAUDE_AUTH_FAILURE, rc=1)
    _use_binary(monkeypatch, "claude_code", binary)

    with pytest.raises(HarnessAuthError, match="Failed to authenticate") as info:
        await ClaudeCodeAdapter().run_structured(role="judge", model="opus", system="s", user="u")
    assert "claude auth login" in str(info.value)
    assert '"session_id"' not in str(info.value)  # the message is the error, not the JSON blob


async def test_claude_budget_stop_is_not_auth_and_keeps_billed_usage(monkeypatch, tmp_path):
    binary, _ = _fake_claude(tmp_path, envelope=CLAUDE_BUDGET_EXHAUSTED, rc=1)
    _use_binary(monkeypatch, "claude_code", binary)

    with pytest.raises(HarnessError) as info:
        await ClaudeCodeAdapter().run_structured(role="judge", model="opus", system="s", user="u")

    assert not isinstance(info.value, HarnessAuthError)  # "401" in the session id is ignored
    assert "Reached maximum budget" in str(info.value)
    assert "error_max_budget_usd" in str(info.value)
    assert info.value.usage["prompt_tokens"] == 2105
    assert info.value.usage["total_tokens"] == 2147


async def test_router_records_tokens_of_a_failed_but_billed_harness_call(monkeypatch, tmp_path):
    binary, _ = _fake_claude(tmp_path, envelope=CLAUDE_BUDGET_EXHAUSTED, rc=1)
    _use_binary(monkeypatch, "claude_code", binary)
    router = _judge_router(tmp_path)

    with pytest.raises(HarnessError):
        await router.complete_structured("opus", "s", "u", role="judge")

    [call] = router.call_log
    assert call["error"].startswith("HarnessError")
    assert call["prompt_tokens"] == 2105
    assert call["cached_tokens"] == 2000
    assert call["total_tokens"] == 2147
    assert call["cost_source"] == "subscription"
    assert router.usage_summary().total_tokens == 2147


async def test_claude_is_error_with_exit_zero_still_fails(monkeypatch, tmp_path):
    envelope = dict(CLAUDE_AUTH_FAILURE, api_error_status=500, api_error_type="api_error")
    envelope["result"] = "API Error: 500 overloaded"
    binary, _ = _fake_claude(tmp_path, envelope=envelope, rc=0)
    _use_binary(monkeypatch, "claude_code", binary)

    with pytest.raises(HarnessError, match="500 overloaded") as info:
        await ClaudeCodeAdapter().run_structured(role="judge", model="opus", system="s", user="u")
    assert not isinstance(info.value, HarnessAuthError)


@pytest.mark.parametrize(
    ("status", "rc", "auth", "authenticated", "auth_mode", "detail"),
    [
        ({"loggedIn": True, "authMethod": "claude.ai"}, 0, "subscription", True,
         "subscription", "authMethod=claude.ai"),
        ({"loggedIn": False, "authMethod": "none"}, 1, "subscription", False,
         "subscription", "loggedIn"),
        ({"loggedIn": True, "authMethod": "api_key"}, 0, "subscription", True,
         "api_key", "billed per token"),
    ],
)
async def test_claude_check_auth_reads_auth_status_json(
    monkeypatch, tmp_path, status, rc, auth, authenticated, auth_mode, detail
):
    binary, _ = _fake_claude(tmp_path, auth_status=status, auth_rc=rc)
    _use_binary(monkeypatch, "claude_code", binary)

    result = await ClaudeCodeAdapter().check_auth(auth=auth)

    assert result.available is True
    assert result.authenticated is authenticated
    assert result.auth_mode == auth_mode
    assert detail in result.details
    if not authenticated:
        assert "claude auth login" in result.login_hint


async def test_claude_check_auth_survives_an_unrunnable_binary(monkeypatch, tmp_path):
    broken = tmp_path / "claude"
    broken.write_text("not a program", encoding="utf-8")  # exists, not executable
    _use_binary(monkeypatch, "claude_code", broken)

    status = await ClaudeCodeAdapter().check_auth(auth="subscription")

    assert status.available is True
    assert status.authenticated is False
    assert "Could not start Claude Code CLI" in status.details

    with pytest.raises(HarnessUnavailableError):
        await ClaudeCodeAdapter().run_structured(role="judge", model="opus", system="s", user="u")


async def test_cli_runners_kill_a_hung_process_on_timeout(tmp_path):
    hung = _write_cli(tmp_path / "hung", "import time\ntime.sleep(30)\n")
    started = asyncio.get_running_loop().time()
    with pytest.raises(HarnessError, match="timed out"):
        await _run_cli([str(hung)], env=dict(os.environ), timeout_seconds=0.5, input_text="hi")
    with pytest.raises(HarnessError, match="timed out"):
        await _run_agy([str(hung)], env=dict(os.environ), timeout_seconds=0.5)
    assert asyncio.get_running_loop().time() - started < 10


# --- Antigravity CLI ----------------------------------------------------------------------


async def test_agy_maps_documented_usage_fields(monkeypatch, tmp_path):
    binary, record = _fake_agy(tmp_path)
    _use_binary(monkeypatch, "antigravity", binary)
    schema = {"type": "object", "properties": {"major": {"type": "integer"}}}

    result = await AntigravityCLIAdapter().run_structured(
        role="parser", model="gemini-4-hydrogen-high", system="Parse.", user="v2.14.3",
        json_schema=schema, timeout_seconds=60,
    )

    assert json.loads(result.text) == {"major": 2, "minor": 14, "patch": 3}
    assert result.prompt_tokens == 10415
    assert result.completion_tokens == 657
    assert result.cached_tokens == 8113
    assert result.reasoning_tokens == 616
    assert result.total_tokens == 11072
    assert result.cost_usd is None and result.cost_source == "unavailable"
    assert result.harness_version == "1.5.15"
    seen = json.loads(record.read_text())
    args = seen["args"]
    assert args[args.index("--model") + 1] == "gemini-4-hydrogen-high"
    assert args[args.index("--print-timeout") + 1] == "60s"
    assert "--sandbox" in args and "--disable-slash-commands" in args and seen["schema"] == schema
    prompt_arg = args[args.index("-p") + 1]
    assert "Do NOT call any tools" in prompt_arg

    # Passing cwd selects the strictly read-only review constraint instead of tool-ban text.
    await AntigravityCLIAdapter().run_structured(
        role="supervisor", model="gemini-4-neon-high", system="Review.", user="app", cwd=tmp_path
    )
    sup_args = json.loads(record.read_text())["args"]
    sup_prompt = sup_args[sup_args.index("-p") + 1]
    assert "strictly read-only review mode" in sup_prompt


async def test_agy_print_timeout_error_is_deduplicated(monkeypatch, tmp_path):
    stderr_blob = (
        "[agy] print timeout after 5m0s with turn in progress; returning partial output\n"
        "error: print timeout after 5m0s with the turn still in progress; the response is "
        "partial\n"
        'AGY_ERROR: {"short_error":"print timeout after 5m0s with the turn still in progress; '
        'the response is partial","status":"DEADLINE_EXCEEDED","retryable":false,'
        '"reason":"print_timeout"}'
    )
    envelope = {
        "status": "DEADLINE_EXCEEDED",
        "error": (
            "print timeout after 5m0s with the turn still in progress; the response is partial"
        ),
        "usage": AGY_SUCCESS["usage"],
    }
    binary, _ = _fake_agy(tmp_path, envelope=envelope, rc=3, stderr=stderr_blob)
    _use_binary(monkeypatch, "antigravity", binary)

    with pytest.raises(HarnessError) as info:
        await AntigravityCLIAdapter().run_structured(
            role="e2e", model="gemini-4-neon-high", system="s", user="u"
        )

    msg = str(info.value)
    assert "DEADLINE_EXCEEDED" in msg
    assert "AGY_ERROR:" not in msg
    assert msg.lower().count("the turn still in progress") == 1
    assert info.value.usage["total_tokens"] == 11072


async def test_agy_env_carries_only_google_settings_and_guards_api_keys(monkeypatch, tmp_path):
    binary, record = _fake_agy(tmp_path)
    _use_binary(monkeypatch, "antigravity", binary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-key")  # e.g. loaded from `.env`
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")

    # Default: cached login. The key in `.env` must not silently switch agy to API billing.
    result = await AntigravityCLIAdapter().run_structured(
        role="parser", model="m", system="s", user="u"
    )
    env = json.loads(record.read_text())["env"]
    assert "GEMINI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env and "OPENROUTER_API_KEY" not in env
    assert env["GOOGLE_CLOUD_PROJECT"] == "my-project"
    assert result.auth_mode == "cached_login"

    # `auth: api` opts in to passing the key.
    result = await AntigravityCLIAdapter().run_structured(
        role="parser", model="m", system="s", user="u", auth="api"
    )
    env = json.loads(record.read_text())["env"]
    assert env["GEMINI_API_KEY"] == "gm-key"
    assert "ANTHROPIC_API_KEY" not in env
    assert result.auth_mode == "api_key"


async def test_agy_error_status_with_exit_zero_keeps_usage(monkeypatch, tmp_path):
    envelope = {"status": "ERROR", "error": "tool call denied", "usage": AGY_SUCCESS["usage"]}
    binary, _ = _fake_agy(tmp_path, envelope=envelope, rc=0)
    _use_binary(monkeypatch, "antigravity", binary)

    with pytest.raises(HarnessError, match="status=ERROR: tool call denied") as info:
        await AntigravityCLIAdapter().run_structured(role="parser", model="m", system="s", user="u")
    assert not isinstance(info.value, HarnessAuthError)
    assert info.value.usage["total_tokens"] == 11072


async def test_agy_nonzero_exit_auth_and_generic_errors(monkeypatch, tmp_path):
    binary, _ = _fake_agy(
        tmp_path, envelope={"status": "ERROR"}, rc=1, stderr="authentication required"
    )
    _use_binary(monkeypatch, "antigravity", binary)
    with pytest.raises(HarnessAuthError, match="authentication required"):
        await AntigravityCLIAdapter().run_structured(role="parser", model="m", system="s", user="u")

    other = tmp_path / "other"
    other.mkdir()
    binary, _ = _fake_agy(other, envelope={"status": "ERROR", "error": "unknown model"}, rc=1)
    _use_binary(monkeypatch, "antigravity", binary)
    with pytest.raises(HarnessError, match="unknown model") as info:
        await AntigravityCLIAdapter().run_structured(role="parser", model="m", system="s", user="u")
    assert not isinstance(info.value, HarnessAuthError)


async def test_agy_oversized_prompt_raises_harness_error_for_router_fallback(monkeypatch, tmp_path):
    binary, _ = _fake_agy(tmp_path)
    _use_binary(monkeypatch, "antigravity", binary)

    # 3 MB in one argv entry exceeds both Linux (128 KiB/arg) and macOS (1 MiB total) limits.
    with pytest.raises(HarnessUnavailableError, match="Could not start Antigravity CLI"):
        await AntigravityCLIAdapter().run_structured(
            role="researcher", model="m", system="s", user="x" * 3_000_000
        )


async def test_agy_check_auth_via_models_command(monkeypatch, tmp_path):
    binary, _ = _fake_agy(tmp_path)
    _use_binary(monkeypatch, "antigravity", binary)
    ok = await AntigravityCLIAdapter().check_auth()
    assert ok.authenticated is True and ok.version == "1.5.15"
    assert ok.auth_mode == "cached_login"

    other = tmp_path / "signed-out"
    other.mkdir()
    binary, _ = _fake_agy(other, models_rc=1)
    _use_binary(monkeypatch, "antigravity", binary)
    out = await AntigravityCLIAdapter().check_auth()
    assert out.authenticated is False
    assert "authentication required" in out.details
    assert "Run `agy` interactively" in out.login_hint

    broken = tmp_path / "agy-broken"
    broken.write_text("not a program", encoding="utf-8")
    _use_binary(monkeypatch, "antigravity", broken)
    gone = await AntigravityCLIAdapter().check_auth()
    assert gone.available is True and gone.authenticated is False
    assert "Could not start Antigravity CLI" in gone.details


# --- Optional: the real `claude` binary against a local mock API ---------------------------


class _MockAnthropic(BaseHTTPRequestHandler):
    """Just enough of the Messages API (SSE) for `claude -p` to complete a run."""

    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []
    fail_status: int | None = None

    def log_message(self, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(200, {})

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        type(self).requests.append({"path": self.path, **req})
        if type(self).fail_status:
            code = type(self).fail_status or 500
            kind = "authentication_error" if code == 401 else "api_error"
            self._send_json(code, {"type": "error", "error": {"type": kind, "message": "mock"}})
            return
        if "count_tokens" in self.path:
            self._send_json(200, {"input_tokens": 10})
            return
        tools = [t.get("name") for t in req.get("tools", [])]
        structured = next((t for t in tools if t and "structured" in t.lower()), None)
        usage = {
            "input_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 2000,
            "output_tokens": 1,
        }
        if structured:
            block: dict[str, Any] = {"type": "tool_use", "id": "toolu_1", "name": structured,
                                     "input": {}}
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps({"verdict": "pass", "score": 9})}
            stop = "tool_use"
        else:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": "MOCK_OK"}
            stop = "end_turn"
        message = {"id": "msg_1", "type": "message", "role": "assistant",
                   "model": req.get("model"), "content": [], "stop_reason": None,
                   "stop_sequence": None, "usage": usage}
        events = [
            ("message_start", {"type": "message_start", "message": message}),
            ("content_block_start",
             {"type": "content_block_start", "index": 0, "content_block": block}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": 42}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("connection", "close")
        self.end_headers()
        for name, data in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()
        self.close_connection = True


@pytest.fixture
def mock_anthropic(monkeypatch, tmp_path):
    _MockAnthropic.requests = []
    _MockAnthropic.fail_status = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockAnthropic)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))  # never touch the user's real Claude login
    monkeypatch.setenv("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mock")
    monkeypatch.setenv("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
    monkeypatch.setenv("CLAUDE_CODE_MAX_RETRIES", "0")
    try:
        yield _MockAnthropic
    finally:
        server.shutdown()


@pytest.mark.skipif(
    shutil.which("claude") is None or os.environ.get("ATLAS_CLAUDE_CONTRACT") != "1",
    reason="set ATLAS_CLAUDE_CONTRACT=1 with `claude` on PATH to check the real CLI",
)
async def test_real_claude_cli_against_mock_api(mock_anthropic, tmp_path):
    adapter = ClaudeCodeAdapter()
    status = await adapter.check_auth(auth="api")
    assert status.authenticated is True and status.version

    result = await adapter.run_structured(
        role="supervisor",
        model="opus",
        system="Review the app read-only.",
        user="ATLAS-CONTRACT-PROMPT " + "y" * 200_000,
        json_schema={
            "type": "object",
            "properties": {"verdict": {"type": "string"}, "score": {"type": "integer"}},
            "required": ["verdict", "score"],
        },
        tools=["Read", "Grep", "Glob"],
        auth="api",
        max_budget_usd=1.5,
        cwd=tmp_path,
        timeout_seconds=120,
    )

    assert json.loads(result.text) == {"verdict": "pass", "score": 9}
    assert result.prompt_tokens >= 2105 and result.cached_tokens >= 2000
    assert result.total_tokens == result.prompt_tokens + result.completion_tokens
    assert result.cost_source == "reported" and (result.cost_usd or 0) > 0
    main = [r for r in mock_anthropic.requests if "ATLAS-CONTRACT-PROMPT" in json.dumps(r)]
    assert main, "the stdin prompt never reached the API"
    offered = {t.get("name") for t in main[0].get("tools", [])}
    assert {"Read", "Grep", "Glob"} <= offered
    assert not offered & {"Bash", "Edit", "Write", "MultiEdit", "NotebookEdit", "WebFetch"}

    with pytest.raises(HarnessError, match="error_max_budget_usd") as info:
        await adapter.run_structured(
            role="judge", model="opus", system="s", user="u", auth="api",
            max_budget_usd=0.0001, timeout_seconds=120,
        )
    assert info.value.usage["total_tokens"] > 0

    mock_anthropic.fail_status = 401
    with pytest.raises(HarnessAuthError):
        await adapter.run_structured(
            role="judge", model="opus", system="s", user="u", auth="api", timeout_seconds=120
        )


# --- Antigravity SDK (local builder) -------------------------------------------------------


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, agent_cls: type) -> None:
    from types import SimpleNamespace

    from atlas.harnesses.antigravity import AntigravitySDKAdapter

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def lightweight(self) -> FakeConfig:
            return self

    modules = {
        "google.antigravity": SimpleNamespace(Agent=agent_cls, LocalOpenAIAgentConfig=FakeConfig),
        "google.antigravity.hooks": SimpleNamespace(
            policy=SimpleNamespace(workspace_only=lambda paths: [], auto=lambda: None)
        ),
    }
    real_import = __import__("importlib").import_module
    monkeypatch.setattr(
        "atlas.harnesses.antigravity.importlib.import_module",
        lambda name: modules.get(name) or real_import(name),
    )
    monkeypatch.setattr(AntigravitySDKAdapter, "package_version", lambda self: "0.1.24")


@pytest.mark.parametrize("crash", [False, True])
async def test_sdk_builder_hides_credentials_from_the_local_harness(monkeypatch, tmp_path, crash):
    from types import SimpleNamespace

    from atlas.harnesses.antigravity import AntigravitySDKAdapter
    from atlas.local_models import LocalModelCandidate

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
    monkeypatch.setenv("ATLAS_HARMLESS_SETTING", "kept")
    seen_env: dict[str, str] = {}

    class FakeAgent:
        def __init__(self, cfg: Any) -> None:
            pass

        async def __aenter__(self) -> FakeAgent:
            # The real SDK spawns its harness here with a copy of os.environ.
            seen_env.update(os.environ)
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def chat(self, prompt_text: str) -> Any:
            if crash:
                raise RuntimeError("model server went away")

            class Response:
                usage_metadata = SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=5, total_token_count=15
                )

                async def text(self) -> str:
                    return "done"

            return Response()

    _fake_sdk(monkeypatch, FakeAgent)
    candidate = LocalModelCandidate(
        provider="ollama", name="gemma4:12b", base_url="http://localhost:11434/v1"
    )
    run = AntigravitySDKAdapter().run_workspace_task(
        candidate=candidate, prompt_text="Build.", app_dir=tmp_path / "app"
    )
    if crash:
        with pytest.raises(RuntimeError, match="went away"):
            await run
    else:
        _, call = await run
        assert call.total_tokens == 15 and call.cost_source == "local_free"

    assert "OPENROUTER_API_KEY" not in seen_env
    assert "ANTHROPIC_API_KEY" not in seen_env and "GITHUB_TOKEN" not in seen_env
    assert seen_env["ATLAS_HARMLESS_SETTING"] == "kept"
    # ...and Atlas gets them back afterwards, even when the build crashed.
    assert os.environ["OPENROUTER_API_KEY"] == "or-secret"
    assert os.environ["GITHUB_TOKEN"] == "gh-secret"


def test_installed_antigravity_sdk_matches_what_the_adapter_calls():
    """Checks the alpha SDK's API surface (no model call). Runs when the SDK is installed."""
    import importlib
    import inspect

    sdk = pytest.importorskip("google.antigravity")
    policy = importlib.import_module("google.antigravity.hooks").policy
    sdk_types = importlib.import_module("google.antigravity.types")

    assert "workspaces" in inspect.signature(policy.workspace_only).parameters
    assert callable(policy.auto)
    fields = set(sdk.LocalOpenAIAgentConfig.model_fields)
    assert {"model", "base_url", "workspaces", "policies", "skills_paths"} <= fields
    assert callable(sdk.LocalOpenAIAgentConfig.lightweight)
    lightweight_tools = {t.name for t in sdk.BuiltinTools.minimal()}
    assert {"RUN_COMMAND", "VIEW_FILE", "CREATE_FILE", "EDIT_FILE"} <= lightweight_tools
    assert hasattr(sdk.Agent, "__aenter__") and callable(sdk.Agent.chat)
    assert inspect.iscoroutinefunction(sdk_types.ChatResponse.text)
    assert isinstance(inspect.getattr_static(sdk_types.ChatResponse, "usage_metadata"), property)
    assert {
        "prompt_token_count",
        "candidates_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
        "total_token_count",
    } <= set(sdk.UsageMetadata.model_fields)


# --- `atlas doctor` end to end over the shipped hybrid profile -----------------------------


async def test_doctor_checks_every_harness_in_the_hybrid_profile(monkeypatch, tmp_path):
    from atlas.harnesses import AntigravitySDKAdapter, AuthStatus, run_doctor
    from atlas.local_models import LocalModelCandidate, LocalResolutionResult

    repo = Path(__file__).resolve().parents[1]
    agy_dir, claude_dir = tmp_path / "agy", tmp_path / "claude"
    agy_dir.mkdir()
    claude_dir.mkdir()
    agy, _ = _fake_agy(agy_dir)
    claude, _ = _fake_claude(claude_dir)
    _use_binary(monkeypatch, "antigravity", agy)
    _use_binary(monkeypatch, "claude_code", claude)
    monkeypatch.setenv("GEMINI_API_KEY", "from-dotenv")  # must not flip agy to API billing
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-dotenv")  # must not flip claude either

    async def sdk_ready(self):
        return AuthStatus(provider="antigravity_sdk", available=True, authenticated=True,
                          auth_mode="local", version="0.1.24", details="ready")

    seen: dict[str, Any] = {}

    async def local_ready(**kwargs):
        seen.update(kwargs)
        cand = LocalModelCandidate(provider="ollama", name="gemma4:12b",
                                   base_url="http://localhost:11434/v1")
        return LocalResolutionResult(selected=cand, candidates=[cand], reason="ok")

    monkeypatch.setattr(AntigravitySDKAdapter, "check_auth", sdk_ready)
    monkeypatch.setattr("atlas.harnesses.resolve_local_model", local_ready)

    report = await run_doctor(ModelRouter(repo / "config" / "harness.yaml"))

    by_provider = {c["provider"]: c for c in report["checks"]}
    assert set(by_provider) == {
        "antigravity", "claude_code", "antigravity_sdk", "local_model_daemon", "openrouter"
    }
    assert by_provider["antigravity"]["authenticated"] is True
    assert by_provider["antigravity"]["auth_mode"] == "cached_login"
    assert by_provider["antigravity"]["version"] == "1.5.15"
    assert by_provider["claude_code"]["auth_mode"] == "subscription"
    assert by_provider["claude_code"]["version"] == "2.1.331 (Claude Code)"
    assert by_provider["local_model_daemon"]["selected_model"] == "gemma4:12b"
    builder_cfg = ModelRouter(repo / "config" / "harness.yaml").cfg["backends"]["builder"]
    assert seen["preferred"] == builder_cfg["preferred"]
    assert seen["min_context_tokens"] == builder_cfg["min_context_tokens"]
    assert seen["require_tool_smoke_check"] is False
    # No OPENROUTER_API_KEY (needed by the fixer and the `fallback: router` stages).
    assert by_provider["openrouter"]["authenticated"] is False
    assert report["healthy"] is False

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    report = await run_doctor(ModelRouter(repo / "config" / "harness.yaml"), live=True)
    assert report["healthy"] is True and seen["require_tool_smoke_check"] is True


async def test_doctor_reports_signed_out_harness_and_missing_sdk(monkeypatch, tmp_path):
    from atlas.harnesses import get_harness_adapter, run_doctor

    repo = Path(__file__).resolve().parents[1]
    agy, _ = _fake_agy(tmp_path, models_rc=1)
    _use_binary(monkeypatch, "antigravity", agy)
    _use_binary(monkeypatch, "claude_code", None)
    monkeypatch.setattr(
        "atlas.harnesses.AntigravitySDKAdapter.package_version", lambda self: None
    )

    async def nothing_running(**kwargs):
        from atlas.local_models import LocalResolutionResult

        return LocalResolutionResult(reason="No local generative models available.")

    monkeypatch.setattr("atlas.harnesses.resolve_local_model", nothing_running)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    report = await run_doctor(ModelRouter(repo / "config" / "harness.yaml"))

    by_provider = {c["provider"]: c for c in report["checks"]}
    assert report["healthy"] is False
    assert by_provider["antigravity"]["authenticated"] is False
    assert by_provider["claude_code"]["available"] is False
    assert "npm install -g @anthropic-ai/claude-code" in by_provider["claude_code"]["login_hint"]
    assert by_provider["antigravity_sdk"]["available"] is False
    assert by_provider["local_model_daemon"]["ready"] is False
    with pytest.raises(ValueError, match="Unsupported harness provider"):
        get_harness_adapter("codex")


async def test_claude_workspace_task_enforces_builder_role_and_captures_diff(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="only permitted for role 'builder'"):
        await ClaudeCodeAdapter().run_workspace_task(
            role="judge", model="sonnet", prompt_text="Build.", app_dir=tmp_path
        )

    record = tmp_path / "claude-ws.json"
    app_dir = tmp_path / "app" / "demo"
    app_dir.mkdir(parents=True)
    script = f"""
import json, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("2.1.331 (Claude Code)")
    sys.exit(0)
stdin = sys.stdin.read()
pathlib.Path({str(record)!r}).write_text(json.dumps({{"args": args, "stdin": stdin}}))
pathlib.Path("app.py").write_text("x = 1\\n", encoding="utf-8")
print({json.dumps(json.dumps(CLAUDE_SUCCESS))})
"""
    binary = _write_cli(tmp_path / "claude", script)
    _use_binary(monkeypatch, "claude_code", binary)

    build, call = await ClaudeCodeAdapter().run_workspace_task(
        role="builder",
        model="sonnet",
        prompt_text="Write app.py",
        app_dir=app_dir,
        auth="subscription",
        max_budget_usd=3.0,
    )
    assert build.created_files == ["app.py"]
    assert call.harness == "claude_code" and call.auth_mode == "subscription"
    args = json.loads(record.read_text())["args"]
    assert args[args.index("--permission-mode") + 1] == "acceptEdits"
    assert args[args.index("--tools") + 1] == "Read,Grep,Glob,Edit,Write"
    assert "--restricted" in args
    assert "--allowedTools" not in args
    assert "--no-session-persistence" in args
    assert "--dangerously-skip-permissions" not in args

    await ClaudeCodeAdapter().run_workspace_task(
        role="builder",
        model="sonnet",
        prompt_text="Write app.py",
        app_dir=app_dir,
        sandbox=False,
    )
    unsandboxed_args = json.loads(record.read_text())["args"]
    assert "--restricted" not in unsandboxed_args
