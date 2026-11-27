"""Gemma 4 12B local builder: model selection, fast failure, and offload to cloud builders.

No model is called: local servers are faked at the probe/HTTP boundary, the SDK is faked at
import time, `agy` is a fake script, and the OpenRouter call is a patched `router.complete`.
"""

from __future__ import annotations

import asyncio
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from atlas.harnesses import (
    AntigravityCLIAdapter,
    AntigravitySDKAdapter,
    ClaudeCodeAdapter,
    HarnessError,
)
from atlas.harnesses.base import HarnessInvocationResult
from atlas.local_models import (
    LocalModelCandidate,
    LocalResolutionResult,
    lm_studio_runtime_context,
    local_resolution_kwargs,
    resolve_local_model,
)
from atlas.pipeline.build import (
    build_app_workspace,
    fix_app,
    fixer_model_for_round,
    fixer_target_for_round,
)
from atlas.pipeline.render import render_build_brief
from atlas.router import ModelRouter
from atlas.runner import _flow_description
from atlas.schemas import (
    CandidateTool,
    CheckResult,
    ClarificationItem,
    ClarificationReport,
    ParsedSpec,
    Plan,
    Requirement,
    ToolDecision,
    VerificationRound,
    WorkspaceBuildResult,
)

LMS = "http://localhost:1234/v1"
OLLAMA = "http://localhost:11434/v1"
TWELVE_B = "google/gemma-4-12b-qat"
SPEC = ParsedSpec(
    summary="Greeting CLI",
    requirements=[Requirement(id="R1", title="Greet", description="Print a greeting")],
)


def _cand(name: str, provider: str = "lm_studio", **kw) -> LocalModelCandidate:
    from atlas.local_models import parse_parameter_billions

    return LocalModelCandidate(
        provider=provider,  # type: ignore[arg-type]
        name=name,
        base_url=LMS if provider == "lm_studio" else OLLAMA,
        parameter_size_b=parse_parameter_billions(None, name),
        supports_tools=True,
        **kw,
    )


def _fake_probes(monkeypatch, lm_studio: list, ollama: list) -> None:
    async def probe_lms(*args, **kwargs):
        return list(lm_studio), None

    async def probe_ollama(*args, **kwargs):
        return list(ollama), None

    monkeypatch.setattr("atlas.local_models.probe_lm_studio", probe_lms)
    monkeypatch.setattr("atlas.local_models.probe_ollama", probe_ollama)


async def _passes(candidate):
    return True, "ok"


async def _no_runtime_ctx(candidate):
    return None


BUILDER_CFG = {
    "preferred": [TWELVE_B, "gemma4:12b"],
    "only_preferred": True,
    "max_parameter_size_b": 14,
    "min_context_tokens": 32768,
    "smoke_timeout_seconds": 90,
}


# --- model selection -------------------------------------------------------------------


async def test_only_the_12b_is_picked_even_when_the_26b_is_loaded(monkeypatch):
    _fake_probes(
        monkeypatch,
        lm_studio=[_cand("google/gemma-4-26b-a4b-qat", loaded=True), _cand(TWELVE_B)],
        ollama=[_cand("qwen3:8b", "ollama"), _cand("gemma4:12b", "ollama")],
    )
    kwargs = local_resolution_kwargs(BUILDER_CFG)
    del kwargs["smoke_timeout_seconds"]
    res = await resolve_local_model(
        smoke_check_fn=_passes, runtime_context_fn=_no_runtime_ctx, **kwargs
    )

    assert res.selected is not None and res.selected.name == TWELVE_B
    assert "only_preferred" in res.rejected["google/gemma-4-26b-a4b-qat"]
    assert "only_preferred" in res.rejected["qwen3:8b"]


async def test_parameter_cap_rejects_a_preferred_26b(monkeypatch):
    _fake_probes(monkeypatch, lm_studio=[_cand("google/gemma-4-26b-a4b-qat")], ollama=[])
    res = await resolve_local_model(
        preferred=["google/gemma-4-26b-a4b-qat"],
        max_parameter_size_b=14,
        smoke_check_fn=_passes,
    )
    assert res.selected is None
    assert "26B parameters > max_parameter_size_b 14B" in res.rejected[
        "google/gemma-4-26b-a4b-qat"
    ]


async def test_missing_12b_says_which_models_were_wanted(monkeypatch):
    _fake_probes(monkeypatch, lm_studio=[_cand("google/gemma-4-26b-a4b-qat")], ollama=[])
    res = await resolve_local_model(
        preferred=[TWELVE_B], only_preferred=True, smoke_check_fn=_passes
    )
    assert res.selected is None
    assert res.reason.startswith("None of the preferred local models")
    assert TWELVE_B in res.reason and "gemma-4-26b-a4b-qat" in res.reason


async def test_lm_studio_jit_load_with_small_context_is_rejected_with_fix(monkeypatch):
    _fake_probes(monkeypatch, lm_studio=[_cand(TWELVE_B)], ollama=[])

    async def jit_4k(candidate):
        return 4096

    res = await resolve_local_model(
        preferred=[TWELVE_B],
        only_preferred=True,
        min_context_tokens=32768,
        smoke_check_fn=_passes,
        runtime_context_fn=jit_4k,
    )
    assert res.selected is None
    problem = res.rejected[TWELVE_B]
    assert "LM Studio serves it with a 4096-token context" in problem
    assert f"lms load {TWELVE_B} --context-length 32768" in problem


async def test_out_of_memory_smoke_failure_suggests_unloading(monkeypatch):
    _fake_probes(monkeypatch, lm_studio=[_cand(TWELVE_B)], ollama=[])

    async def oom(candidate):
        return False, "HTTP 400: Model loading was stopped due to insufficient system resources."

    res = await resolve_local_model(preferred=[TWELVE_B], smoke_check_fn=oom)
    assert res.selected is None
    assert "lms unload --all" in res.rejected[TWELVE_B]


async def test_smoke_timeout_is_configurable(monkeypatch):
    _fake_probes(monkeypatch, lm_studio=[_cand(TWELVE_B)], ollama=[])
    seen: dict = {}

    async def fake_smoke(candidate, *, timeout=180.0):
        seen["timeout"] = timeout
        return True, "ok"

    monkeypatch.setattr("atlas.local_models.run_tool_smoke_check", fake_smoke)
    res = await resolve_local_model(
        preferred=[TWELVE_B], smoke_timeout_seconds=90, runtime_context_fn=_no_runtime_ctx
    )
    assert res.selected is not None and seen["timeout"] == 90


async def test_lm_studio_runtime_context_reads_the_loaded_window(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v0/models")
        return httpx.Response(200, json={"data": [
            {"id": "other", "state": "loaded", "loaded_context_length": 99},
            {"id": TWELVE_B, "state": "loaded", "loaded_context_length": 32768,
             "max_context_length": 262144},
        ]})

    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "atlas.local_models.httpx.AsyncClient", lambda **kw: real(transport=transport, **kw)
    )
    assert await lm_studio_runtime_context(_cand(TWELVE_B)) == 32768
    assert await lm_studio_runtime_context(_cand("not-listed")) is None


# --- SDK stall detection ---------------------------------------------------------------


def _fake_sdk(monkeypatch, chat_impl) -> None:
    class FakeConfig:
        def __init__(self, **kwargs):
            pass

        def lightweight(self):
            return self

    class FakeResponse:
        usage_metadata = None

        async def text(self):
            return "done"

    class FakeAgent:
        def __init__(self, cfg):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def chat(self, prompt_text):
            await chat_impl()
            return FakeResponse()

    fake_sdk = SimpleNamespace(Agent=FakeAgent, LocalOpenAIAgentConfig=FakeConfig)
    fake_hooks = SimpleNamespace(
        policy=SimpleNamespace(
            workspace_only=lambda _: [],
            deny=lambda *a, **kw: None,
            allow_all=lambda: None,
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


async def test_local_agent_that_stops_writing_files_is_stopped_early(monkeypatch, tmp_path):
    async def think_forever():
        await asyncio.sleep(30)

    _fake_sdk(monkeypatch, think_forever)
    started = asyncio.get_running_loop().time()
    with pytest.raises(HarnessError, match="stalled: no file changes"):
        await AntigravitySDKAdapter().run_workspace_task(
            candidate=_cand(TWELVE_B),
            prompt_text="Build.",
            app_dir=tmp_path / "app",
            timeout_seconds=20,
            stall_seconds=0.3,
            stall_poll_seconds=0.05,
        )
    assert asyncio.get_running_loop().time() - started < 5


async def test_local_agent_that_keeps_writing_is_not_stalled(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"

    async def write_steadily():
        for i in range(8):
            (app_dir / f"part{i}.py").write_text(f"X = {i}\n", encoding="utf-8")
            await asyncio.sleep(0.1)

    _fake_sdk(monkeypatch, write_steadily)
    build, call = await AntigravitySDKAdapter().run_workspace_task(
        candidate=_cand(TWELVE_B),
        prompt_text="Build.",
        app_dir=app_dir,
        timeout_seconds=20,
        stall_seconds=0.3,
        stall_poll_seconds=0.05,
    )
    assert len(build.created_files) == 8
    assert call.text == "done" and call.cost_source == "local_free"


# --- offload ---------------------------------------------------------------------------


def _router(tmp_path: Path, offload_yaml: str) -> ModelRouter:
    cfg = tmp_path / "builder.yaml"
    cfg.write_text(
        f"""
roles:
  builder: [local-auto]
backends:
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    preferred: [{TWELVE_B}]
    only_preferred: true
    min_context_tokens: 32768
    stall_seconds: 120
    offload:
{offload_yaml}
limits:
  max_parallel_calls: 2
  max_tokens: 2000
  timeout_seconds: 30
  attempts_per_model: 1
  repair_attempts: 0
""",
        encoding="utf-8",
    )
    return ModelRouter(cfg)


HYDROGEN_THEN_SONNET = """      - name: hydrogen
        type: harness
        provider: antigravity
        model: gemini-4-hydrogen-high
        timeout_seconds: 60
      - name: sonnet
        type: harness
        provider: claude_code
        model: sonnet
        auth: subscription
        max_budget_usd: 5.0
        timeout_seconds: 60"""


def _fake_claude_builder(tmp_path: Path, rc: int = 0) -> tuple[Path, Path]:
    record = tmp_path / "claude-build-record.json"
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": bool(rc),
        "result": "Built with Claude Code." if rc == 0 else "credit exhausted",
        "total_cost_usd": 0.15,
        "usage": {"input_tokens": 50, "output_tokens": 20},
    }
    script = f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("2.1.331 (Claude Code)"); sys.exit(0)
if args[:2] == ["auth", "status"]:
    print('{{"loggedIn": true, "authMethod": "claude.ai"}}'); sys.exit(0)
stdin = sys.stdin.read()
pathlib.Path({str(record)!r}).write_text(json.dumps({{
    "args": args, "stdin": stdin, "env": dict(os.environ), "cwd": os.getcwd(),
}}))
if {rc} == 0:
    pathlib.Path("main.py").write_text("print('hi from claude')\\n", encoding="utf-8")
print({json.dumps(json.dumps(envelope))})
sys.exit({rc})
"""
    path = tmp_path / "claude"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path, record


def _fake_agy(tmp_path: Path, rc: int = 0) -> Path:
    record = tmp_path / "agy-args.json"
    envelope = {"status": "SUCCESS", "response": "Built it.", "usage": {"total_tokens": 9}}
    if rc:
        envelope = {"status": "ERROR", "error": "quota exhausted"}
    script = f"""#!{sys.executable}
import json, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("1.5.15"); sys.exit(0)
pathlib.Path({str(record)!r}).write_text(json.dumps(args))
if {rc} == 0:
    pathlib.Path("main.py").write_text("print('hello')\\n", encoding="utf-8")
print({json.dumps(json.dumps(envelope))})
sys.exit({rc})
"""
    path = tmp_path / "agy"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _no_local_model(monkeypatch):
    async def nothing(**kwargs):
        assert kwargs["only_preferred"] is True
        return LocalResolutionResult(reason="None of the preferred local models are served")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", nothing)


async def test_build_goes_to_hydrogen_when_the_12b_is_not_available(monkeypatch, tmp_path):
    _no_local_model(monkeypatch)
    binary = _fake_agy(tmp_path)
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, HYDROGEN_THEN_SONNET)
    app_dir = tmp_path / "app" / "greeting"

    result = await build_app_workspace(router, SPEC, app_dir, plan_markdown="# Plan\n")

    assert result.harness == "agy" and result.model == "gemini-4-hydrogen-high"
    assert result.created_files == ["main.py"]
    args = json.loads((tmp_path / "agy-args.json").read_text())
    prompt = args[args.index("-p") + 1]
    assert "Read `PLAN.md` on disk" in prompt  # the cloud model gets the full plan
    assert "smaller local model attempted this build" in prompt
    assert [c["harness"] for c in router.call_log] == ["agy"]


async def test_build_goes_to_sonnet_via_claude_code_when_hydrogen_fails(monkeypatch, tmp_path):
    from atlas.events import RunEventBus, attach, detach
    from atlas.harnesses import AntigravityCLIAdapter, ClaudeCodeAdapter

    _no_local_model(monkeypatch)
    agy_bin = _fake_agy(tmp_path, rc=1)
    claude_bin, record = _fake_claude_builder(tmp_path, rc=0)
    monkeypatch.setattr(AntigravityCLIAdapter, "_find_binary", lambda self: str(agy_bin))
    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", lambda self: str(claude_bin))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-must-not-bill-api")

    router = _router(tmp_path, HYDROGEN_THEN_SONNET)
    app_dir = tmp_path / "app" / "greeting"
    bus = RunEventBus(persist=False)
    attach(bus)
    try:
        result = await build_app_workspace(router, SPEC, app_dir)
    finally:
        detach(bus)

    assert result.harness == "claude_code"
    assert result.model == "sonnet"
    assert result.created_files == ["main.py"]
    assert (app_dir / "main.py").read_text() == "print('hi from claude')\n"

    seen = json.loads(record.read_text())
    args = seen["args"]
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--permission-mode") + 1] == "acceptEdits"
    assert args[args.index("--tools") + 1] == "Read,Grep,Glob,Edit,Write"
    assert "--restricted" in args
    assert "--allowedTools" not in args
    assert "--no-session-persistence" in args
    assert "--dangerously-skip-permissions" not in args
    assert args[args.index("--max-budget-usd") + 1] == "5"
    assert "ANTHROPIC_API_KEY" not in seen["env"]
    assert "PARSED SPECIFICATION" in seen["stdin"]
    assert "Greeting CLI" in seen["stdin"]

    # Both failed agy attempt and subscription-mode Claude Code build are logged honestly.
    assert router.call_log[0]["harness"] == "antigravity" and router.call_log[0]["error"]
    assert router.call_log[1]["harness"] == "claude_code"
    assert router.call_log[1]["auth_mode"] == "subscription"
    assert router.call_log[1]["cost_source"] == "subscription"

    attempts = [
        (e["data"].get("attempt"), e["status"])
        for e in bus.events_after(0)
        if e["kind"] == "attempt"
    ]
    assert ("local", "offloaded") in attempts
    assert ("hydrogen", "offloaded") in attempts
    assert ("sonnet", "passed") in attempts


async def test_openrouter_offload_still_works_when_configured_explicitly(monkeypatch, tmp_path):
    _no_local_model(monkeypatch)
    or_yaml = """      - name: or-sonnet
        type: openrouter
        model: openrouter/anthropic/claude-sonnet-5.5"""
    router = _router(tmp_path, or_yaml)
    seen: dict = {}

    async def fake_complete(model, system, user, role=None):
        seen.update(model=model, role=role)
        return model, json.dumps({
            "summary": "Greeting CLI",
            "stack": "python",
            "files": [{"path": "main.py", "content": "print('hi')", "purpose": "entry"}],
            "run_instructions": "python main.py",
        })

    monkeypatch.setattr(router, "complete", fake_complete)
    app_dir = tmp_path / "app" / "greeting"
    result = await build_app_workspace(router, SPEC, app_dir)
    assert seen == {"model": "openrouter/anthropic/claude-sonnet-5.5", "role": "builder"}
    assert result.harness == "openrouter"
    assert result.run_instructions == "python main.py"
    assert (app_dir / "main.py").read_text() == "print('hi')\n"


async def test_every_target_failing_reports_each_reason(monkeypatch, tmp_path):
    from atlas.harnesses import AntigravityCLIAdapter, ClaudeCodeAdapter

    _no_local_model(monkeypatch)
    agy_bin = _fake_agy(tmp_path, rc=1)
    claude_bin, _ = _fake_claude_builder(tmp_path, rc=1)
    monkeypatch.setattr(AntigravityCLIAdapter, "_find_binary", lambda self: str(agy_bin))
    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", lambda self: str(claude_bin))
    router = _router(tmp_path, HYDROGEN_THEN_SONNET)

    with pytest.raises(HarnessError) as info:
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")
    message = str(info.value)
    assert "local: HarnessUnavailableError" in message
    assert "hydrogen: HarnessError" in message and "quota exhausted" in message
    assert "sonnet: HarnessError" in message and "credit exhausted" in message


async def test_12b_that_writes_nothing_is_offloaded(monkeypatch, tmp_path):
    cand = _cand(TWELVE_B)

    async def ready(**kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand])

    seen: dict = {}

    async def idle_sdk(self, **kwargs):
        seen.update(kwargs)
        return (
            WorkspaceBuildResult(
                summary="Nothing to do", harness="antigravity_sdk",
                model=f"lm_studio/{TWELVE_B}", app_dir=str(kwargs["app_dir"]),
            ),
            SimpleNamespace(
                used_model=f"lm_studio/{TWELVE_B}", harness="antigravity_sdk",
                harness_version="0.1.24", auth_mode="local", elapsed_ms=1.0, prompt_tokens=0,
                completion_tokens=0, cached_tokens=0, reasoning_tokens=0, total_tokens=0,
                cost_usd=0.0, cost_source="local_free", usage_reported=False,
            ),
        )

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", idle_sdk)
    binary = _fake_agy(tmp_path)
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, HYDROGEN_THEN_SONNET)

    result = await build_app_workspace(router, SPEC, tmp_path / "app" / "x", clean=True)

    assert seen["stall_seconds"] == 120
    assert result.harness == "agy"
    args = json.loads((tmp_path / "agy-args.json").read_text())
    # A clean build starts the cloud model on a clean folder, so no "partial files" note.
    assert "smaller local model attempted" not in args[args.index("-p") + 1]


async def test_without_offload_targets_the_local_error_is_raised(monkeypatch, tmp_path):
    _no_local_model(monkeypatch)
    router = _router(tmp_path, "      []")
    with pytest.raises(HarnessError, match="None of the preferred"):
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")


# --- build brief -----------------------------------------------------------------------


def _plan() -> Plan:
    spec = ParsedSpec(
        summary="Listing page",
        requirements=[Requirement(id="R1", title="Headline", description="Show it")],
        constraints=["Single HTML file"],
    )
    decision = ToolDecision(
        requirement_id="R1",
        options=[
            CandidateTool(name="Semantic h1", pros=["x" * 5000]),
            CandidateTool(name="Canvas text"),
        ],
        pick="Semantic h1",
        rationale="Accessible and simple. " * 40,
    )
    return Plan(spec=spec, decisions=[decision], tasks=["Write index.html"])


def test_build_brief_is_short_and_keeps_what_the_builder_acts_on():
    clarifications = ClarificationReport(items=[
        ClarificationItem(question="Currency?", answer="ZAR", requirement_id="R1"),
        ClarificationItem(question="Unanswered?", answer=""),
    ])
    brief = render_build_brief(_plan(), clarifications=clarifications)

    assert "**R1:** Semantic h1 — Accessible and simple." in brief
    assert "Q (R1): Currency? — A: ZAR" in brief and "Unanswered?" not in brief
    assert "1. Write index.html" in brief and "Single HTML file" in brief
    assert "x" * 100 not in brief  # option pros/cons stay in PLAN.md
    assert len(brief) < 2000


async def test_local_build_reads_the_brief_not_the_full_plan(monkeypatch, tmp_path):
    cand = _cand(TWELVE_B)

    async def ready(**kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand])

    seen: dict = {}

    app_dir = tmp_path / "app" / "x"
    app_dir.mkdir(parents=True)
    (app_dir / "PLAN.md").write_text("# Old plan from an earlier run\n", encoding="utf-8")

    async def fake_sdk(self, **kwargs):
        seen.update(kwargs)
        seen["plan_on_disk"] = (app_dir / "PLAN.md").exists()
        raise HarnessError("stop here")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk)
    router = _router(tmp_path, "      []")

    with pytest.raises(HarnessError, match="stop here"):
        await build_app_workspace(
            router, SPEC, app_dir, _plan(), plan_markdown="# Plan\n" + "long " * 30000
        )

    brief_path = app_dir.resolve() / "BUILD_BRIEF.md"
    assert brief_path.is_file()
    prompt = seen["prompt_text"]
    assert f"Read `{brief_path}` first" in prompt
    assert "Always pass full absolute paths" in prompt
    assert "no absolute paths" not in prompt
    assert "Return ONLY JSON" not in prompt  # workspace agents write files, not a JSON reply
    # The ~30K-token plan stays out of the 12B's reach (even a stale one) and comes back after.
    assert seen["plan_on_disk"] is False
    assert "PLAN.md" not in prompt
    assert (app_dir / "PLAN.md").read_text(encoding="utf-8").startswith("# Plan")
    # File tools only unless the config sets allow_shell.
    assert "You have file tools only" in prompt
    assert "allow_shell" not in seen


# --- config validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("offload_yaml", "message"),
    [
        ("      - {type: harness, provider: codex}", "provider must be one of"),
        ("      - {type: harness, provider: claude_code, auth: magic}", "auth must be one of"),
        ("      - {type: harness, provider: claude_code, max_budget_usd: 0}", "max_budget_usd"),
        ("      - {type: openrouter}", "model must be a non-empty string"),
        ("      - {type: magic, model: x}", "type must be 'harness', 'openrouter'"),
        ("      - {type: harness, timeout_seconds: 0}", "timeout_seconds must be > 0"),
    ],
)
def test_bad_offload_targets_are_rejected(tmp_path, offload_yaml, message):
    with pytest.raises(ValueError, match=message):
        _router(tmp_path, offload_yaml)


def test_offload_is_only_for_local_builders(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        """
roles: {builder: [x]}
backends:
  builder:
    type: harness
    provider: antigravity
    offload: [{type: openrouter, model: m}]
limits: {max_parallel_calls: 1, max_tokens: 10, timeout_seconds: 5}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only supported with type local_harness"):
        ModelRouter(cfg)


def test_shipped_harness_config_uses_the_12b_with_offload():
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    builder = router.backend_for("builder")
    assert builder["preferred"][0] == TWELVE_B
    assert all("26b" not in m.lower() for m in builder["preferred"])
    assert builder["only_preferred"] is True and builder["max_parameter_size_b"] <= 14
    assert [t["name"] for t in builder["offload"]] == ["sonnet", "hydrogen", "sonnet-openrouter"]
    assert builder["offload"][0]["type"] == "harness"
    assert builder["offload"][0]["provider"] == "claude_code"
    assert builder["offload"][0]["model"] == "sonnet"
    assert builder["offload"][0]["auth"] == "subscription"
    assert builder["offload"][1]["type"] == "harness"
    assert builder["offload"][1]["provider"] == "antigravity"
    assert builder["offload"][1]["model"] == "gemini-4-hydrogen-high"
    assert router.models_for("builder") == ["openrouter/anthropic/claude-sonnet-5.5"]
    assert "openrouter/google/gemini-4-hydrogen" in router.candidates_for(
        "openrouter/anthropic/claude-sonnet-5.5", role="builder"
    )


def test_shipped_fixer_uses_only_sonnet_and_hydrogen():
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    rotate = router.backend_for("fixer")["rotate"]
    assert len(rotate) == 2
    assert rotate[0]["type"] == "harness"
    assert rotate[0]["provider"] == "claude_code"
    assert rotate[0]["model"] == "sonnet"
    assert rotate[0]["read_only"] is True
    assert rotate[0]["auth"] == "subscription"
    assert rotate[0]["fallback"] == "router"
    assert rotate[0]["router_model"] == "openrouter/anthropic/claude-sonnet-5.5"

    assert rotate[1]["type"] == "harness"
    assert rotate[1]["provider"] == "antigravity"
    assert rotate[1]["model"] == "gemini-4-hydrogen-medium"
    assert rotate[1]["fallback"] == "router"
    assert rotate[1]["router_model"] == "openrouter/google/gemini-4-hydrogen"

    assert fixer_model_for_round(router, 1) == "claude_code/sonnet"
    assert fixer_model_for_round(router, 2) == "antigravity/gemini-4-hydrogen-medium"
    fb1, ov1, lbl1 = fixer_target_for_round(router, 1)
    assert (fb1, (ov1 or {}).get("model"), lbl1) == (
        "openrouter/anthropic/claude-sonnet-5.5",
        "sonnet",
        "claude_code/sonnet",
    )
    fb2, ov2, lbl2 = fixer_target_for_round(router, 2)
    assert (fb2, (ov2 or {}).get("model"), lbl2) == (
        "openrouter/google/gemini-4-hydrogen",
        "gemini-4-hydrogen-medium",
        "antigravity/gemini-4-hydrogen-medium",
    )

    for model in (
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/google/gemini-4-hydrogen",
    ):
        candidates = router.candidates_for(model, role="fixer")
        assert all("sonnet" in m or "hydrogen" in m for m in candidates)
        assert len(candidates) == 3  # the requested model, the other one, direct Gemini


def test_every_harness_stage_falls_back_to_openrouter():
    """User rule: run on the local harnesses first, then fall back to OpenRouter."""
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    backends = router.cfg["backends"]
    for role, cfg in backends.items():
        if cfg.get("type") == "harness":
            assert cfg.get("fallback") == "router", f"{role} has no OpenRouter fallback"
            assert router.models_for(role)[0].startswith("openrouter/"), role
        for idx, entry in enumerate(cfg.get("rotate") or []):
            if isinstance(entry, dict) and entry.get("type") == "harness":
                assert entry.get("fallback") == "router", f"{role}.rotate[{idx}] missing fallback"
                assert str(entry.get("router_model", "")).startswith("openrouter/")
    for reviewer in router.supervisor_reviewers():
        if reviewer.get("type") == "harness":
            assert reviewer.get("fallback") == "router", reviewer["name"]
            assert str(reviewer.get("router_model", "")).startswith("openrouter/")
    offload = backends["builder"]["offload"]
    assert [t.get("type") for t in offload[:-1]] == ["harness"] * (len(offload) - 1)
    assert offload[-1]["type"] == "openrouter"
    assert offload[-1]["model"].startswith("openrouter/")


def test_fixer_runs_harnesses_first_and_falls_back_only_on_error(monkeypatch, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app.py").write_text("x = 0\n", encoding="utf-8")
    failed = VerificationRound(
        round_number=1,
        passed=False,
        stack_detected="python",
        checks=[
            CheckResult(
                name="python-pytest",
                command="pytest",
                passed=False,
                exit_code=1,
                stderr="AssertionError",
            )
        ],
    )

    calls: list[tuple[str, str, list[str] | None]] = []
    router_fallbacks: list[str] = []

    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", staticmethod(lambda: "/bin/claude"))
    monkeypatch.setattr(AntigravityCLIAdapter, "_find_binary", staticmethod(lambda: "/bin/agy"))

    patch_json = json.dumps(
        {
            "diagnosis": "fixed value",
            "files": [{"path": "app.py", "content": "x = 1\n"}],
            "delete_files": [],
        }
    )

    async def fake_claude_structured(
        self, *, role, model, system, user, tools=None, **kwargs
    ):
        calls.append(("claude_code", model, tools))
        return HarnessInvocationResult(
            text=patch_json,
            used_model=model,
            harness="claude_code",
            auth_mode="subscription",
            cost_source="subscription",
            usage_reported=True,
        )

    async def fake_agy_structured(self, *, role, model, system, user, **kwargs):
        calls.append(("antigravity", model, None))
        return HarnessInvocationResult(
            text=patch_json,
            used_model=model,
            harness="antigravity",
            auth_mode="cached_login",
            cost_source="unavailable",
            usage_reported=False,
        )

    async def fail_if_called(model, system, user, **kwargs):
        router_fallbacks.append(model)
        return model, patch_json

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", fake_claude_structured)
    monkeypatch.setattr(AntigravityCLIAdapter, "run_structured", fake_agy_structured)
    monkeypatch.setattr(router, "_complete_litellm", fail_if_called)

    async def run_happy():
        r1 = await fix_app(router, SPEC, app_dir, failed, fix_round=1)
        r2 = await fix_app(router, SPEC, app_dir, failed, fix_round=2)
        return r1, r2

    r1, r2 = asyncio.run(run_happy())
    assert r1.diagnosis == "fixed value"
    assert r2.diagnosis == "fixed value"
    assert calls == [
        ("claude_code", "sonnet", None),
        ("antigravity", "gemini-4-hydrogen-medium", None),
    ]
    assert router_fallbacks == []

    # Now fail both harnesses and confirm each falls back to its configured OpenRouter model.
    async def broken_claude(self, **kwargs):
        raise HarnessError("claude down")

    async def broken_agy(self, **kwargs):
        raise HarnessError("agy down")

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", broken_claude)
    monkeypatch.setattr(AntigravityCLIAdapter, "run_structured", broken_agy)

    asyncio.run(fix_app(router, SPEC, app_dir, failed, fix_round=1))
    asyncio.run(fix_app(router, SPEC, app_dir, failed, fix_round=2))
    assert router_fallbacks == [
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/google/gemini-4-hydrogen",
    ]


def test_invalid_fixer_rotate_dict_is_rejected(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((repo / "config" / "harness.yaml").read_text(encoding="utf-8"))
    data["backends"]["fixer"]["rotate"][0]["read_only"] = False
    cfg = tmp_path / "bad_fixer.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="strictly read-only"):
        ModelRouter(cfg)


def test_flow_description_formats_harness_fixer_entries():
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    flow = _flow_description(
        router,
        skip_research=False,
        use_e2e=True,
        max_fix_rounds=3,
        local_model=None,
        from_run=None,
    )
    assert flow["fixer"] == ["claude_code/sonnet", "antigravity/gemini-4-hydrogen-medium"]


def test_builder_handover_order_local_sonnet_hydrogen(monkeypatch, tmp_path):
    from atlas.events import RunEventBus, attach, detach

    _no_local_model(monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    app_dir = tmp_path / "app" / "greeting"
    seen_builders: list[str] = []

    agy_bin = _fake_agy(tmp_path, rc=0)
    monkeypatch.setattr(AntigravityCLIAdapter, "_find_binary", staticmethod(lambda: str(agy_bin)))
    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", staticmethod(lambda: "/bin/claude"))

    async def empty_claude_build(self, *, role, model, prompt_text, app_dir, **kwargs):
        seen_builders.append(f"claude_code:{model}")
        return (
            WorkspaceBuildResult(
                summary="could not finish",
                harness="claude_code",
                model=model,
                app_dir=str(app_dir),
                stack="python",
            ),
            HarnessInvocationResult(
                text="",
                used_model=model,
                harness="claude_code",
                auth_mode="subscription",
                cost_source="subscription",
                usage_reported=True,
            ),
        )

    orig_agy_run = AntigravityCLIAdapter.run_workspace_task

    async def tracked_agy_build(self, **kwargs):
        seen_builders.append(f"antigravity:{kwargs['model']}")
        return await orig_agy_run(self, **kwargs)

    async def fail_openrouter(*args, **kwargs):
        raise AssertionError("OpenRouter must not run when hydrogen succeeds")

    monkeypatch.setattr(ClaudeCodeAdapter, "run_workspace_task", empty_claude_build)
    monkeypatch.setattr(AntigravityCLIAdapter, "run_workspace_task", tracked_agy_build)
    monkeypatch.setattr(router, "_complete_litellm", fail_openrouter)

    bus = RunEventBus(persist=False)
    attach(bus)
    try:
        result = asyncio.run(build_app_workspace(router, SPEC, app_dir))
    finally:
        detach(bus)

    assert seen_builders == ["claude_code:sonnet", "antigravity:gemini-4-hydrogen-high"]
    assert result.harness == "agy"
    assert result.model == "gemini-4-hydrogen-high"
    assert (app_dir / "main.py").read_text() == "print('hello')\n"

    attempt_events = [e for e in bus.events_after(0) if e["kind"] == "attempt"]
    transitions = [
        (e["data"].get("attempt"), e["status"], e["data"].get("next"))
        for e in attempt_events
    ]
    assert ("local", "offloaded", "sonnet") in transitions
    assert ("sonnet", "offloaded", "hydrogen") in transitions
    assert ("hydrogen", "passed", None) in transitions


def test_shipped_e2e_uses_claude_code_sonnet_high(monkeypatch):
    from atlas.pipeline.e2e import generate_e2e_tests
    from atlas.pipeline.util import role_target_label

    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    e2e_cfg = router.backend_for("e2e")
    assert e2e_cfg["type"] == "harness"
    assert e2e_cfg["provider"] == "claude_code"
    assert e2e_cfg["model"] == "sonnet"
    assert e2e_cfg["effort"] == "high"
    assert e2e_cfg["read_only"] is True
    assert e2e_cfg["auth"] == "subscription"
    assert e2e_cfg["fallback"] == "router"
    assert router.models_for("e2e") == ["openrouter/anthropic/claude-sonnet-5.5"]
    assert role_target_label(router, "e2e") == "claude_code/sonnet (high)"

    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", staticmethod(lambda: "/bin/claude"))
    suite_json = json.dumps(
        {
            "summary": "Black-box E2E tests",
            "product_interface": "CLI: python main.py",
            "verifiable": True,
            "files": [
                {"path": "tests/test_e2e_product.py", "content": "def test_ok(): assert True\n"}
            ],
        }
    )
    seen_calls: list[tuple[str, str | None, list[str] | None]] = []
    router_fallbacks: list[str] = []

    async def fake_claude_structured(
        self, *, role, model, system, user, tools=None, effort=None, **kwargs
    ):
        seen_calls.append((model, effort, tools))
        return HarnessInvocationResult(
            text=suite_json,
            used_model=model,
            harness="claude_code",
            auth_mode="subscription",
            cost_source="subscription",
            usage_reported=True,
        )

    async def fail_openrouter(model, system, user, **kwargs):
        router_fallbacks.append(model)
        return model, suite_json

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", fake_claude_structured)
    monkeypatch.setattr(router, "_complete_litellm", fail_openrouter)

    suite = asyncio.run(generate_e2e_tests(router, SPEC))
    assert suite.verifiable is True
    assert seen_calls == [("sonnet", "high", None)]
    assert router_fallbacks == []

    async def broken_claude(self, **kwargs):
        raise HarnessError("claude e2e failed")

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", broken_claude)
    fallback_suite = asyncio.run(generate_e2e_tests(router, SPEC))
    assert fallback_suite.verifiable is True
    assert router_fallbacks == ["openrouter/anthropic/claude-sonnet-5.5"]


def test_local_builder_prompt_uses_absolute_paths_for_e2e_and_skills(monkeypatch, tmp_path):
    from atlas.schemas import E2ETestSuite, GeneratedFile

    skill_dir = tmp_path / "skills" / "writing-clean-code"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: writing-clean-code\ndescription: Clean style\n---\nKeep functions small.\n",
        encoding="utf-8",
    )
    cand = LocalModelCandidate(
        provider="lm_studio", name=TWELVE_B, base_url=LMS, context_length=49152
    )

    async def ready(**kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand])

    seen: dict = {}

    async def fake_sdk(self, **kwargs):
        seen.update(kwargs)
        raise HarnessError("stop here")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk)

    cfg = tmp_path / "harness.yaml"
    cfg.write_text(
        f"""
roles:
  parser: [openrouter/m]
  researcher: [openrouter/m]
  judge: [openrouter/m]
  builder: [openrouter/m]
  fixer: [openrouter/m]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
skills:
  sources: [{str(tmp_path / "skills")!r}]
  allowlist: [writing-clean-code]
  discovery_mode: progressive
  by_role:
    builder: [writing-clean-code]
backends:
  builder:
    type: local_harness
    preferred: [{TWELVE_B}]
    only_preferred: true
    require_tool_smoke_check: false
    offload: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)
    app_dir = tmp_path / "app" / "x"
    suite = E2ETestSuite(
        summary="E2E",
        product_interface="CLI: python main.py",
        verifiable=True,
        files=[GeneratedFile(path="tests/test_e2e_product.py", content="def test_ok(): pass\n")],
    )

    with pytest.raises(HarnessError, match="stop here"):
        asyncio.run(build_app_workspace(router, SPEC, app_dir, _plan(), e2e_suite=suite))

    resolved = app_dir.resolve()
    prompt = seen["prompt_text"]
    assert str(resolved / ".agents" / "skills" / "writing-clean-code" / "SKILL.md") in prompt
    assert str(resolved / "tests" / "test_e2e_product.py") in prompt


def test_snapshot_app_files_excludes_atlas_planning_files_and_does_not_starve_source(tmp_path):
    from atlas.pipeline.build import snapshot_app_files

    app_dir = tmp_path / "app" / "demo"
    (app_dir / "tests").mkdir(parents=True)
    (app_dir / "PLAN.md").write_text("P" * 120_000, encoding="utf-8")
    (app_dir / "BUILD_BRIEF.md").write_text("B" * 20_000, encoding="utf-8")
    (app_dir / ".ignore").write_text(".venv/\n", encoding="utf-8")
    (app_dir / "index.html").write_text("<h1>Hello</h1>\n", encoding="utf-8")
    (app_dir / "tests" / "test_e2e_product.py").write_text(
        "def test_r1(): pass\n", encoding="utf-8"
    )

    snap = snapshot_app_files(app_dir)
    assert "PLAN.md" not in snap
    assert "BUILD_BRIEF.md" not in snap
    assert ".ignore" not in snap
    assert snap["index.html"] == "<h1>Hello</h1>\n"
    assert snap["tests/test_e2e_product.py"] == "def test_r1(): pass\n"


def test_runner_rotates_to_next_fixer_when_one_round_fails(monkeypatch, tmp_path):
    from atlas.router import OutputTruncatedError
    from atlas.runner import run_build
    from atlas.schemas import CheckResult

    spec = tmp_path / "demo.md"
    spec.write_text("# Demo\n\n## Requirements\n- R001: Greet\n", encoding="utf-8")
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
roles:
  parser: [m-p]
  researcher: [m-r]
  judge: [m-j]
  builder: [m-build]
  fixer: [m-a]
backends:
  fixer:
    type: openrouter
    rotate: [m-a, m-b]
limits:
  max_parallel_calls: 1
  max_tokens: 1000
  timeout_seconds: 10
  attempts_per_model: 1
  repair_attempts: 0
  max_fix_rounds: 3
verification:
  require_requirement_tests: false
  require_product_contract: false
  require_e2e_tests: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    app_dir = tmp_path / "app" / "demo"
    fixer_calls: list[str] = []

    async def fake_acompletion(*, model, messages, **kwargs):
        if model == "m-p":
            txt = json.dumps(
                {
                    "summary": "Demo",
                    "requirements": [
                        {
                            "id": "R001",
                            "title": "Greet",
                            "kind": "functional",
                            "description": "Greet",
                            "acceptance_criteria": [],
                            "depends_on": [],
                        }
                    ],
                }
            )
        elif model == "m-build":
            txt = json.dumps(
                {
                    "summary": "Built",
                    "files": [{"path": "app.py", "content": "OK = False\n"}],
                    "run_instructions": "pytest",
                }
            )
        elif model == "m-a":
            fixer_calls.append("m-a")
            raise OutputTruncatedError("cut off on round 1")
        else:
            fixer_calls.append("m-b")
            txt = json.dumps(
                {
                    "diagnosis": "flipped OK to True",
                    "files": [{"path": "app.py", "content": "OK = True\n"}],
                    "delete_files": [],
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=txt), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def fake_verify(root):
        passed = "OK = True" in (root / "app.py").read_text(encoding="utf-8")
        return "python", [
            CheckResult(
                name="python-pytest",
                command="pytest",
                passed=passed,
                exit_code=0 if passed else 1,
            )
        ]

    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_acompletion)
    run_dir = asyncio.run(
        run_build(
            spec,
            tmp_path / "runs",
            app_dir=app_dir,
            config_path=cfg,
            skip_research=True,
            verify_fn=fake_verify,
        )
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert fixer_calls == ["m-a", "m-b"]


def test_builder_records_and_publishes_chosen_skills(monkeypatch, tmp_path, caplog):
    import logging

    from atlas.events import RunEventBus, attach, detach

    skills_root = tmp_path / "skills"
    for s in ("developing-fullstack-apps", "writing-clean-code"):
        d = skills_root / s
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {s}\ndescription: {s}\n---\nRule for {s}.\n",
            encoding="utf-8",
        )

    cfg = tmp_path / "harness.yaml"
    cfg.write_text(
        f"""
roles:
  parser: [openrouter/m]
  researcher: [openrouter/m]
  judge: [openrouter/m]
  builder: [openrouter/m]
  fixer: [openrouter/m]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
skills:
  sources: [{str(skills_root)!r}]
  allowlist: [developing-fullstack-apps, writing-clean-code]
  discovery_mode: progressive
  by_role:
    builder: [developing-fullstack-apps, writing-clean-code]
backends:
  builder:
    type: harness
    provider: claude_code
    model: sonnet
    auth: subscription
""".strip()
        + "\n",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)
    app_dir = tmp_path / "app" / "demo"

    async def fake_ws(self, *, role, model, prompt_text, app_dir, **kwargs):
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "app.py").write_text("x = 1\n", encoding="utf-8")
        return (
            WorkspaceBuildResult(
                summary="Built app.\nSkills used: writing-clean-code",
                harness="claude_code",
                model=model,
                app_dir=str(app_dir),
                stack="python",
                created_files=["app.py"],
                files_snapshot=["app.py"],
            ),
            HarnessInvocationResult(
                text="ok",
                used_model=model,
                harness="claude_code",
                auth_mode="subscription",
                cost_source="subscription",
                usage_reported=True,
            ),
        )

    monkeypatch.setattr(ClaudeCodeAdapter, "_find_binary", staticmethod(lambda: "/bin/claude"))
    monkeypatch.setattr(ClaudeCodeAdapter, "run_workspace_task", fake_ws)

    bus = RunEventBus(persist=False)
    attach(bus)
    try:
        with caplog.at_level(logging.INFO, logger="atlas"):
            result = asyncio.run(build_app_workspace(router, SPEC, app_dir))
    finally:
        detach(bus)

    assert result.available_skills == ["developing-fullstack-apps", "writing-clean-code"]
    assert result.used_skills == ["writing-clean-code"]
    assert result.skills_mode == "progressive"
    assert (
        "[build] Builder skills used: writing-clean-code "
        "(available: developing-fullstack-apps, writing-clean-code; mode=progressive)"
    ) in caplog.text
    attempt_passed = next(
        e for e in bus.events_after(0) if e["kind"] == "attempt" and e["status"] == "passed"
    )
    assert attempt_passed["data"]["used_skills"] == ["writing-clean-code"]
    assert attempt_passed["data"]["skills_mode"] == "progressive"


def test_local_builder_ignores_startup_atime_scan_noise(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "writing-clean-code"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\nname: writing-clean-code\ndescription: Clean\n---\nKeep clean.\n",
        encoding="utf-8",
    )

    cfg = tmp_path / "local.yaml"
    cfg.write_text(
        f"""
roles:
  builder: [local-auto]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
skills:
  sources: [{str(skills_root)!r}]
  allowlist: [writing-clean-code]
  discovery_mode: progressive
  by_role:
    builder: [writing-clean-code]
backends:
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    preferred: [google/gemma-4-12b-qat]
    min_context_tokens: 49152
""".strip()
        + "\n",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)
    app_dir = tmp_path / "app" / "demo"
    candidate = LocalModelCandidate(
        provider="lm_studio", name="google/gemma-4-12b-qat", base_url="http://localhost:1234/v1"
    )

    async def ready(**kwargs):
        return LocalResolutionResult(selected=candidate, candidates=[candidate])

    async def fake_sdk(self, *, app_dir, **kwargs):
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "app.py").write_text("x = 1\n", encoding="utf-8")
        return (
            WorkspaceBuildResult(
                summary="Built app cleanly without citing skills.",
                harness="antigravity_sdk",
                model="lm_studio/google/gemma-4-12b-qat",
                app_dir=str(app_dir),
                stack="python",
                created_files=["app.py"],
                files_snapshot=["app.py"],
            ),
            HarnessInvocationResult(
                text="ok",
                used_model="lm_studio/google/gemma-4-12b-qat",
                harness="antigravity_sdk",
                auth_mode="local",
                cost_source="local_free",
                usage_reported=True,
            ),
        )

    real_stat = Path.stat
    stat_calls = 0

    class _BumpedStat:
        def __init__(self, wrapped, st_atime_ns: int) -> None:
            self._wrapped = wrapped
            self.st_atime_ns = st_atime_ns

        def __getattr__(self, name: str):
            return getattr(self._wrapped, name)

    def fake_stat(self: Path, **kw):
        nonlocal stat_calls
        real = real_stat(self, **kw)
        if self == skill_md:
            stat_calls += 1
            return _BumpedStat(real, real.st_atime_ns + stat_calls * 1_000_000)
        return real

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk)

    result = asyncio.run(build_app_workspace(router, SPEC, app_dir))
    assert result.available_skills == ["writing-clean-code"]
    assert result.used_skills == []


def test_local_build_prompt_advises_small_files_first(tmp_path):
    from atlas.pipeline.build import _build_workspace_prompt

    prompt_text = _build_workspace_prompt(
        SPEC,
        plan=None,
        raw_spec="# Demo\nSimple app.",
        skill_prompt="",
        brief_file="BUILD_BRIEF.md",
        workspace_root=tmp_path,
        require_absolute_paths=True,
    )
    assert "Write small files first" in prompt_text


# --- local OpenAI shaping proxy ---------------------------------------------------------


def test_local_openai_proxy_noop_when_unconfigured():
    from atlas.harnesses.antigravity import _local_openai_proxy

    upstream = "http://localhost:1234/v1"
    with _local_openai_proxy(upstream) as url:
        assert url == upstream


def test_local_openai_proxy_rewrites_chat_completions_and_streams_sse():
    import http.server
    import socketserver
    import threading

    from atlas.harnesses.antigravity import _local_openai_proxy

    captured: list[tuple[str, str, dict | None]] = []

    class _UpstreamHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            captured.append(("GET", self.path, None))
            body = json.dumps({"data": [{"id": "google/gemma-4-12b-qat"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            captured.append(("POST", self.path, payload))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = (
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                b"data: [DONE]\n\n",
            )
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()

    class _Srv(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = _Srv(("127.0.0.1", 0), _UpstreamHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        upstream_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        with _local_openai_proxy(
            upstream_url, reasoning_effort="none", max_completion_tokens=8192
        ) as proxy_url:
            models_resp = httpx.get(f"{proxy_url}/models", timeout=5.0)
            assert models_resp.status_code == 200
            assert models_resp.json()["data"][0]["id"] == "google/gemma-4-12b-qat"

            # SDK large default (65535) gets capped to 8192 and reasoning_effort injected.
            r1 = httpx.post(
                f"{proxy_url}/chat/completions",
                json={"model": "m", "messages": [], "max_completion_tokens": 65535},
                timeout=5.0,
            )
            assert r1.status_code == 200
            assert 'data: {"choices":[{"delta":{"content":"hi"}}]}' in r1.text
            assert "data: [DONE]" in r1.text

            # Smaller caller max_completion_tokens (512) is preserved.
            r2 = httpx.post(
                f"{proxy_url}/chat/completions",
                json={"model": "m", "messages": [], "max_completion_tokens": 512},
                timeout=5.0,
            )
            assert r2.status_code == 200
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2.0)

    assert captured[0] == ("GET", "/v1/models", None)
    assert captured[1][0:2] == ("POST", "/v1/chat/completions")
    assert captured[1][2] == {
        "model": "m",
        "messages": [],
        "reasoning_effort": "none",
        "max_completion_tokens": 8192,
    }
    assert captured[2][2]["max_completion_tokens"] == 512


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"reasoning_effort": "extreme"}, "reasoning_effort"),
        ({"max_completion_tokens": 0}, "max_completion_tokens"),
        ({"max_completion_tokens": True}, "max_completion_tokens"),
    ],
)
def test_builder_rejects_invalid_reasoning_or_token_cap(tmp_path, extra, match):
    cfg = {
        "roles": {"builder": ["local-auto"]},
        "backends": {
            "builder": {
                "type": "local_harness",
                "local_harness": "antigravity_sdk",
                "local_providers": {"lm_studio": LMS},
                **extra,
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
    cfg_path = tmp_path / "h.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        ModelRouter(cfg_path)


async def test_run_local_build_forwards_reasoning_effort_and_token_cap(monkeypatch, tmp_path):
    cand = _cand(TWELVE_B)
    seen: dict = {}

    async def ready(**kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand])

    async def fake_sdk(self, **kwargs):
        seen.update(kwargs)
        app_dir = kwargs["app_dir"]
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        return (
            WorkspaceBuildResult(
                summary="done",
                harness="antigravity_sdk",
                model=f"lm_studio/{TWELVE_B}",
                app_dir=str(app_dir),
                created_files=["main.py"],
                files_snapshot=["main.py"],
            ),
            HarnessInvocationResult(
                text="ok",
                used_model=f"lm_studio/{TWELVE_B}",
                harness="antigravity_sdk",
                auth_mode="local",
                cost_source="local_free",
                usage_reported=True,
            ),
        )

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk)
    router = _router(tmp_path, "      []")
    router.cfg["backends"]["builder"]["reasoning_effort"] = "none"
    router.cfg["backends"]["builder"]["max_completion_tokens"] = 8192

    await build_app_workspace(router, SPEC, tmp_path / "app" / "x")
    assert seen["reasoning_effort"] == "none"
    assert seen["max_completion_tokens"] == 8192
