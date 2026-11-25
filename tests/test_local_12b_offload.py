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

from atlas.harnesses import AntigravitySDKAdapter, HarnessError
from atlas.local_models import (
    LocalModelCandidate,
    LocalResolutionResult,
    lm_studio_runtime_context,
    local_resolution_kwargs,
    resolve_local_model,
)
from atlas.pipeline.build import build_app_workspace
from atlas.pipeline.render import render_build_brief
from atlas.router import ModelRouter
from atlas.schemas import (
    CandidateTool,
    ClarificationItem,
    ClarificationReport,
    ParsedSpec,
    Plan,
    Requirement,
    ToolDecision,
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
        type: openrouter
        model: openrouter/anthropic/claude-sonnet-5.5"""


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


async def test_build_goes_to_sonnet_when_hydrogen_fails_too(monkeypatch, tmp_path):
    _no_local_model(monkeypatch)
    binary = _fake_agy(tmp_path, rc=1)
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, HYDROGEN_THEN_SONNET)
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
    assert result.model == "openrouter/anthropic/claude-sonnet-5.5"
    assert result.run_instructions == "python main.py"
    assert (app_dir / "main.py").read_text() == "print('hi')\n"
    # The failed agy attempt is still recorded (it may have been billed).
    assert router.call_log[0]["harness"] == "antigravity" and router.call_log[0]["error"]


async def test_every_target_failing_reports_each_reason(monkeypatch, tmp_path):
    _no_local_model(monkeypatch)
    binary = _fake_agy(tmp_path, rc=1)
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, HYDROGEN_THEN_SONNET)

    async def broken(model, system, user, role=None):
        raise RuntimeError("HTTP 402: no credit")

    monkeypatch.setattr(router, "complete", broken)
    with pytest.raises(HarnessError) as info:
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")
    message = str(info.value)
    assert "local: HarnessUnavailableError" in message
    assert "hydrogen: HarnessError" in message and "quota exhausted" in message
    assert "sonnet: RuntimeError: HTTP 402" in message


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

    async def fake_sdk(self, **kwargs):
        seen.update(kwargs)
        raise HarnessError("stop here")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr("atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk)
    router = _router(tmp_path, "      []")
    app_dir = tmp_path / "app" / "x"

    with pytest.raises(HarnessError, match="stop here"):
        await build_app_workspace(
            router, SPEC, app_dir, _plan(), plan_markdown="# Plan\n" + "long " * 30000
        )

    assert (app_dir / "BUILD_BRIEF.md").is_file()
    prompt = seen["prompt_text"]
    assert "Read `BUILD_BRIEF.md` first" in prompt
    assert "do not read it whole" in prompt
    assert "Return ONLY JSON" not in prompt  # workspace agents write files, not a JSON reply


# --- config validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("offload_yaml", "message"),
    [
        ("      - {type: harness, provider: claude_code}", "provider must be 'antigravity'"),
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
    builder = ModelRouter(repo / "config" / "harness.yaml").backend_for("builder")
    assert builder["preferred"][0] == TWELVE_B
    assert all("26b" not in m.lower() for m in builder["preferred"])
    assert builder["only_preferred"] is True and builder["max_parameter_size_b"] <= 14
    assert [t["name"] for t in builder["offload"]] == ["hydrogen", "sonnet"]


def test_shipped_fixer_uses_only_sonnet_and_hydrogen():
    repo = Path(__file__).resolve().parents[1]
    router = ModelRouter(repo / "config" / "harness.yaml")
    assert router.backend_for("fixer")["rotate"] == [
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/google/gemini-4-hydrogen",
    ]
    for model in ("openrouter/anthropic/claude-sonnet-5.5", "openrouter/google/gemini-4-hydrogen"):
        candidates = router.candidates_for(model, role="fixer")
        assert all("sonnet" in m or "hydrogen" in m for m in candidates)
        assert len(candidates) == 3  # the requested model, the other one, direct Gemini
