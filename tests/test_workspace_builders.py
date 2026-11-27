"""`build_app_workspace` through each builder backend, with telemetry on success and failure.

The `agy` builder runs a real subprocess (a fake `agy` that edits files in its cwd); the local
SDK builder is faked at the adapter boundary. No model is called.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from atlas.harnesses import HarnessError, HarnessUnavailableError
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.pipeline.build import build_app_workspace
from atlas.router import ModelRouter
from atlas.schemas import ParsedSpec, Requirement

SPEC = ParsedSpec(
    summary="Greeting CLI",
    requirements=[Requirement(id="R1", title="Greet", description="Print a greeting")],
)
USAGE = {
    "input_tokens": 10415,
    "output_tokens": 657,
    "thinking_tokens": 616,
    "cache_read_tokens": 8113,
    "total_tokens": 11072,
}


def _router(tmp_path: Path, builder_yaml: str) -> ModelRouter:
    cfg = tmp_path / "builder.yaml"
    cfg.write_text(
        f"""
roles:
  builder: [local-auto]
backends:
  builder:
{builder_yaml}
limits:
  max_parallel_calls: 2
  max_tokens: 2000
  timeout_seconds: 30
  attempts_per_model: 1
""",
        encoding="utf-8",
    )
    return ModelRouter(cfg)


def _fake_agy(tmp_path: Path, envelope: dict, rc: int = 0) -> Path:
    record = tmp_path / "agy-build.json"
    script = f"""#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("1.5.15"); sys.exit(0)
pathlib.Path("main.py").write_text("print('hello')\\n", encoding="utf-8")
pathlib.Path({str(record)!r}).write_text(json.dumps({{"args": args, "cwd": os.getcwd()}}))
print({json.dumps(json.dumps(envelope))})
sys.exit({rc})
"""
    path = tmp_path / "agy"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


AGY_BUILDER = """    type: harness
    provider: antigravity
    model: gemini-4-hydrogen-high
    timeout_seconds: 60
    sandbox: true"""


async def test_agy_workspace_builder_writes_files_and_records_usage(monkeypatch, tmp_path):
    binary = _fake_agy(tmp_path, {"status": "SUCCESS", "response": "Built it.", "usage": USAGE})
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, AGY_BUILDER)
    app_dir = tmp_path / "app" / "greeting"

    result = await build_app_workspace(
        router, SPEC, app_dir, raw_spec="# Greeting\nPrint hello.", plan_markdown="# Plan\n"
    )

    assert result.created_files == ["main.py"]  # PLAN.md was written before the build
    assert (app_dir / "PLAN.md").is_file()
    seen = json.loads((tmp_path / "agy-build.json").read_text())
    assert Path(seen["cwd"]).resolve() == app_dir.resolve()
    args = seen["args"]
    assert args[args.index("--mode") + 1] == "auto" and "--sandbox" in args
    prompt = args[args.index("-p") + 1]
    # With PLAN.md on disk the prompt carries the compact spec (fits small local contexts).
    assert "COMPACT SPECIFICATION" in prompt and '"id": "R1"' in prompt
    assert "Read `PLAN.md` on disk" in prompt
    [call] = router.call_log
    assert call["role"] == "builder" and call["harness"] == "agy"
    assert call["total_tokens"] == 11072 and call["cached_tokens"] == 8113
    assert call["reasoning_tokens"] == 616


async def test_agy_workspace_builder_failure_still_records_billed_tokens(monkeypatch, tmp_path):
    binary = _fake_agy(
        tmp_path, {"status": "ERROR", "error": "ran out of turns", "usage": USAGE}, rc=1
    )
    monkeypatch.setattr("atlas.harnesses.antigravity.shutil.which", lambda _: str(binary))
    router = _router(tmp_path, AGY_BUILDER)

    with pytest.raises(HarnessError, match="ran out of turns"):
        await build_app_workspace(router, SPEC, tmp_path / "app" / "greeting")

    [call] = router.call_log
    assert call["error"].startswith("HarnessError")
    assert call["total_tokens"] == 11072


async def test_unsupported_workspace_builder_backend_is_rejected(tmp_path):
    router = _router(tmp_path, "    type: openrouter")
    with pytest.raises(HarnessError, match="Unsupported workspace builder backend type"):
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")


LOCAL_BUILDER = """    type: local_harness
    local_harness: antigravity_sdk
    local_model: null
    preferred: [gemma4:12b]
    min_context_tokens: 32768
    require_tool_smoke_check: true"""


async def test_local_builder_without_a_ready_model_fails_before_any_call(monkeypatch, tmp_path):
    async def no_model(**kwargs):
        return LocalResolutionResult(reason="Ollama and LM Studio are not running")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", no_model)
    router = _router(tmp_path, LOCAL_BUILDER)

    with pytest.raises(HarnessUnavailableError, match="not running"):
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")
    assert router.call_log == []


async def test_local_builder_sdk_crash_is_logged_as_free_local_call(monkeypatch, tmp_path):
    candidate = LocalModelCandidate(
        provider="ollama", name="gemma4:12b", base_url="http://localhost:11434/v1"
    )
    seen: dict = {}

    async def ready(**kwargs):
        seen.update(kwargs)
        return LocalResolutionResult(selected=candidate, candidates=[candidate])

    async def crash(self, **kwargs):
        raise RuntimeError("SDK session closed")

    from types import SimpleNamespace

    clock = 100.0

    def fake_mono() -> float:
        nonlocal clock
        clock += 0.125
        return clock

    monkeypatch.setattr("atlas.pipeline.build.time", SimpleNamespace(monotonic=fake_mono))
    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready)
    monkeypatch.setattr(
        "atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", crash
    )
    router = _router(tmp_path, LOCAL_BUILDER)

    with pytest.raises(RuntimeError, match="SDK session closed"):
        await build_app_workspace(
            router, SPEC, tmp_path / "app" / "x", local_model_override="gemma4:12b"
        )

    assert seen["explicit_model"] == "gemma4:12b"
    assert seen["min_context_tokens"] == 32768
    [call] = router.call_log
    assert call["used_model"] == "ollama/gemma4:12b"
    assert call["cost_source"] == "local_free" and call["auth_mode"] == "local"
    assert "SDK session closed" in call["error"]
    assert call["elapsed_ms"] > 0


async def test_harness_builder_failure_records_nonzero_elapsed_ms(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from atlas.harnesses import AntigravityCLIAdapter

    clock = 10.0

    def fake_mono() -> float:
        nonlocal clock
        clock += 0.25
        return clock

    monkeypatch.setattr("atlas.pipeline.build.time", SimpleNamespace(monotonic=fake_mono))

    async def fail_ws(self, **kwargs):
        raise HarnessError("agy crashed mid-build")

    monkeypatch.setattr(AntigravityCLIAdapter, "run_workspace_task", fail_ws)
    router = _router(
        tmp_path,
        """    type: harness
    provider: antigravity
    model: gemini-4-hydrogen-high""",
    )
    with pytest.raises(HarnessError, match="crashed"):
        await build_app_workspace(router, SPEC, tmp_path / "app" / "x")

    [call] = router.call_log
    assert call["elapsed_ms"] == 250.0
    assert "agy crashed" in call["error"]
