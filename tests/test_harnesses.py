import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.harnesses import (
    AntigravityCLIAdapter,
    ClaudeCodeAdapter,
    HarnessError,
    HarnessUnavailableError,
    run_doctor,
)
from atlas.harnesses.base import (
    HarnessInvocationResult,
    capture_workspace_build_result,
    find_git_changes_outside_app,
    scoped_subprocess_env,
    snapshot_file_hashes,
    snapshot_git_status,
)
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.router import ModelRouter
from atlas.runner import run_build
from atlas.schemas import CheckResult, WorkspaceBuildResult
from atlas.telemetry import ledger_entry, render_token_report_md, summarize_usage


def _harness_yaml(tmp_path: Path, *, supervisor_enabled: bool = True) -> Path:
    skills_dir = tmp_path / "skills"
    (skills_dir / "writing-clean-code").mkdir(parents=True)
    (skills_dir / "writing-clean-code" / "SKILL.md").write_text(
        "---\nname: writing-clean-code\ndescription: Clean code rules\n---\n\nKeep code clean.\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "harness.yaml"
    cfg.write_text(
        f"""
roles:
  parser: [openrouter/google/gemini-4-hydrogen]
  researcher: [openrouter/google/gemini-4-neon]
  judge: [openrouter/anthropic/claude-opus-5.5]
  builder: [local-auto]
  fixer: [openrouter/anthropic/claude-sonnet-5.5]
backends:
  parser:
    type: harness
    provider: antigravity
    model: gemini-4-hydrogen-high
    timeout_seconds: 180
    fallback: router
  researcher:
    type: harness
    provider: antigravity
    model: gemini-4-neon-high
    timeout_seconds: 300
    max_parallel: 3
    fallback: router
  judge:
    type: harness
    provider: claude_code
    model: opus
    read_only: true
    auth: subscription
    max_budget_usd: 1.50
  supervisor:
    enabled: {str(supervisor_enabled).lower()}
    type: harness
    provider: claude_code
    model: opus
    tools: [Read, Grep, Glob]
    auth: subscription
    max_budget_usd: 1.50
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    local_providers:
      ollama: http://localhost:11434/v1
      lm_studio: http://localhost:1234/v1
    local_model: gpt-oss:20b
    preferred: [gpt-oss:20b, gemma4:26b]
    require_tools: true
    min_context_tokens: 64000
    memory_budget_fraction: 0.70
    require_tool_smoke_check: true
  fixer:
    type: openrouter
    rotate:
      - openrouter/anthropic/claude-sonnet-5.5
      - openrouter/openai/gpt-6.2-sol
skills:
  sources: ["{skills_dir}"]
  allowlist: [writing-clean-code]
  by_role:
    builder: [writing-clean-code]
    fixer: [writing-clean-code]
limits:
  max_parallel_calls: 4
  max_tokens: 2000
  timeout_seconds: 60
  attempts_per_model: 1
  repair_attempts: 0
  max_fix_rounds: 3
  verify_timeout_seconds: 60
limits_extra:
  max_total_tokens_per_run: 50000
verification:
  require_requirement_tests: true
  require_product_contract: true
""",
        encoding="utf-8",
    )
    return cfg


def test_router_validates_v4_harness_config_and_rejects_unsafe_options(tmp_path):
    valid = ModelRouter(_harness_yaml(tmp_path))
    assert valid.backend_for("judge")["provider"] == "claude_code"
    assert valid.max_total_tokens == 50000
    assert valid.require_product_contract is True

    # Rejects cloud fallback on builder and unsafe supervisor tools.
    bad_fallback = tmp_path / "bad_fb.yaml"
    bad_fallback.write_text(
        """
roles:
  builder: [local]
backends:
  builder:
    type: local_harness
    allow_cloud_fallback: true
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="local-only"):
        ModelRouter(bad_fallback)

    bad_tools = tmp_path / "bad_tools.yaml"
    bad_tools.write_text(
        """
roles:
  supervisor: [opus]
backends:
  supervisor:
    type: harness
    provider: claude_code
    tools: [Read, Bash]
limits:
  max_parallel_calls: 1
  max_tokens: 100
  timeout_seconds: 10
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="subset of"):
        ModelRouter(bad_tools)


def test_scoped_subprocess_env_and_claude_billing_guard(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret")

    claude_env = scoped_subprocess_env(
        ("ANTHROPIC_", "CLAUDE_"), {"NO_COLOR"}, drop_names={"ANTHROPIC_API_KEY"}
    )
    assert "ANTHROPIC_API_KEY" not in claude_env
    assert "OPENROUTER_API_KEY" not in claude_env
    assert "GEMINI_API_KEY" not in claude_env


async def test_claude_code_billing_guard_and_strict_readonly_flags(monkeypatch, tmp_path):
    seen_calls: list[tuple[list[str], dict[str, str], Path | None]] = []

    async def fake_run_cli(args, *, env, timeout_seconds, cwd=None):
        if args[1] == "--version":
            return 0, "2.1.330 (Claude Code)", ""
        seen_calls.append((list(args), dict(env), cwd))
        return (
            0,
            json.dumps(
                {
                    "structured_output": {"approved": True, "summary": "ok"},
                    "total_cost_usd": 0.42,
                    "usage": {"input_tokens": 120, "output_tokens": 45},
                }
            ),
            "",
        )

    monkeypatch.setattr(
        "atlas.harnesses.claude_code.shutil.which", lambda _: "/usr/local/bin/claude"
    )
    monkeypatch.setattr("atlas.harnesses.claude_code._run_cli", fake_run_cli)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-be-stripped-in-subscription")

    adapter = ClaudeCodeAdapter()

    # Subscription auth strips ANTHROPIC_API_KEY, passes --tools "", omits --max-turns and
    # --dangerously-skip-permissions, and records cost_source="subscription" with cost_usd=None.
    res_sub = await adapter.run_structured(
        role="judge",
        model="opus",
        system="sys",
        user="usr",
        json_schema={"type": "object"},
        auth="subscription",
        max_budget_usd=1.50,
    )
    args_sub, env_sub, _ = seen_calls[0]
    assert "--dangerously-skip-permissions" not in args_sub
    assert "--max-turns" not in args_sub
    assert args_sub[args_sub.index("--tools") + 1] == ""
    assert "--json-schema" in args_sub
    assert args_sub[args_sub.index("--max-budget-usd") + 1] == "1.50"
    assert "ANTHROPIC_API_KEY" not in env_sub
    assert res_sub.cost_usd is None
    assert res_sub.cost_source == "subscription"
    assert res_sub.auth_mode == "subscription"

    # Supervisor with auth="api" preserves ANTHROPIC_API_KEY and passes Read,Grep,Glob + cwd.
    res_api = await adapter.run_structured(
        role="supervisor",
        model="opus",
        system="sys",
        user="usr",
        auth="api",
        tools=["Read", "Grep", "Glob"],
        cwd=tmp_path,
    )
    args_api, env_api, cwd_api = seen_calls[1]
    assert env_api["ANTHROPIC_API_KEY"] == "should-be-stripped-in-subscription"
    assert args_api[args_api.index("--tools") + 1] == "Read,Grep,Glob"
    assert cwd_api == tmp_path
    assert res_api.cost_usd == 0.42
    assert res_api.cost_source == "reported"

    with pytest.raises(ValueError, match="disallowed tools"):
        await adapter.run_structured(
            role="supervisor", model="opus", system="s", user="u", tools=["Bash"]
        )


async def test_agy_status_check_schema_timeout_and_router_fallback(monkeypatch, tmp_path):
    seen_args: list[list[str]] = []

    async def fake_run_agy(args, *, env, timeout_seconds, cwd=None):
        if args[1] == "--version":
            return 0, "1.5.15", ""
        seen_args.append(list(args))
        # Exit code 0 with status != SUCCESS must fail loudly!
        return (
            0,
            json.dumps({"status": "DENIED", "error": "permission denied by sandbox"}),
            "",
        )

    monkeypatch.setattr(
        "atlas.harnesses.antigravity.shutil.which", lambda _: "/opt/homebrew/bin/agy"
    )
    monkeypatch.setattr("atlas.harnesses.antigravity._run_agy", fake_run_agy)

    adapter = AntigravityCLIAdapter()
    with pytest.raises(HarnessError, match="status=DENIED"):
        await adapter.run_structured(
            role="parser",
            model="gemini-4-hydrogen-high",
            system="sys",
            user="usr",
            json_schema={"type": "object"},
            timeout_seconds=180,
        )

    assert "--sandbox" in seen_args[0]
    assert "--json-schema" in seen_args[0]
    assert "--print-timeout" in seen_args[0]

    # When fallback: router is set, Router falls through to LiteLLM when agy fails.
    router = ModelRouter(_harness_yaml(tmp_path))

    async def fake_acompletion(*, model, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, cost=0.001),
        )

    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_acompletion)
    used, text = await router.complete_structured(
        "openrouter/google/gemini-4-hydrogen", "sys", "usr", role="parser"
    )
    assert used == "openrouter/google/gemini-4-hydrogen"
    assert text == '{"ok": true}'
    assert len(router.call_log) == 2
    assert router.call_log[0]["error"]
    assert router.call_log[1]["cost_source"] == "reported"


def test_workspace_guards_reject_symlink_escape_and_git_escape(tmp_path):
    repo = tmp_path / "repo"
    app_dir = repo / "app" / "demo"
    app_dir.mkdir(parents=True)
    (repo / "README.md").write_text("# Root\n", encoding="utf-8")

    def git(*args: str):
        subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-b", "main")
    git("config", "user.name", "Tester")
    git("config", "user.email", "tester@example.com")
    git("add", "README.md")
    git("commit", "-m", "init")

    before_git = snapshot_git_status(app_dir)
    before_hashes = snapshot_file_hashes(app_dir)
    (app_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
    # Modification inside app_dir is allowed:
    assert find_git_changes_outside_app(app_dir, before_git) == []

    # Modification outside app_dir in repo IS caught!
    (repo / "src_leak.py").write_text("bad\n", encoding="utf-8")
    escaped = find_git_changes_outside_app(app_dir, before_git)
    assert escaped == ["src_leak.py"]

    res = capture_workspace_build_result(
        app_dir,
        before_hashes,
        summary="Built",
        harness="antigravity_sdk",
        harness_version="0.1.24",
        model="ollama/gpt-oss:20b",
    )
    assert isinstance(res, WorkspaceBuildResult)
    assert res.created_files == ["main.py"]

    outside = tmp_path / "secret.txt"
    outside.write_text("nope", encoding="utf-8")
    (app_dir / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink escapes workspace"):
        snapshot_file_hashes(app_dir)


def test_honest_telemetry_tracks_unpriced_auth_mode_and_run_metadata():
    summary = summarize_usage(
        [
            {
                "role": "judge",
                "requested_model": "opus",
                "used_model": "opus",
                "harness": "claude_code",
                "auth_mode": "subscription",
                "cost_usd": None,
                "cost_source": "subscription",
                "prompt_tokens": 200,
                "completion_tokens": 50,
            },
            {
                "role": "builder",
                "requested_model": "gpt-oss:20b",
                "used_model": "ollama/gpt-oss:20b",
                "harness": "antigravity_sdk",
                "auth_mode": "local",
                "cost_usd": 0.0,
                "cost_source": "local_free",
            },
        ]
    )
    assert summary.cost_usd == 0.0
    assert summary.unpriced_calls == 1
    assert summary.calls[0].auth_mode == "subscription"
    assert summary.calls[1].cost_source == "local_free"
    assert "partial" in render_token_report_md(summary)
    entry = ledger_entry(
        run_id="r1",
        mode="build",
        spec_path="specs/x.md",
        status="verified",
        summary=summary,
        local_model="ollama/gpt-oss:20b",
        builder_backend="local_harness",
        fix_rounds=2,
    )
    assert entry["local_model"] == "ollama/gpt-oss:20b"
    assert entry["builder_backend"] == "local_harness"
    assert entry["fix_rounds"] == 2
    assert entry["unpriced_calls"] == 1


async def test_local_builder_rotation_and_supervisor_loop(tmp_path, monkeypatch):
    cfg_path = _harness_yaml(tmp_path, supervisor_enabled=True)
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nBuild add(a, b).\n", encoding="utf-8")

    async def fake_agy_structured(self, *, role, model, system, user, **kwargs):
        if role == "parser":
            payload = {
                "summary": "Adder demo",
                "requirements": [{"id": "R001", "title": "Add", "description": "Add two ints."}],
            }
        else:
            payload = {
                "requirement_id": "R001",
                "model": model,
                "candidates": [{"name": "stdlib", "pros": ["built-in"]}],
                "findings": ["Pure Python works"],
            }
        return HarnessInvocationResult(
            text=json.dumps(payload),
            used_model=model,
            harness="agy",
            harness_version="1.5.15",
            auth_mode="cached_login",
            cost_usd=None,
            cost_source="unavailable",
        )

    supervisor_calls = 0

    async def fake_claude_structured(self, *, role, model, system, user, **kwargs):
        nonlocal supervisor_calls
        if role == "judge":
            text = json.dumps(
                {
                    "decisions": [
                        {
                            "requirement_id": "R001",
                            "options": [{"name": "stdlib"}, {"name": "numpy"}],
                            "pick": "stdlib",
                            "rationale": "Zero deps",
                        }
                    ],
                    "risks": [],
                    "tasks": ["Implement add"],
                }
            )
        else:
            supervisor_calls += 1
            # Reject on first supervisor review, approve after Sol 6.2 applies cleanup patch!
            text = json.dumps(
                {
                    "approved": supervisor_calls > 1,
                    "summary": "Needs docstring" if supervisor_calls == 1 else "Clean!",
                    "issues": ["Add docstring"] if supervisor_calls == 1 else [],
                    "requirement_notes": {"R001": "Verified"},
                }
            )
        return HarnessInvocationResult(
            text=text,
            used_model=model,
            harness="claude_code",
            harness_version="2.1.330",
            auth_mode="subscription",
            cost_usd=None,
            cost_source="subscription",
        )

    monkeypatch.setattr(AntigravityCLIAdapter, "run_structured", fake_agy_structured)
    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", fake_claude_structured)

    # Offline -> fails cleanly with no cloud builder fallback.
    async def offline_local(**kwargs):
        return LocalResolutionResult(reason="No local servers running.")

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", offline_local)
    app_dir = tmp_path / "out_app"
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "usage" / "ledger.jsonl"

    with pytest.raises(HarnessUnavailableError, match="No local servers"):
        await run_build(spec_file, runs_root, app_dir, config_path=cfg_path, ledger_path=ledger)

    # Local ready -> SDK builds buggy code -> Fix Round 1 (Sonnet 5.5) fixes bug ->
    # Supervisor rejects -> Fix Round 2 (Sol 6.2 + previous_diagnoses) adds docstring -> approved!
    candidate = LocalModelCandidate(
        provider="ollama",
        name="gpt-oss:20b",
        base_url="http://localhost:11434/v1",
        parameter_size_b=20.0,
        supports_tools=True,
        tool_smoke_passed=True,
    )

    async def ready_local(**kwargs):
        return LocalResolutionResult(selected=candidate, candidates=[candidate], reason="Ready")

    async def fake_sdk_build(self, *, candidate, prompt_text, app_dir, **kwargs):
        assert "Keep code clean." in prompt_text
        (app_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
        (app_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (app_dir / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (app_dir / "test_app.py").write_text(
            "# R001\nfrom app import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )
        return (
            WorkspaceBuildResult(
                summary="Local workspace build",
                harness="antigravity_sdk",
                harness_version="0.1.24",
                model="ollama/gpt-oss:20b",
                app_dir=str(app_dir),
                stack="python",
                created_files=["README.md", "app.py", "requirements.txt", "test_app.py"],
                files_snapshot=["README.md", "app.py", "requirements.txt", "test_app.py"],
            ),
            HarnessInvocationResult(
                text="Built files",
                used_model="ollama/gpt-oss:20b",
                harness="antigravity_sdk",
                harness_version="0.1.24",
                auth_mode="local",
                cost_usd=0.0,
                cost_source="local_free",
            ),
        )

    fix_models_seen: list[str] = []
    fix_user_payloads: list[dict] = []

    async def fake_fix_acompletion(*, model, messages, **kwargs):
        fix_models_seen.append(model)
        assert "Keep code clean." in messages[0]["content"]
        fix_user_payloads.append(json.loads(messages[1]["content"]))
        if len(fix_models_seen) == 1:
            diag = "Fixed subtraction to addition"
            content = "def add(a, b):\n    return a + b\n"
        else:
            diag = "Added docstring per supervisor"
            content = 'def add(a, b):\n    """Add two integers."""\n    return a + b\n'
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "diagnosis": diag,
                                "files": [{"path": "app.py", "content": content, "purpose": "Fix"}],
                            }
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=80, completion_tokens=30, cost=0.002),
        )

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready_local)
    monkeypatch.setattr(
        "atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk_build
    )
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_fix_acompletion)

    async def fake_verify(path: Path):
        code = (path / "app.py").read_text(encoding="utf-8")
        passed = "a + b" in code
        return (
            "python",
            [
                CheckResult(
                    name="python-pytest",
                    command="pytest",
                    passed=passed,
                    exit_code=0 if passed else 1,
                )
            ],
        )

    out_dir = await run_build(
        spec_file,
        runs_root,
        app_dir,
        config_path=cfg_path,
        clean_app_dir=True,
        verify_fn=fake_verify,
        ledger_path=ledger,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["local_model"] == "ollama/gpt-oss:20b"
    assert manifest["builder_backend"] == "local_harness"
    assert manifest["fix_rounds"] == 2
    assert fix_models_seen == [
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/openai/gpt-6.2-sol",
    ]
    assert fix_user_payloads[1]["previous_diagnoses"] == ["Fixed subtraction to addition"]
    assert supervisor_calls == 2
    assert (out_dir / "supervisor_review.json").is_file()
    assert not (app_dir / ".agents").exists()

    ledger_lines = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ledger_lines[-1]["local_model"] == "ollama/gpt-oss:20b"
    assert ledger_lines[-1]["fix_rounds"] == 2

    doc = await run_doctor(ModelRouter(cfg_path))
    assert doc["skills"]["resolved"] == ["writing-clean-code"]


def test_atlas_run_alias_cli_help():
    proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--local-model" in proc.stdout
