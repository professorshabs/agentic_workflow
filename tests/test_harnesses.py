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
from atlas.pipeline.build import merge_supervisor_reviews
from atlas.router import ModelRouter
from atlas.runner import run_build
from atlas.schemas import CheckResult, ReviewerVerdict, WorkspaceBuildResult
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
    seen_stdin: list[str | None] = []

    async def fake_run_cli(args, *, env, timeout_seconds, cwd=None, input_text=None):
        if args[1] == "--version":
            return 0, "2.1.330 (Claude Code)", ""
        seen_calls.append((list(args), dict(env), cwd))
        seen_stdin.append(input_text)
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
    assert args_sub[args_sub.index("--max-budget-usd") + 1] == "1.5"
    assert "ANTHROPIC_API_KEY" not in env_sub
    # The prompt travels on stdin, never in argv (argv entries are capped at 128 KiB on Linux).
    assert seen_stdin[0] is not None and "USER REQUEST:\nusr" in seen_stdin[0]
    assert not any("USER REQUEST" in a for a in args_sub)
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

    # Ignored workspace dirs (such as Atlas's own .agents/skills symlinks) may target external
    # skill directories without triggering the workspace-escape guard.
    external_skill = tmp_path / "skills" / "demo-skill"
    external_skill.mkdir(parents=True)
    (external_skill / "SKILL.md").write_text("# Demo Skill\n", encoding="utf-8")
    (app_dir / ".agents" / "skills").mkdir(parents=True)
    (app_dir / ".agents" / "skills" / "demo-skill").symlink_to(external_skill)
    hashes_with_skill = snapshot_file_hashes(app_dir)
    assert "main.py" in hashes_with_skill
    assert not any(k.startswith(".agents/") for k in hashes_with_skill)

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


def test_dual_supervisor_config_validation(tmp_path):
    repo_harness = Path(__file__).resolve().parents[1] / "config" / "harness.yaml"
    real_router = ModelRouter(repo_harness)
    assert [r["name"] for r in real_router.supervisor_reviewers()] == ["opus", "neon"]
    assert real_router.backend_for("supervisor")["enabled"] is True

    def _write_sup(body: str, name: str) -> Path:
        p = tmp_path / f"{name}.yaml"
        p.write_text(
            f"""
roles:
  supervisor: [openrouter/google/gemini-4-neon]
backends:
  supervisor:
{body}
limits:
  max_parallel_calls: 2
  max_tokens: 1000
  timeout_seconds: 30
""",
            encoding="utf-8",
        )
        return p

    with pytest.raises(ValueError, match="non-empty list"):
        ModelRouter(_write_sup("    enabled: true\n    reviewers: []", "empty"))

    with pytest.raises(ValueError, match="duplicate name"):
        ModelRouter(
            _write_sup(
                "    reviewers:\n"
                "      - name: opus\n"
                "        type: harness\n"
                "        provider: claude_code\n"
                "      - name: OPUS\n"
                "        type: openrouter\n"
                "        model: openrouter/google/gemini-4-neon",
                "dup",
            )
        )

    with pytest.raises(ValueError, match="subset of"):
        ModelRouter(
            _write_sup(
                "    reviewers:\n"
                "      - name: opus\n"
                "        type: harness\n"
                "        provider: claude_code\n"
                "        tools: [Read, Bash]",
                "bad_tool",
            )
        )

    with pytest.raises(ValueError, match="strictly read-only"):
        ModelRouter(
            _write_sup(
                "    reviewers:\n"
                "      - name: opus\n"
                "        type: harness\n"
                "        provider: claude_code\n"
                "        read_only: false",
                "rw_claude",
            )
        )

    with pytest.raises(ValueError, match="mode must be one of"):
        ModelRouter(
            _write_sup(
                "    mode: majority\n"
                "    reviewers:\n"
                "      - name: neon\n"
                "        type: openrouter\n"
                "        model: openrouter/google/gemini-4-neon",
                "bad_mode",
            )
        )


def test_merge_supervisor_reviews_merges_and_deduplicates():
    v_opus_ok = ReviewerVerdict(
        reviewer="opus",
        model="opus",
        harness="claude_code",
        approved=True,
        summary="Tests pass",
        requirement_notes={"R001": "Functional behavior verified"},
    )
    v_neon_ok = ReviewerVerdict(
        reviewer="neon",
        model="gemini-4-neon-high",
        harness="agy",
        approved=True,
        summary="Clean edge cases",
        requirement_notes={"R001": "Input validation verified"},
    )
    both_ok = merge_supervisor_reviews([v_opus_ok, v_neon_ok], mode="unanimous")
    assert both_ok.approved is True
    assert "[opus: APPROVED] Tests pass" in both_ok.summary
    assert "[neon: APPROVED] Clean edge cases" in both_ok.summary
    assert both_ok.requirement_notes["R001"] == (
        "[opus] Functional behavior verified | [neon] Input validation verified"
    )

    v_opus_no = ReviewerVerdict(
        reviewer="opus",
        model="opus",
        harness="claude_code",
        approved=False,
        summary="Missing error check",
        issues=["Reject negative inputs in app.py", "Add CLI exit test"],
        improvement_plan=["Validate negative inputs in app.py"],
        requirement_notes={"R001": "Allows negative inputs"},
    )
    v_neon_no = ReviewerVerdict(
        reviewer="neon",
        model="openrouter/google/gemini-4-neon",
        harness="openrouter",
        approved=False,
        summary="Edge cases incomplete",
        issues=["reject negative inputs in app.py"],  # duplicates Opus with different case
        improvement_plan=[],  # omitted plan should fall back to its issue
        requirement_notes={"R001": "Missing boundary check"},
    )
    merged = merge_supervisor_reviews([v_opus_no, v_neon_ok], mode="unanimous")
    assert merged.approved is False
    assert "[opus] Reject negative inputs in app.py" in merged.issues

    any_ok = merge_supervisor_reviews([v_opus_no, v_neon_ok], mode="any_approve")
    assert any_ok.approved is True

    both_no = merge_supervisor_reviews([v_opus_no, v_neon_no], mode="unanimous")
    assert both_no.approved is False
    assert both_no.issues == [
        "[opus, neon] Reject negative inputs in app.py",
        "[opus] Add CLI exit test",
    ]
    assert both_no.improvement_plan == [
        "[opus] Validate negative inputs in app.py",
        "[neon] reject negative inputs in app.py",
    ]

    with pytest.raises(ValueError, match="empty"):
        merge_supervisor_reviews([])


async def test_dual_post_build_supervisor_run_build_and_doctor(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    (skills_dir / "writing-clean-code").mkdir(parents=True)
    (skills_dir / "writing-clean-code" / "SKILL.md").write_text(
        "---\nname: writing-clean-code\ndescription: Clean code rules\n---\n\nKeep code clean.\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "dual_harness.yaml"
    cfg_path.write_text(
        f"""
roles:
  parser: [openrouter/google/gemini-4-hydrogen]
  researcher: [openrouter/google/gemini-4-neon]
  judge: [openrouter/anthropic/claude-opus-5.5]
  supervisor: [openrouter/google/gemini-4-neon]
  builder: [local-auto]
  fixer: [openrouter/anthropic/claude-sonnet-5.5]
backends:
  parser:
    type: harness
    provider: antigravity
    model: gemini-4-hydrogen-high
    fallback: router
  supervisor:
    enabled: true
    mode: unanimous
    reviewers:
      - name: opus
        type: harness
        provider: claude_code
        model: opus
        tools: [Read, Grep, Glob]
        auth: subscription
        max_budget_usd: 1.50
      - name: neon
        type: harness
        provider: antigravity
        model: gemini-4-neon-high
        fallback: router
        router_model: openrouter/google/gemini-4-neon
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    local_model: gpt-oss:20b
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
verification:
  require_requirement_tests: true
  require_product_contract: true
""",
        encoding="utf-8",
    )

    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nBuild add(a, b) handling non-negative ints.\n", encoding="utf-8")

    agy_sup_calls = 0

    async def fake_agy_structured(self, *, role, model, system, user, **kwargs):
        nonlocal agy_sup_calls
        if role == "parser":
            payload = {
                "summary": "Adder demo",
                "requirements": [{"id": "R001", "title": "Add", "description": "Add ints."}],
            }
            return HarnessInvocationResult(
                text=json.dumps(payload),
                used_model=model,
                harness="agy",
                harness_version="1.5.15",
                auth_mode="cached_login",
            )
        if role == "supervisor":
            agy_sup_calls += 1
            if agy_sup_calls == 1:
                # Round 1: agy fails -> falls back to OpenRouter neon reviewer!
                raise HarnessError("agy supervisor transient failure")
            return HarnessInvocationResult(
                text=json.dumps(
                    {
                        "approved": True,
                        "summary": "Neon confirmed negative guard fixed",
                        "issues": [],
                        "improvement_plan": [],
                        "requirement_notes": {"R001": "Negative input check verified"},
                    }
                ),
                used_model=model,
                harness="agy",
                harness_version="1.5.15",
                auth_mode="cached_login",
            )
        raise AssertionError(f"Unexpected agy role: {role}")

    opus_calls = 0

    async def fake_claude_structured(self, *, role, model, system, user, **kwargs):
        nonlocal opus_calls
        assert role == "supervisor"
        opus_calls += 1
        return HarnessInvocationResult(
            text=json.dumps(
                {
                    "approved": True,
                    "summary": "Opus sees working add",
                    "issues": [],
                    "improvement_plan": [],
                    "requirement_notes": {"R001": "Basic test passes"},
                }
            ),
            used_model=model,
            harness="claude_code",
            harness_version="2.1.330",
            auth_mode="subscription",
            cost_usd=None,
            cost_source="subscription",
        )

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
        (app_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
        (app_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (app_dir / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
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

    neon_router_payloads: list[dict] = []
    fix_user_payloads: list[dict] = []

    async def fake_acompletion(*, model, messages, **kwargs):
        user_payload = json.loads(messages[1]["content"])
        if model == "openrouter/google/gemini-4-neon":
            # OpenRouter Neon fallback must receive full file contents!
            assert "app.py" in user_payload.get("current_files", {})
            neon_router_payloads.append(user_payload)
            content = json.dumps(
                {
                    "approved": False,
                    "summary": "Missing negative input validation",
                    "issues": ["Reject negative inputs in app.py"],
                    "improvement_plan": ["Validate inputs before adding in app.py"],
                    "requirement_notes": {"R001": "Needs negative guard"},
                }
            )
        else:
            assert model == "openrouter/anthropic/claude-sonnet-5.5"
            fix_user_payloads.append(user_payload)
            content = json.dumps(
                {
                    "diagnosis": "Added negative input check per dual supervisor plan",
                    "files": [
                        {
                            "path": "app.py",
                            "content": (
                                "def add(a, b):\n"
                                "    if a < 0 or b < 0:\n"
                                "        raise ValueError('negative')\n"
                                "    return a + b\n"
                            ),
                            "purpose": "Validate non-negative inputs",
                        }
                    ],
                }
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=60, completion_tokens=25, cost=0.001),
        )

    monkeypatch.setattr(AntigravityCLIAdapter, "run_structured", fake_agy_structured)
    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", fake_claude_structured)
    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready_local)
    monkeypatch.setattr(
        "atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk_build
    )
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_acompletion)

    async def fake_verify(path: Path):
        return (
            "python",
            [CheckResult(name="python-pytest", command="pytest", passed=True, exit_code=0)],
        )

    app_dir = tmp_path / "out_app"
    runs_root = tmp_path / "runs"
    ledger = tmp_path / "usage" / "ledger.jsonl"

    out_dir = await run_build(
        spec_file,
        runs_root,
        app_dir,
        config_path=cfg_path,
        skip_research=True,
        clean_app_dir=True,
        verify_fn=fake_verify,
        ledger_path=ledger,
    )

    assert len(neon_router_payloads) == 1
    assert opus_calls == 2
    assert agy_sup_calls == 2
    assert len(fix_user_payloads) == 1
    sup_err = fix_user_payloads[0]["failing_checks"][0]["stderr"]
    assert "Issues:" in sup_err
    assert "[neon] Reject negative inputs in app.py" in sup_err
    assert "Improvement plan:" in sup_err
    assert "[neon] Validate inputs before adding in app.py" in sup_err
    assert "Requirement notes:" in sup_err
    assert "R001: [opus] Basic test passes | [neon] Needs negative guard" in sup_err

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["supervisor_approved"] is True
    assert manifest["supervisor_reviewers"] == ["opus", "neon"]
    assert manifest["fix_rounds"] == 1

    review_json = json.loads((out_dir / "supervisor_review.json").read_text(encoding="utf-8"))
    assert review_json["approved"] is True
    assert [r["reviewer"] for r in review_json["reviewers"]] == ["opus", "neon"]

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Supervisor review" in report_md
    assert "| `opus` |" in report_md
    assert "| `neon` |" in report_md

    doc = await run_doctor(ModelRouter(cfg_path))
    providers = {chk["provider"] for chk in doc["checks"]}
    assert {"antigravity", "claude_code", "openrouter", "local_model_daemon"}.issubset(providers)


@pytest.mark.asyncio
async def test_supervise_app_records_failed_usage_and_repairs_then_falls_back(
    tmp_path, monkeypatch
):
    from atlas.pipeline import build as build_mod
    from atlas.schemas import ParsedSpec, Requirement, SupervisorOutput, VerificationRound

    cfg = tmp_path / "sup.yaml"
    cfg.write_text(
        """
roles:
  supervisor: [openrouter/anthropic/claude-opus-5.5]
backends:
  supervisor:
    enabled: true
    mode: unanimous
    reviewers:
      - name: opus
        type: harness
        provider: claude_code
        model: opus
        auth: subscription
        repair_attempts: 1
        fallback: router
        router_model: openrouter/anthropic/claude-opus-5.5
limits:
  max_parallel_calls: 2
  max_tokens: 1000
  timeout_seconds: 10
  repair_attempts: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)
    spec = ParsedSpec(
        summary="App",
        requirements=[Requirement(id="R001", title="Req", description="Desc")],
    )
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "app.py").write_text("x = 1\n", encoding="utf-8")
    vround = VerificationRound(
        round_number=1,
        passed=True,
        stack_detected="python",
        checks=[CheckResult(name="pytest", command="pytest", passed=True, exit_code=0)],
    )

    err = HarnessError("timed out after billing")
    err.usage = {
        "used_model": "claude-opus-5-5",
        "auth_mode": "subscription",
        "prompt_tokens": 111,
        "completion_tokens": 22,
        "total_tokens": 133,
        "cost_usd": 0.04,
        "cost_source": "reported",
        "usage_reported": True,
    }

    async def fail_with_usage(self, **kwargs):
        raise err

    fallback_calls: list[str] = []

    async def fake_complete_json(rtr, *, model, role, system, user, schema, **kwargs):
        fallback_calls.append(model)
        return model, SupervisorOutput(approved=True, summary="Fallback approved")

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", fail_with_usage)
    monkeypatch.setattr(build_mod, "complete_json_model", fake_complete_json)

    review = await build_mod.supervise_app(router, spec, app_dir, vround)
    assert review.approved is True
    assert fallback_calls == ["openrouter/anthropic/claude-opus-5.5"]
    failed_record = next(c for c in router.call_log if c.get("error"))
    assert failed_record["total_tokens"] == 133
    assert failed_record["cost_usd"] == 0.04
    assert failed_record["usage_reported"] is True

    # Also test invalid JSON/schema output -> repair attempt -> router fallback
    router2 = ModelRouter(cfg)
    harness_calls = 0

    async def invalid_json(self, **kwargs):
        nonlocal harness_calls
        harness_calls += 1
        return HarnessInvocationResult(
            text="not json at all",
            used_model="claude-opus-5-5",
            harness="claude_code",
            auth_mode="subscription",
            cost_source="subscription",
            usage_reported=True,
        )

    monkeypatch.setattr(ClaudeCodeAdapter, "run_structured", invalid_json)
    review2 = await build_mod.supervise_app(router2, spec, app_dir, vround)
    assert harness_calls == 2  # initial + 1 repair_attempt
    assert review2.approved is True


@pytest.mark.asyncio
async def test_run_doctor_mixed_auth_provider_keys_and_offload_flag(tmp_path, monkeypatch):
    cfg = tmp_path / "doctor_mixed.yaml"
    cfg.write_text(
        """
roles:
  parser: [anthropic/claude-sonnet-4]
  builder: [local-auto]
  fixer: [openrouter/m]
backends:
  parser:
    type: litellm
    model: anthropic/claude-sonnet-4
  judge:
    type: harness
    provider: claude_code
    auth: subscription
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    offload:
      - name: sonnet-api
        type: harness
        provider: claude_code
        auth: api
  fixer:
    type: harness
    provider: antigravity
limits:
  max_parallel_calls: 2
  max_tokens: 1000
  timeout_seconds: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )
    from atlas.harnesses.base import AuthStatus

    seen_auths: list[str | None] = []

    async def fake_claude_auth(self, *, auth=None):
        seen_auths.append(auth)
        return AuthStatus(
            provider="claude_code",
            available=True,
            authenticated=True,
            auth_mode="api_key" if auth == "api" else (auth or "subscription"),
        )

    async def fake_agy_auth(self, *, auth=None):
        return AuthStatus(provider="antigravity", available=True, authenticated=True)

    async def fake_sdk_auth(self, *, auth=None):
        return AuthStatus(provider="antigravity_sdk", available=True, authenticated=True)

    async def fake_local(**kwargs):
        return LocalResolutionResult(reason="stopped")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(ClaudeCodeAdapter, "check_auth", fake_claude_auth)
    monkeypatch.setattr(AntigravityCLIAdapter, "check_auth", fake_agy_auth)
    monkeypatch.setattr("atlas.harnesses.AntigravitySDKAdapter.check_auth", fake_sdk_auth)
    monkeypatch.setattr("atlas.harnesses.resolve_local_model", fake_local)

    report = await run_doctor(ModelRouter(cfg))
    assert set(seen_auths) == {"api", "subscription"}
    assert {
        c["auth_mode"] for c in report["checks"] if c["provider"] == "claude_code"
    } == {"subscription", "api_key"}
    by_provider = {c["provider"]: c for c in report["checks"]}
    assert "anthropic" in by_provider and by_provider["anthropic"]["authenticated"] is True
    assert "openrouter" not in by_provider
    assert by_provider["local_model_daemon"]["offload_configured"] is True


@pytest.mark.asyncio
async def test_run_doctor_uses_router_model_for_harness_fallback_credentials(tmp_path, monkeypatch):
    cfg = tmp_path / "doctor_fallback.yaml"
    cfg.write_text(
        """
roles:
  parser: [gemini-4-hydrogen-high]
backends:
  parser:
    type: harness
    provider: antigravity
    model: gemini-4-hydrogen-high
    fallback: router
    router_model: openai/gpt-4.1
limits:
  max_parallel_calls: 1
  max_tokens: 1000
  timeout_seconds: 10
""".strip()
        + "\n",
        encoding="utf-8",
    )
    from atlas.harnesses.base import AuthStatus

    async def fake_agy_auth(self, *, auth=None):
        return AuthStatus(provider="antigravity", available=True, authenticated=True)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(AntigravityCLIAdapter, "check_auth", fake_agy_auth)

    report = await run_doctor(ModelRouter(cfg))
    by_provider = {check["provider"]: check for check in report["checks"]}
    assert by_provider["openai"]["authenticated"] is True
    assert "openrouter" not in by_provider
