"""Offline tests for researcher clarifying questions and black-box E2E product tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.harnesses.base import HarnessInvocationResult
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.pipeline.clarify import (
    apply_clarifications_to_spec,
    clarify_with_user,
    collect_clarifying_questions,
    load_answers_file,
)
from atlas.pipeline.render import render_run_report
from atlas.pipeline.verify import check_e2e_tests
from atlas.router import ModelRouter
from atlas.runner import run_build, run_plan
from atlas.schemas import (
    ClarificationReport,
    E2ETestSuite,
    GeneratedFile,
    ParsedSpec,
    Requirement,
    ResearchNote,
    ResearchReport,
    TokenUsageSummary,
    WorkspaceBuildResult,
)


def test_clarify_collection_answering_and_spec_application(tmp_path):
    spec = ParsedSpec(
        summary="CLI greeter",
        requirements=[
            Requirement(id="R001", title="Greet CLI", description="Greet user from CLI args"),
        ],
        open_questions=["Should --upper uppercase the entire output?"],
    )
    report = ResearchReport(
        notes=[
            ResearchNote(
                requirement_id="R001",
                model="m-res",
                clarifying_questions=[
                    "What prefix should the greeting use?",
                    "should --upper uppercase the entire output?",  # duplicates open_question
                ],
                e2e_scenarios=["Run python -m greeter Ada -> exit 0 and stdout 'HELLO: Ada'"],
            )
        ]
    )

    items = collect_clarifying_questions(spec, report)
    assert len(items) == 2
    assert items[0].source == "spec"
    assert items[1].source == "research"
    assert items[1].requirement_id == "R001"

    answers_path = tmp_path / "answers.json"
    answers_path.write_text(
        json.dumps({"1": "Yes, uppercase the full output line."}),
        encoding="utf-8",
    )
    prefilled = load_answers_file(answers_path)

    asked_prompts: list[str] = []

    def fake_ask(prompt_text: str) -> str:
        asked_prompts.append(prompt_text)
        return "Prefix every greeting with 'HELLO:'"

    clar = clarify_with_user(items, ask_fn=fake_ask, prefilled_answers=prefilled)
    assert clar.interactive is True
    assert len(asked_prompts) == 1
    assert "(R001)" in asked_prompts[0]
    assert clar.items[0].answer == "Yes, uppercase the full output line."
    assert clar.items[1].answer == "Prefix every greeting with 'HELLO:'"

    updated = apply_clarifications_to_spec(spec, clar)
    assert updated.open_questions == []
    assert any("HELLO:" in a for a in updated.assumptions)
    assert any("uppercase the full output line" in a for a in updated.assumptions)

    # Non-interactive with no ask_fn and no answers never blocks.
    silent = clarify_with_user(items, interactive=False)
    assert silent.interactive is False
    assert all(item.answer == "" for item in silent.items)

    bad_answers = tmp_path / "bad.json"
    bad_answers.write_text('["not a dict"]', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_answers_file(bad_answers)


def test_e2e_suite_schema_and_integrity_check(tmp_path):
    with pytest.raises(ValidationError, match="only create test files"):
        E2ETestSuite(
            summary="Invalid non-test file",
            product_interface="python -m demo",
            stack="python",
            files=[GeneratedFile(path="demo/main.py", content="print('nope')\n")],
        )

    suite = E2ETestSuite(
        summary="CLI E2E suite",
        product_interface="python -m greeter <name>",
        stack="python",
        files=[
            GeneratedFile(
                path="tests/test_e2e_product.py",
                content="def test_cli():\n    assert 1 == 1\n",
            )
        ],
    )
    assert suite.files[0].path == "tests/test_e2e_product.py"

    # Missing file fails integrity gate.
    missing = check_e2e_tests(tmp_path, ["tests/test_e2e_product.py"])
    assert missing.passed is False
    assert "Missing required E2E product test file" in missing.stderr

    # Skipped test fails integrity gate.
    test_file = tmp_path / "tests" / "test_e2e_product.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "import pytest\n\n@pytest.mark.skip\ndef test_cli():\n    assert True\n",
        encoding="utf-8",
    )
    skipped = check_e2e_tests(tmp_path, ["tests/test_e2e_product.py"])
    assert skipped.passed is False
    assert "unconditional skip marker" in skipped.stderr

    # Valid runnable test passes integrity gate.
    test_file.write_text("def test_cli():\n    assert 2 + 2 == 4\n", encoding="utf-8")
    ok = check_e2e_tests(tmp_path, ["tests/test_e2e_product.py"])
    assert ok.passed is True


class _ClarifyE2ERouter(ModelRouter):
    def __init__(self, responses: dict[str, list[str]]):
        self.path = Path("fake")
        self.cfg = {
            "roles": {
                "parser": ["m-parse"],
                "researcher": ["m-res"],
                "judge": ["m-judge"],
                "builder": ["m-build"],
                "fixer": ["m-fix"],
            },
            "role_fallbacks": {},
            "fallbacks": [],
            "limits": {
                "max_parallel_calls": 4,
                "max_tokens": 4000,
                "timeout_seconds": 30,
                "attempts_per_model": 1,
                "repair_attempts": 0,
                "max_fix_rounds": 2,
                "verify_timeout_seconds": 60,
            },
            "verification": {
                "require_requirement_tests": True,
                "require_product_contract": True,
                "require_e2e_tests": True,
            },
        }
        self._semaphores = {}
        self.call_log = []
        self._responses = {k: list(v) for k, v in responses.items()}
        self.seen_prompts: list[tuple[str, str]] = []

    async def complete(self, model: str, system: str, user: str, role: str | None = None):
        rkey = role or ""
        if rkey == "researcher" and "black-box end-to-end" in system.lower():
            rkey = "e2e"
        self.seen_prompts.append((rkey, user))
        payload = self._responses[rkey].pop(0)
        self.call_log.append(
            {
                "role": role,
                "requested_model": model,
                "used_model": model,
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "cost_usd": 0.001,
            }
        )
        return model, payload


async def test_research_clarify_and_real_subprocess_e2e_fixes_cli(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Greeter CLI\nBuild python -m greeter <name>.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "CLI greeter that outputs a formatted greeting.",
            "requirements": [
                {
                    "id": "R001",
                    "title": "CLI greeting command",
                    "description": "Accept a name argument on CLI and print formatted greeting.",
                    "kind": "functional",
                    "acceptance_criteria": ["python -m greeter Ada exits 0 and prints greeting"],
                }
            ],
        }
    )
    research_json = json.dumps(
        {
            "candidates": [{"name": "stdlib-argparse"}, {"name": "click"}],
            "findings": ["argparse handles positional CLI args with zero deps"],
            "clarifying_questions": ["What exact greeting prefix should python -m greeter print?"],
            "e2e_scenarios": ["Run python -m greeter Ada via subprocess -> stdout 'HELLO: Ada'"],
        }
    )
    judge_json = json.dumps(
        {
            "decisions": [
                {
                    "requirement_id": "R001",
                    "options": [{"name": "stdlib-argparse"}, {"name": "click"}],
                    "pick": "stdlib-argparse",
                    "rationale": "Zero dependencies needed.",
                }
            ],
            "risks": [],
            "tasks": ["Implement greeter/__main__.py with argparse"],
        }
    )
    e2e_code = (
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n\n"
        "def test_cli_greets_end_to_end():\n"
        "    # R001\n"
        "    root = Path(__file__).resolve().parents[1]\n"
        "    proc = subprocess.run(\n"
        '        [sys.executable, "-m", "greeter", "Ada"],\n'
        "        cwd=root,\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "    )\n"
        "    assert proc.returncode == 0\n"
        '    assert proc.stdout.strip() == "HELLO: Ada"\n'
    )
    e2e_json = json.dumps(
        {
            "summary": "Subprocess E2E check for python -m greeter.",
            "product_interface": "python -m greeter <name> -> stdout 'HELLO: <name>'",
            "stack": "python",
            "scenarios": ["Run python -m greeter Ada and assert exact HELLO prefix"],
            "files": [
                {
                    "path": "tests/test_e2e_product.py",
                    "content": e2e_code,
                    "purpose": "Black-box subprocess E2E test",
                }
            ],
        }
    )
    # Builder writes a buggy prefix ("Hi") instead of clarified "HELLO:".
    buggy_app_json = json.dumps(
        {
            "summary": "Greeter CLI package",
            "stack": "python",
            "files": [
                {"path": "README.md", "content": "# Greeter\nRun `python -m greeter Ada`.\n"},
                {"path": "requirements.txt", "content": "pytest\n"},
                {"path": "greeter/__init__.py", "content": ""},
                {
                    "path": "greeter/__main__.py",
                    "content": (
                        "import sys\n\n\n"
                        "def main() -> None:\n"
                        '    name = sys.argv[1] if len(sys.argv) > 1 else "World"\n'
                        '    print(f"Hi {name}")\n\n\n'
                        'if __name__ == "__main__":\n'
                        "    main()\n"
                    ),
                },
            ],
        }
    )
    fix_json = json.dumps(
        {
            "diagnosis": "E2E subprocess test expects 'HELLO: Ada', updated greeting prefix.",
            "files": [
                {
                    "path": "greeter/__main__.py",
                    "content": (
                        "import sys\n\n\n"
                        "def main() -> None:\n"
                        '    name = sys.argv[1] if len(sys.argv) > 1 else "World"\n'
                        '    print(f"HELLO: {name}")\n\n\n'
                        'if __name__ == "__main__":\n'
                        "    main()\n"
                    ),
                }
            ],
            "delete_files": ["tests/test_e2e_product.py"],  # must be ignored by runner!
        }
    )

    router = _ClarifyE2ERouter(
        {
            "parser": [parsed_json],
            "researcher": [research_json],
            "judge": [judge_json],
            "e2e": [e2e_json],
            "builder": [buggy_app_json],
            "fixer": [fix_json],
        }
    )

    def ask_user(question: str) -> str:
        assert "greeting prefix" in question
        return "HELLO:"

    app_dir = tmp_path / "greeter_app"
    out_dir = await run_build(
        spec_file,
        tmp_path / "runs",
        app_dir,
        router=router,
        ask_fn=ask_user,
    )

    # Judge and Builder both received the clarified assumption and E2E contract.
    judge_prompt = next(u for r, u in router.seen_prompts if r == "judge")
    assert "HELLO:" in judge_prompt
    build_prompt = next(u for r, u in router.seen_prompts if r == "builder")
    assert "END-TO-END PRODUCT TEST CONTRACT" in build_prompt
    assert "python -m greeter <name>" in build_prompt

    # Round 1 failed real subprocess pytest on Hi vs HELLO; Fixer patched; Round 2 passed!
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["clarifications_count"] == 1
    assert manifest["clarifications_answered"] == 1
    assert manifest["stages"]["clarify"] == "completed"
    assert manifest["stages"]["e2e_tests"] == "completed"
    assert manifest["e2e_test_files"] == ["tests/test_e2e_product.py"]
    assert manifest["verification_rounds"] == 2
    assert (out_dir / "clarifications.json").is_file()
    assert (out_dir / "e2e_tests.json").is_file()
    assert (app_dir / "tests" / "test_e2e_product.py").is_file()

    clar_obj = ClarificationReport.model_validate_json(
        (out_dir / "clarifications.json").read_text(encoding="utf-8")
    )
    assert clar_obj.items[0].answer == "HELLO:"

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Clarifications" in report_md
    assert "HELLO:" in report_md
    assert "## End-to-end product tests" in report_md
    assert "tests/test_e2e_product.py" in report_md

    # Verify plan workflow also records clarifications + E2E tests into plan.md.
    plan_router = _ClarifyE2ERouter(
        {
            "parser": [parsed_json],
            "researcher": [research_json],
            "judge": [judge_json],
            "e2e": [e2e_json],
        }
    )
    plan_dir = await run_plan(
        spec_file,
        tmp_path / "plan_runs",
        router=plan_router,
        clarification_answers={"1": "HELLO:"},
        generate_e2e=True,
    )
    assert (plan_dir / "clarifications.json").is_file()
    assert (plan_dir / "e2e_tests.json").is_file()
    plan_md = (plan_dir / "plan.md").read_text(encoding="utf-8")
    assert "## Clarifications" in plan_md
    assert "## End-to-end product tests" in plan_md


def test_cli_clarify_and_e2e_flags_in_help():
    for subcmd in ("plan", "build", "run"):
        proc = subprocess.run(
            [sys.executable, "-m", "atlas.cli", subcmd, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0
        assert "--clarify" in proc.stdout
        assert "--answers" in proc.stdout
        assert "--e2e-tests" in proc.stdout


async def test_workspace_preseeding_progressive_lean_prompt_and_tamper_restoration(
    tmp_path, monkeypatch
):
    skills_dir = tmp_path / "skills"
    (skills_dir / "writing-clean-code").mkdir(parents=True)
    (skills_dir / "writing-clean-code" / "SKILL.md").write_text(
        "---\nname: writing-clean-code\ndescription: Clean code rules\n---\n\n"
        "UNIQUE_BODY_MARKER_NEVER_IN_PROMPT\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "prog.yaml"
    cfg_path.write_text(
        f"""
roles:
  parser: [m-parse]
  researcher: [m-res]
  judge: [m-judge]
  builder: [local-auto]
  fixer: [m-fix]
backends:
  builder:
    type: local_harness
    local_harness: antigravity_sdk
    local_model: gpt-oss:20b
skills:
  sources: ["{skills_dir}"]
  discovery_mode: progressive
  allowlist: [writing-clean-code]
  by_role:
    builder: [writing-clean-code]
limits:
  max_parallel_calls: 2
  max_tokens: 4000
  timeout_seconds: 30
  attempts_per_model: 1
  repair_attempts: 0
  max_fix_rounds: 1
  verify_timeout_seconds: 60
verification:
  require_requirement_tests: true
  require_product_contract: true
  require_e2e_tests: true
""",
        encoding="utf-8",
    )

    spec_file = tmp_path / "spec.md"
    spec_file.write_text(
        "# Greeter CLI\nBuild python -m greeter <name> outputting HELLO: <name>.\n",
        encoding="utf-8",
    )

    parsed_json = json.dumps(
        {
            "summary": "CLI greeter",
            "requirements": [
                {
                    "id": "R001",
                    "title": "CLI greet",
                    "description": "Print HELLO: <name> from python -m greeter <name>.",
                    "kind": "functional",
                    "acceptance_criteria": ["Exit 0 with stdout 'HELLO: Ada'"],
                }
            ],
        }
    )
    research_json = json.dumps(
        {
            "candidates": [{"name": "stdlib"}, {"name": "click"}],
            "findings": ["Use stdlib sys.argv"],
            "e2e_scenarios": ["python -m greeter Ada -> 'HELLO: Ada'"],
        }
    )
    judge_json = json.dumps(
        {
            "decisions": [
                {
                    "requirement_id": "R001",
                    "options": [{"name": "stdlib"}, {"name": "click"}],
                    "pick": "stdlib",
                    "rationale": "Zero dependencies",
                }
            ],
            "tasks": ["Write greeter/__main__.py"],
        }
    )
    e2e_code = (
        "# EXPECTED_TAMPER_CHECK_TOKEN\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n\n"
        "def test_cli_greets_end_to_end():\n"
        "    # R001\n"
        "    root = Path(__file__).resolve().parents[1]\n"
        "    proc = subprocess.run(\n"
        '        [sys.executable, "-m", "greeter", "Ada"],\n'
        "        cwd=root,\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        check=False,\n"
        "    )\n"
        "    assert proc.returncode == 0\n"
        '    assert proc.stdout.strip() == "HELLO: Ada"\n'
    )
    e2e_json = json.dumps(
        {
            "summary": "Subprocess E2E check for greeter",
            "product_interface": "python -m greeter <name> -> 'HELLO: <name>'",
            "stack": "python",
            "verifiable": True,
            "scenarios": ["Run python -m greeter Ada"],
            "files": [{"path": "tests/test_e2e_product.py", "content": e2e_code}],
        }
    )

    queues = {
        "parser": [parsed_json],
        "researcher": [research_json],
        "judge": [judge_json],
        "e2e": [e2e_json],
    }
    router = ModelRouter(cfg_path)

    async def fake_complete(self, model: str, system: str, user: str, role: str | None = None):
        rkey = role or ""
        if rkey == "researcher" and "black-box end-to-end" in system.lower():
            rkey = "e2e"
        self.call_log.append(
            {
                "role": role,
                "requested_model": model,
                "used_model": model,
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "cost_usd": 0.0,
            }
        )
        return model, queues[rkey].pop(0)

    monkeypatch.setattr(ModelRouter, "complete", fake_complete)

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

    seen_prompts: list[str] = []

    async def fake_sdk_build(
        self, *, candidate, prompt_text, app_dir, timeout_seconds=600.0, skill_dirs=None
    ):
        seen_prompts.append(prompt_text)
        # The brief and the authoritative E2E test file must ALREADY exist before the local
        # builder runs; the full PLAN.md is kept away from the small model until it is done.
        assert (app_dir / "BUILD_BRIEF.md").is_file()
        assert not (app_dir / "PLAN.md").exists()
        assert (app_dir / "tests" / "test_e2e_product.py").is_file()
        assert skill_dirs == [app_dir.resolve() / ".agents" / "skills"]
        # Lean prompt: no full skill body or full E2E source inline!
        assert "UNIQUE_BODY_MARKER_NEVER_IN_PROMPT" not in prompt_text
        assert "EXPECTED_TAMPER_CHECK_TOKEN" not in prompt_text
        assert "BUILD_BRIEF.md" in prompt_text and "PLAN.md" not in prompt_text
        assert "tests/test_e2e_product.py" in prompt_text
        assert ".agents/skills/writing-clean-code/SKILL.md" in prompt_text

        (app_dir / "README.md").write_text("# Greeter\n", encoding="utf-8")
        (app_dir / "requirements.txt").write_text("pytest\n", encoding="utf-8")
        (app_dir / "greeter").mkdir(exist_ok=True)
        (app_dir / "greeter" / "__init__.py").write_text("", encoding="utf-8")
        (app_dir / "greeter" / "__main__.py").write_text(
            "import sys\n\n\n"
            "def main() -> None:\n"
            '    print(f"HELLO: {sys.argv[1]}")\n\n\n'
            'if __name__ == "__main__":\n'
            "    main()\n",
            encoding="utf-8",
        )
        # Simulate accidental tampering by local model -> Atlas must restore authoritative test!
        (app_dir / "tests" / "test_e2e_product.py").write_text(
            "def test_tampered():\n    assert True\n",
            encoding="utf-8",
        )
        return (
            WorkspaceBuildResult(
                summary="Built CLI greeter",
                harness="antigravity_sdk",
                harness_version="0.1.24",
                model="ollama/gpt-oss:20b",
                app_dir=str(app_dir),
                stack="python",
                created_files=["README.md", "requirements.txt", "greeter/__main__.py"],
                files_snapshot=[
                    "PLAN.md",
                    "README.md",
                    "greeter/__init__.py",
                    "greeter/__main__.py",
                    "requirements.txt",
                    "tests/test_e2e_product.py",
                ],
            ),
            HarnessInvocationResult(
                text="Built",
                used_model="ollama/gpt-oss:20b",
                harness="antigravity_sdk",
                harness_version="0.1.24",
                auth_mode="local",
                cost_usd=0.0,
                cost_source="local_free",
            ),
        )

    monkeypatch.setattr("atlas.pipeline.build.resolve_local_model", ready_local)
    monkeypatch.setattr(
        "atlas.pipeline.build.AntigravitySDKAdapter.run_workspace_task", fake_sdk_build
    )

    app_dir = tmp_path / "greeter_ws"
    out_dir = await run_build(
        spec_file,
        tmp_path / "runs",
        app_dir,
        router=router,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["verification_rounds"] == 1
    assert manifest["e2e_verifiable"] is True
    assert (app_dir / "PLAN.md").is_file()
    restored = (app_dir / "tests" / "test_e2e_product.py").read_text(encoding="utf-8")
    assert "EXPECTED_TAMPER_CHECK_TOKEN" in restored
    assert not (app_dir / ".agents").exists()
    assert len(seen_prompts) == 1


def test_unverifiable_e2e_contract_and_report_rendering(tmp_path):
    with pytest.raises(ValidationError, match="unverifiable_reason"):
        E2ETestSuite(
            summary="No reason",
            product_interface="ble pair",
            verifiable=False,
            files=[],
        )
    with pytest.raises(ValidationError, match="at least one test file"):
        E2ETestSuite(
            summary="Missing files",
            product_interface="python -m x",
            verifiable=True,
            files=[],
        )

    suite = E2ETestSuite(
        summary="Hardware E2E check deferred",
        product_interface="ble-cli pair <mac>",
        stack="python",
        verifiable=False,
        unverifiable_reason="Requires physical Bluetooth Low Energy hardware.",
        files=[],
    )
    chk = check_e2e_tests(
        tmp_path,
        [],
        verifiable=suite.verifiable,
        unverifiable_reason=suite.unverifiable_reason,
    )
    assert chk.passed is True
    assert "unverifiable offline" in chk.stdout

    report = render_run_report(
        run_id="r1",
        spec_path="specs/ble.md",
        status="verified",
        spec=None,
        generated_app=None,
        verification=None,
        token_summary=TokenUsageSummary(
            total_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
        ),
        written_files=[],
        e2e_suite=suite,
    )
    assert "Verifiable offline:** No" in report
    assert "Requires physical Bluetooth Low Energy hardware." in report


async def test_harness_yaml_e2e_role_limits_and_compact_e2e_prompt():
    from atlas.pipeline.e2e import generate_e2e_tests
    from atlas.schemas import Plan, ToolDecision

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "harness.yaml"
    router = ModelRouter(cfg_path)
    assert router.models_for("e2e") == ["openrouter/anthropic/claude-sonnet-5.5"]
    assert router.backend_for("e2e")["provider"] == "claude_code"
    assert router.backend_for("e2e")["model"] == "sonnet"
    assert router.backend_for("e2e")["effort"] == "high"
    assert router.backend_for("e2e")["timeout_seconds"] == 600
    assert router.max_tokens_for("e2e") == 32000
    assert router.max_tokens_for("researcher") == 24000
    assert router.timeout_for("e2e") == 300

    reqs = [
        Requirement(
            id=f"R{i:03d}",
            title=f"Requirement {i}",
            description=f"Implement feature {i} cleanly.",
            acceptance_criteria=[f"Feature {i} passes offline"],
        )
        for i in range(1, 21)
    ]
    spec = ParsedSpec(summary="20-req static house ad", requirements=reqs)
    notes = [
        ResearchNote(
            requirement_id=r.id,
            model="m-res",
            candidates=[
                {"name": "chosen-lib", "pros": ["fast"]},
                {"name": f"REJECTED_CANDIDATE_{r.id}", "pros": ["heavy"]},
            ],
            findings=[f"VERBOSE_FINDING_SHOULD_OMIT_{r.id} " * 8],
            e2e_scenarios=[f"Check scenario for {r.id}"],
        )
        for r in reqs
    ]
    decisions = [
        ToolDecision(
            requirement_id=r.id,
            options=[{"name": "chosen-lib"}, {"name": "other-lib"}],
            pick="chosen-lib",
            rationale="Lightweight and offline",
        )
        for r in reqs
    ]
    plan = Plan(spec=spec, decisions=decisions, risks=[], tasks=["Write index.html"])

    seen: list[tuple[str, str]] = []

    class _RecordingRouter(ModelRouter):
        def __init__(self) -> None:
            self.path = router.path
            self.cfg = router.cfg
            self._semaphores = {}
            self.call_log = []

        async def complete_structured(
            self,
            model: str,
            system: str,
            user: str,
            role: str | None = None,
            json_schema: dict | None = None,
            use_harness: bool = True,
        ):
            del model, system, json_schema, use_harness
            seen.append((role or "", user))
            return "gemini-4-neon-high", json.dumps(
                {
                    "summary": "Offline static HTML E2E suite",
                    "product_interface": "index.html",
                    "stack": "python",
                    "verifiable": True,
                    "scenarios": ["Check all sections"],
                    "files": [
                        {
                            "path": "tests/test_e2e_product.py",
                            "content": "def test_html():\n    assert True\n",
                        }
                    ],
                }
            )

    suite = await generate_e2e_tests(
        _RecordingRouter(),
        spec,
        ResearchReport(notes=notes),
        plan,
        raw_spec="# House Ad\nOne static HTML page.\n",
    )
    assert suite.verifiable is True
    assert len(seen) == 1
    used_role, user_prompt = seen[0]
    assert used_role == "e2e"
    assert "REJECTED_CANDIDATE_" not in user_prompt
    assert "VERBOSE_FINDING_SHOULD_OMIT_" not in user_prompt
    assert "Check scenario for R001" in user_prompt
    assert "chosen-lib" in user_prompt
    assert len(user_prompt) < 15_000


def test_e2e_sanitizer_and_sha256_integrity_guard(tmp_path):
    from atlas.pipeline.build import materialize_files
    from atlas.pipeline.e2e import sanitize_e2e_suite_files
    from atlas.pipeline.verify import e2e_file_hashes

    dirty = (
        "import re\n\n"
        "def test_case():\n"
        "    assert re.search('abc', 'ABC', flags=re.I)\n"
    )
    suite = E2ETestSuite(
        summary="Regex flag cleanup",
        product_interface="app",
        stack="python",
        verifiable=True,
        scenarios=["case"],
        files=[GeneratedFile(path="tests/test_e2e_product.py", content=dirty)],
    )
    cleaned = sanitize_e2e_suite_files(suite)
    assert "re.IGNORECASE" in cleaned.files[0].content
    assert "re.I)" not in cleaned.files[0].content

    app_dir = tmp_path / "app"
    (app_dir / "tests").mkdir(parents=True)
    target = app_dir / "tests" / "test_e2e_product.py"
    target.write_text(cleaned.files[0].content, encoding="utf-8")

    hashes = e2e_file_hashes(cleaned.files)
    ok = check_e2e_tests(app_dir, ["tests/test_e2e_product.py"], expected_hashes=hashes)
    assert ok.passed

    target.write_text("def test_case():\n    assert True\n", encoding="utf-8")
    tampered = check_e2e_tests(app_dir, ["tests/test_e2e_product.py"], expected_hashes=hashes)
    assert not tampered.passed
    assert "sha256 mismatch" in tampered.stderr

    no_newline_suite = E2ETestSuite(
        summary="No final newline",
        product_interface="app",
        stack="python",
        verifiable=True,
        scenarios=["preserve test integrity"],
        files=[
            GeneratedFile(
                path="tests/test_no_newline.py",
                content="def test_case():\n    assert True",
            )
        ],
    )
    no_newline_hashes = e2e_file_hashes(no_newline_suite.files)
    materialize_files(app_dir, no_newline_suite.files)
    intact = check_e2e_tests(
        app_dir,
        ["tests/test_no_newline.py"],
        expected_hashes=no_newline_hashes,
    )
    assert intact.passed
    missing_digest = check_e2e_tests(
        app_dir,
        ["tests/test_no_newline.py"],
        expected_hashes={},
    )
    assert not missing_digest.passed
    assert "no authoritative SHA-256" in missing_digest.stderr


@pytest.mark.asyncio
async def test_fix_loop_ignores_protected_e2e_patches_and_restores_suite(tmp_path, monkeypatch):
    from atlas import runner as runner_mod
    from atlas.schemas import CheckResult, GeneratedApp, PatchResult, Plan, ToolDecision

    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nA demo app\n", encoding="utf-8")
    spec = ParsedSpec(
        summary="Demo",
        requirements=[Requirement(id="R001", title="Do", description="Do thing")],
    )
    e2e_content = "def test_r001_product():\n    assert True\n"
    e2e_suite = E2ETestSuite(
        summary="E2E suite",
        product_interface="app",
        stack="python",
        verifiable=True,
        scenarios=["check"],
        files=[GeneratedFile(path="tests/test_e2e_product.py", content=e2e_content)],
    )
    from_run = tmp_path / "from_run"
    from_run.mkdir()
    (from_run / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (from_run / "plan.json").write_text(
        Plan(
            spec=spec,
            decisions=[
                ToolDecision(
                    requirement_id="R001",
                    options=[{"name": "stdlib"}, {"name": "alt"}],
                    pick="stdlib",
                    rationale="Standard library",
                )
            ],
            risks=[],
            tasks=["t"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (from_run / "e2e_tests.json").write_text(e2e_suite.model_dump_json(indent=2), encoding="utf-8")

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
roles:
  parser: [openrouter/m]
  researcher: [openrouter/m]
  judge: [openrouter/m]
  builder: [openrouter/m]
  fixer: [openrouter/m]
limits:
  max_parallel_calls: 2
  max_tokens: 1000
  timeout_seconds: 10
  max_fix_rounds: 1
verification:
  require_requirement_tests: false
  require_product_contract: false
  require_e2e_tests: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def fake_build(*args, **kwargs):
        return GeneratedApp(
            summary="Built",
            stack="python",
            files=[GeneratedFile(path="app.py", content="ok = False\n", purpose="app")],
        )

    async def fake_verify(app_dir: Path):
        ok = "ok = True" in (app_dir / "app.py").read_text(encoding="utf-8")
        return (
            "python",
            [CheckResult(name="pytest", command="pytest", passed=ok, exit_code=0 if ok else 1)],
        )

    seen_protected: list[list[str] | None] = []

    async def sneaky_fix(*args, protected_files=None, **kwargs):
        seen_protected.append(protected_files)
        return PatchResult(
            diagnosis="Tried editing E2E test alongside app.py",
            files=[
                GeneratedFile(path="app.py", content="ok = True\n", purpose="fix app"),
                GeneratedFile(
                    path="tests/test_e2e_product.py",
                    content="def test_weakened():\n    assert 1 == 1\n",
                    purpose="weaken test",
                ),
            ],
            delete_files=["tests/test_e2e_product.py"],
        )

    monkeypatch.setattr(runner_mod, "build_app", fake_build)
    monkeypatch.setattr(runner_mod, "fix_app", sneaky_fix)

    app_dir = tmp_path / "out_app"
    out_dir = await run_build(
        spec_file,
        tmp_path / "runs",
        app_dir,
        config_path=cfg,
        from_run=from_run,
        verify_fn=fake_verify,
        ledger_path=tmp_path / "usage" / "ledger.jsonl",
    )
    assert seen_protected == [["tests/test_e2e_product.py"]]
    assert (app_dir / "tests" / "test_e2e_product.py").read_text(encoding="utf-8") == e2e_content
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
