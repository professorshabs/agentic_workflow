"""Offline tests for real-time pipeline progress logging and CLI stream behavior."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas.harnesses.base import HarnessInvocationResult
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.pipeline.e2e import generate_e2e_tests
from atlas.router import ModelRouter
from atlas.runner import run_build, run_plan
from atlas.schemas import CheckResult, ParsedSpec, WorkspaceBuildResult

PARSED_TWO_REQS = {
    "summary": "CLI calculator.",
    "requirements": [
        {
            "id": "R001",
            "title": "Add numbers",
            "description": "Adds two numbers via CLI.",
            "kind": "functional",
            "acceptance_criteria": ["add 1 2 returns 3"],
        },
        {
            "id": "R002",
            "title": "Subtract numbers",
            "description": "Subtracts two numbers via CLI.",
            "kind": "functional",
            "acceptance_criteria": ["sub 5 2 returns 3"],
        },
    ],
    "open_questions": ["Should decimals be supported?"],
}

APP_INITIAL = {
    "summary": "Calculator",
    "stack": "python",
    "files": [
        {"path": "README.md", "content": "# Calc\n"},
        {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"},
    ],
}

E2E_SUITE = {
    "summary": "Subprocess E2E tests.",
    "product_interface": "python calc.py",
    "stack": "python",
    "scenarios": ["Run calculator commands"],
    "files": [
        {
            "path": "tests/test_e2e_product.py",
            "content": "# R001 R002\ndef test_calc_e2e():\n    assert 1 + 2 == 3\n",
            "purpose": "E2E check",
        }
    ],
}

E2E_UNVERIFIABLE = {
    "summary": "No offline browser runner.",
    "product_interface": "Static HTML",
    "stack": "other",
    "verifiable": False,
    "unverifiable_reason": "Pure static HTML",
    "scenarios": [],
    "files": [],
}

PATCH_FIX = {
    "diagnosis": "Implement subtract support in calc.py",
    "files": [
        {
            "path": "calc.py",
            "content": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
        }
    ],
    "delete_files": [],
}

SUPERVISOR_OK = {
    "approved": True,
    "summary": "All requirements satisfied cleanly.",
    "issues": [],
    "improvement_plan": [],
    "requirement_notes": {"R001": "Verified", "R002": "Verified"},
}

SUPERVISOR_CHANGES = {
    "approved": False,
    "summary": "Missing edge case handling.",
    "issues": ["Handle negative numbers"],
    "improvement_plan": ["Add negative test coverage"],
    "requirement_notes": {"R001": "Needs edge case"},
}


class _FakeRouter(ModelRouter):
    def __init__(
        self,
        *,
        single_req: bool = False,
        fail_judge: bool = False,
        reject_first_review: bool = False,
    ) -> None:
        self.path = Path("fake.yaml")
        self.cfg = {
            "roles": {
                r: [f"openrouter/fake/{r}"]
                for r in ("parser", "researcher", "judge", "e2e", "builder", "fixer", "supervisor")
            },
            "role_fallbacks": {},
            "fallbacks": [],
            "backends": {
                "supervisor": {
                    "enabled": True,
                    "mode": "unanimous",
                    "reviewers": [
                        {
                            "name": "fake-reviewer",
                            "type": "openrouter",
                            "model": "openrouter/fake/supervisor",
                        }
                    ],
                }
            },
            "limits": {
                "max_parallel_calls": 4,
                "max_tokens": 4000,
                "timeout_seconds": 30,
                "attempts_per_model": 1,
                "repair_attempts": 0,
                "max_fix_rounds": 1,
                "verify_timeout_seconds": 60,
            },
            "verification": {
                "require_requirement_tests": False,
                "require_product_contract": False,
                "require_e2e_tests": True,
            },
        }
        self._semaphores = {}
        self.call_log: list[dict[str, Any]] = []
        self._single_req = single_req
        self._fail_judge = fail_judge
        self._reject_first_review = reject_first_review
        self._supervisor_calls = 0

    async def complete_structured(
        self,
        model: str,
        system: str,
        user: str,
        role: str | None = None,
        json_schema: dict[str, Any] | None = None,
        use_harness: bool = True,
    ):
        del json_schema, use_harness
        return await self.complete(model, system, user, role=role)

    async def complete(self, model: str, system: str, user: str, role: str | None = None):
        r = role or ""
        self.call_log.append(
            {
                "role": r,
                "requested_model": model,
                "used_model": model,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cost_usd": 0.001,
                "cost_source": "reported",
            }
        )
        if r == "parser":
            if self._single_req:
                single = {**PARSED_TWO_REQS, "requirements": PARSED_TWO_REQS["requirements"][:1]}
                return model, json.dumps(single)
            return model, json.dumps(PARSED_TWO_REQS)
        if r == "researcher":
            if "black-box end-to-end" in system.lower():
                return model, json.dumps(E2E_SUITE)
            req_id = "R002" if "Requirement R002:" in user else "R001"
            return model, json.dumps(
                {
                    "requirement_id": req_id,
                    "model": model,
                    "candidates": [{"name": "stdlib", "pros": ["Built-in support"]}],
                    "findings": ["Standard Python suffices"],
                    "uncertainties": [],
                    "clarifying_questions": [],
                    "e2e_scenarios": [f"Verify {req_id}"],
                }
            )
        if r == "judge":
            if self._fail_judge:
                raise RuntimeError("synthetic judge outage")
            req_id = "R002" if '"id":"R002"' in user else "R001"
            return model, json.dumps(
                {
                    "decisions": [
                        {
                            "requirement_id": req_id,
                            "options": [
                                {"name": "stdlib", "pros": ["Simple"]},
                                {"name": "click", "pros": ["External CLI helper"]},
                            ],
                            "pick": "stdlib",
                            "rationale": "Zero external dependencies",
                            "confidence": "high",
                            "needs_verification": [],
                        }
                    ],
                    "risks": ["Edge case inputs"],
                    "tasks": [f"Implement {req_id}"],
                }
            )
        if r == "e2e":
            return model, json.dumps(E2E_SUITE)
        if r == "builder":
            return model, json.dumps(APP_INITIAL)
        if r == "fixer":
            return model, json.dumps(PATCH_FIX)
        if r == "supervisor":
            self._supervisor_calls += 1
            if self._reject_first_review and self._supervisor_calls == 1:
                return model, json.dumps(SUPERVISOR_CHANGES)
            return model, json.dumps(SUPERVISOR_OK)
        raise AssertionError(f"Unexpected role {r}")


async def test_run_build_emits_step_by_step_progress_logs(tmp_path, caplog):
    spec = tmp_path / "calc.md"
    spec.write_text("# Calculator\nBuild add and subtract.\n", encoding="utf-8")
    router = _FakeRouter()
    verify_calls = 0

    async def fake_verify(_app_dir: Path) -> tuple[str, list[CheckResult]]:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            return "python", [
                CheckResult(
                    name="pytest",
                    command="pytest -q",
                    passed=False,
                    exit_code=1,
                    duration_ms=420.0,
                    stderr="sub missing",
                )
            ]
        return "python", [
            CheckResult(
                name="pytest",
                command="pytest -q",
                passed=True,
                exit_code=0,
                duration_ms=310.0,
            )
        ]

    with caplog.at_level(logging.INFO, logger="atlas"):
        out = await run_build(
            spec,
            tmp_path / "runs",
            tmp_path / "app",
            router=router,
            verify_fn=fake_verify,
            clarification_answers={"1": "Integers only"},
            generate_e2e=True,
        )

    assert (out / "manifest.json").is_file()
    text = caplog.text
    for expected in (
        "Atlas build started:",
        "[parse] Parsing specification",
        "[parse] Done in ",
        "[research] Researching 2 requirement(s)",
        "[research] 1/2 done:",
        "[research] 2/2 done:",
        "[clarify] 1 clarifying question(s) (prefilled)",
        "[clarify] Done — 1/1 answered",
        "[judge] Judging architecture decisions for 2 requirement(s)",
        "[judge] 1/2 done:",
        "[judge] 2/2 done:",
        "[e2e] Generating authoritative E2E product tests",
        "[e2e] Calling role=e2e target=openrouter/fake/e2e",
        "[build] Building product in ",
        "[build] Done in ",
        "[verify] Round 1: running verification checks",
        "[verify] Round 1 FAILED",
        "[verify]   [FAIL] pytest",
        "[fix] Round 1/1: fixing [pytest]",
        "[fix] Calling model openrouter/fake/fixer",
        "[fix] Round 1/1 done in ",
        "[verify] Round 2 PASSED",
        "[verify]   [PASS] pytest",
        "[supervisor] Running acceptance review (fake-reviewer)",
        "[supervisor] Reviewer 'fake-reviewer' (openrouter/openrouter/fake/supervisor): APPROVED",
        "[done] Build verified",
    ):
        assert expected in text, f"Missing expected log fragment: {expected!r}\n\n{text}"


async def test_run_plan_emits_progress_and_failure_logs(tmp_path, caplog):
    spec = tmp_path / "one.md"
    spec.write_text("# One\nSingle requirement.\n", encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="atlas"):
        out = await run_plan(
            spec,
            tmp_path / "runs",
            router=_FakeRouter(single_req=True),
            clarification_answers={"1": "Yes"},
            generate_e2e=False,
        )

    assert (out / "plan.md").is_file()
    for expected in (
        "Atlas plan started:",
        "[parse] Done in ",
        "[research] 1/1 done: R001",
        "[judge] 1/1 done: R001",
        "[plan] Rendered ",
        "[done] Plan ready in ",
    ):
        assert expected in caplog.text

    caplog.clear()
    with (
        caplog.at_level(logging.INFO, logger="atlas"),
        pytest.raises(RuntimeError, match="Judge failed"),
    ):
        await run_plan(
            spec,
            tmp_path / "runs2",
            router=_FakeRouter(single_req=True, fail_judge=True),
            generate_e2e=False,
        )

    assert "[judge] 1/1 FAILED: R001" in caplog.text
    assert "[failed] Plan " in caplog.text


async def test_harness_target_labels_and_tty_prefilled_clarifications(
    tmp_path, monkeypatch, caplog
):
    router = _FakeRouter(single_req=True)
    router.cfg["roles"].pop("e2e", None)
    router.cfg["backends"]["researcher"] = {
        "type": "harness",
        "provider": "antigravity",
        "model": "gemini-4-neon-high",
    }
    parsed_spec = ParsedSpec.model_validate(
        {
            **PARSED_TWO_REQS,
            "open_questions": ["First question?", "Second question?"],
        }
    )

    with caplog.at_level(logging.INFO, logger="atlas"):
        await generate_e2e_tests(router, parsed_spec)

    assert "[e2e] Calling role=researcher target=antigravity/gemini-4-neon-high" in caplog.text

    spec = tmp_path / "clarify.md"
    spec.write_text("# App\nTwo questions.\n", encoding="utf-8")
    router_two = _FakeRouter(single_req=True)
    router_two.cfg["roles"]["parser"] = ["openrouter/fake/parser"]

    async def fake_parse_two(model, system, user, role=None):
        if role == "parser":
            return model, json.dumps(
                {**PARSED_TWO_REQS, "requirements": PARSED_TWO_REQS["requirements"][:1],
                 "open_questions": ["Dark or light?", "Single or multi-page?"]}
            )
        return await _FakeRouter.complete(router_two, model, system, user, role=role)

    monkeypatch.setattr(router_two, "complete", fake_parse_two)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="atlas"):
        await run_plan(
            spec,
            tmp_path / "runs_prefilled",
            router=router_two,
            interactive_clarify=True,
            clarification_answers={"1": "Dark", "2": "Single"},
            generate_e2e=False,
        )
    assert "[clarify] 2 clarifying question(s) (prefilled)" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="atlas"):
        await run_plan(
            spec,
            tmp_path / "runs_partial",
            router=router_two,
            interactive_clarify=True,
            ask_fn=lambda _q: "Single",
            clarification_answers={"1": "Dark"},
            generate_e2e=False,
        )
    assert "[clarify] 2 clarifying question(s) (prefilled + interactive)" in caplog.text


async def test_workspace_builder_skip_research_unverifiable_e2e_and_failed_build(
    tmp_path, monkeypatch, caplog
):
    from atlas.pipeline import build as build_mod

    spec = tmp_path / "site.md"
    spec.write_text("# Site\nStatic HTML.\n", encoding="utf-8")
    router = _FakeRouter(single_req=True, reject_first_review=True)
    router.cfg["backends"]["builder"] = {
        "type": "local_harness",
        "local_harness": "antigravity_sdk",
        "preferred": ["gemma-4"],
    }

    cand = LocalModelCandidate(
        provider="lm_studio",
        name="gemma-4",
        base_url="http://localhost:1234/v1",
        loaded=True,
        tool_smoke_passed=True,
    )

    async def fake_resolve(**_kwargs):
        return LocalResolutionResult(selected=cand, candidates=[cand], reason="ok")

    async def fake_ws(self, *, candidate, prompt_text, app_dir, timeout_seconds, skill_dirs=None):
        del self, prompt_text, timeout_seconds, skill_dirs
        (app_dir / "README.md").write_text("# Site\n", encoding="utf-8")
        res = WorkspaceBuildResult(
            summary="Built static site",
            harness="antigravity_sdk",
            model=f"{candidate.provider}/{candidate.name}",
            app_dir=str(app_dir),
            stack="other",
            created_files=["README.md"],
            files_snapshot=["README.md"],
        )
        inv = HarnessInvocationResult(
            text="done",
            harness="antigravity_sdk",
            used_model=f"{candidate.provider}/{candidate.name}",
            auth_mode="local",
            elapsed_ms=25.0,
            cost_usd=0.0,
            cost_source="local_free",
        )
        return res, inv

    async def fake_e2e_complete(model, system, user, role=None):
        if role == "e2e":
            return model, json.dumps(E2E_UNVERIFIABLE)
        return await _FakeRouter.complete(router, model, system, user, role=role)

    async def fake_verify(_app_dir: Path) -> tuple[str, list[CheckResult]]:
        return "python", [
            CheckResult(
                name="pytest",
                command="pytest -q",
                passed=True,
                exit_code=0,
                duration_ms=10.0,
            )
        ]

    monkeypatch.setattr(build_mod, "resolve_local_model", fake_resolve)
    monkeypatch.setattr(build_mod.AntigravitySDKAdapter, "run_workspace_task", fake_ws)
    monkeypatch.setattr(router, "complete", fake_e2e_complete)

    with caplog.at_level(logging.INFO, logger="atlas"):
        await run_build(
            spec,
            tmp_path / "runs",
            tmp_path / "app",
            router=router,
            skip_research=True,
            verify_fn=fake_verify,
            clarification_answers={"1": "Static"},
            generate_e2e=True,
        )

    text = caplog.text
    for expected in (
        "[research] Skipped (--skip-research)",
        "[judge] Skipped (--skip-research)",
        "[e2e] Done in ",
        "marked unverifiable (Pure static HTML)",
        "[build] Prepared workspace ",
        "[build] Resolving local model (explicit=auto)...",
        "[build] Selected local model lm_studio/gemma-4 — starting Antigravity SDK agent...",
        "model=lm_studio/gemma-4",
        "CHANGES REQUESTED",
        "[fix] Round 1/1: fixing [supervisor-review]",
        "[done] Build verified",
    ):
        assert expected in text, f"Missing expected log fragment: {expected!r}\n\n{text}"

    caplog.clear()
    with (
        caplog.at_level(logging.INFO, logger="atlas"),
        pytest.raises(RuntimeError, match="All research calls failed"),
    ):
        fail_router = _FakeRouter(single_req=True)

        async def fail_research(model, system, user, role=None):
            if role == "researcher":
                return model, "not json"
            return await _FakeRouter.complete(fail_router, model, system, user, role=role)

        monkeypatch.setattr(fail_router, "complete", fail_research)
        await run_build(
            spec,
            tmp_path / "runs_fail",
            tmp_path / "app_fail",
            router=fail_router,
            verify_fn=fake_verify,
            generate_e2e=False,
        )

    assert "[research] 1/1 FAILED: R001" in caplog.text
    assert "[failed] Build " in caplog.text


def test_cli_subprocess_default_quiet_and_verbose_streams(tmp_path):
    snippet = """
import json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
import atlas.cli as cli
from atlas.router import ModelRouter

class QuickRouter(ModelRouter):
    def __init__(self, path=None):
        self.path = Path("fake.yaml")
        self.cfg = {
            "roles": {r: [f"m/{r}"] for r in ("parser", "researcher", "judge", "builder", "fixer")},
            "role_fallbacks": {},
            "fallbacks": [],
            "limits": {
                "max_parallel_calls": 2,
                "max_tokens": 4000,
                "timeout_seconds": 10,
                "attempts_per_model": 1,
                "repair_attempts": 0,
                "max_fix_rounds": 0,
                "verify_timeout_seconds": 10,
            },
            "verification": {
                "require_requirement_tests": False,
                "require_product_contract": False,
                "require_e2e_tests": False,
            },
        }
        self._semaphores = {}
        self.call_log = []

    async def complete(self, model, system, user, role=None):
        self.call_log.append(
            {
                "role": role,
                "requested_model": model,
                "used_model": model,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        )
        if role == "parser":
            return model, json.dumps(
                {
                    "summary": "S",
                    "requirements": [
                        {
                            "id": "R001",
                            "title": "T",
                            "description": "D",
                            "acceptance_criteria": ["A"],
                        }
                    ],
                }
            )
        if role == "researcher":
            return model, json.dumps(
                {
                    "requirement_id": "R001",
                    "model": model,
                    "candidates": [{"name": "a"}],
                    "findings": ["f"],
                }
            )
        if role == "judge":
            return model, json.dumps(
                {
                    "decisions": [
                        {
                            "requirement_id": "R001",
                            "options": [{"name": "a"}, {"name": "b"}],
                            "pick": "a",
                            "rationale": "r",
                        }
                    ],
                    "risks": [],
                    "tasks": ["t"],
                }
            )
        raise AssertionError(role)

extra = sys.argv[1:]
spec = Path(sys.argv[1])
out_root = Path(sys.argv[2])
flags = sys.argv[3:]
with (
    patch("atlas.cli.load_dotenv", return_value=False),
    patch("atlas.runner.ModelRouter", QuickRouter),
):
    sys.argv = [
        "atlas",
        "plan",
        str(spec),
        "--out",
        str(out_root),
        "--no-ledger",
        "--no-clarify",
        "--no-e2e-tests",
        *flags,
    ]
    cli.main()
"""
    spec = tmp_path / "sub.md"
    spec.write_text("# Sub\nCheck CLI streams.\n", encoding="utf-8")
    env = {
        **os.environ,
        "LITELLM_MODE": "PRODUCTION",
        # The fake router never needs remote pricing data. Keep LiteLLM's
        # startup cost-map refresh from polluting the CLI stderr assertions
        # when tests run without network access.
        "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    }

    default_proc = subprocess.run(
        [sys.executable, "-c", snippet, str(spec), str(tmp_path / "r_default")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert default_proc.returncode == 0, default_proc.stderr
    assert "Plan written to" in default_proc.stdout
    assert "Tokens: 45 total" in default_proc.stdout
    assert re.search(r"^\[\d{2}:\d{2}:\d{2}\] Atlas plan started:", default_proc.stderr, re.M)
    assert "[done] Plan ready" in default_proc.stderr

    quiet_proc = subprocess.run(
        [sys.executable, "-c", snippet, str(spec), str(tmp_path / "r_quiet"), "--quiet"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert quiet_proc.returncode == 0, quiet_proc.stderr
    assert "Plan written to" in quiet_proc.stdout
    assert "Tokens: 45 total" in quiet_proc.stdout
    assert quiet_proc.stderr.strip() == ""

    verbose_proc = subprocess.run(
        [sys.executable, "-c", snippet, str(spec), str(tmp_path / "r_verbose"), "--verbose"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert verbose_proc.returncode == 0, verbose_proc.stderr
    stdout_lines = [line for line in verbose_proc.stdout.strip().splitlines() if line.strip()]
    assert len(stdout_lines) == 2, verbose_proc.stdout
    assert stdout_lines[0].startswith("Plan written to ")
    assert stdout_lines[1].startswith("Tokens: 45 total")
    assert "LiteLLM" not in verbose_proc.stdout
    assert "INFO atlas.runner: Atlas plan started:" in verbose_proc.stderr
