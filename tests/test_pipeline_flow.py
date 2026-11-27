import json
from pathlib import Path

import pytest

from atlas.pipeline.judge import judge
from atlas.pipeline.parse import parse_spec
from atlas.pipeline.research import research
from atlas.router import ModelRouter
from atlas.runner import record_approval, run_build, run_plan
from atlas.schemas import (
    CandidateTool,
    ParsedSpec,
    Requirement,
    ResearchFailure,
    ResearchNote,
    ResearchReport,
)


class FakeRouter(ModelRouter):
    def __init__(self, responses: dict[tuple[str, str], list[str | Exception]]):
        self.path = Path("fake")
        self.cfg = {
            "roles": {
                "parser": ["m-parse"],
                "researcher": ["m-res-1", "m-res-2"],
                "judge": ["m-judge"],
            },
            "role_fallbacks": {},
            "fallbacks": [],
            "limits": {
                "max_parallel_calls": 4,
                "max_tokens": 1000,
                "timeout_seconds": 10,
                "attempts_per_model": 1,
                "repair_attempts": 1,
            },
        }
        self._semaphores = {}
        self.call_log = []
        self._responses = {k: list(v) for k, v in responses.items()}

    async def complete(self, model: str, system: str, user: str, role: str | None = None):
        queue = self._responses[(role or "", model)]
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        self.call_log.append(
            {
                "role": role,
                "requested_model": model,
                "used_model": model,
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "cost_usd": 0.001,
            }
        )
        return model, item


async def test_research_retains_partial_success_and_records_failure():
    spec = ParsedSpec(
        summary="Test",
        requirements=[Requirement(id="R001", title="T", description="D")],
    )
    router = FakeRouter(
        {
            ("researcher", "m-res-1"): [
                json.dumps(
                    {"candidates": [{"name": "dbt"}], "findings": ["ok"], "uncertainties": []}
                )
            ],
            ("researcher", "m-res-2"): [RuntimeError("provider down")],
        }
    )
    report = await research(router, spec)
    assert len(report.notes) == 1
    assert len(report.failures) == 1
    assert report.failures[0].requirement_id == "R001"
    assert report.failures[0].model == "m-res-2"


async def test_run_plan_and_approval_flow(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nBuild a pipeline.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "Demo pipeline",
            "requirements": [
                {"id": "R001", "title": "Transform", "description": "Run SQL transforms"}
            ],
        }
    )
    res_json = json.dumps({"candidates": [{"name": "dbt"}, {"name": "SQLMesh"}]})
    judge_json = json.dumps(
        {
            "decisions": [
                {
                    "requirement_id": "R001",
                    "options": [{"name": "dbt"}, {"name": "SQLMesh"}],
                    "pick": "dbt",
                    "rationale": "Fits stack",
                    "confidence": "high",
                }
            ],
            "risks": ["Schema drift"],
            "tasks": ["Init dbt project"],
        }
    )

    router = FakeRouter(
        {
            ("parser", "m-parse"): [parsed_json],
            ("researcher", "m-res-1"): [res_json],
            ("researcher", "m-res-2"): [RuntimeError("timeout")],
            ("judge", "m-judge"): [judge_json],
        }
    )

    out_dir = await run_plan(spec_file, tmp_path / "runs", router=router)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "plan_ready"
    assert manifest["research_notes_count"] == 1
    assert manifest["research_failures_count"] == 1
    assert len(manifest["plan_sha256"]) == 64
    assert (out_dir / "spec.md").is_file()
    assert (out_dir / "plan.md").is_file()

    # Wrong hash must be rejected
    with pytest.raises(ValueError, match="plan_sha256"):
        record_approval(out_dir, plan_sha256="badhash", approved_by="owner")

    # Exact hash succeeds
    approval = record_approval(
        out_dir,
        plan_sha256=manifest["plan_sha256"],
        approved_by="owner",
        source="telegram",
    )
    assert approval.run_id == manifest["run_id"]
    updated = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert updated["status"] == "approved"


async def test_parse_research_all_fail_judge_variants_and_run_plan_errors(tmp_path):
    spec_obj = await parse_spec(
        FakeRouter(
            {
                ("parser", "m-parse"): [
                    json.dumps(
                        {
                            "summary": "S",
                            "requirements": [{"id": "R001", "title": "T", "description": "D"}],
                        }
                    )
                ]
            }
        ),
        "# S\nDo D.",
    )
    assert spec_obj.requirements[0].id == "R001"

    with pytest.raises(RuntimeError, match="All research calls failed"):
        await research(
            FakeRouter(
                {
                    ("researcher", "m-res-1"): [RuntimeError("err1")],
                    ("researcher", "m-res-2"): [RuntimeError("err2")],
                }
            ),
            spec_obj,
        )

    judge_router = FakeRouter(
        {
            ("judge", "m-judge"): [
                json.dumps(
                    {
                        "decisions": [
                            {
                                "requirement_id": "R001",
                                "options": [{"name": "A"}, {"name": "B"}],
                                "pick": "A",
                                "rationale": "Simple",
                            }
                        ]
                    }
                )
            ]
        }
    )
    plan = await judge(
        judge_router,
        spec_obj,
        [
            ResearchNote(
                requirement_id="R001",
                model="m-res-1",
                candidates=[CandidateTool(name="A")],
            )
        ],
    )
    assert plan.decisions[0].pick == "A"

    with pytest.raises(FileNotFoundError):
        await run_plan(tmp_path / "missing.md", tmp_path / "runs")

    empty_spec = tmp_path / "empty.md"
    empty_spec.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        await run_plan(empty_spec, tmp_path / "runs")

    runs_root = tmp_path / "fail_runs"
    failing_router = FakeRouter({("parser", "m-parse"): [RuntimeError("parse down")]})
    with pytest.raises(RuntimeError, match="parse down"):
        spec_ok = tmp_path / "ok.md"
        spec_ok.write_text("# Demo\nBuild x.", encoding="utf-8")
        await run_plan(spec_ok, runs_root, router=failing_router)

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert (run_dirs[0] / "tokens.json").is_file()


async def test_run_build_with_research_and_judge_end_to_end(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Greeter\nBuild greet(name) in Python.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "Greeter lib",
            "requirements": [
                {"id": "R001", "title": "Greet", "description": "Return greeting string"}
            ],
        }
    )
    res_json = json.dumps({"candidates": [{"name": "stdlib"}, {"name": "click"}]})
    judge_json = json.dumps(
        {
            "decisions": [
                {
                    "requirement_id": "R001",
                    "options": [{"name": "stdlib"}, {"name": "click"}],
                    "pick": "stdlib",
                    "rationale": "Minimal",
                }
            ],
            "tasks": ["Implement greet"],
        }
    )
    app_json = json.dumps(
        {
            "summary": "Greeter",
            "stack": "python",
            "files": [
                {
                    "path": "greeter.py",
                    "content": "def greet(n: str) -> str:\n    return f'Hi {n}'\n",
                },
                {
                    "path": "tests/test_greeter.py",
                    "content": (
                        "from greeter import greet\n\n\n"
                        "def test_greet():\n"
                        "    assert greet('A') == 'Hi A'\n"
                    ),
                },
            ],
        }
    )

    router = FakeRouter(
        {
            ("parser", "m-parse"): [parsed_json],
            ("researcher", "m-res-1"): [res_json],
            ("researcher", "m-res-2"): [res_json],
            ("judge", "m-judge"): [judge_json],
            ("builder", "m-judge"): [app_json],
        }
    )
    # Provide explicit builder role on router
    router.cfg["roles"]["builder"] = ["m-build"]
    router._responses[("builder", "m-build")] = [app_json]

    out_dir = await run_build(spec_file, tmp_path / "runs", tmp_path / "app", router=router)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["stages"]["research"] == "completed"
    assert manifest["stages"]["judge"] == "completed"
    assert (out_dir / "research.json").is_file()
    assert (out_dir / "plan.json").is_file()
    assert (out_dir / "plan.md").is_file()
    assert (out_dir / "build.json").is_file()
    assert (out_dir / "verification.json").is_file()

    tokens = json.loads((out_dir / "tokens.json").read_text(encoding="utf-8"))
    assert tokens["total_calls"] == 5
    assert [b["name"] for b in tokens["by_stage"]] == ["parser", "researcher", "judge", "builder"]


async def test_judge_evaluates_multiple_requirements_in_parallel_with_scoped_research():
    spec = ParsedSpec(
        summary="Three-part app",
        requirements=[
            Requirement(id="R001", title="Slug", description="Slugify titles"),
            Requirement(id="R002", title="Store", description="Persist JSON mappings"),
            Requirement(id="R003", title="CLI", description="Parse CLI args"),
        ],
    )
    report = ResearchReport(
        notes=[
            ResearchNote(
                requirement_id="R001",
                model="m-res-1",
                candidates=[
                    CandidateTool(name="unicodedata"),
                    CandidateTool(name="python-slugify"),
                ],
            ),
            ResearchNote(
                requirement_id="R002",
                model="m-res-1",
                candidates=[CandidateTool(name="json"), CandidateTool(name="sqlite3")],
            ),
            ResearchNote(
                requirement_id="R003",
                model="m-res-1",
                candidates=[CandidateTool(name="argparse"), CandidateTool(name="click")],
            ),
        ],
        failures=[
            ResearchFailure(
                requirement_id="R002",
                model="m-res-2",
                error_type="TimeoutError",
                error_message="timed out",
            )
        ],
    )

    seen_users: list[str] = []

    class RecordingJudgeRouter(ModelRouter):
        def __init__(self):
            self.path = "fake"
            self.cfg = {
                "roles": {"judge": ["m-judge"]},
                "limits": {
                    "max_parallel_calls": 4,
                    "max_tokens": 1000,
                    "timeout_seconds": 10,
                    "repair_attempts": 1,
                },
            }
            self._semaphores = {}
            self.call_log = []

        async def complete(self, model: str, system: str, user: str, role: str | None = None):
            seen_users.append(user)
            if "RESEARCH FOR R001" in user:
                req_id, a, b, extra_risk, extra_task = (
                    "R001",
                    "unicodedata",
                    "python-slugify",
                    "Unicode edge case",
                    "Implement slugify",
                )
            elif "RESEARCH FOR R002" in user:
                req_id, a, b, extra_risk, extra_task = (
                    "R002",
                    "json",
                    "sqlite3",
                    "Shared risk",
                    "Implement store",
                )
            else:
                req_id, a, b, extra_risk, extra_task = (
                    "R003",
                    "argparse",
                    "click",
                    "shared risk",
                    "Implement CLI",
                )
            payload = {
                "decisions": [
                    {
                        "requirement_id": req_id,
                        "options": [{"name": a}, {"name": b}],
                        "pick": a,
                        "rationale": f"Fits {req_id}",
                    }
                ],
                "risks": [extra_risk],
                "tasks": [extra_task, "Write tests"],
            }
            return model, json.dumps(payload)

    plan = await judge(RecordingJudgeRouter(), spec, report)
    assert len(seen_users) == 3
    assert [d.requirement_id for d in plan.decisions] == ["R001", "R002", "R003"]

    r1_prompt = next(u for u in seen_users if "RESEARCH FOR R001" in u)
    r2_prompt = next(u for u in seen_users if "RESEARCH FOR R002" in u)
    assert "python-slugify" in r1_prompt
    assert "sqlite3" not in r1_prompt
    assert "1 failed calls" in r2_prompt
    assert "0 failed calls" in r1_prompt

    # Case-insensitive deduplication while preserving order
    assert plan.risks == ["Unicode edge case", "Shared risk"]
    assert plan.tasks == ["Implement slugify", "Write tests", "Implement store", "Implement CLI"]


@pytest.mark.asyncio
async def test_research_harness_backend_deduplicates_multi_model_roles(monkeypatch):
    from atlas.pipeline import research as research_mod

    spec = ParsedSpec(
        summary="Multi-model config under harness",
        requirements=[
            Requirement(id="R001", title="Req 1", description="Do A"),
            Requirement(id="R002", title="Req 2", description="Do B"),
        ],
    )

    class _HarnessResearchRouter(ModelRouter):
        def __init__(self) -> None:
            self.path = Path("fake")
            self.cfg = {
                "roles": {"researcher": ["m-one", "m-two", "m-three"]},
                "backends": {
                    "researcher": {"type": "harness", "provider": "antigravity", "model": "neon"}
                },
                "limits": {"max_parallel_calls": 4, "max_tokens": 1000, "timeout_seconds": 10},
            }
            self._semaphores = {}
            self.call_log = []

    calls: list[str] = []

    async def fake_complete_json_model(router, *, model, role, system, user, **kwargs):
        calls.append(model)
        req_id = "R001" if "Requirement R001" in user else "R002"
        return "neon", ResearchNote(
            requirement_id=req_id,
            model="neon",
            candidates=[CandidateTool(name="stdlib")],
        )

    monkeypatch.setattr(research_mod, "complete_json_model", fake_complete_json_model)
    report = await research(_HarnessResearchRouter(), spec)
    assert len(calls) == 2
    assert {n.requirement_id for n in report.notes} == {"R001", "R002"}


@pytest.mark.asyncio
async def test_run_build_preserves_verification_json_and_fix_rounds_on_mid_loop_crash(
    tmp_path, monkeypatch
):
    from atlas import runner as runner_mod
    from atlas.schemas import (
        CheckResult,
        GeneratedApp,
        GeneratedFile,
        PatchResult,
        Plan,
        ToolDecision,
    )

    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nA small app\n", encoding="utf-8")
    spec = ParsedSpec(
        summary="Demo app",
        requirements=[Requirement(id="R001", title="Run", description="Runs cleanly")],
    )
    plan = Plan(
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
        tasks=["Write app.py"],
    )
    from_run = tmp_path / "prev_run"
    from_run.mkdir()
    (from_run / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (from_run / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

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
  max_fix_rounds: 3
verification:
  require_requirement_tests: false
  require_product_contract: false
  require_e2e_tests: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def fake_build(*args, **kwargs):
        return GeneratedApp(
            summary="Built",
            stack="python",
            files=[GeneratedFile(path="app.py", content="x = 1\n", purpose="app")],
        )

    verify_calls = 0

    async def fake_verify(app_dir: Path):
        nonlocal verify_calls
        verify_calls += 1
        return (
            "python",
            [
                CheckResult(
                    name="pytest",
                    command="pytest",
                    passed=False,
                    exit_code=1,
                    stderr=f"failure round {verify_calls}",
                )
            ],
        )

    fix_calls = 0

    async def fake_fix(*args, **kwargs):
        nonlocal fix_calls
        fix_calls += 1
        if fix_calls == 1:
            return PatchResult(
                diagnosis="First round attempt",
                files=[GeneratedFile(path="app.py", content="x = 2\n", purpose="update")],
            )
        raise RuntimeError("OutputTruncatedError: mid-loop crash")

    monkeypatch.setattr(runner_mod, "build_app", fake_build)
    monkeypatch.setattr(runner_mod, "fix_app", fake_fix)

    runs_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="mid-loop crash"):
        await run_build(
            spec_file,
            runs_root,
            tmp_path / "app" / "demo",
            config_path=cfg,
            from_run=from_run,
            verify_fn=fake_verify,
            ledger_path=tmp_path / "usage" / "ledger.jsonl",
        )

    [out_dir] = list(runs_root.iterdir())
    assert (out_dir / "verification.json").is_file()
    vjson = json.loads((out_dir / "verification.json").read_text(encoding="utf-8"))
    assert vjson["passed"] is False
    assert [r["round_number"] for r in vjson["rounds"]] == [1, 2]
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["verification_rounds"] == 2
    assert manifest["fix_rounds"] == 2
