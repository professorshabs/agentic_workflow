import json
from pathlib import Path

import pytest

from atlas.pipeline.research import research
from atlas.router import ModelRouter
from atlas.runner import record_approval, run_plan
from atlas.schemas import ParsedSpec, Requirement


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
        self.call_log.append({"role": role, "requested_model": model, "used_model": model})
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
