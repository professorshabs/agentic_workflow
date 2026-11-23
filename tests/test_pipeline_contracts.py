import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.pipeline.util import complete_json_model, extract_json, extract_json_object
from atlas.router import ModelRouter
from atlas.schemas import (
    CandidateTool,
    GeneratedApp,
    GeneratedFile,
    ParsedSpec,
    PatchResult,
    Plan,
    Requirement,
    ToolDecision,
)


def test_extract_json_handles_fences_prose_and_trailing_braces():
    assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert extract_json('Result:\n```json\n{"ok": true}\n```\nTrailing {note}') == {"ok": True}
    assert extract_json('Result:\n{"ok": true}\nDone {extra}') == {"ok": True}


def test_extract_json_object_rejects_non_dict():
    with pytest.raises(ValueError, match="Expected a JSON object"):
        extract_json_object("[1, 2, 3]")


def test_parsed_spec_rejects_empty_requirements():
    with pytest.raises(ValidationError):
        ParsedSpec(summary="x", requirements=[])


def test_requirement_ids_must_be_unique():
    with pytest.raises(ValidationError):
        ParsedSpec(
            summary="x",
            requirements=[
                Requirement(id="R1", title="a", description="a"),
                Requirement(id="R1", title="b", description="b"),
            ],
        )


def test_requirement_rejects_self_dependency_and_cycles():
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        ParsedSpec(
            summary="x",
            requirements=[
                Requirement(id="R1", title="a", description="a", depends_on=["R1"]),
            ],
        )
    with pytest.raises(ValidationError, match="Cyclic requirement dependency"):
        ParsedSpec(
            summary="x",
            requirements=[
                Requirement(id="R1", title="a", description="a", depends_on=["R2"]),
                Requirement(id="R2", title="b", description="b", depends_on=["R1"]),
            ],
        )


def test_decision_requires_two_or_three_unique_options_and_valid_pick():
    with pytest.raises(ValidationError):
        ToolDecision(
            requirement_id="R1",
            options=[CandidateTool(name="A")],
            pick="A",
            rationale="x",
        )
    with pytest.raises(ValidationError, match="case-insensitive"):
        ToolDecision(
            requirement_id="R1",
            options=[CandidateTool(name="dbt"), CandidateTool(name="DBT")],
            pick="dbt",
            rationale="x",
        )
    with pytest.raises(ValidationError):
        ToolDecision(
            requirement_id="R1",
            options=[CandidateTool(name="A"), CandidateTool(name="B")],
            pick="C",
            rationale="x",
        )


def test_plan_requires_decision_for_each_requirement_and_hashes_deterministically():
    spec = ParsedSpec(
        summary="x",
        requirements=[Requirement(id="R1", title="a", description="a")],
    )
    with pytest.raises(ValidationError):
        Plan(spec=spec, decisions=[])

    decision = ToolDecision(
        requirement_id="R1",
        options=[CandidateTool(name="A"), CandidateTool(name="B")],
        pick="A",
        rationale="fits",
    )
    plan = Plan(spec=spec, decisions=[decision])
    assert len(plan.content_sha256()) == 64
    assert plan.content_sha256() == Plan(spec=spec, decisions=[decision]).content_sha256()

    with pytest.raises(ValidationError, match="Duplicate requirement decisions"):
        Plan(spec=spec, decisions=[decision, decision])


def test_generated_app_patch_and_relative_path_contracts():
    gf_backslash = GeneratedFile(path=r".\src\main.py", content="print('hi')\n")
    assert gf_backslash.path == "src/main.py"

    with pytest.raises(ValidationError, match="Absolute drive paths"):
        GeneratedFile(path="C:/temp/app.py", content="x = 1")

    with pytest.raises(ValidationError, match="Writing inside .github"):
        GeneratedFile(path=".github/workflows/evil.yml", content="name: x")

    with pytest.raises(ValidationError, match="unique"):
        GeneratedApp(
            summary="dup",
            files=[
                GeneratedFile(path="a.py", content="1"),
                GeneratedFile(path="./a.py", content="2"),
            ],
        )

    with pytest.raises(ValidationError, match="modify or delete at least one file"):
        PatchResult(diagnosis="nothing changed", files=[], delete_files=[])

    with pytest.raises(ValidationError, match="unique"):
        PatchResult(
            diagnosis="dup",
            files=[
                GeneratedFile(path="a.py", content="1"),
                GeneratedFile(path="a.py", content="2"),
            ],
        )

    with pytest.raises(ValidationError, match="Parent traversal"):
        PatchResult(diagnosis="bad delete", delete_files=["../secret.txt"])


def test_extract_json_rejects_empty_and_non_json():
    with pytest.raises(ValueError, match="empty"):
        extract_json("   \n\t  ")
    with pytest.raises(ValueError, match="valid JSON object"):
        extract_json("This is plain prose with no JSON.")


class _RepairRouter(ModelRouter):
    def __init__(self, outputs: list[str], repair_attempts: int = 1):
        self.path = Path("fake")
        self.cfg = {
            "roles": {"parser": ["m"]},
            "limits": {
                "max_parallel_calls": 1,
                "max_tokens": 100,
                "timeout_seconds": 10,
                "repair_attempts": repair_attempts,
            },
        }
        self._semaphores = {}
        self.call_log = []
        self._outputs = list(outputs)
        self.seen_users: list[str] = []

    async def complete(self, model: str, system: str, user: str, role: str | None = None):
        self.seen_users.append(user)
        return model, self._outputs.pop(0)


async def test_complete_json_model_repairs_invalid_output_and_raises_when_exhausted():
    valid = json.dumps(
        {
            "summary": "Repaired spec",
            "requirements": [{"id": "R1", "title": "T", "description": "D"}],
        }
    )
    router = _RepairRouter(["not json at all", valid], repair_attempts=1)
    used, parsed = await complete_json_model(
        router,
        model="m",
        role="parser",
        system="sys",
        user="orig user",
        schema=ParsedSpec,
    )
    assert used == "m"
    assert parsed.summary == "Repaired spec"
    assert len(router.seen_users) == 2
    assert "failed JSON/schema validation" in router.seen_users[1]

    exhausted = _RepairRouter(["still broken", "{}"], repair_attempts=1)
    with pytest.raises(ValidationError):
        await complete_json_model(
            exhausted,
            model="m",
            role="parser",
            system="sys",
            user="orig",
            schema=ParsedSpec,
        )
