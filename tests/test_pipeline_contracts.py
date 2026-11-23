import pytest
from pydantic import ValidationError

from atlas.pipeline.util import extract_json, extract_json_object
from atlas.schemas import (
    CandidateTool,
    ParsedSpec,
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
