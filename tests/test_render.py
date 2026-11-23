from atlas.pipeline.render import render_plan
from atlas.schemas import (
    CandidateTool,
    ParsedSpec,
    Plan,
    Requirement,
    ResearchFailure,
    ToolDecision,
)


def test_render_contains_title_table_cleaned_verify_and_warning():
    spec = ParsedSpec(
        summary="Daily roaming pipeline",
        requirements=[Requirement(id="R001", title="Daily ingest", description="Ingest files")],
    )
    decision = ToolDecision(
        requirement_id="R001",
        options=[
            CandidateTool(name="dbt", verification_flags=["VERIFY: Check adapter version"]),
            CandidateTool(name="Spark SQL"),
        ],
        pick="dbt",
        rationale="fits stack",
        needs_verification=["Confirm warehouse sizing"],
    )
    plan = Plan(spec=spec, decisions=[decision])
    md = render_plan(
        plan,
        run_id="20261121T010000Z-12345678",
        plan_sha256="a" * 64,
        research_failures=[
            ResearchFailure(
                requirement_id="R001",
                model="openai/gpt-5",
                error_type="TimeoutError",
                error_message="timed out",
            )
        ],
    )
    assert "### R001 — Daily ingest: recommend **dbt**" in md
    assert "| Candidate | Effort | Maintenance | Licensing | Selected |" in md
    assert "- **VERIFY:** Check adapter version" in md
    assert "VERIFY: VERIFY:" not in md
    assert "**Outstanding verification:**" in md
    assert "Partial research warning:" in md
    assert "Plan reviewed and approved" in md
