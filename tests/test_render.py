from atlas.pipeline.render import render_plan, render_run_report
from atlas.schemas import (
    CandidateTool,
    CheckResult,
    GeneratedApp,
    GeneratedFile,
    ParsedSpec,
    Plan,
    Requirement,
    ResearchFailure,
    TokenUsageSummary,
    ToolDecision,
    UsageBucket,
    VerificationReport,
    VerificationRound,
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


def test_render_run_report_covers_verified_failed_and_error_states():
    spec = ParsedSpec(
        summary="CLI calculator",
        requirements=[Requirement(id="R001", title="Add", description="Add numbers")],
    )
    app = GeneratedApp(
        summary="Python calculator",
        stack="python",
        files=[GeneratedFile(path="calc.py", content="def add(a, b): return a + b\n")],
        run_instructions="python -m calc",
    )
    fail_chk = CheckResult(
        name="python-pytest",
        command="python -m pytest",
        passed=False,
        exit_code=1,
        duration_ms=42.0,
        stdout="1 failed",
        stderr="AssertionError",
    )
    verification = VerificationReport(
        passed=False,
        stack_detected="python",
        rounds=[
            VerificationRound(
                round_number=1,
                passed=False,
                stack_detected="python",
                checks=[fail_chk],
                diagnosis="Fixed return expression",
            )
        ],
        final_checks=[fail_chk],
    )
    summary = TokenUsageSummary(
        total_calls=2,
        prompt_tokens=150,
        completion_tokens=75,
        cached_tokens=0,
        reasoning_tokens=0,
        total_tokens=225,
        cost_usd=0.002,
        elapsed_ms=55.0,
        by_stage=[
            UsageBucket(
                name="builder",
                calls=2,
                prompt_tokens=150,
                completion_tokens=75,
                cached_tokens=0,
                reasoning_tokens=0,
                total_tokens=225,
                cost_usd=0.002,
                elapsed_ms=55.0,
            )
        ],
    )

    md = render_run_report(
        run_id="20261123T120000Z-abcdef12",
        spec_path="specs/calc.md",
        status="verification_failed",
        spec=spec,
        generated_app=app,
        verification=verification,
        token_summary=summary,
        written_files=["calc.py"],
        error="Verification failed",
    )
    assert "# Atlas Build & Verification Report" in md
    assert "`verification_failed`" in md
    assert "### Auto-fix rounds" in md
    assert "Fixed return expression" in md
    assert "<summary>Failing verification output</summary>" in md
    assert "AssertionError" in md
    assert "## Token Usage & Cost" in md
    assert "- `calc.py`" in md
    assert "python -m calc" in md
