import re

from atlas.schemas import (
    GeneratedApp,
    ParsedSpec,
    Plan,
    ResearchFailure,
    TokenUsageSummary,
    VerificationReport,
    WorkspaceBuildResult,
)
from atlas.telemetry import render_token_report_md

_VERIFY_PREFIX_RE = re.compile(r"^verify\s*:\s*", flags=re.IGNORECASE)


def _clean_verify_flag(flag: str) -> str:
    return _VERIFY_PREFIX_RE.sub("", flag.strip()).strip()


def render_plan(
    plan: Plan,
    *,
    run_id: str | None = None,
    plan_sha256: str | None = None,
    research_failures: list[ResearchFailure] | None = None,
) -> str:
    spec = plan.spec
    req_titles = {req.id: req.title for req in spec.requirements}

    lines = [
        "# Engineering plan",
        "",
        spec.summary,
        "",
        "> Status: DRAFT — no implementation or repository changes have been authorized.",
    ]
    if run_id or plan_sha256:
        meta_parts = []
        if run_id:
            meta_parts.append(f"Run ID: `{run_id}`")
        if plan_sha256:
            meta_parts.append(f"Plan SHA-256: `{plan_sha256[:16]}...`")
        lines.append(f"> {' | '.join(meta_parts)}")

    if research_failures:
        affected = sorted({f.requirement_id for f in research_failures})
        lines.extend([
            "",
            f"> **Partial research warning:** {len(research_failures)} research call(s) failed "
            f"(affected requirements: {', '.join(affected)}). Review confidence scores and "
            "verification flags carefully.",
        ])

    lines.extend(["", "## Requirements"])
    for req in spec.requirements:
        lines.append(f"- **{req.id} — {req.title}** ({req.kind}): {req.description}")
        for criterion in req.acceptance_criteria:
            lines.append(f"  - Acceptance: {criterion}")
        if req.depends_on:
            lines.append(f"  - Depends on: {', '.join(req.depends_on)}")

    for heading, values in (
        ("Constraints", spec.constraints),
        ("Assumptions", spec.assumptions),
        ("Open questions", spec.open_questions),
        ("Risks", plan.risks),
    ):
        if values:
            lines.extend(["", f"## {heading}"])
            lines.extend(f"- {value}" for value in values)

    lines.extend(["", "## Tool decisions"])
    for decision in plan.decisions:
        title = req_titles.get(decision.requirement_id)
        heading_id = f"{decision.requirement_id} — {title}" if title else decision.requirement_id
        lines.extend([
            "",
            f"### {heading_id}: recommend **{decision.pick}**",
            f"Confidence: {decision.confidence}",
            "",
            decision.rationale,
            "",
            "| Candidate | Effort | Maintenance | Licensing | Selected |",
            "| :--- | :---: | :---: | :--- | :---: |",
        ])
        for option in decision.options:
            selected = "**Yes**" if option.name == decision.pick else "No"
            licensing = option.licensing or "Not established"
            lines.append(
                f"| {option.name} | {option.integration_effort} | "
                f"{option.maintenance_burden} | {licensing} | {selected} |"
            )

        for option in decision.options:
            lines.extend([
                "",
                f"#### {option.name}",
                f"- Advantages: {'; '.join(option.pros) or 'Not specified'}",
                f"- Disadvantages: {'; '.join(option.cons) or 'Not specified'}",
                f"- Licensing: {option.licensing or 'Not established'}",
                f"- Cost: {option.cost_notes or 'Not established'}",
                f"- Integration effort: {option.integration_effort}",
                f"- Maintenance burden: {option.maintenance_burden}",
                f"- Documentation: {option.docs_url or 'Not supplied'}",
            ])
            for evidence in option.evidence:
                source = evidence.source_url or evidence.source_title or evidence.source_type
                lines.append(f"- Evidence: {evidence.claim} ({source})")
            for flag in option.verification_flags:
                cleaned_flag = _clean_verify_flag(flag)
                if cleaned_flag:
                    lines.append(f"- **VERIFY:** {cleaned_flag}")

        if decision.needs_verification:
            lines.extend([
                "",
                "**Outstanding verification:**",
                *[f"- {item}" for item in decision.needs_verification],
            ])

    if plan.tasks:
        lines.extend(["", "## Implementation tasks"])
        lines.extend(f"{i}. {task}" for i, task in enumerate(plan.tasks, 1))

    lines.extend([
        "",
        "## Approval",
        "- [ ] Plan reviewed and approved by the project owner",
        "",
        "Approval is a human workflow action recorded by Atlas. This checkbox is documentary only; "
        "it does not trigger development, merge a pull request, or grant repository permissions.",
        "",
    ])
    return "\n".join(lines)


def render_run_report(
    *,
    run_id: str,
    spec_path: str,
    status: str,
    spec: ParsedSpec | None,
    generated_app: GeneratedApp | WorkspaceBuildResult | None,
    verification: VerificationReport | None,
    token_summary: TokenUsageSummary,
    written_files: list[str],
    app_dir: str | None = None,
    error: str | None = None,
) -> str:
    """Render a full Markdown build, verification, and token usage report."""
    stack_label = (
        verification.stack_detected
        if verification is not None
        else (generated_app.stack if generated_app is not None else "unknown")
    )
    rounds_count = len(verification.rounds) if verification is not None else 0

    lines = [
        "# Atlas Build & Verification Report",
        "",
        f"- **Run ID:** `{run_id}`",
        f"- **Spec:** `{spec_path}`",
        *(
            [f"- **Output dir:** `{app_dir}`"]
            if app_dir
            else []
        ),
        f"- **Status:** `{status}`",
        f"- **Stack detected:** `{stack_label}`",
        f"- **Verification rounds:** {rounds_count}",
    ]

    if spec is not None:
        lines.extend(["", "## Specification summary", "", spec.summary])

    if generated_app is not None and generated_app.summary:
        lines.extend(["", "## Application summary", "", generated_app.summary])

    if error:
        lines.extend(["", "## Error", "", f"```\n{error}\n```"])

    if verification is not None:
        lines.extend(["", "## Verification results", ""])
        if verification.final_checks:
            lines.extend([
                "| Check | Command | Status | Exit | Duration |",
                "| :--- | :--- | :---: | ---: | ---: |",
            ])
            for chk in verification.final_checks:
                badge = "PASS" if chk.passed else "FAIL"
                lines.append(
                    f"| `{chk.name}` | `{chk.command}` | {badge} | "
                    f"{chk.exit_code} | {chk.duration_ms:.1f} ms |"
                )

        round_diagnoses = [r for r in verification.rounds if r.diagnosis]
        if round_diagnoses:
            lines.extend(["", "### Auto-fix rounds", ""])
            for rnd in round_diagnoses:
                lines.append(f"- **Round {rnd.round_number}:** {rnd.diagnosis}")

        failing = [c for c in verification.final_checks if not c.passed]
        if failing:
            lines.extend([
                "",
                "<details>",
                "<summary>Failing verification output</summary>",
                "",
            ])
            for chk in failing:
                lines.append(f"#### `{chk.name}` (`{chk.command}`)")
                if chk.stdout:
                    lines.extend(["", "```text", chk.stdout, "```"])
                if chk.stderr:
                    lines.extend(["", "```text", chk.stderr, "```"])
            lines.extend(["", "</details>"])

    lines.extend(["", render_token_report_md(token_summary)])

    if written_files:
        lines.extend(["", "## Generated files", ""])
        lines.extend(f"- `{path}`" for path in written_files)

    if generated_app is not None and generated_app.run_instructions:
        lines.extend(["", "## Run instructions", "", generated_app.run_instructions])

    lines.append("")
    return "\n".join(lines)
