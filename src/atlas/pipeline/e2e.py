"""Authoritative black-box end-to-end product test generation before build."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from atlas.router import ModelRouter
from atlas.schemas import (
    ClarificationReport,
    E2ETestSuite,
    ParsedSpec,
    Plan,
    ResearchReport,
)

from .util import complete_json_model, prompt, role_target_label

logger = logging.getLogger("atlas.pipeline.e2e")


def _compact(value: Any) -> str:
    """Serialize JSON compactly to avoid wasting prompt tokens on indentation."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _ruff_executable() -> str | None:
    """Locate `ruff` either next to the running Python interpreter or on PATH."""
    sibling = Path(sys.executable).parent / "ruff"
    if sibling.is_file():
        return str(sibling)
    return shutil.which("ruff")


def sanitize_e2e_suite_files(suite: E2ETestSuite) -> E2ETestSuite:
    """Auto-fix deterministic Python lint issues in authoritative E2E files."""
    ruff = _ruff_executable()
    if not ruff or not suite.files:
        return suite
    updated = []
    changed = False
    for file_entry in suite.files:
        content = file_entry.content
        if file_entry.path.endswith(".py") and content.strip():
            try:
                proc = subprocess.run(
                    [
                        ruff,
                        "check",
                        "--isolated",
                        "--no-cache",
                        "--fix",
                        "--quiet",
                        "--stdin-filename",
                        file_entry.path,
                        "-",
                    ],
                    input=content,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                proc = None
            if proc is not None and proc.returncode in (0, 1) and proc.stdout:
                if proc.stdout != content:
                    changed = True
                updated.append(file_entry.model_copy(update={"content": proc.stdout}))
                continue
        updated.append(file_entry)
    return suite.model_copy(update={"files": updated}) if changed else suite


async def generate_e2e_tests(
    router: ModelRouter,
    spec: ParsedSpec,
    research_report: ResearchReport | None = None,
    plan: Plan | None = None,
    clarifications: ClarificationReport | None = None,
    *,
    raw_spec: str = "",
) -> E2ETestSuite:
    """Generate black-box end-to-end product tests covering the finished product contract."""
    role = "researcher"
    model = ""
    for candidate_role in ("e2e", "researcher", "judge", "parser"):
        try:
            models = router.models_for(candidate_role)
            if models:
                role = candidate_role
                model = models[0]
                break
        except ValueError:
            continue
    if not model:
        raise ValueError("No model configured for e2e/researcher/judge/parser roles")

    parts = [f"PARSED SPECIFICATION:\n{_compact(spec.model_dump(mode='json'))}"]
    if raw_spec.strip():
        parts.append(f"RAW SPECIFICATION:\n{raw_spec.strip()[:12_000]}")

    if clarifications is not None and clarifications.items:
        clar_payload = [
            item.model_dump(mode="json")
            for item in clarifications.items
            if item.answer.strip()
        ]
        if clar_payload:
            parts.append(f"USER CLARIFICATIONS:\n{_compact(clar_payload)}")

    if research_report is not None and research_report.notes:
        scenarios_by_req: dict[str, list[str]] = {}
        findings_by_req: dict[str, list[str]] = {}
        for note in research_report.notes:
            req_scenarios = scenarios_by_req.setdefault(note.requirement_id, [])
            for scen in note.e2e_scenarios:
                cleaned = scen.strip()
                if cleaned and cleaned not in req_scenarios:
                    req_scenarios.append(cleaned)
            if plan is None and note.findings:
                req_findings = findings_by_req.setdefault(note.requirement_id, [])
                for finding in note.findings[:3]:
                    cleaned = finding.strip()
                    if cleaned and cleaned not in req_findings:
                        req_findings.append(cleaned)
        scenarios: list[dict[str, Any]] = []
        for req_id, req_scenarios in scenarios_by_req.items():
            entry: dict[str, Any] = {"requirement_id": req_id}
            if req_scenarios:
                entry["e2e_scenarios"] = req_scenarios
            if plan is None and findings_by_req.get(req_id):
                entry["findings"] = findings_by_req[req_id][:3]
            if len(entry) > 1:
                scenarios.append(entry)
        if scenarios:
            label = (
                "RESEARCH E2E SCENARIOS:"
                if plan is not None
                else "RESEARCH E2E SCENARIOS & FINDINGS:"
            )
            parts.append(f"{label}\n{_compact(scenarios)}")

    if plan is not None:
        decisions_payload = [
            {
                "requirement_id": d.requirement_id,
                "pick": d.pick,
                "rationale": d.rationale[:300],
            }
            for d in plan.decisions
        ]
        plan_payload = {"decisions": decisions_payload, "tasks": plan.tasks}
        parts.append(f"ENGINEERING PLAN DECISIONS & TASKS:\n{_compact(plan_payload)}")

    target = role_target_label(router, role, fallback_model=model)
    logger.info(
        "[e2e] Calling role=%s target=%s for %d requirement(s)...",
        role,
        target,
        len(spec.requirements),
    )
    _, suite = await complete_json_model(
        router,
        model=model,
        role=role,
        system=prompt("e2e_tests"),
        user="\n\n".join(parts),
        schema=E2ETestSuite,
    )
    return sanitize_e2e_suite_files(suite)
