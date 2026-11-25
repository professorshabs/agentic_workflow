import asyncio
import json
import logging
import time
from typing import Any

from atlas.router import ModelRouter
from atlas.schemas import (
    JudgeOutput,
    ParsedSpec,
    Plan,
    Requirement,
    ResearchFailure,
    ResearchNote,
    ResearchReport,
)

from .util import complete_json_model, prompt

logger = logging.getLogger("atlas.pipeline.judge")


def _compact(value: Any) -> str:
    """JSON without indentation: whitespace is billed as prompt tokens."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


async def judge(
    router: ModelRouter,
    spec: ParsedSpec,
    research_input: ResearchReport | list[ResearchNote],
) -> Plan:
    model = router.models_for("judge")[0]
    system = prompt("judge")

    if isinstance(research_input, ResearchReport):
        notes = research_input.notes
        failures = research_input.failures
    else:
        notes = research_input
        failures = []

    project_overview = {
        "summary": spec.summary,
        "constraints": spec.constraints,
        "assumptions": spec.assumptions,
        "open_questions": spec.open_questions,
        "requirements_overview": [
            {
                "id": r.id,
                "title": r.title,
                "kind": r.kind,
                "depends_on": r.depends_on,
            }
            for r in spec.requirements
        ],
    }

    notes_by_req: dict[str, list[ResearchNote]] = {}
    for note in notes:
        notes_by_req.setdefault(note.requirement_id, []).append(note)

    failures_by_req: dict[str, list[ResearchFailure]] = {}
    for failure in failures:
        failures_by_req.setdefault(failure.requirement_id, []).append(failure)

    total = len(spec.requirements)
    completed = 0

    async def judge_one(req: Requirement) -> JudgeOutput:
        nonlocal completed
        started = time.monotonic()
        req_notes = notes_by_req.get(req.id, [])
        req_failures = failures_by_req.get(req.id, [])
        research_payload = {
            "notes": [n.model_dump(mode="json", exclude_none=True) for n in req_notes],
            "failures": [f.model_dump(mode="json") for f in req_failures],
        }
        completeness_note = (
            f"RESEARCH FOR {req.id} ({len(req_notes)} notes, {len(req_failures)} failed calls — "
            "treat gaps with lower confidence):"
            if req_failures
            else f"RESEARCH FOR {req.id} ({len(req_notes)} notes, 0 failed calls):"
        )
        user = (
            f"PROJECT OVERVIEW:\n{_compact(project_overview)}\n\n"
            f"TARGET REQUIREMENT:\n{req.model_dump_json()}\n\n"
            f"{completeness_note}\n"
            f"{_compact(research_payload)}"
        )

        def validate_single_decision(data: dict[str, Any], _used_model: str) -> dict[str, Any]:
            decisions = data.get("decisions")
            if not isinstance(decisions, list) or len(decisions) != 1:
                got = len(decisions) if isinstance(decisions, list) else type(decisions).__name__
                raise ValueError(f"Expected exactly 1 decision for '{req.id}', got {got}")
            first = decisions[0]
            if not isinstance(first, dict):
                raise ValueError(f"Decision for '{req.id}' must be a JSON object")
            # We asked about exactly this requirement, so set its ID rather than paying for a
            # repair call when the model writes "r001", "R1" or leaves it out.
            first["requirement_id"] = req.id
            return data

        try:
            used_model, judged = await complete_json_model(
                router,
                model=model,
                role="judge",
                system=system,
                user=user,
                schema=JudgeOutput,
                postprocess=validate_single_decision,
            )
        except Exception as exc:
            completed += 1
            logger.info(
                "[judge] %d/%d FAILED: %s (%s, %.1fs)",
                completed,
                total,
                req.id,
                type(exc).__name__,
                time.monotonic() - started,
            )
            raise
        completed += 1
        logger.info(
            "[judge] %d/%d done: %s -> %s (%s, %.1fs)",
            completed,
            total,
            req.id,
            judged.decisions[0].pick[:60],
            used_model,
            time.monotonic() - started,
        )
        return judged

    outcomes = await asyncio.gather(
        *(judge_one(req) for req in spec.requirements), return_exceptions=True
    )
    failed = [
        f"{req.id}: {type(out).__name__}: {out}"
        for req, out in zip(spec.requirements, outcomes, strict=True)
        if isinstance(out, BaseException)
    ]
    if failed:
        raise RuntimeError(
            f"Judge failed for {len(failed)} of {len(outcomes)} requirement(s): "
            + " | ".join(failed)
        )
    results = [out for out in outcomes if isinstance(out, JudgeOutput)]

    decisions = [res.decisions[0] for res in results]
    risks = _dedupe_preserve_order([risk for res in results for risk in res.risks])
    tasks = _dedupe_preserve_order([task for res in results for task in res.tasks])

    return Plan(
        spec=spec,
        decisions=decisions,
        risks=risks,
        tasks=tasks,
    )
