import asyncio
import logging
from typing import Any

from atlas.router import ModelRouter
from atlas.schemas import ParsedSpec, Requirement, ResearchFailure, ResearchNote, ResearchReport

from .util import complete_json_model, prompt

logger = logging.getLogger("atlas.pipeline.research")


async def research(router: ModelRouter, spec: ParsedSpec) -> ResearchReport:
    system = prompt("research")
    context = (
        f"Project: {spec.summary}\n"
        f"Constraints: {spec.constraints}\n"
        f"Assumptions: {spec.assumptions}\n"
    )

    async def one(req: Requirement, model: str) -> ResearchNote:
        user = (
            f"{context}\nRequirement {req.id}: {req.title}\n{req.description}\n"
            f"Acceptance criteria: {req.acceptance_criteria}\n"
            "Return structured JSON matching the research schema."
        )

        def attach_metadata(data: dict[str, Any], used_model: str) -> dict[str, Any]:
            data["requirement_id"] = req.id
            data["model"] = used_model
            return data

        _, note = await complete_json_model(
            router,
            model=model,
            role="researcher",
            system=system,
            user=user,
            schema=ResearchNote,
            postprocess=attach_metadata,
        )
        return note

    targets: list[tuple[Requirement, str]] = [
        (req, model)
        for req in spec.requirements
        for model in router.models_for("researcher")
    ]
    results = await asyncio.gather(
        *(one(req, model) for req, model in targets),
        return_exceptions=True,
    )

    notes: list[ResearchNote] = []
    failures: list[ResearchFailure] = []
    for (req, model), result in zip(targets, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Research call failed (requirement_id=%s, model=%s): %s: %s",
                req.id,
                model,
                type(result).__name__,
                result,
            )
            failures.append(
                ResearchFailure(
                    requirement_id=req.id,
                    model=model,
                    error_type=type(result).__name__,
                    error_message=str(result) or type(result).__name__,
                )
            )
        else:
            notes.append(result)

    if not notes:
        summary = " | ".join(
            f"{f.requirement_id}/{f.model}: {f.error_type}: {f.error_message}" for f in failures
        )
        raise RuntimeError(f"All research calls failed: {summary}")

    return ResearchReport(notes=notes, failures=failures)
