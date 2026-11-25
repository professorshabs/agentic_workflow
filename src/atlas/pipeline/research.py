import asyncio
import logging
import time
from typing import Any

from atlas.events import publish
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

    # When a provider is down (e.g. no credit), several researcher models can fall back to the
    # same model. Asking one model the same question three times only triples the tokens.
    configured = router.models_for("researcher")
    models = list(dict.fromkeys(router.first_available(m, "researcher") for m in configured))
    if len(models) < len(configured):
        logger.warning(
            "Research: %d configured researcher models resolve to %d available model(s) %s; "
            "running each once per requirement.",
            len(configured),
            len(models),
            models,
        )
    targets: list[tuple[Requirement, str]] = [
        (req, model) for req in spec.requirements for model in models
    ]
    total = len(targets)
    completed = 0

    async def one(req: Requirement, model: str) -> ResearchNote:
        nonlocal completed
        started = time.monotonic()
        user = (
            f"{context}\nRequirement {req.id}: {req.title}\n{req.description}\n"
            f"Acceptance criteria: {req.acceptance_criteria}\n"
            "Return structured JSON matching the research schema."
        )

        def attach_metadata(data: dict[str, Any], used_model: str) -> dict[str, Any]:
            data["requirement_id"] = req.id
            data["model"] = used_model
            return data

        try:
            used_model, note = await complete_json_model(
                router,
                model=model,
                role="researcher",
                system=system,
                user=user,
                schema=ResearchNote,
                postprocess=attach_metadata,
            )
        except Exception as exc:
            completed += 1
            logger.info(
                "[research] %d/%d FAILED: %s (%s, %.1fs)",
                completed,
                total,
                req.id,
                type(exc).__name__,
                time.monotonic() - started,
            )
            publish(
                "item",
                "research",
                "failed",
                requirement_id=req.id,
                title=req.title,
                completed=completed,
                total=total,
                model=model,
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(time.monotonic() - started, 1),
            )
            raise
        completed += 1
        logger.info(
            "[research] %d/%d done: %s — %s (%s, %.1fs)",
            completed,
            total,
            req.id,
            req.title[:50],
            used_model,
            time.monotonic() - started,
        )
        publish(
            "item",
            "research",
            "passed",
            requirement_id=req.id,
            title=req.title,
            completed=completed,
            total=total,
            model=used_model,
            seconds=round(time.monotonic() - started, 1),
        )
        return note

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
