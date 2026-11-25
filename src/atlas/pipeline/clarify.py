"""Interactive and prefilled clarifying-question collection for specifications."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

from atlas.schemas import ClarificationItem, ClarificationReport, ParsedSpec, ResearchReport


def collect_clarifying_questions(
    spec: ParsedSpec,
    research_report: ResearchReport | None = None,
    *,
    max_questions: int = 8,
) -> list[ClarificationItem]:
    """Gather deduplicated clarifying questions from `spec` and `research_report`."""
    if max_questions <= 0:
        return []

    seen: set[str] = set()
    items: list[ClarificationItem] = []

    def _add(raw_q: str, *, source: str, requirement_id: str | None) -> None:
        if len(items) >= max_questions:
            return
        cleaned = raw_q.strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        items.append(
            ClarificationItem(
                question=cleaned,
                source="spec" if source == "spec" else "research",
                requirement_id=requirement_id,
            )
        )

    for question in spec.open_questions:
        _add(question, source="spec", requirement_id=None)

    if research_report is not None:
        for note in research_report.notes:
            for question in note.clarifying_questions:
                _add(question, source="research", requirement_id=note.requirement_id)

    return items


def load_answers_file(path: Path) -> dict[str, str]:
    """Load prefilled clarification answers (`{question_or_index: answer}`) from JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"Answers file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Answers file must contain a JSON object: {path}")
    return {str(k).strip(): str(v).strip() for k, v in raw.items() if v is not None}


def clarify_with_user(
    questions: list[ClarificationItem],
    *,
    ask_fn: Callable[[str], str] | None = None,
    prefilled_answers: dict[str, str] | None = None,
    interactive: bool = False,
) -> ClarificationReport:
    """Resolve answers from prefilled mapping or interactive prompts (never blocking non-TTY)."""
    raw_prefilled = prefilled_answers or {}
    prefilled_lower = {str(k).strip().lower(): str(v).strip() for k, v in raw_prefilled.items()}
    can_prompt = ask_fn is not None or (interactive and sys.stdin.isatty())
    prompter: Callable[[str], str] = ask_fn if ask_fn is not None else input

    resolved: list[ClarificationItem] = []
    total = len(questions)
    for idx, item in enumerate(questions, 1):
        answer = (
            raw_prefilled.get(item.question)
            or prefilled_lower.get(item.question.strip().lower())
            or prefilled_lower.get(str(idx))
            or item.answer
            or ""
        ).strip()

        if not answer and can_prompt:
            prefix = f"[{idx}/{total}]"
            if item.requirement_id:
                prefix = f"{prefix} ({item.requirement_id})"
            prompt_msg = f"{prefix} {item.question}\n> "
            try:
                answer = str(prompter(prompt_msg) or "").strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""

        resolved.append(
            ClarificationItem(
                question=item.question,
                answer=answer,
                source=item.source,
                requirement_id=item.requirement_id,
            )
        )

    return ClarificationReport(items=resolved, interactive=can_prompt)


def apply_clarifications_to_spec(
    spec: ParsedSpec, clarifications: ClarificationReport
) -> ParsedSpec:
    """Fold answered clarifications into `spec.assumptions` and clear resolved open questions."""
    new_assumptions = list(spec.assumptions)
    answered_keys: set[str] = set()

    for item in clarifications.items:
        answer = item.answer.strip()
        if not answer:
            continue
        question = item.question.strip()
        answered_keys.add(question.lower())
        entry = f"Clarification — Q: {question} | A: {answer}"
        if entry not in new_assumptions:
            new_assumptions.append(entry)

    if not answered_keys:
        return spec

    remaining_open = [
        q for q in spec.open_questions if q.strip().lower() not in answered_keys
    ]
    return spec.model_copy(
        update={
            "assumptions": new_assumptions,
            "open_questions": remaining_open,
        }
    )
