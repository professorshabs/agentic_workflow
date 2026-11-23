import json

from atlas.router import ModelRouter
from atlas.schemas import JudgeOutput, ParsedSpec, Plan, ResearchNote, ResearchReport

from .util import complete_json_model, prompt


async def judge(
    router: ModelRouter,
    spec: ParsedSpec,
    research_input: ResearchReport | list[ResearchNote],
) -> Plan:
    model = router.models_for("judge")[0]
    if isinstance(research_input, ResearchReport):
        notes = research_input.notes
        failures = research_input.failures
    else:
        notes = research_input
        failures = []

    research_payload = {
        "notes": [n.model_dump(mode="json") for n in notes],
        "failures": [f.model_dump(mode="json") for f in failures],
    }
    completeness_note = (
        f"RESEARCH ({len(notes)} notes, {len(failures)} failed calls — "
        "treat gaps with lower confidence):"
        if failures
        else f"RESEARCH ({len(notes)} notes, 0 failed calls):"
    )
    user = (
        f"SPECIFICATION:\n{spec.model_dump_json(indent=2)}\n\n"
        f"{completeness_note}\n"
        f"{json.dumps(research_payload, indent=2)}"
    )
    _, judged = await complete_json_model(
        router,
        model=model,
        role="judge",
        system=prompt("judge"),
        user=user,
        schema=JudgeOutput,
    )
    return Plan(
        spec=spec,
        decisions=judged.decisions,
        risks=judged.risks,
        tasks=judged.tasks,
    )
