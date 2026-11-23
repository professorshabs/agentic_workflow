from atlas.router import ModelRouter
from atlas.schemas import ParsedSpec

from .util import complete_json_model, prompt


async def parse_spec(router: ModelRouter, spec_text: str) -> ParsedSpec:
    model = router.models_for("parser")[0]
    _, parsed = await complete_json_model(
        router,
        model=model,
        role="parser",
        system=prompt("parse"),
        user=spec_text,
        schema=ParsedSpec,
    )
    return parsed
