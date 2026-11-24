import json
import logging
import re
from collections.abc import Callable
from importlib import resources
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.router import ModelRouter

logger = logging.getLogger("atlas.pipeline")
T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)


def prompt(name: str) -> str:
    return resources.files("atlas.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def extract_json(text: str) -> Any:
    """Parse JSON from plain output, fenced code blocks, or prose-wrapped JSON objects."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model response was empty")

    # 1. Try fenced code blocks first (handles prose before/after fences).
    for match in _FENCE_RE.finditer(cleaned):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 2. Try parsing the entire trimmed string directly.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Scan for the first valid JSON object using raw_decode (safe against trailing braces).
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = cleaned.find("{", idx)
        if start < 0:
            break
        try:
            obj, _ = decoder.raw_decode(cleaned, start)
            return obj
        except json.JSONDecodeError:
            idx = start + 1

    raise ValueError("Model response did not contain a valid JSON object")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON and guarantee the top-level value is a JSON object (dict)."""
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from model, got {type(parsed).__name__}")
    return parsed


async def complete_json_model(
    router: ModelRouter,
    *,
    model: str,
    role: str,
    system: str,
    user: str,
    schema: type[T],
    postprocess: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
) -> tuple[str, T]:
    """Call a model, extract a JSON object, validate against Pydantic, with repair retry."""
    max_repairs = router.repair_attempts
    current_user = user
    last_err: Exception | None = None

    for attempt in range(max_repairs + 1):
        backend_fn = getattr(router, "backend_for", None)
        complete_structured = getattr(router, "complete_structured", None)
        if (
            callable(backend_fn)
            and backend_fn(role).get("type") == "harness"
            and callable(complete_structured)
        ):
            used_model, raw_output = await complete_structured(
                model, system, current_user, role=role, json_schema=schema.model_json_schema()
            )
        else:
            used_model, raw_output = await router.complete(model, system, current_user, role=role)
        try:
            data = extract_json_object(raw_output)
            if postprocess is not None:
                data = postprocess(data, used_model)
            return used_model, schema.model_validate(data)
        except (ValueError, ValidationError) as exc:
            last_err = exc
            if attempt >= max_repairs:
                break
            logger.warning(
                "Validation failed for role=%s model=%s (repair %d/%d): %s",
                role,
                used_model,
                attempt + 1,
                max_repairs,
                exc,
            )
            current_user = (
                f"{user}\n\n"
                "Your previous response failed JSON/schema validation with the following error:\n"
                f"{exc}\n\n"
                "Return ONLY a valid JSON object matching the exact schema."
            )

    assert last_err is not None
    raise last_err
