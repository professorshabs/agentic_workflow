"""Local model discovery, suitability filtering, and tool-use readiness probes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Literal

import httpx

from atlas.schemas import StrictModel

logger = logging.getLogger("atlas.local_models")

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1"

_EXCLUDED_SUBSTRINGS = (
    "embed",
    "bge",
    "nomic-embed",
    "mxbai-embed",
    "rerank",
    "all-minilm",
    "snowflake-arctic-embed",
)
_PARAM_RE = re.compile(r"(?:^|[:\-_/\s])(\d+(?:\.\d+)?)\s*([bBmM])(?=$|[:\-_/\s])")


class LocalModelCandidate(StrictModel):
    """A locally served model candidate discovered from Ollama or LM Studio."""

    provider: Literal["ollama", "lm_studio"]
    name: str
    base_url: str
    parameter_size_b: float | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    supports_tools: bool | None = None
    loaded: bool = False
    tool_smoke_passed: bool | None = None


class LocalResolutionResult(StrictModel):
    """Outcome of resolving a ready local model for workspace builds."""

    selected: LocalModelCandidate | None = None
    candidates: list[LocalModelCandidate] = []
    rejected: dict[str, str] = {}
    errors: dict[str, str] = {}
    reason: str = ""


def system_memory_bytes() -> int | None:
    """Return total physical RAM in bytes when available from POSIX sysconf."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if (
            isinstance(page_size, int)
            and isinstance(phys_pages, int)
            and page_size > 0
            and phys_pages > 0
        ):
            return page_size * phys_pages
    except (AttributeError, ValueError, OSError):
        pass
    return None


def parse_parameter_billions(raw: str | None, fallback_name: str = "") -> float | None:
    """Normalize parameter sizes such as `'26B'` or `'500M'` to billions of parameters."""
    for candidate_text in (raw or "", fallback_name):
        text = candidate_text.strip()
        if not text:
            continue
        match = _PARAM_RE.search(text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return round(value / 1000.0 if unit == "m" else value, 4)
    return None


def _is_excluded_model(name: str, family: str = "") -> bool:
    combined = f"{name} {family}".lower()
    return any(marker in combined for marker in _EXCLUDED_SUBSTRINGS)


def _normalize_v1_url(url: str) -> str:
    clean = url.rstrip("/")
    return clean if clean.endswith("/v1") else f"{clean}/v1"


def _strip_v1_root(url: str) -> str:
    clean = url.rstrip("/")
    return clean[:-3] if clean.endswith("/v1") else clean


def _extract_context_from_model_info(model_info: Any) -> int | None:
    if not isinstance(model_info, dict):
        return None
    for key, val in model_info.items():
        if (
            str(key).endswith(".context_length")
            and isinstance(val, int)
            and not isinstance(val, bool)
        ):
            return val
    return None


async def probe_ollama(
    base_url: str = DEFAULT_OLLAMA_URL,
    *,
    timeout: float = 1.5,
) -> tuple[list[LocalModelCandidate], str | None]:
    """Probe Ollama `/api/tags`, `/api/ps`, and `/api/show` for capabilities and context."""
    root = _strip_v1_root(base_url)
    v1_url = _normalize_v1_url(root)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            tags_resp = await client.get(f"{root}/api/tags")
            tags_resp.raise_for_status()
            tags_data = tags_resp.json()

            loaded_info: dict[str, int | None] = {}
            try:
                ps_resp = await client.get(f"{root}/api/ps")
                if ps_resp.status_code == 200:
                    ps_data = ps_resp.json()
                    ps_models = ps_data.get("models") if isinstance(ps_data, dict) else None
                    for item in ps_models or []:
                        if isinstance(item, dict) and item.get("name"):
                            ctx = item.get("context_length")
                            loaded_info[str(item["name"])] = (
                                int(ctx)
                                if isinstance(ctx, int) and not isinstance(ctx, bool)
                                else None
                            )
            except Exception:
                pass

            candidates: list[LocalModelCandidate] = []
            raw_models = tags_data.get("models") if isinstance(tags_data, dict) else None
            for entry in raw_models or []:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or entry.get("model") or "").strip()
                if not name:
                    continue
                raw_details = entry.get("details")
                details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
                family = str(details.get("family") or "")
                if _is_excluded_model(name, family):
                    continue
                raw_param = str(details.get("parameter_size") or "").strip() or None
                raw_size = entry.get("size")
                size_bytes = (
                    int(raw_size)
                    if isinstance(raw_size, int) and not isinstance(raw_size, bool)
                    else None
                )
                raw_ctx = (
                    details.get("context_length")
                    or entry.get("context_length")
                    or loaded_info.get(name)
                )
                ctx_len = (
                    int(raw_ctx)
                    if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool)
                    else None
                )

                supports_tools: bool | None = None
                try:
                    show_resp = await client.post(f"{root}/api/show", json={"name": name})
                    if show_resp.status_code == 200:
                        show_data = show_resp.json()
                        if isinstance(show_data, dict):
                            caps = show_data.get("capabilities")
                            if isinstance(caps, list):
                                supports_tools = "tools" in {str(c).lower() for c in caps}
                            if ctx_len is None:
                                ctx_len = _extract_context_from_model_info(
                                    show_data.get("model_info")
                                )
                except Exception:
                    pass

                candidates.append(
                    LocalModelCandidate(
                        provider="ollama",
                        name=name,
                        base_url=v1_url,
                        parameter_size_b=parse_parameter_billions(raw_param, name),
                        size_bytes=size_bytes,
                        context_length=ctx_len,
                        supports_tools=supports_tools,
                        loaded=name in loaded_info,
                    )
                )
    except Exception as exc:
        return [], f"Ollama unreachable at {root}: {type(exc).__name__}: {exc}"

    return candidates, None


async def probe_lm_studio(
    base_url: str = DEFAULT_LM_STUDIO_URL,
    *,
    timeout: float = 1.5,
) -> tuple[list[LocalModelCandidate], str | None]:
    """Probe LM Studio `/v1/models` and optional `/api/v0/models` metadata."""
    v1_url = _normalize_v1_url(base_url)
    root = _strip_v1_root(v1_url)
    v0_by_id: dict[str, dict[str, Any]] = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{v1_url}/models")
            resp.raise_for_status()
            payload = resp.json()
            try:
                v0_resp = await client.get(f"{root}/api/v0/models")
                if v0_resp.status_code == 200:
                    v0_payload = v0_resp.json()
                    v0_items = v0_payload.get("data") if isinstance(v0_payload, dict) else None
                    for item in v0_items or []:
                        if isinstance(item, dict) and item.get("id"):
                            v0_by_id[str(item["id"]).strip()] = item
            except Exception:
                v0_by_id = {}
    except Exception as exc:
        return [], f"LM Studio unreachable at {v1_url}: {type(exc).__name__}: {exc}"

    candidates: list[LocalModelCandidate] = []
    raw_data = payload.get("data") if isinstance(payload, dict) else None
    for entry in raw_data or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("id") or "").strip()
        if not name or _is_excluded_model(name):
            continue
        v0 = v0_by_id.get(name) or {}
        model_type = str(v0.get("type") or entry.get("type") or "").strip().lower()
        if model_type == "embeddings":
            continue
        loaded = (
            bool(entry.get("loaded", False))
            or str(v0.get("state") or "").strip().lower() == "loaded"
        )
        raw_loaded_ctx = v0.get("loaded_context_length") if loaded else None
        if isinstance(raw_loaded_ctx, int) and not isinstance(raw_loaded_ctx, bool):
            ctx_len: int | None = raw_loaded_ctx
        else:
            raw_ctx = (
                entry.get("max_context_length")
                or entry.get("context_length")
                or v0.get("max_context_length")
                or v0.get("context_length")
            )
            ctx_len = (
                int(raw_ctx)
                if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool)
                else None
            )
        raw_size = entry.get("size_bytes") or entry.get("size")
        size_bytes = (
            int(raw_size)
            if isinstance(raw_size, int) and not isinstance(raw_size, bool)
            else None
        )
        raw_caps = (
            entry.get("capabilities")
            if entry.get("capabilities") is not None
            else v0.get("capabilities")
        )
        supports_tools: bool | None = None
        if isinstance(raw_caps, list):
            supports_tools = bool({"tools", "tool_use"} & {str(c).lower() for c in raw_caps})

        candidates.append(
            LocalModelCandidate(
                provider="lm_studio",
                name=name,
                base_url=v1_url,
                parameter_size_b=parse_parameter_billions(
                    str(entry.get("parameter_size") or "") or None, name
                ),
                size_bytes=size_bytes,
                context_length=ctx_len,
                supports_tools=supports_tools,
                loaded=loaded,
            )
        )
    return candidates, None


async def run_tool_smoke_check(
    candidate: LocalModelCandidate,
    *,
    timeout: float = 180.0,
) -> tuple[bool, str]:
    """Verify the local model responds cleanly to a minimal tool-capable chat request."""
    url = f"{candidate.base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": candidate.name,
        "messages": [
            {"role": "system", "content": "Call the ping tool exactly once."},
            {"role": "user", "content": "Use the ping tool now."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Respond to a readiness ping.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        "tool_choice": "required",
        "max_tokens": 256,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}: {resp.text[:240]}"
        body = resp.json()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices:
        return False, "No choices returned by local endpoint"
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return False, "Malformed message object in local response"
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False, "Local model did not call the required ping tool"
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "ping":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict):
            return True, "ok"
    return False, "Local model did not return a valid ping tool call"


def _matches_model(candidate: LocalModelCandidate, model_name: str) -> bool:
    target = model_name.strip().lower()
    name = candidate.name.lower()
    if name == target or (":" not in target and name == f"{target}:latest"):
        return True
    name_base = name.rsplit("/", 1)[-1]
    target_base = target.rsplit("/", 1)[-1]
    if "/" not in target and "/" in name:
        return name_base == target or (":" not in target and name_base == f"{target}:latest")
    if "/" in target and "/" not in name:
        return name == target_base or (
            ":" not in target_base and name == f"{target_base}:latest"
        )
    return False


def _check_suitability(
    candidate: LocalModelCandidate,
    *,
    require_tools: bool,
    min_context_tokens: int | None,
    max_model_size_gb: float | None,
    memory_budget_fraction: float | None,
    total_memory_bytes: int | None,
    max_parameter_size_b: float | None = None,
) -> str | None:
    if require_tools and candidate.supports_tools is False:
        return "capabilities do not include 'tools'"
    if (
        max_parameter_size_b is not None
        and candidate.parameter_size_b is not None
        and candidate.parameter_size_b > max_parameter_size_b
    ):
        return (
            f"{candidate.parameter_size_b:g}B parameters > max_parameter_size_b "
            f"{max_parameter_size_b:g}B"
        )
    if (
        min_context_tokens is not None
        and candidate.context_length is not None
        and candidate.context_length < min_context_tokens
    ):
        return f"context_length {candidate.context_length} < required {min_context_tokens}"
    if max_model_size_gb is not None and candidate.size_bytes is not None:
        max_bytes = int(max_model_size_gb * (1024**3))
        if candidate.size_bytes > max_bytes:
            gb = candidate.size_bytes / (1024**3)
            return f"size {gb:.1f} GB > max_model_size_gb {max_model_size_gb:.1f} GB"
    if (
        memory_budget_fraction is not None
        and total_memory_bytes is not None
        and candidate.size_bytes is not None
    ):
        budget = int(total_memory_bytes * memory_budget_fraction)
        estimated_bytes = int(candidate.size_bytes * 1.15)
        if estimated_bytes > budget:
            est_gb = estimated_bytes / (1024**3)
            budget_gb = budget / (1024**3)
            return (
                f"estimated weights+KV footprint {est_gb:.1f} GB exceeds RAM budget "
                f"{budget_gb:.1f} GB ({memory_budget_fraction:.0%})"
            )
    return None


SmokeCheckFn = Callable[[LocalModelCandidate], Awaitable[tuple[bool, str]]]
RuntimeContextFn = Callable[[LocalModelCandidate], Awaitable[int | None]]


def local_resolution_kwargs(builder_cfg: dict[str, Any]) -> dict[str, Any]:
    """`resolve_local_model` keyword arguments taken from a `local_harness` builder config.

    Used by the build stage, `atlas doctor` and `atlas detect-local` so all three pick the
    same model. Optional keys are passed only when set.
    """
    preferred = builder_cfg.get("preferred")
    kwargs: dict[str, Any] = {
        "preferred": list(preferred) if preferred else None,
        "require_tools": bool(builder_cfg.get("require_tools", True)),
        "min_context_tokens": builder_cfg.get("min_context_tokens"),
        "max_model_size_gb": builder_cfg.get("max_model_size_gb"),
        "memory_budget_fraction": builder_cfg.get("memory_budget_fraction"),
    }
    if builder_cfg.get("only_preferred"):
        kwargs["only_preferred"] = True
    if builder_cfg.get("max_parameter_size_b") is not None:
        kwargs["max_parameter_size_b"] = float(builder_cfg["max_parameter_size_b"])
    if builder_cfg.get("smoke_timeout_seconds") is not None:
        kwargs["smoke_timeout_seconds"] = float(builder_cfg["smoke_timeout_seconds"])
    return kwargs


async def ollama_runtime_context(
    candidate: LocalModelCandidate, *, timeout: float = 1.5
) -> int | None:
    """Context window Ollama actually loaded `candidate` with (`/api/ps`), if it is loaded.

    `/api/show` reports the model's trained maximum (often 128K), but Ollama serves
    OpenAI-compatible requests with its server setting (4096 by default unless
    `OLLAMA_CONTEXT_LENGTH` is set), and silently truncates anything longer.
    """
    root = _strip_v1_root(candidate.base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{root}/api/ps")
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None
    models = data.get("models") if isinstance(data, dict) else None
    for item in models or []:
        if not isinstance(item, dict):
            continue
        if candidate.name in (str(item.get("name") or ""), str(item.get("model") or "")):
            ctx = item.get("context_length")
            if isinstance(ctx, int) and not isinstance(ctx, bool):
                return ctx
    return None


async def lm_studio_runtime_context(
    candidate: LocalModelCandidate, *, timeout: float = 1.5
) -> int | None:
    """Context window LM Studio actually loaded `candidate` with, if it is loaded.

    LM Studio reports the model's trained maximum (256K for Gemma 4) as `max_context_length`,
    but a just-in-time load uses the model's default load setting (often 4096), and prompts
    longer than that get truncated, which makes an agent loop forever.
    """
    root = _strip_v1_root(candidate.base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{root}/api/v0/models")
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None
    items = data.get("data") if isinstance(data, dict) else None
    for item in items or []:
        if not isinstance(item, dict) or str(item.get("id") or "") != candidate.name:
            continue
        if str(item.get("state") or "").strip().lower() != "loaded":
            return None
        ctx = item.get("loaded_context_length")
        if isinstance(ctx, int) and not isinstance(ctx, bool):
            return ctx
    return None


async def local_runtime_context(candidate: LocalModelCandidate) -> int | None:
    """Loaded context window for an Ollama or LM Studio candidate (None when unknown)."""
    if candidate.provider == "ollama":
        return await ollama_runtime_context(candidate)
    if candidate.provider == "lm_studio":
        return await lm_studio_runtime_context(candidate)
    return None


def _context_fix_hint(candidate: LocalModelCandidate, min_context_tokens: int) -> str:
    if candidate.provider == "lm_studio":
        return (
            f"reload it with `lms unload --all && lms load {candidate.name} "
            f"--context-length {min_context_tokens}` (or set its default context length "
            "in LM Studio > My Models)"
        )
    return f"restart with `OLLAMA_CONTEXT_LENGTH={min_context_tokens} ollama serve`"


def _smoke_failure_hint(candidate: LocalModelCandidate, message: str) -> str:
    low = message.lower()
    if "insufficient system resources" in low or "out of memory" in low:
        if candidate.provider == "lm_studio":
            return f"{message} (free memory first: `lms unload --all`)"
        return f"{message} (free memory first: `ollama stop <other-model>`)"
    return message


async def resolve_local_model(
    *,
    explicit_model: str | None = None,
    providers: dict[str, str] | Iterable[str] = ("ollama", "lm_studio"),
    preferred: list[str] | None = None,
    require_tools: bool = True,
    min_context_tokens: int | None = None,
    max_model_size_gb: float | None = None,
    memory_budget_fraction: float | None = None,
    total_memory_bytes: int | None = None,
    require_tool_smoke_check: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    lm_studio_url: str = DEFAULT_LM_STUDIO_URL,
    probe_timeout: float = 1.5,
    smoke_check_fn: SmokeCheckFn | None = None,
    runtime_context_fn: RuntimeContextFn | None = None,
    only_preferred: bool = False,
    max_parameter_size_b: float | None = None,
    smoke_timeout_seconds: float | None = None,
) -> LocalResolutionResult:
    """Resolve an explicit or discovered local model checked for tools, context, and memory."""
    provider_urls: dict[str, str] = {}
    if isinstance(providers, dict):
        provider_urls = {str(k): str(v) for k, v in providers.items()}
    else:
        for prov in providers:
            if prov == "ollama":
                provider_urls["ollama"] = ollama_url
            elif prov == "lm_studio":
                provider_urls["lm_studio"] = lm_studio_url

    probe_tasks: list[tuple[str, Awaitable[tuple[list[LocalModelCandidate], str | None]]]] = []
    for prov, url in provider_urls.items():
        if prov == "ollama":
            probe_tasks.append(("ollama", probe_ollama(url, timeout=probe_timeout)))
        elif prov == "lm_studio":
            probe_tasks.append(("lm_studio", probe_lm_studio(url, timeout=probe_timeout)))

    if not probe_tasks:
        return LocalResolutionResult(reason="No local model providers are configured.")

    raw_results = await asyncio.gather(*(task for _, task in probe_tasks))
    all_candidates: list[LocalModelCandidate] = []
    errors: dict[str, str] = {}
    for (prov, _), (found, err) in zip(probe_tasks, raw_results, strict=True):
        all_candidates.extend(found)
        if err:
            errors[prov] = err

    if not all_candidates:
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        ctx = int(min_context_tokens or 32768)
        hint = (
            "Start LM Studio and load Gemma 4 12B "
            f"(`lms load google/gemma-4-12b-qat --context-length {ctx}`), or start Ollama "
            f"with `OLLAMA_CONTEXT_LENGTH={ctx} ollama serve` and `ollama pull gemma4:12b`."
        )
        return LocalResolutionResult(
            candidates=[],
            errors=errors,
            reason=f"No local generative models available. {hint}"
            + (f" ({detail})" if detail else ""),
        )

    ram_bytes = total_memory_bytes if total_memory_bytes is not None else system_memory_bytes()
    rejected: dict[str, str] = {}

    if explicit_model:
        matches = [c for c in all_candidates if _matches_model(c, explicit_model)]
        if not matches:
            available_names = ", ".join(c.name for c in all_candidates)
            return LocalResolutionResult(
                candidates=all_candidates,
                errors=errors,
                reason=(
                    f"Configured local_model '{explicit_model}' was not found on running local "
                    f"servers (`ollama pull {explicit_model}`). Available: {available_names}"
                ),
            )
        ordered: list[LocalModelCandidate] = []
        for cand in matches:
            bad_reason = _check_suitability(
                cand,
                require_tools=require_tools,
                min_context_tokens=min_context_tokens,
                max_model_size_gb=max_model_size_gb,
                memory_budget_fraction=memory_budget_fraction,
                total_memory_bytes=ram_bytes,
                max_parameter_size_b=max_parameter_size_b,
            )
            if bad_reason:
                rejected[cand.name] = bad_reason
            else:
                ordered.append(cand)
        if not ordered:
            return LocalResolutionResult(
                candidates=all_candidates,
                rejected=rejected,
                errors=errors,
                reason=f"Configured local_model '{explicit_model}' failed suitability checks: "
                + "; ".join(f"{k}: {v}" for k, v in rejected.items()),
            )
    else:
        pref_list = list(preferred or [])
        pool = all_candidates
        if only_preferred and pref_list:
            pool = [c for c in all_candidates if any(_matches_model(c, p) for p in pref_list)]
            for cand in all_candidates:
                if cand not in pool:
                    rejected[cand.name] = "not in the preferred list (only_preferred: true)"
            if not pool:
                available_names = ", ".join(c.name for c in all_candidates) or "none"
                return LocalResolutionResult(
                    candidates=all_candidates,
                    rejected=rejected,
                    errors=errors,
                    reason=(
                        "None of the preferred local models are installed or served "
                        f"({', '.join(pref_list)}). Available: {available_names}"
                    ),
                )
        suitable: list[LocalModelCandidate] = []
        for cand in pool:
            bad_reason = _check_suitability(
                cand,
                require_tools=require_tools,
                min_context_tokens=min_context_tokens,
                max_model_size_gb=max_model_size_gb,
                memory_budget_fraction=memory_budget_fraction,
                total_memory_bytes=ram_bytes,
                max_parameter_size_b=max_parameter_size_b,
            )
            if bad_reason:
                rejected[cand.name] = bad_reason
            else:
                suitable.append(cand)

        if not suitable:
            return LocalResolutionResult(
                candidates=all_candidates,
                rejected=rejected,
                errors=errors,
                reason="No local models satisfied tools/context/memory checks: "
                + "; ".join(f"{k} ({v})" for k, v in rejected.items()),
            )

        def _rank_key(c: LocalModelCandidate) -> tuple[int, bool, float, int]:
            pref_score = 0
            for idx, pref in enumerate(pref_list):
                if _matches_model(c, pref):
                    pref_score = len(pref_list) - idx
                    break
            return (pref_score, c.loaded, c.parameter_size_b or 0.0, c.size_bytes or 0)

        ordered = sorted(suitable, key=_rank_key, reverse=True)

    smoke_fn: SmokeCheckFn
    if smoke_check_fn is not None:
        smoke_fn = smoke_check_fn
    elif smoke_timeout_seconds is not None:
        smoke_timeout = float(smoke_timeout_seconds)

        async def smoke_fn(cand: LocalModelCandidate) -> tuple[bool, str]:
            return await run_tool_smoke_check(cand, timeout=smoke_timeout)
    else:
        smoke_fn = run_tool_smoke_check
    if not require_tool_smoke_check:
        chosen = ordered[0]
        return LocalResolutionResult(
            selected=chosen,
            candidates=all_candidates,
            rejected=rejected,
            errors=errors,
            reason=f"Selected local {chosen.provider} model '{chosen.name}' (smoke check skipped).",
        )

    context_fn = runtime_context_fn or local_runtime_context
    smoke_failures: list[str] = []
    for candidate in ordered:
        passed, msg = await smoke_fn(candidate)
        candidate.tool_smoke_passed = passed
        if not passed:
            msg = _smoke_failure_hint(candidate, msg)
        if passed and min_context_tokens:
            # The smoke call loaded the model, so the server now reports the real window.
            runtime_ctx = await context_fn(candidate)
            if runtime_ctx is not None:
                candidate.context_length = runtime_ctx
                if runtime_ctx < min_context_tokens:
                    server = "Ollama" if candidate.provider == "ollama" else "LM Studio"
                    problem = (
                        f"{server} serves it with a {runtime_ctx}-token context "
                        f"(< required {min_context_tokens}); "
                        + _context_fix_hint(candidate, min_context_tokens)
                    )
                    rejected[candidate.name] = problem
                    smoke_failures.append(f"{candidate.name}: {problem}")
                    continue
        if passed:
            return LocalResolutionResult(
                selected=candidate,
                candidates=all_candidates,
                rejected=rejected,
                errors=errors,
                reason=(
                    f"Selected local {candidate.provider} model '{candidate.name}' "
                    "(passed tool smoke check)."
                ),
            )
        rejected[candidate.name] = f"tool smoke check failed: {msg}"
        smoke_failures.append(f"{candidate.name}: {msg}")

    return LocalResolutionResult(
        candidates=all_candidates,
        rejected=rejected,
        errors=errors,
        reason="Local model(s) failed readiness checks: " + "; ".join(smoke_failures),
    )
