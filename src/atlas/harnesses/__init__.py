"""Agentic harness adapters and pre-flight doctor diagnostics."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from atlas.local_models import (
    DEFAULT_LM_STUDIO_URL,
    DEFAULT_OLLAMA_URL,
    local_resolution_kwargs,
    resolve_local_model,
)
from atlas.skills import discover_skills

from .antigravity import AntigravityCLIAdapter, AntigravitySDKAdapter
from .base import (
    AuthStatus,
    HarnessAdapter,
    HarnessAuthError,
    HarnessError,
    HarnessInvocationResult,
    HarnessUnavailableError,
)
from .claude_code import ClaudeCodeAdapter
from .local_openai import LocalOpenAIAdapter

if TYPE_CHECKING:
    from atlas.router import ModelRouter


def get_harness_adapter(provider: str) -> HarnessAdapter:
    """Return the configured CLI harness adapter by provider name."""
    if provider == "antigravity":
        return AntigravityCLIAdapter()
    if provider == "claude_code":
        return ClaudeCodeAdapter()
    raise ValueError(f"Unsupported harness provider: {provider}")


def get_local_harness_adapter(name: str) -> AntigravitySDKAdapter | LocalOpenAIAdapter:
    """Return the configured local builder harness adapter."""
    if name == "local_openai":
        return LocalOpenAIAdapter()
    if name == "antigravity_sdk":
        return AntigravitySDKAdapter()
    raise ValueError(f"Unsupported local harness: {name}")


_API_KEY_ENVS_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


# LiteLLM prefixes for models served on this machine: they need no API key.
_LOCAL_MODEL_PROVIDERS = frozenset(
    {"ollama", "ollama_chat", "lm_studio", "hosted_vllm", "llamafile", "local"}
)


def _cloud_provider_for_model(model: str, default_type: str) -> str:
    prefix = model.split("/", 1)[0].strip().lower() if "/" in model else ""
    if default_type == "openrouter" and prefix in ("", "openrouter"):
        return "openrouter"
    return prefix or default_type or "openrouter"


async def run_doctor(
    router: ModelRouter,
    *,
    live: bool = False,
) -> dict[str, Any]:
    """Run live pre-flight checks for configured harnesses, local daemons, API keys, and skills."""
    backends = router.cfg.get("backends") or {}
    harness_configs: dict[tuple[str, str | None], None] = {}
    cloud_providers: set[str] = set()
    local_harnesses: set[str] = set()
    offload_configured = False

    def _collect_backend_deps(cfg: dict[str, Any], *, role_name: str | None = None) -> None:
        nonlocal offload_configured
        btype = str(cfg.get("type", "openrouter") or "openrouter")
        if btype == "harness":
            prov = str(cfg.get("provider") or "")
            if prov:
                auth_val = str(cfg["auth"]) if cfg.get("auth") is not None else None
                harness_configs[(prov, auth_val)] = None
        elif btype == "local_harness":
            local_harnesses.add(str(cfg.get("local_harness") or "antigravity_sdk"))
            offload = [t for t in (cfg.get("offload") or []) if isinstance(t, dict)]
            if offload:
                offload_configured = True
            for target in offload:
                _collect_backend_deps({"type": "harness", **target})
        elif btype in ("openrouter", "litellm"):
            explicit_model = str(cfg.get("model") or "").strip()
            if explicit_model:
                cloud_providers.add(_cloud_provider_for_model(explicit_model, btype))
            elif role_name and role_name in (router.cfg.get("roles") or {}):
                for m in router.models_for(role_name):
                    cloud_providers.add(_cloud_provider_for_model(m, btype))
            else:
                cloud_providers.add("openrouter" if btype == "openrouter" else "litellm")
        if cfg.get("fallback") == "router":
            fb_model = str(cfg.get("router_model") or cfg.get("fallback_model") or "").strip()
            if fb_model:
                cloud_providers.add(_cloud_provider_for_model(fb_model, "openrouter"))
            else:
                cloud_providers.add("openrouter")
        for entry in cfg.get("rotate") or []:
            if isinstance(entry, dict):
                _collect_backend_deps(entry)

    if not backends:
        cloud_providers.add("openrouter")
    else:
        for role_name, role_cfg in backends.items():
            if not isinstance(role_cfg, dict):
                continue
            if role_name == "supervisor" and not role_cfg.get("enabled", False):
                continue
            if role_name == "supervisor" and isinstance(role_cfg.get("reviewers"), list):
                for reviewer_cfg in role_cfg["reviewers"]:
                    if isinstance(reviewer_cfg, dict):
                        _collect_backend_deps(reviewer_cfg)
                continue
            _collect_backend_deps(role_cfg, role_name=role_name)

    checks: list[dict[str, Any]] = []
    all_ok = True

    for provider, auth_cfg in sorted(harness_configs, key=lambda item: (item[0], item[1] or "")):
        status = await get_harness_adapter(provider).check_auth(auth=auth_cfg)
        checks.append(status.model_dump(mode="json"))
        if not (status.available and status.authenticated):
            all_ok = False

    if local_harnesses:
        for lh_name in sorted(local_harnesses):
            lh_status = await get_local_harness_adapter(lh_name).check_auth()
            checks.append(lh_status.model_dump(mode="json"))
            if not (lh_status.available and lh_status.authenticated):
                all_ok = False

        builder_cfg = backends.get("builder") or {}
        providers = builder_cfg.get("local_providers") or {
            "ollama": DEFAULT_OLLAMA_URL,
            "lm_studio": DEFAULT_LM_STUDIO_URL,
        }
        local_res = await resolve_local_model(
            explicit_model=builder_cfg.get("local_model"),
            providers=providers,
            require_tool_smoke_check=live,
            **local_resolution_kwargs(builder_cfg),
        )
        local_ok = local_res.selected is not None
        if not local_ok:
            all_ok = False
        checks.append(
            {
                "provider": "local_model_daemon",
                "available": bool(local_res.candidates),
                "authenticated": True,
                "auth_mode": "local",
                "ready": local_ok,
                "offload_configured": offload_configured,
                "selected_model": local_res.selected.name if local_res.selected else None,
                "candidates_count": len(local_res.candidates),
                "rejected": local_res.rejected,
                "details": local_res.reason,
            }
        )

    for prov in sorted(cloud_providers):
        if prov in _LOCAL_MODEL_PROVIDERS:
            checks.append(
                {
                    "provider": prov,
                    "available": True,
                    "authenticated": True,
                    "auth_mode": "local",
                    "details": "Model served on this machine; no API key needed.",
                    "login_hint": "",
                }
            )
            continue
        env_candidates = _API_KEY_ENVS_BY_PROVIDER.get(prov, (f"{prov.upper()}_API_KEY",))
        found_var = next((v for v in env_candidates if os.environ.get(v, "").strip()), None)
        has_key = found_var is not None
        if not has_key:
            all_ok = False
        env_hint = " or ".join(env_candidates)
        checks.append(
            {
                "provider": prov,
                "available": True,
                "authenticated": has_key,
                "auth_mode": "api_key",
                "details": (
                    f"{found_var} is set."
                    if found_var is not None
                    else f"{env_hint} is missing from environment."
                ),
                "login_hint": "" if has_key else f"Export {env_hint} in `.env` or shell.",
            }
        )

    skills_cfg = router.skills_config()
    skills_summary: dict[str, Any] = {"configured": bool(skills_cfg), "resolved": []}
    if skills_cfg:
        discovered = discover_skills(
            list(skills_cfg.get("sources") or []),
            allowlist=list(skills_cfg["allowlist"]) if "allowlist" in skills_cfg else None,
        )
        skills_summary["resolved"] = sorted(discovered.keys())

    return {
        "healthy": all_ok,
        "checks": checks,
        "skills": skills_summary,
    }


__all__ = [
    "AntigravityCLIAdapter",
    "AntigravitySDKAdapter",
    "AuthStatus",
    "ClaudeCodeAdapter",
    "HarnessAdapter",
    "HarnessAuthError",
    "HarnessError",
    "HarnessInvocationResult",
    "HarnessUnavailableError",
    "LocalOpenAIAdapter",
    "get_harness_adapter",
    "get_local_harness_adapter",
    "run_doctor",
]
