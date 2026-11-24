"""Centralized LiteLLM access with role-specific fallback, timeout, telemetry and concurrency."""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
import yaml

from atlas.schemas import TokenUsageSummary
from atlas.telemetry import extract_usage_metrics, summarize_usage

logger = logging.getLogger("atlas.router")

# LiteLLM prints a "Provider List" banner after every failed call; Atlas logs failures itself.
litellm.suppress_debug_info = True

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class TokenBudgetExceededError(RuntimeError):
    """Raised when a configured token or cost limit has been reached."""


class OutputTruncatedError(RuntimeError):
    """Raised when a model stops at the max_tokens limit, so its output is incomplete."""


_TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}
# A bad key (401) or no credit left (402) fails for every model of that provider.
_PROVIDER_DOWN_STATUSES = {401, 402}
# Not allowed (403) or unknown model (404): skip that model for the rest of the run.
_MODEL_DOWN_STATUSES = {403, 404}
# Client errors that the same request will hit again: do not retry the same model.
_NO_RETRY_STATUSES = {400, 401, 402, 403, 404, 422}
_ALLOWED_BACKEND_TYPES = {"openrouter", "litellm", "harness", "local_harness"}
_ALLOWED_HARNESS_PROVIDERS = {"antigravity", "claude_code"}
_ALLOWED_LOCAL_HARNESSES = {"antigravity_sdk"}
_ALLOWED_LOCAL_PROVIDERS = {"ollama", "lm_studio"}
_ALLOWED_AUTH_MODES = {"subscription", "oauth_token", "api"}
_ALLOWED_SUPERVISOR_TOOLS = {"Read", "Grep", "Glob"}


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int) and not isinstance(code, bool):
            return code
    text = str(exc).lower()
    if "requires more credits" in text or '"code":402' in text.replace(" ", ""):
        return 402
    return None


def _provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


def resolve_default_config() -> Path:
    """Resolve config path from ATLAS_CONFIG env var, CWD, or package-relative default."""
    env_path = os.environ.get("ATLAS_CONFIG")
    if env_path:
        return Path(env_path)
    cwd_candidate = Path.cwd() / "config" / "models.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate
    return DEFAULT_CONFIG


class ModelRouter:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else resolve_default_config()
        if not self.path.is_file():
            raise FileNotFoundError(f"Model config not found: {self.path}")
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Model config must be a YAML mapping: {self.path}")
        self.cfg: dict[str, Any] = raw
        self._validate_config()
        self._semaphores: dict[int, asyncio.Semaphore] = {}
        self.call_log: list[dict[str, Any]] = []
        # Models or whole providers ("openrouter/*") that failed in a way retrying cannot fix.
        self._unavailable: dict[str, str] = {}
        self._active_api_keys: dict[str, str] = {}
        self._used_fallback_keys: set[str] = set()

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(int(self.cfg["limits"]["max_parallel_calls"]))
            self._semaphores[loop_id] = sem
        return sem

    def _get_role_semaphore(self, role: str | None) -> asyncio.Semaphore:
        backend = self.backend_for(role)
        max_parallel = backend.get("max_parallel")
        if max_parallel is None:
            return self._get_semaphore()
        limit = int(max_parallel)
        loop_id = id(asyncio.get_running_loop())
        role_map = self.__dict__.setdefault("_role_semaphores", {})
        key = (loop_id, role or "")
        sem = role_map.get(key)
        if sem is None:
            sem = asyncio.Semaphore(limit)
            role_map[key] = sem
        return sem

    def _validate_config(self) -> None:
        for key in ("roles", "limits"):
            if key not in self.cfg or not isinstance(self.cfg[key], dict):
                raise ValueError(f"Missing or invalid '{key}' mapping in {self.path}")

        if not self.cfg["roles"]:
            raise ValueError(f"'roles' must define at least one role in {self.path}")

        for role, models in self.cfg["roles"].items():
            if not isinstance(models, list) or not models or not all(
                isinstance(m, str) and m.strip() for m in models
            ):
                raise ValueError(f"Role '{role}' must be a non-empty list of model strings")

        role_fallbacks = self.cfg.get("role_fallbacks", {})
        if not isinstance(role_fallbacks, dict):
            raise ValueError("'role_fallbacks' must be a mapping")
        for role, models in role_fallbacks.items():
            if not isinstance(models, list) or not all(
                isinstance(m, str) and m.strip() for m in models
            ):
                raise ValueError(f"role_fallbacks['{role}'] must be a list of model strings")

        fallbacks = self.cfg.get("fallbacks", [])
        if not isinstance(fallbacks, list) or not all(
            isinstance(m, str) and m.strip() for m in fallbacks
        ):
            raise ValueError("'fallbacks' must be a list of model strings")

        limits = self.cfg["limits"]
        if int(limits.get("max_parallel_calls", 0)) < 1:
            raise ValueError("limits.max_parallel_calls must be >= 1")
        if int(limits.get("max_tokens", 0)) < 1:
            raise ValueError("limits.max_tokens must be >= 1")
        if float(limits.get("timeout_seconds", 0)) <= 0:
            raise ValueError("limits.timeout_seconds must be > 0")
        if int(limits.get("attempts_per_model", 1)) < 1:
            raise ValueError("limits.attempts_per_model must be >= 1")
        if int(limits.get("repair_attempts", 1)) < 0:
            raise ValueError("limits.repair_attempts must be >= 0")
        if int(limits.get("max_fix_rounds", 3)) < 0:
            raise ValueError("limits.max_fix_rounds must be >= 0")
        if float(limits.get("verify_timeout_seconds", 180)) <= 0:
            raise ValueError("limits.verify_timeout_seconds must be > 0")
        max_total_tokens = limits.get("max_total_tokens")
        if max_total_tokens is not None and int(max_total_tokens) < 1:
            raise ValueError("limits.max_total_tokens must be >= 1")
        max_cost_usd = limits.get("max_cost_usd")
        if max_cost_usd is not None and float(max_cost_usd) <= 0:
            raise ValueError("limits.max_cost_usd must be > 0")
        by_role = limits.get("max_tokens_by_role") or {}
        if not isinstance(by_role, dict):
            raise ValueError("limits.max_tokens_by_role must be a mapping of role -> tokens")
        for role, value in by_role.items():
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"limits.max_tokens_by_role['{role}'] must be >= 1")
        timeouts_by_role = limits.get("timeout_seconds_by_role") or {}
        if not isinstance(timeouts_by_role, dict):
            raise ValueError("limits.timeout_seconds_by_role must be a mapping of role -> seconds")
        for role, value in timeouts_by_role.items():
            if isinstance(value, bool) or float(value) <= 0:
                raise ValueError(f"limits.timeout_seconds_by_role['{role}'] must be > 0")
        verification = self.cfg.get("verification", {})
        if not isinstance(verification, dict):
            raise ValueError("'verification' must be a mapping")

        backends = self.cfg.get("backends")
        if backends is not None:
            if not isinstance(backends, dict):
                raise ValueError("'backends' must be a mapping of role -> config")
            for role, bcfg in backends.items():
                if not isinstance(bcfg, dict):
                    raise ValueError(f"backends['{role}'] must be a mapping")
                btype = str(bcfg.get("type", "openrouter")).strip()
                if btype not in _ALLOWED_BACKEND_TYPES:
                    raise ValueError(
                        f"backends['{role}'].type '{btype}' must be one of "
                        f"{sorted(_ALLOWED_BACKEND_TYPES)}"
                    )
                if btype == "harness":
                    prov = bcfg.get("provider")
                    if prov not in _ALLOWED_HARNESS_PROVIDERS:
                        raise ValueError(
                            f"backends['{role}'].provider must be one of "
                            f"{sorted(_ALLOWED_HARNESS_PROVIDERS)}"
                        )
                    if prov == "claude_code" and bcfg.get("read_only") is False:
                        raise ValueError(
                            f"backends['{role}']: claude_code backend is strictly read-only"
                        )
                    if prov == "claude_code" and bcfg.get("tools") is not None:
                        tools = bcfg["tools"]
                        if not isinstance(tools, list) or any(
                            t not in _ALLOWED_SUPERVISOR_TOOLS for t in tools
                        ):
                            raise ValueError(
                                f"backends['{role}'].tools must be a subset of "
                                f"{sorted(_ALLOWED_SUPERVISOR_TOOLS)}"
                            )
                    auth_val = bcfg.get("auth")
                    if auth_val is not None and auth_val not in _ALLOWED_AUTH_MODES:
                        raise ValueError(
                            f"backends['{role}'].auth must be one of {sorted(_ALLOWED_AUTH_MODES)}"
                        )
                    fb_mode = bcfg.get("fallback")
                    if fb_mode is not None and fb_mode != "router":
                        raise ValueError(
                            f"backends['{role}'].fallback must be 'router' or null"
                        )
                    if bcfg.get("max_parallel") is not None and int(bcfg["max_parallel"]) < 1:
                        raise ValueError(f"backends['{role}'].max_parallel must be >= 1")
                    if (
                        bcfg.get("max_budget_usd") is not None
                        and float(bcfg["max_budget_usd"]) <= 0
                    ):
                        raise ValueError(f"backends['{role}'].max_budget_usd must be > 0")
                elif btype == "local_harness":
                    lh = bcfg.get("local_harness", "antigravity_sdk")
                    if lh not in _ALLOWED_LOCAL_HARNESSES:
                        raise ValueError(
                            f"backends['{role}'].local_harness must be one of "
                            f"{sorted(_ALLOWED_LOCAL_HARNESSES)}"
                        )
                    if bcfg.get("allow_cloud_fallback"):
                        raise ValueError(
                            f"backends['{role}'].allow_cloud_fallback is not supported; "
                            "builder is local-only"
                        )
                    lp = bcfg.get("local_providers", ["ollama", "lm_studio"])
                    if isinstance(lp, dict):
                        if not lp or any(k not in _ALLOWED_LOCAL_PROVIDERS for k in lp):
                            raise ValueError(
                                f"backends['{role}'].local_providers keys must be in "
                                f"{sorted(_ALLOWED_LOCAL_PROVIDERS)}"
                            )
                    elif not isinstance(lp, list) or not lp or any(
                        p not in _ALLOWED_LOCAL_PROVIDERS for p in lp
                    ):
                        raise ValueError(
                            f"backends['{role}'].local_providers must be a non-empty dict or "
                            f"list of {sorted(_ALLOWED_LOCAL_PROVIDERS)}"
                        )
                elif role == "fixer" and bcfg.get("rotate") is not None:
                    rot = bcfg["rotate"]
                    if not isinstance(rot, list) or not rot or not all(
                        isinstance(m, str) and m.strip() for m in rot
                    ):
                        raise ValueError("backends['fixer'].rotate must be a non-empty list")

        limits_extra = self.cfg.get("limits_extra")
        if limits_extra is not None:
            if not isinstance(limits_extra, dict):
                raise ValueError("'limits_extra' must be a mapping")
            max_run_tok = limits_extra.get("max_total_tokens_per_run")
            if max_run_tok is not None and (
                isinstance(max_run_tok, bool) or int(max_run_tok) < 1
            ):
                raise ValueError("limits_extra.max_total_tokens_per_run must be >= 1")

        skills = self.cfg.get("skills")
        if skills is not None:
            if not isinstance(skills, dict):
                raise ValueError("'skills' must be a mapping")
            sources = skills.get("sources", [])
            if not isinstance(sources, list) or not all(
                isinstance(s, str) and s.strip() for s in sources
            ):
                raise ValueError("skills.sources must be a list of directory strings")
            allowlist = skills.get("allowlist")
            if allowlist is not None:
                if not isinstance(allowlist, list) or not all(
                    isinstance(a, str) and a.strip() for a in allowlist
                ):
                    raise ValueError("skills.allowlist must be a list of skill names")
            by_role = skills.get("by_role", {})
            if not isinstance(by_role, dict):
                raise ValueError("skills.by_role must be a mapping of role -> skill names")
            allowed_set = set(allowlist) if allowlist is not None else None
            for role, names in by_role.items():
                if not isinstance(names, list) or not all(
                    isinstance(n, str) and n.strip() for n in names
                ):
                    raise ValueError(f"skills.by_role['{role}'] must be a list of skill names")
                if allowed_set is not None:
                    disallowed = [n for n in names if n not in allowed_set]
                    if disallowed:
                        raise ValueError(
                            f"skills.by_role['{role}'] references non-allowlisted skill(s): "
                            f"{disallowed}"
                        )

    @property
    def repair_attempts(self) -> int:
        return int(self.cfg["limits"].get("repair_attempts", 1))

    @property
    def max_fix_rounds(self) -> int:
        return int(self.cfg["limits"].get("max_fix_rounds", 3))

    @property
    def verify_timeout_seconds(self) -> float:
        return float(self.cfg["limits"].get("verify_timeout_seconds", 180))

    @property
    def max_total_tokens(self) -> int | None:
        extra = (self.cfg.get("limits_extra") or {}).get("max_total_tokens_per_run")
        if extra is not None:
            return int(extra)
        val = self.cfg["limits"].get("max_total_tokens")
        return int(val) if val is not None else None

    @property
    def max_cost_usd(self) -> float | None:
        val = self.cfg["limits"].get("max_cost_usd")
        return float(val) if val is not None else None

    @property
    def require_requirement_tests(self) -> bool:
        """Fail verification when a functional requirement has no test mentioning its ID."""
        return bool((self.cfg.get("verification") or {}).get("require_requirement_tests", False))

    @property
    def require_product_contract(self) -> bool:
        """Fail verification when `app/<spec-name>/` is not self-contained."""
        return bool((self.cfg.get("verification") or {}).get("require_product_contract", False))

    def max_tokens_for(self, role: str | None) -> int:
        """Output token limit for a role: limits.max_tokens_by_role, else limits.max_tokens."""
        by_role = self.cfg["limits"].get("max_tokens_by_role") or {}
        if role is not None and role in by_role:
            return int(by_role[role])
        return int(self.cfg["limits"]["max_tokens"])

    def timeout_for(self, role: str | None) -> float:
        """Per-call timeout: limits.timeout_seconds_by_role, else limits.timeout_seconds."""
        by_role = self.cfg["limits"].get("timeout_seconds_by_role") or {}
        if role is not None and role in by_role:
            return float(by_role[role])
        return float(self.cfg["limits"].get("timeout_seconds", 90))

    def models_for(self, role: str, *, fallback_role: str | None = None) -> list[str]:
        if role in self.cfg["roles"]:
            return list(self.cfg["roles"][role])
        if fallback_role is not None and fallback_role in self.cfg["roles"]:
            return list(self.cfg["roles"][fallback_role])
        raise ValueError(f"Unknown model role: {role}")

    def usage_summary(self) -> TokenUsageSummary:
        return summarize_usage(self.call_log)

    def _check_budget(self) -> None:
        token_cap = self.max_total_tokens
        cost_cap = self.max_cost_usd
        if token_cap is None and cost_cap is None:
            return
        summary = self.usage_summary()
        if token_cap is not None and summary.total_tokens >= token_cap:
            raise TokenBudgetExceededError(
                f"Token budget exceeded ({summary.total_tokens} >= {token_cap})"
            )
        if cost_cap is not None and summary.cost_usd is not None and summary.cost_usd >= cost_cap:
            raise TokenBudgetExceededError(
                f"Cost budget exceeded (${summary.cost_usd:.6f} >= ${cost_cap:.6f})"
            )

    def unavailable_reason(self, model: str) -> str | None:
        """Why `model` is skipped for the rest of this run, or None if it can be used."""
        blocked: dict[str, str] = getattr(self, "_unavailable", {})
        return blocked.get(model) or blocked.get(f"{_provider(model)}/*")

    def first_available(self, model: str, role: str | None = None) -> str:
        """The model a call for `model` would actually reach right now (skips blocked ones)."""
        for candidate in self.candidates_for(model, role=role):
            if self.unavailable_reason(candidate) is None:
                return candidate
        return model

    def _mark_unavailable(self, model: str, status: int, exc: BaseException) -> None:
        blocked = self.__dict__.setdefault("_unavailable", {})
        key = f"{_provider(model)}/*" if status in _PROVIDER_DOWN_STATUSES else model
        if key not in blocked:
            blocked[key] = f"HTTP {status}: {type(exc).__name__}: {str(exc)[:300]}"
            what = "no credit or bad key" if status in _PROVIDER_DOWN_STATUSES else "not usable"
            logger.warning("Skipping %s for the rest of this run (HTTP %s, %s).", key, status, what)

    def _switch_to_fallback_key(self, model: str, status: int) -> bool:
        prov = _provider(model)
        used_fallback: set[str] = self.__dict__.setdefault("_used_fallback_keys", set())
        if prov in used_fallback:
            return False
        env_name = f"{prov.upper()}_API_KEY_FALLBACK"
        fallback_key = os.environ.get(env_name, "").strip()
        primary_key = os.environ.get(f"{prov.upper()}_API_KEY", "").strip()
        if fallback_key and fallback_key != primary_key:
            used_fallback.add(prov)
            self.__dict__.setdefault("_active_api_keys", {})[prov] = fallback_key
            logger.warning(
                "Provider '%s' returned HTTP %s; switching to %s for the rest of this run.",
                prov,
                status,
                env_name,
            )
            return True
        return False

    def candidates_for(self, model: str, role: str | None = None) -> list[str]:
        """Return deduplicated candidates: requested model -> role fallbacks -> global fallbacks."""
        role_fb = self.cfg.get("role_fallbacks", {}).get(role, []) if role else []
        global_fb = self.cfg.get("fallbacks", [])
        return list(dict.fromkeys([model, *role_fb, *global_fb]))

    def backend_for(self, role: str | None) -> dict[str, Any]:
        """Return backend config for `role`, defaulting safely to OpenRouter/LiteLLM."""
        backends = self.cfg.get("backends") or {}
        if role is not None and isinstance(backends.get(role), dict):
            return dict(backends[role])
        return {"type": "openrouter"}

    def skills_config(self) -> dict[str, Any]:
        """Return parsed `skills` configuration mapping, or empty dict when omitted."""
        skills = self.cfg.get("skills")
        return dict(skills) if isinstance(skills, dict) else {}

    def record_harness_call(
        self,
        *,
        role: str,
        requested_model: str,
        used_model: str,
        harness: str,
        harness_version: str | None,
        elapsed_ms: float,
        auth_mode: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float | None = None,
        cost_source: str = "unavailable",
        usage_reported: bool = False,
        error: str | None = None,
    ) -> None:
        """Append one harness invocation's honest telemetry into `call_log`."""
        self.call_log.append(
            {
                "role": role,
                "requested_model": requested_model,
                "used_model": used_model,
                "harness": harness,
                "harness_version": harness_version,
                **({"auth_mode": auth_mode} if auth_mode else {}),
                "attempts": 1,
                "elapsed_ms": round(float(elapsed_ms), 1),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cached_tokens": int(cached_tokens),
                "reasoning_tokens": int(reasoning_tokens),
                "total_tokens": int(total_tokens),
                "cost_usd": cost_usd,
                "cost_source": cost_source,
                "usage_reported": usage_reported,
                "timestamp": datetime.now(UTC).isoformat(),
                **({"error": error} if error else {}),
            }
        )

    async def _complete_with_harness(
        self,
        backend: dict[str, Any],
        model: str,
        system: str,
        user: str,
        *,
        role: str | None,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        from atlas.harnesses import HarnessError, get_harness_adapter

        provider = str(backend["provider"])
        target_model = str(backend.get("model") or model)
        timeout = float(backend.get("timeout_seconds") or self.timeout_for(role))
        sem = self._get_role_semaphore(role)
        started = time.monotonic()
        adapter = get_harness_adapter(provider)
        try:
            async with sem:
                self._check_budget()
                result = await adapter.run_structured(
                    role=role or "unknown",
                    model=target_model,
                    system=system,
                    user=user,
                    json_schema=json_schema,
                    timeout_seconds=timeout,
                    auth=backend.get("auth"),
                    max_budget_usd=(
                        float(backend["max_budget_usd"])
                        if backend.get("max_budget_usd") is not None
                        else None
                    ),
                    tools=list(backend["tools"]) if backend.get("tools") is not None else None,
                )
            self.record_harness_call(
                role=role or "unknown",
                requested_model=target_model,
                used_model=result.used_model,
                harness=result.harness,
                harness_version=result.harness_version,
                auth_mode=result.auth_mode,
                elapsed_ms=result.elapsed_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cached_tokens=result.cached_tokens,
                reasoning_tokens=result.reasoning_tokens,
                total_tokens=result.total_tokens,
                cost_usd=result.cost_usd,
                cost_source=result.cost_source,
                usage_reported=result.usage_reported,
            )
            return result.used_model, result.text
        except TokenBudgetExceededError:
            raise
        except Exception as exc:
            self.record_harness_call(
                role=role or "unknown",
                requested_model=target_model,
                used_model=target_model,
                harness=provider,
                harness_version=None,
                elapsed_ms=round((time.monotonic() - started) * 1000, 1),
                cost_usd=None,
                cost_source="unavailable",
                usage_reported=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, HarnessError) and backend.get("fallback") == "router":
                logger.warning(
                    "Harness failed for role '%s' (%s); falling back to LiteLLM ModelRouter.",
                    role,
                    exc,
                )
                return await self._complete_litellm(model, system, user, role=role)
            raise

    async def complete_structured(
        self,
        model: str,
        system: str,
        user: str,
        *,
        role: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Dispatch structured JSON request with optional JSON Schema enforcement for harnesses."""
        backend = self.backend_for(role)
        if backend.get("type") == "harness":
            return await self._complete_with_harness(
                backend, model, system, user, role=role, json_schema=json_schema
            )
        return await self._complete_litellm(model, system, user, role=role)

    async def complete(
        self, model: str, system: str, user: str, role: str | None = None
    ) -> tuple[str, str]:
        """Try the configured backend (harness or LiteLLM model + fallbacks)."""
        backend = self.backend_for(role)
        if backend.get("type") == "harness":
            return await self._complete_with_harness(backend, model, system, user, role=role)
        return await self._complete_litellm(model, system, user, role=role)

    async def _complete_litellm(
        self, model: str, system: str, user: str, role: str | None = None
    ) -> tuple[str, str]:
        last_err: Exception | None = None
        attempts = int(self.cfg["limits"].get("attempts_per_model", 1))
        timeout = self.timeout_for(role)
        candidates = self.candidates_for(model, role=role)
        max_tokens = self.max_tokens_for(role)
        sem = self._get_semaphore()
        total_attempts = 0
        started = time.monotonic()

        def record(response: Any, candidate: str, error: str | None = None) -> None:
            # Every response the provider returned is logged, including unusable ones:
            # they are normally billed, so leaving them out would under-report spend.
            self.call_log.append(
                {
                    "role": role,
                    "requested_model": model,
                    "used_model": candidate,
                    "attempts": total_attempts,
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                    "timestamp": datetime.now(UTC).isoformat(),
                    **extract_usage_metrics(response),
                    **({"error": error} if error else {}),
                }
            )

        for candidate in candidates:
            skip_reason = self.unavailable_reason(candidate)
            if skip_reason is not None:
                last_err = last_err or RuntimeError(f"{candidate} skipped earlier: {skip_reason}")
                continue
            switched_for_candidate = False
            attempt = 0
            while attempt < attempts:
                total_attempts += 1
                response: Any = None
                extra_kwargs: dict[str, Any] = {}
                active_key = self.__dict__.get("_active_api_keys", {}).get(_provider(candidate))
                if active_key:
                    extra_kwargs["api_key"] = active_key
                try:
                    async with sem:
                        # Checked here, not when the call is queued: parallel calls (research,
                        # judge) all queue at once, before any of them has spent anything.
                        self._check_budget()
                        response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=candidate,
                                max_tokens=max_tokens,
                                timeout=timeout,
                                messages=[
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                                **extra_kwargs,
                            ),
                            timeout=timeout + 5,
                        )
                    choices = getattr(response, "choices", None)
                    if not choices:
                        raise ValueError(f"Model {candidate} returned no choices")
                    if getattr(choices[0], "finish_reason", None) in _TRUNCATED_FINISH_REASONS:
                        raise OutputTruncatedError(
                            f"Model {candidate} stopped at the output limit of {max_tokens} "
                            f"tokens for role '{role}', so its answer is incomplete. Raise "
                            f"limits.max_tokens_by_role.{role} in the model config, or ask "
                            "for less output. Fallback models were not tried: they share the "
                            "same limit and prompt, so each would be cut off and billed too."
                        )
                    content = getattr(choices[0].message, "content", None)
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError(f"Model {candidate} returned empty content")

                    record(response, candidate)
                    return candidate, content.strip()
                except TokenBudgetExceededError:
                    raise  # a spending cap is not a model failure: do not try fallbacks
                except OutputTruncatedError as exc:
                    record(response, candidate, error=str(exc))
                    raise exc from None
                except Exception as exc:  # model/provider failures are isolated here
                    last_err = exc
                    if response is not None:
                        record(response, candidate, error=f"{type(exc).__name__}: {exc}")
                    status = _status_code(exc)
                    if (
                        status in _PROVIDER_DOWN_STATUSES
                        and not switched_for_candidate
                        and self._switch_to_fallback_key(candidate, status)
                    ):
                        switched_for_candidate = True
                        continue
                    if status in _PROVIDER_DOWN_STATUSES | _MODEL_DOWN_STATUSES:
                        self._mark_unavailable(candidate, status, exc)
                    logger.warning(
                        "Model call failed (role=%s, candidate=%s, attempt=%d/%d): %s: %s",
                        role,
                        candidate,
                        attempt + 1,
                        attempts,
                        type(exc).__name__,
                        exc,
                    )
                    if status in _NO_RETRY_STATUSES:
                        break  # the same request would fail the same way; try the next model
                    attempt += 1
                    if attempt < attempts:
                        await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(
            f"All configured models failed for role request '{model}'. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        ) from last_err
