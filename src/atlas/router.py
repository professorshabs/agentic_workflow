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

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class TokenBudgetExceededError(RuntimeError):
    """Raised when a configured token or cost limit has been reached."""


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

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._semaphores.get(loop_id)
        if sem is None:
            sem = asyncio.Semaphore(int(self.cfg["limits"]["max_parallel_calls"]))
            self._semaphores[loop_id] = sem
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
        val = self.cfg["limits"].get("max_total_tokens")
        return int(val) if val is not None else None

    @property
    def max_cost_usd(self) -> float | None:
        val = self.cfg["limits"].get("max_cost_usd")
        return float(val) if val is not None else None

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

    def candidates_for(self, model: str, role: str | None = None) -> list[str]:
        """Return deduplicated candidates: requested model -> role fallbacks -> global fallbacks."""
        role_fb = self.cfg.get("role_fallbacks", {}).get(role, []) if role else []
        global_fb = self.cfg.get("fallbacks", [])
        return list(dict.fromkeys([model, *role_fb, *global_fb]))

    async def complete(
        self, model: str, system: str, user: str, role: str | None = None
    ) -> tuple[str, str]:
        """Try the requested model followed by role-specific then global fallbacks."""
        self._check_budget()
        last_err: Exception | None = None
        attempts = int(self.cfg["limits"].get("attempts_per_model", 1))
        timeout = float(self.cfg["limits"].get("timeout_seconds", 90))
        candidates = self.candidates_for(model, role=role)
        sem = self._get_semaphore()
        total_attempts = 0
        started = time.monotonic()

        for candidate in candidates:
            for attempt in range(attempts):
                total_attempts += 1
                try:
                    async with sem:
                        response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=candidate,
                                max_tokens=int(self.cfg["limits"]["max_tokens"]),
                                timeout=timeout,
                                messages=[
                                    {"role": "system", "content": system},
                                    {"role": "user", "content": user},
                                ],
                            ),
                            timeout=timeout + 5,
                        )
                    choices = getattr(response, "choices", None)
                    if not choices:
                        raise ValueError(f"Model {candidate} returned no choices")
                    content = getattr(choices[0].message, "content", None)
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError(f"Model {candidate} returned empty content")

                    metrics = extract_usage_metrics(response)
                    self.call_log.append(
                        {
                            "role": role,
                            "requested_model": model,
                            "used_model": candidate,
                            "attempts": total_attempts,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                            "timestamp": datetime.now(UTC).isoformat(),
                            **metrics,
                        }
                    )
                    return candidate, content.strip()
                except Exception as exc:  # model/provider failures are isolated here
                    last_err = exc
                    logger.warning(
                        "Model call failed (role=%s, candidate=%s, attempt=%d/%d): %s: %s",
                        role,
                        candidate,
                        attempt + 1,
                        attempts,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt + 1 < attempts:
                        await asyncio.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"All configured models failed for role request '{model}'. "
            f"Last error: {type(last_err).__name__}: {last_err}"
        ) from last_err
