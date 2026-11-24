"""Strictly read-only Claude Code CLI adapter for Judge / Supervisor checkpoints."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

from atlas.schemas import AuthMode, CostSource, WorkspaceBuildResult

from .base import (
    AuthStatus,
    HarnessAdapter,
    HarnessAuthError,
    HarnessError,
    HarnessInvocationResult,
    HarnessUnavailableError,
    scoped_subprocess_env,
)

ALLOWED_READONLY_TOOLS = {"Read", "Grep", "Glob"}

_AUTH_ERROR_MARKERS = (
    "not logged in",
    "please run `claude login`",
    "authentication",
    "unauthorized",
    "api key",
    "oauth",
    "401",
)


def _claude_env(auth: str | None = None) -> dict[str, str]:
    """Billing guard: strip `ANTHROPIC_API_KEY` unless `auth == 'api'` is explicitly set."""
    normalized = (auth or "subscription").strip().lower()
    drop = set() if normalized == "api" else {"ANTHROPIC_API_KEY"}
    return scoped_subprocess_env(("ANTHROPIC_", "CLAUDE_"), {"NO_COLOR"}, drop_names=drop)


def _resolved_auth_mode(auth: str | None, env: dict[str, str]) -> AuthMode:
    mode = (auth or "subscription").strip().lower()
    if mode == "api" and env.get("ANTHROPIC_API_KEY"):
        return "api_key"
    if mode == "oauth_token" or env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "oauth_token"
    return "subscription"


async def _run_cli(
    args: list[str],
    *,
    env: dict[str, str],
    timeout_seconds: float,
    cwd: Path | None = None,
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise HarnessError(
            f"Claude Code CLI timed out after {timeout_seconds:.0f}s."
        ) from None
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _extract_claude_text_and_usage(
    stdout: str, *, auth_mode: AuthMode
) -> tuple[str, dict[str, Any]]:
    raw = stdout.strip()
    if not raw:
        return "", {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}

    if not isinstance(parsed, dict):
        return json.dumps(parsed), {}

    text = ""
    if parsed.get("structured_output") is not None:
        so = parsed["structured_output"]
        text = so if isinstance(so, str) else json.dumps(so)
    elif parsed.get("result") is not None:
        res = parsed["result"]
        text = res if isinstance(res, str) else json.dumps(res)
    else:
        text = raw

    raw_usage = parsed.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    cached_tokens = int(usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    usage_reported = bool(usage)

    raw_cost = parsed.get("total_cost_usd")
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    if auth_mode == "api_key":
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            cost_usd = round(float(raw_cost), 6)
            cost_source = "reported"
    else:
        # Flat-rate Pro/Max subscription or OAuth token: keep `cost_usd` None rather than zeroing.
        cost_source = "subscription"

    return text, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": 0,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "usage_reported": usage_reported,
    }


class ClaudeCodeAdapter(HarnessAdapter):
    """Read-only harness adapter wrapping the `claude` CLI."""

    provider = "claude_code"

    def _find_binary(self) -> str | None:
        return shutil.which("claude")

    async def _version(self, binary: str, env: dict[str, str]) -> str | None:
        try:
            rc, out, _ = await _run_cli([binary, "--version"], env=env, timeout_seconds=5.0)
            if rc == 0 and out.strip():
                return out.strip().splitlines()[0]
        except Exception:
            pass
        return None

    async def check_auth(self, auth: str | None = None) -> AuthStatus:
        binary = self._find_binary()
        if not binary:
            return AuthStatus(
                provider=self.provider,
                available=False,
                authenticated=False,
                details="`claude` CLI binary not found on PATH.",
                login_hint=(
                    "Install with `npm install -g @anthropic-ai/claude-code` then run "
                    "`claude login` (Pro/Max+), or configure Judge on `agy` "
                    "(`claude-opus-5-5-high`)."
                ),
            )

        env = _claude_env(auth)
        version = await self._version(binary, env)
        auth_mode = _resolved_auth_mode(auth, env)

        if auth == "api":
            has_api = bool(env.get("ANTHROPIC_API_KEY"))
            return AuthStatus(
                provider=self.provider,
                available=True,
                authenticated=has_api,
                auth_mode="api_key",
                binary_path=binary,
                version=version,
                details=(
                    "ANTHROPIC_API_KEY configured (`auth: api`)."
                    if has_api
                    else "`auth: api` configured but ANTHROPIC_API_KEY is missing."
                ),
                login_hint=(
                    ""
                    if has_api
                    else "Set ANTHROPIC_API_KEY or switch `auth: subscription`."
                ),
            )

        rc, out, err = await _run_cli([binary, "auth", "status"], env=env, timeout_seconds=10.0)
        combined = f"{out}\n{err}".strip()
        if rc == 0:
            lowered = combined.lower()
            logged_out = "not logged" in lowered or '"loggedin":false' in lowered.replace(" ", "")
            if not logged_out:
                return AuthStatus(
                    provider=self.provider,
                    available=True,
                    authenticated=True,
                    auth_mode=auth_mode,
                    binary_path=binary,
                    version=version,
                    details=combined or f"Claude Code authenticated ({auth_mode}).",
                )

        if env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return AuthStatus(
                provider=self.provider,
                available=True,
                authenticated=True,
                auth_mode="oauth_token",
                binary_path=binary,
                version=version,
                details="CLAUDE_CODE_OAUTH_TOKEN configured (`auth: oauth_token`).",
            )

        return AuthStatus(
            provider=self.provider,
            available=True,
            authenticated=False,
            auth_mode=auth_mode,
            binary_path=binary,
            version=version,
            details=combined or "Claude Code CLI is not signed in.",
            login_hint="Run `claude login` (Pro/Max+), or export `CLAUDE_CODE_OAUTH_TOKEN`.",
        )

    async def run_structured(
        self,
        *,
        role: str,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        timeout_seconds: float = 300.0,
        auth: str | None = None,
        max_budget_usd: float | None = None,
        tools: list[str] | None = None,
        cwd: Path | None = None,
    ) -> HarnessInvocationResult:
        """Execute a strictly read-only evaluation (`--tools ""` or `--tools Read,Grep,Glob`)."""
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError(
                "Claude Code CLI (`claude`) is not installed on PATH. "
                "Install it or switch Judge to `antigravity` (`claude-opus-5-5-high`)."
            )

        if tools is not None:
            disallowed = [t for t in tools if t not in ALLOWED_READONLY_TOOLS]
            if disallowed:
                raise ValueError(
                    f"ClaudeCodeAdapter is strictly read-only; disallowed tools: {disallowed}"
                )
            tools_arg = ",".join(tools)
        else:
            tools_arg = ""

        env = _claude_env(auth)
        auth_mode = _resolved_auth_mode(auth, env)
        version = await self._version(binary, env)
        prompt_text = f"SYSTEM INSTRUCTIONS:\n{system.strip()}\n\nUSER REQUEST:\n{user.strip()}"
        args = [
            binary,
            "-p",
            prompt_text,
            "--model",
            model,
            "--output-format",
            "json",
            "--tools",
            tools_arg,
        ]
        if json_schema is not None:
            args.extend(["--json-schema", json.dumps(json_schema)])
        if max_budget_usd is not None:
            args.extend(["--max-budget-usd", f"{max_budget_usd:.2f}"])

        started = time.monotonic()
        rc, out, err = await _run_cli(args, env=env, timeout_seconds=timeout_seconds, cwd=cwd)
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)

        if rc != 0:
            msg = err.strip() or out.strip() or f"claude exited with code {rc}"
            if any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS):
                raise HarnessAuthError(
                    f"Claude Code authentication required for role '{role}': {msg}. "
                    "Run `claude login` or set `auth: api` with `ANTHROPIC_API_KEY`."
                )
            raise HarnessError(f"Claude Code failed for role '{role}' (exit={rc}): {msg}")

        text, usage = _extract_claude_text_and_usage(out, auth_mode=auth_mode)
        if not text.strip():
            raise HarnessError(f"Claude Code returned empty output for role '{role}'.")

        return HarnessInvocationResult(
            text=text.strip(),
            used_model=model,
            harness=self.provider,
            harness_version=version,
            auth_mode=auth_mode,
            elapsed_ms=elapsed_ms,
            exit_code=rc,
            **usage,
        )

    async def run_workspace_task(
        self,
        *,
        role: str,
        model: str,
        prompt_text: str,
        app_dir: Path,
        timeout_seconds: float = 600.0,
        sandbox: bool = True,
        skill_dirs: list[Path] | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        raise NotImplementedError(
            "ClaudeCodeAdapter is strictly read-only and cannot modify workspace files."
        )
