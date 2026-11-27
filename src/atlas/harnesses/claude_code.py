"""Claude Code CLI adapter: read-only for Judge/Supervisor and file-scoped for Builder offload."""

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
    capture_workspace_build_result,
    find_git_changes_outside_app,
    scoped_subprocess_env,
    snapshot_file_hashes,
    snapshot_git_status,
)

ALLOWED_READONLY_TOOLS = {"Read", "Grep", "Glob"}
ALLOWED_WORKSPACE_TOOLS = ("Read", "Grep", "Glob", "Edit", "Write")
ALLOWED_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

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
    input_text: str | None = None,
) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        # Missing/non-executable binary, or E2BIG when an argument is too long.
        raise HarnessUnavailableError(f"Could not start Claude Code CLI: {exc}") from exc
    stdin_bytes = input_text.encode("utf-8") if input_text is not None else None
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(stdin_bytes), timeout=timeout_seconds
        )
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


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _parse_envelope(stdout: str) -> dict[str, Any] | None:
    raw = stdout.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _claude_usage(parsed: dict[str, Any], *, auth_mode: AuthMode) -> dict[str, Any]:
    """Normalise Claude Code token usage and cost from a `--output-format json` result.

    ``modelUsage`` is preferred: it covers every model the run billed (including Claude
    Code's background Haiku calls and sub-agents), whereas top-level ``usage`` covers only
    the main loop and is zeroed on some error results (e.g. ``error_max_budget_usd``).
    Anthropic reports ``input_tokens`` *excluding* cache reads and writes, so prompt tokens
    are the sum of all three, matching LiteLLM's convention that ``cached_tokens`` is a
    subset of ``prompt_tokens``.
    """
    input_tokens = cache_read = cache_write = output = thinking = 0
    reported = False
    model_usage = parsed.get("modelUsage")
    if isinstance(model_usage, dict):
        for entry in model_usage.values():
            if not isinstance(entry, dict):
                continue
            reported = True
            input_tokens += _as_count(entry.get("inputTokens"))
            cache_read += _as_count(entry.get("cacheReadInputTokens"))
            cache_write += _as_count(entry.get("cacheCreationInputTokens"))
            output += _as_count(entry.get("outputTokens"))
            thinking += _as_count(entry.get("thinkingTokens"))
    if not reported:
        raw_usage = parsed.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        if usage:
            reported = True
            input_tokens = _as_count(usage.get("input_tokens"))
            cache_read = _as_count(usage.get("cache_read_input_tokens"))
            cache_write = _as_count(usage.get("cache_creation_input_tokens"))
            output = _as_count(usage.get("output_tokens"))
            details = usage.get("output_tokens_details")
            if isinstance(details, dict):
                thinking = _as_count(details.get("thinking_tokens"))

    prompt_tokens = input_tokens + cache_read + cache_write
    raw = parsed.get("total_cost_usd")
    raw_cost = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    cost_usd: float | None = None
    api_cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    if auth_mode == "api_key":
        if raw_cost is not None:
            cost_usd = round(raw_cost, 6)
            cost_source = "reported"
    else:
        # Flat-rate Pro/Max subscription or OAuth token: keep `cost_usd` None rather than zeroing.
        # Claude Code still prices the tokens at API rates; keep that only as an estimate.
        cost_source = "subscription"
        if raw_cost is not None:
            api_cost_usd = round(raw_cost, 6)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output,
        "cached_tokens": cache_read,
        "reasoning_tokens": thinking,
        "total_tokens": prompt_tokens + output,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "api_cost_usd": api_cost_usd,
        "usage_reported": reported,
    }


def _extract_claude_text_and_usage(
    stdout: str, *, auth_mode: AuthMode
) -> tuple[str, dict[str, Any]]:
    raw = stdout.strip()
    if not raw:
        return "", {}
    parsed = _parse_envelope(raw)
    if parsed is None:
        try:
            return json.dumps(json.loads(raw)), {}
        except json.JSONDecodeError:
            return raw, {}

    if parsed.get("structured_output") is not None:
        so = parsed["structured_output"]
        text = so if isinstance(so, str) else json.dumps(so)
    elif parsed.get("result") is not None:
        res = parsed["result"]
        text = res if isinstance(res, str) else json.dumps(res)
    else:
        text = raw
    return text, _claude_usage(parsed, auth_mode=auth_mode)


def _claude_failure(
    *, role: str, rc: int, stdout: str, stderr: str, auth_mode: AuthMode
) -> HarnessError:
    """Turn a failed `claude -p` run into a typed error that keeps its billed usage.

    On failure the CLI exits non-zero, leaves stderr empty, and prints the JSON result with
    ``is_error``, ``api_error_status``/``api_error_type`` and either ``result`` (the error
    text) or ``errors`` (e.g. ``error_max_budget_usd``). Classify from those fields, never
    from the raw JSON blob, whose UUIDs can contain strings such as ``401``.
    """
    parsed = _parse_envelope(stdout)
    usage: dict[str, Any] = {}
    if parsed is None:
        msg = stderr.strip() or stdout.strip() or f"claude exited with code {rc}"
        is_auth = any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS)
    else:
        usage = _claude_usage(parsed, auth_mode=auth_mode)
        errors = parsed.get("errors")
        parts = [str(parsed.get("result") or "").strip(), stderr.strip()]
        if isinstance(errors, list):
            parts.extend(str(e) for e in errors)
        subtype = str(parsed.get("subtype") or "")
        if subtype and subtype != "success":
            parts.append(f"subtype={subtype}")
        msg = "; ".join(p for p in parts if p) or f"claude exited with code {rc}"
        status = parsed.get("api_error_status")
        error_type = str(parsed.get("api_error_type") or "")
        is_auth = status in (401, 403) or error_type in (
            "authentication_error",
            "permission_error",
        )
        if not is_auth and parsed.get("api_error_status") is None and not parsed.get("errors"):
            is_auth = any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS)
    if is_auth:
        return HarnessAuthError(
            f"Claude Code authentication required for role '{role}': {msg}. "
            "Run `claude auth login` or set `auth: api` with `ANTHROPIC_API_KEY`.",
            usage=usage,
        )
    return HarnessError(f"Claude Code failed for role '{role}' (exit={rc}): {msg}", usage=usage)


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
                    "`claude auth login` (Pro/Max+), or configure Judge on `agy` "
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

        try:
            # Prints JSON (`loggedIn`, `authMethod`, ...); exits 0 if logged in, 1 if not.
            rc, out, err = await _run_cli(
                [binary, "auth", "status"], env=env, timeout_seconds=15.0
            )
        except HarnessError as exc:
            rc, out, err = 1, "", str(exc)
        combined = f"{out}\n{err}".strip()
        status_json = _parse_envelope(out)
        auth_method = ""
        if status_json is not None and "loggedIn" in status_json:
            logged_in = rc == 0 and status_json.get("loggedIn") is True
            auth_method = str(status_json.get("authMethod") or "")
        else:
            lowered = combined.lower()
            logged_in = rc == 0 and not (
                "not logged" in lowered or '"loggedin":false' in lowered.replace(" ", "")
            )
        if logged_in:
            if auth_method == "api_key":
                # Something other than ANTHROPIC_API_KEY (stripped above), e.g. an
                # `apiKeyHelper` setting, would bill these runs to the API, not the plan.
                return AuthStatus(
                    provider=self.provider,
                    available=True,
                    authenticated=True,
                    auth_mode="api_key",
                    binary_path=binary,
                    version=version,
                    details=(
                        "Claude Code is signed in with an API key (authMethod=api_key) even "
                        f"though `auth: {auth or 'subscription'}` is configured; runs will be "
                        "billed per token."
                    ),
                    login_hint="Run `claude auth login` with your Pro/Max account, or set "
                    "`auth: api` to accept API billing.",
                )
            return AuthStatus(
                provider=self.provider,
                available=True,
                authenticated=True,
                auth_mode=auth_mode,
                binary_path=binary,
                version=version,
                details=(
                    f"Claude Code signed in (authMethod={auth_method})."
                    if auth_method
                    else combined or f"Claude Code authenticated ({auth_mode})."
                ),
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
            login_hint=(
                "Run `claude auth login` (Pro/Max+), or export `CLAUDE_CODE_OAUTH_TOKEN`."
            ),
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
        effort: str | None = None,
    ) -> HarnessInvocationResult:
        """Execute a strictly read-only evaluation (`--tools ""` or `--tools Read,Grep,Glob`)."""
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError(
                "Claude Code CLI (`claude`) is not installed on PATH. "
                "Install it or switch Judge to `antigravity` (`claude-opus-5-5-high`)."
            )

        if effort is not None and effort not in ALLOWED_EFFORT_LEVELS:
            raise ValueError(
                f"Unsupported Claude Code effort level '{effort}'; expected one of "
                f"{sorted(ALLOWED_EFFORT_LEVELS)}"
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
        # The prompt goes in on stdin (`claude -p` with no positional prompt reads it, up to
        # 10 MB): judge/supervisor prompts embed app code and can exceed the per-argument
        # argv limit (128 KiB on Linux).
        args = [
            binary,
            "-p",
            "--model",
            model,
            *(["--effort", effort] if effort is not None else []),
            "--output-format",
            "json",
            "--tools",
            tools_arg,
        ]
        if json_schema is not None:
            args.extend(["--json-schema", json.dumps(json_schema)])
        if max_budget_usd is not None:
            # `g` keeps small caps exact (`.2f` would turn 0.004 into "0.00").
            args.extend(["--max-budget-usd", format(max_budget_usd, "g")])

        started = time.monotonic()
        rc, out, err = await _run_cli(
            args, env=env, timeout_seconds=timeout_seconds, cwd=cwd, input_text=prompt_text
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)

        envelope = _parse_envelope(out)
        if rc != 0 or (envelope is not None and envelope.get("is_error") is True):
            raise _claude_failure(role=role, rc=rc, stdout=out, stderr=err, auth_mode=auth_mode)

        text, usage = _extract_claude_text_and_usage(out, auth_mode=auth_mode)
        if not text.strip():
            raise HarnessError(
                f"Claude Code returned empty output for role '{role}'.", usage=usage
            )

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
        auth: str | None = None,
        max_budget_usd: float | None = None,
        effort: str | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        """Build or edit workspace files inside `app_dir` using file-only tools (`acceptEdits`)."""
        del skill_dirs
        if role != "builder":
            raise ValueError(
                "ClaudeCodeAdapter.run_workspace_task is only permitted for role 'builder'."
            )
        if effort is not None and effort not in ALLOWED_EFFORT_LEVELS:
            raise ValueError(
                f"Unsupported Claude Code effort level '{effort}'; expected one of "
                f"{sorted(ALLOWED_EFFORT_LEVELS)}"
            )
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError(
                "Claude Code CLI (`claude`) is not installed on PATH. "
                "Install it or configure a different builder offload target."
            )

        env = _claude_env(auth)
        auth_mode = _resolved_auth_mode(auth, env)
        version = await self._version(binary, env)
        app_dir.mkdir(parents=True, exist_ok=True)
        resolved_app_dir = app_dir.resolve()
        before_hashes = snapshot_file_hashes(resolved_app_dir)
        before_git = snapshot_git_status(resolved_app_dir)

        tools_csv = ",".join(ALLOWED_WORKSPACE_TOOLS)
        args = [
            binary,
            "-p",
            "--model",
            model,
            *(["--effort", effort] if effort is not None else []),
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            *(["--restricted"] if sandbox else []),
            "--tools",
            tools_csv,
            "--no-session-persistence",
        ]
        if max_budget_usd is not None:
            args.extend(["--max-budget-usd", format(max_budget_usd, "g")])

        started = time.monotonic()
        rc, out, err = await _run_cli(
            args,
            env=env,
            timeout_seconds=timeout_seconds,
            cwd=resolved_app_dir,
            input_text=prompt_text,
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)

        envelope = _parse_envelope(out)
        if rc != 0 or (envelope is not None and envelope.get("is_error") is True):
            raise _claude_failure(role=role, rc=rc, stdout=out, stderr=err, auth_mode=auth_mode)

        text, usage = _extract_claude_text_and_usage(out, auth_mode=auth_mode)
        escaped_paths = find_git_changes_outside_app(resolved_app_dir, before_git)
        if escaped_paths:
            raise HarnessError(
                f"Harness modified files outside '{app_dir}': {', '.join(escaped_paths)}",
                usage=usage,
            )

        build_result = capture_workspace_build_result(
            resolved_app_dir,
            before_hashes,
            summary=text or f"Workspace built by claude_code ({model})",
            harness=self.provider,
            harness_version=version,
            model=model,
            exit_code=rc,
        )
        invocation = HarnessInvocationResult(
            text=text.strip() or build_result.summary,
            used_model=model,
            harness=self.provider,
            harness_version=version,
            auth_mode=auth_mode,
            elapsed_ms=elapsed_ms,
            exit_code=rc,
            **usage,
        )
        return build_result, invocation
