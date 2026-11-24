"""Google Antigravity CLI (`agy`) and Python SDK (`LocalOpenAIAgentConfig`) adapters."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from atlas.local_models import LocalModelCandidate
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

_AUTH_ERROR_MARKERS = (
    "auth",
    "login",
    "sign in",
    "credential",
    "unauthorized",
    "api key",
    "401",
)


def _antigravity_env() -> dict[str, str]:
    return scoped_subprocess_env(("GEMINI_", "GOOGLE_", "ANTIGRAVITY_"), {"NO_COLOR"})


def _agy_auth_mode(env: dict[str, str]) -> AuthMode:
    if any(env.get(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTIGRAVITY_API_KEY")):
        return "api_key"
    return "cached_login"


async def _run_agy(
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
            f"Antigravity CLI timed out after {timeout_seconds:.0f}s."
        ) from None
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _parse_agy_output(stdout: str, *, requested_model: str) -> tuple[str, dict[str, Any]]:
    raw = stdout.strip()
    if not raw:
        return "", {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}

    if not isinstance(parsed, dict):
        return json.dumps(parsed), {}

    # Check explicit `status` in agy JSON envelope: exit code 0 still occurs on soft-denies.
    raw_status = parsed.get("status")
    if isinstance(raw_status, str) and raw_status.strip().upper() != "SUCCESS":
        err_msg = str(parsed.get("error") or parsed.get("message") or raw_status)
        if any(marker in err_msg.lower() for marker in _AUTH_ERROR_MARKERS):
            raise HarnessAuthError(f"Antigravity authentication required: {err_msg}")
        raise HarnessError(f"Antigravity CLI status={raw_status}: {err_msg}")

    text = raw
    if parsed.get("structured_output") is not None:
        so = parsed["structured_output"]
        text = so if isinstance(so, str) else json.dumps(so)
    else:
        for envelope_key in ("response", "result", "text", "output", "content"):
            if parsed.get(envelope_key) is not None:
                val = parsed[envelope_key]
                text = val if isinstance(val, str) else json.dumps(val)
                break

    raw_usage = parsed.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = int(
        usage.get("input")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
        or 0
    )
    completion_tokens = int(
        usage.get("output")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidates_token_count")
        or 0
    )
    cached_tokens = int(
        usage.get("cache_read")
        or usage.get("cached_tokens")
        or usage.get("cached_content_token_count")
        or usage.get("cache_read_input_tokens")
        or 0
    )
    reasoning_tokens = int(
        usage.get("thinking")
        or usage.get("reasoning_tokens")
        or usage.get("thoughts_token_count")
        or 0
    )
    total_tokens = int(
        usage.get("total")
        or usage.get("total_tokens")
        or usage.get("total_token_count")
        or (prompt_tokens + completion_tokens)
    )
    usage_reported = bool(usage)

    # agy does not emit cost; report real cost only if explicitly included, else None/unavailable.
    raw_cost = parsed.get("cost_usd")
    if raw_cost is None:
        raw_cost = usage.get("cost")
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
        cost_usd = round(float(raw_cost), 6)
        cost_source = "reported"

    used_model = str(parsed.get("model") or requested_model)
    return text, {
        "used_model": used_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "usage_reported": usage_reported,
    }


class AntigravityCLIAdapter(HarnessAdapter):
    """Harness adapter wrapping Google Antigravity CLI (`agy`)."""

    provider = "antigravity"

    def _find_binary(self) -> str | None:
        return shutil.which("agy")

    async def _version(self, binary: str) -> str | None:
        try:
            rc, out, _ = await _run_agy(
                [binary, "--version"], env=_antigravity_env(), timeout_seconds=5.0
            )
            if rc == 0 and out.strip():
                return out.strip().splitlines()[0]
        except Exception:
            pass
        return None

    async def check_auth(self, auth: str | None = None) -> AuthStatus:
        del auth
        binary = self._find_binary()
        if not binary:
            return AuthStatus(
                provider=self.provider,
                available=False,
                authenticated=False,
                details="`agy` CLI binary not found on PATH.",
                login_hint="Install Google Antigravity CLI (`agy`) and sign in interactively.",
            )

        version = await self._version(binary)
        env = _antigravity_env()
        auth_mode = _agy_auth_mode(env)
        rc, out, err = await _run_agy([binary, "models"], env=env, timeout_seconds=15.0)
        combined = f"{out}\n{err}".strip()
        if rc == 0 and out.strip():
            return AuthStatus(
                provider=self.provider,
                available=True,
                authenticated=True,
                auth_mode=auth_mode,
                binary_path=binary,
                version=version,
                details="Antigravity CLI authenticated (`agy models` succeeded).",
            )

        if auth_mode == "api_key":
            return AuthStatus(
                provider=self.provider,
                available=True,
                authenticated=True,
                auth_mode="api_key",
                binary_path=binary,
                version=version,
                details=f"Antigravity API key env present (`agy models` rc={rc}).",
            )

        return AuthStatus(
            provider=self.provider,
            available=True,
            authenticated=False,
            auth_mode="cached_login",
            binary_path=binary,
            version=version,
            details=combined or f"`agy models` exited with code {rc}",
            login_hint="Run `agy` interactively to authenticate, or export `GEMINI_API_KEY`.",
        )

    async def run_structured(
        self,
        *,
        role: str,
        model: str,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        timeout_seconds: float = 180.0,
        auth: str | None = None,
        max_budget_usd: float | None = None,
        tools: list[str] | None = None,
        cwd: Path | None = None,
    ) -> HarnessInvocationResult:
        del auth, max_budget_usd, tools
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError("Antigravity CLI (`agy`) is not installed on PATH.")

        version = await self._version(binary)
        env = _antigravity_env()
        prompt_text = f"SYSTEM INSTRUCTIONS:\n{system.strip()}\n\nUSER REQUEST:\n{user.strip()}"
        args = [
            binary,
            "-p",
            prompt_text,
            "--model",
            model,
            "--sandbox",
            "--output-format",
            "json",
            "--print-timeout",
            f"{max(1, int(timeout_seconds))}s",
        ]

        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="atlas-agy-") as tmpdir:
            if json_schema is not None:
                schema_path = Path(tmpdir) / "schema.json"
                schema_path.write_text(json.dumps(json_schema), encoding="utf-8")
                args.extend(["--json-schema", str(schema_path)])
            run_cwd = cwd if cwd is not None else Path(tmpdir)
            rc, out, err = await _run_agy(
                args, env=env, timeout_seconds=timeout_seconds + 10.0, cwd=run_cwd
            )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)

        if rc != 0:
            msg = err.strip() or out.strip() or f"agy exited with code {rc}"
            if any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS):
                raise HarnessAuthError(
                    f"Antigravity authentication required for role '{role}': {msg}. "
                    "Run `agy` to sign in or set `GEMINI_API_KEY`."
                )
            raise HarnessError(f"Antigravity CLI failed for role '{role}' (exit={rc}): {msg}")

        text, meta = _parse_agy_output(out, requested_model=model)
        if not text.strip():
            raise HarnessError(
                f"Antigravity CLI returned empty output for role '{role}' "
                "(check non-TTY stdout support or set `fallback: router`)."
            )

        used_model = str(meta.pop("used_model", model))
        return HarnessInvocationResult(
            text=text.strip(),
            used_model=used_model,
            harness="agy",
            harness_version=version,
            auth_mode=_agy_auth_mode(env),
            elapsed_ms=elapsed_ms,
            exit_code=rc,
            **meta,
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
        del skill_dirs
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError("Antigravity CLI (`agy`) is not installed on PATH.")

        version = await self._version(binary)
        env = _antigravity_env()
        app_dir.mkdir(parents=True, exist_ok=True)
        resolved_app_dir = app_dir.resolve()
        before_hashes = snapshot_file_hashes(resolved_app_dir)
        before_git = snapshot_git_status(resolved_app_dir)

        args = [
            binary,
            "-p",
            prompt_text,
            "--model",
            model,
            "--mode",
            "auto",
            *(["--sandbox"] if sandbox else []),
            "--output-format",
            "json",
            "--print-timeout",
            f"{max(1, int(timeout_seconds))}s",
        ]
        started = time.monotonic()
        rc, out, err = await _run_agy(
            args, env=env, timeout_seconds=timeout_seconds + 15.0, cwd=resolved_app_dir
        )
        elapsed_ms = round((time.monotonic() - started) * 1000, 1)

        if rc != 0:
            msg = err.strip() or out.strip() or f"agy exited with code {rc}"
            if any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS):
                raise HarnessAuthError(
                    f"Antigravity authentication required for role '{role}': {msg}."
                )
            raise HarnessError(f"Antigravity CLI build failed for role '{role}' (exit={rc}): {msg}")

        text, meta = _parse_agy_output(out, requested_model=model)
        escaped_paths = find_git_changes_outside_app(resolved_app_dir, before_git)
        if escaped_paths:
            raise HarnessError(
                f"Harness modified files outside '{app_dir}': {', '.join(escaped_paths)}"
            )

        used_model = str(meta.pop("used_model", model))
        build_result = capture_workspace_build_result(
            resolved_app_dir,
            before_hashes,
            summary=text or f"Workspace built by agy ({used_model})",
            harness="agy",
            harness_version=version,
            model=used_model,
            exit_code=rc,
        )
        invocation = HarnessInvocationResult(
            text=text.strip() or build_result.summary,
            used_model=used_model,
            harness="agy",
            harness_version=version,
            auth_mode=_agy_auth_mode(env),
            elapsed_ms=elapsed_ms,
            exit_code=rc,
            **meta,
        )
        return build_result, invocation


class AntigravitySDKAdapter:
    """Google Antigravity Python SDK adapter for local Ollama / LM Studio execution."""

    harness = "antigravity_sdk"

    def package_version(self) -> str | None:
        if importlib.util.find_spec("google.antigravity") is None:
            return None
        try:
            return importlib.metadata.version("google-antigravity")
        except importlib.metadata.PackageNotFoundError:
            return None

    async def check_auth(self) -> AuthStatus:
        version = self.package_version()
        if version is None:
            return AuthStatus(
                provider=self.harness,
                available=False,
                authenticated=False,
                auth_mode="local",
                details="`google-antigravity` Python SDK is not installed.",
                login_hint="Install optional harness support with `pip install -e '.[harnesses]'`.",
            )
        return AuthStatus(
            provider=self.harness,
            available=True,
            authenticated=True,
            auth_mode="local",
            version=version,
            details=f"google-antigravity SDK {version} ready for local model endpoints.",
        )

    async def run_workspace_task(
        self,
        *,
        candidate: LocalModelCandidate,
        prompt_text: str,
        app_dir: Path,
        timeout_seconds: float = 600.0,
        skill_dirs: list[Path] | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        version = self.package_version()
        if version is None:
            raise HarnessUnavailableError(
                "`google-antigravity` SDK is not installed. Run `pip install -e '.[harnesses]'`."
            )

        sdk = importlib.import_module("google.antigravity")
        hooks = importlib.import_module("google.antigravity.hooks")
        Agent = sdk.Agent
        LocalOpenAIAgentConfig = sdk.LocalOpenAIAgentConfig

        app_dir.mkdir(parents=True, exist_ok=True)
        resolved_app_dir = app_dir.resolve()
        before_hashes = snapshot_file_hashes(resolved_app_dir)
        before_git = snapshot_git_status(resolved_app_dir)

        policies = [
            *hooks.policy.workspace_only([str(resolved_app_dir)]),
            hooks.policy.auto(),
        ]
        config_kwargs: dict[str, Any] = {
            "model": candidate.name,
            "base_url": candidate.base_url,
            "workspaces": [str(resolved_app_dir)],
            "policies": policies,
        }
        if skill_dirs:
            config_kwargs["skills_paths"] = [str(p.resolve()) for p in skill_dirs]

        cfg = LocalOpenAIAgentConfig(**config_kwargs).lightweight()
        started = time.monotonic()
        prev_cwd = Path.cwd()
        response_text = ""
        usage_meta: Any = None
        try:
            os.chdir(resolved_app_dir)
            async with Agent(cfg) as agent:
                response = await asyncio.wait_for(
                    agent.chat(prompt_text), timeout=timeout_seconds
                )
                response_text = await asyncio.wait_for(
                    response.text(), timeout=timeout_seconds
                )
                usage_meta = getattr(response, "usage_metadata", None)
        except TimeoutError:
            raise HarnessError(
                f"Antigravity SDK local build timed out after {timeout_seconds:.0f}s "
                f"(model={candidate.name})."
            ) from None
        finally:
            os.chdir(prev_cwd)

        escaped_paths = find_git_changes_outside_app(resolved_app_dir, before_git)
        if escaped_paths:
            raise HarnessError(
                f"Local builder modified files outside '{app_dir}': {', '.join(escaped_paths)}"
            )

        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", None) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", None) or 0)
        cached_tokens = int(getattr(usage_meta, "cached_content_token_count", None) or 0)
        reasoning_tokens = int(getattr(usage_meta, "thoughts_token_count", None) or 0)
        total_tokens = int(
            getattr(usage_meta, "total_token_count", None) or (prompt_tokens + completion_tokens)
        )
        usage_reported = usage_meta is not None and any(
            getattr(usage_meta, attr, None) is not None
            for attr in ("prompt_token_count", "candidates_token_count", "total_token_count")
        )

        used_model = f"{candidate.provider}/{candidate.name}"
        build_result = capture_workspace_build_result(
            resolved_app_dir,
            before_hashes,
            summary=response_text.strip() or f"Workspace built locally with {used_model}",
            harness=self.harness,
            harness_version=version,
            model=used_model,
            exit_code=0,
        )
        invocation = HarnessInvocationResult(
            text=response_text.strip() or build_result.summary,
            used_model=used_model,
            harness=self.harness,
            harness_version=version,
            auth_mode="local",
            elapsed_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cost_usd=0.0,
            cost_source="local_free",
            usage_reported=usage_reported,
            exit_code=0,
        )
        return build_result, invocation
