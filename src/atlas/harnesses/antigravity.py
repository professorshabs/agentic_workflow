"""Google Antigravity CLI (`agy`) and Python SDK (`LocalOpenAIAgentConfig`) adapters."""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import importlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import socketserver
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from atlas.local_models import LocalModelCandidate
from atlas.pipeline.verify import credential_free_env
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
    workspace_signature,
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


_AGY_API_KEY_NAMES = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTIGRAVITY_API_KEY"}


@contextlib.contextmanager
def _credentials_hidden_from_children() -> Iterator[None]:
    """Hide credentials from processes the Antigravity SDK starts.

    The SDK launches its local harness with a copy of ``os.environ`` (its ``env`` option can
    only add variables), and the local model's ``run_command`` tool runs inside that harness.
    Without this, every key Atlas loaded from `.env` (OpenRouter, Anthropic, Gemini, GitHub)
    would be readable by the model and could end up in the generated app.
    """
    keep = credential_free_env()
    hidden = {key: value for key, value in os.environ.items() if key not in keep}
    for key in hidden:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        os.environ.update(hidden)


def _antigravity_env(auth: str | None = None) -> dict[str, str]:
    """Scope agy's environment to Google credentials, with a billing guard.

    Atlas (and LiteLLM on import) load `.env` into the process, so a `GEMINI_API_KEY` kept
    there for other tools would otherwise reach `agy` and get every call logged as
    `api_key`. Like the Claude guard, API keys are passed only with `auth: api`; otherwise
    `agy` runs on its cached interactive login.
    """
    normalized = (auth or "subscription").strip().lower()
    drop = set() if normalized == "api" else set(_AGY_API_KEY_NAMES)
    return scoped_subprocess_env(
        ("GEMINI_", "GOOGLE_", "ANTIGRAVITY_"), {"NO_COLOR"}, drop_names=drop
    )


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
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        # Missing/non-executable binary, or E2BIG: `agy -p` takes the prompt as one argv
        # entry, which Linux caps at 128 KiB. Raising HarnessError lets `fallback: router`
        # take over instead of crashing the stage.
        raise HarnessUnavailableError(f"Could not start Antigravity CLI: {exc}") from exc
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


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _first_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if usage.get(key) is not None:
            return _as_count(usage.get(key))
    return 0


def _parse_json_dict(stdout: str) -> dict[str, Any] | None:
    raw = stdout.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _agy_usage(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map the agy envelope's ``usage`` onto Atlas telemetry fields.

    Documented keys: ``input_tokens`` (includes ``cache_read_tokens``), ``output_tokens``
    (includes ``thinking_tokens``), ``total_tokens`` (= input + output). agy reports no
    cost. Older/alternate key spellings are accepted as fallbacks.
    """
    raw_usage = parsed.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    prompt_tokens = _first_count(
        usage, "input_tokens", "prompt_tokens", "input", "prompt_token_count"
    )
    completion_tokens = _first_count(
        usage, "output_tokens", "completion_tokens", "output", "candidates_token_count"
    )
    cached_tokens = _first_count(
        usage,
        "cache_read_tokens",
        "cached_tokens",
        "cache_read",
        "cached_content_token_count",
        "cache_read_input_tokens",
    )
    reasoning_tokens = _first_count(
        usage, "thinking_tokens", "reasoning_tokens", "thinking", "thoughts_token_count"
    )
    total_tokens = _first_count(usage, "total_tokens", "total", "total_token_count") or (
        prompt_tokens + completion_tokens
    )

    # agy does not emit cost; report real cost only if explicitly included, else None/unavailable.
    raw_cost = parsed.get("cost_usd")
    if raw_cost is None:
        raw_cost = usage.get("cost")
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
        cost_usd = round(float(raw_cost), 6)
        cost_source = "reported"
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "usage_reported": bool(usage),
    }


def _compact_agy_clauses(stderr: str, detail: str) -> str:
    clauses: list[str] = []
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("AGY_ERROR:"):
            payload_raw = line[len("AGY_ERROR:") :].strip()
            try:
                err_obj = json.loads(payload_raw)
            except json.JSONDecodeError:
                err_obj = None
            if isinstance(err_obj, dict):
                short = str(err_obj.get("short_error") or "").strip()
                status = str(err_obj.get("status") or "").strip()
                if short:
                    if status and status.lower() not in short.lower():
                        short = f"{short} [{status}]"
                    clauses.append(short)
                    continue
        for prefix in ("[agy] ", "error: "):
            if line.lower().startswith(prefix):
                line = line[len(prefix) :].strip()
        if line:
            clauses.append(line)

    if detail.strip():
        clauses.append(detail.strip())

    kept: list[str] = []
    for candidate in clauses:
        low = candidate.lower()
        if any(low in existing.lower() for existing in kept):
            continue
        kept = [existing for existing in kept if existing.lower() not in low]
        kept.append(candidate)
    return "; ".join(kept)


def _agy_failure(
    *, role: str, rc: int, stdout: str, stderr: str, action: str = "failed"
) -> HarnessError:
    parsed = _parse_json_dict(stdout)
    usage = _agy_usage(parsed) if parsed is not None else {}
    if parsed is not None:
        detail = str(parsed.get("error") or parsed.get("message") or parsed.get("status") or "")
    else:
        detail = stdout.strip()
    msg = _compact_agy_clauses(stderr, detail) or f"agy exited with code {rc}"
    if any(marker in msg.lower() for marker in _AUTH_ERROR_MARKERS):
        return HarnessAuthError(
            f"Antigravity authentication required for role '{role}': {msg}. "
            "Run `agy` to sign in or set `GEMINI_API_KEY`.",
            usage=usage,
        )
    return HarnessError(
        f"Antigravity CLI {action} for role '{role}' (exit={rc}): {msg}", usage=usage
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

    usage = _agy_usage(parsed)
    # Check explicit `status` in agy JSON envelope: exit code 0 still occurs on soft-denies.
    raw_status = parsed.get("status")
    if isinstance(raw_status, str) and raw_status.strip().upper() != "SUCCESS":
        err_msg = str(parsed.get("error") or parsed.get("message") or raw_status)
        if any(marker in err_msg.lower() for marker in _AUTH_ERROR_MARKERS):
            raise HarnessAuthError(f"Antigravity authentication required: {err_msg}", usage=usage)
        raise HarnessError(f"Antigravity CLI status={raw_status}: {err_msg}", usage=usage)

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

    used_model = str(parsed.get("model") or requested_model)
    return text, {"used_model": used_model, **usage}


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
        env = _antigravity_env(auth)
        auth_mode = _agy_auth_mode(env)
        try:
            rc, out, err = await _run_agy([binary, "models"], env=env, timeout_seconds=15.0)
        except HarnessError as exc:
            rc, out, err = 1, "", str(exc)
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

        return AuthStatus(
            provider=self.provider,
            available=True,
            authenticated=False,
            auth_mode=auth_mode,
            binary_path=binary,
            version=version,
            details=combined or f"`agy models` exited with code {rc}",
            login_hint=(
                "Check the Antigravity API key and CLI connectivity; `agy models` did not "
                "succeed."
                if auth_mode == "api_key"
                else "Run `agy` interactively to sign in, or set `auth: api` on this backend "
                "to pass `GEMINI_API_KEY`."
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
        timeout_seconds: float = 180.0,
        auth: str | None = None,
        max_budget_usd: float | None = None,
        tools: list[str] | None = None,
        cwd: Path | None = None,
        effort: str | None = None,
    ) -> HarnessInvocationResult:
        del max_budget_usd, tools, effort
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError("Antigravity CLI (`agy`) is not installed on PATH.")

        version = await self._version(binary)
        env = _antigravity_env(auth)
        constraint = (
            "EXECUTION CONSTRAINT: Respond directly in a single turn with ONLY the requested "
            "JSON object. Do NOT call any tools, read or write files, run shell commands, "
            "search the web, or spawn subagents.\n\n"
            if cwd is None
            else (
                "EXECUTION CONSTRAINT: You are in strictly read-only review mode. Do NOT "
                "create, edit, or delete files, run mutating commands, or spawn subagents; "
                "return ONLY the requested JSON object.\n\n"
            )
        )
        prompt_text = (
            f"SYSTEM INSTRUCTIONS:\n{constraint}{system.strip()}\n\n"
            f"USER REQUEST:\n{user.strip()}"
        )
        args = [
            binary,
            "-p",
            prompt_text,
            "--model",
            model,
            "--sandbox",
            "--disable-slash-commands",
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
            raise _agy_failure(role=role, rc=rc, stdout=out, stderr=err)

        text, meta = _parse_agy_output(out, requested_model=model)
        if not text.strip():
            raise HarnessError(
                f"Antigravity CLI returned empty output for role '{role}' "
                "(check non-TTY stdout support or set `fallback: router`).",
                usage={k: v for k, v in meta.items() if k != "used_model"},
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
        auth: str | None = None,
        max_budget_usd: float | None = None,
        effort: str | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        del skill_dirs, max_budget_usd, effort
        binary = self._find_binary()
        if not binary:
            raise HarnessUnavailableError("Antigravity CLI (`agy`) is not installed on PATH.")

        version = await self._version(binary)
        env = _antigravity_env(auth)
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
            raise _agy_failure(role=role, rc=rc, stdout=out, stderr=err, action="build failed")

        text, meta = _parse_agy_output(out, requested_model=model)
        escaped_paths = find_git_changes_outside_app(resolved_app_dir, before_git)
        if escaped_paths:
            raise HarnessError(
                f"Harness modified files outside '{app_dir}': {', '.join(escaped_paths)}",
                usage={k: v for k, v in meta.items() if k != "used_model"},
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


async def _wait_for_stall(app_dir: Path, stall_seconds: float, poll_seconds: float) -> float:
    """Return (seconds idle) once no file in `app_dir` has changed for `stall_seconds`."""
    last = workspace_signature(app_dir)
    last_change = time.monotonic()
    while True:
        await asyncio.sleep(poll_seconds)
        current = workspace_signature(app_dir)
        now = time.monotonic()
        if current != last:
            last, last_change = current, now
        elif now - last_change >= stall_seconds:
            return now - last_change


async def _cancel_quietly(task: asyncio.Future[Any], grace_seconds: float = 10.0) -> None:
    if task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(task, timeout=grace_seconds)


logger = logging.getLogger("atlas.harnesses.antigravity")


def _is_command_outside_workspace(
    resolved_app_dir: Path,
    args: Any,
    *,
    require_sandbox: bool = False,
) -> bool:
    """Return True when a `run_command` invocation sets a `Cwd` outside `resolved_app_dir`."""
    if not isinstance(args, dict):
        return True
    if require_sandbox and args.get("BypassSandbox"):
        return True
    raw_cwd = args.get("Cwd")
    if raw_cwd is None:
        return False
    if not isinstance(raw_cwd, str) or not raw_cwd.strip():
        return True
    cwd_path = Path(raw_cwd)
    try:
        resolved_cwd = (
            cwd_path.resolve()
            if cwd_path.is_absolute()
            else (resolved_app_dir / cwd_path).resolve()
        )
    except OSError:
        return True
    return resolved_cwd != resolved_app_dir and resolved_app_dir not in resolved_cwd.parents


_HOP_BY_HOP_REQ_HEADERS = frozenset(
    {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}
)
_HOP_BY_HOP_RESP_HEADERS = frozenset(
    {"transfer-encoding", "content-length", "connection", "keep-alive", "content-encoding"}
)


def _rewrite_chat_completion_body(
    raw_body: bytes,
    *,
    reasoning_effort: str | None,
    max_completion_tokens: int | None,
) -> bytes:
    """Shape an OpenAI `/chat/completions` JSON payload for a local model server."""
    if not raw_body:
        return raw_body
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw_body
    if not isinstance(payload, dict):
        return raw_body

    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        existing_mct = payload.get("max_completion_tokens")
        if (
            isinstance(existing_mct, (int, float))
            and not isinstance(existing_mct, bool)
            and int(existing_mct) > 0
        ):
            payload["max_completion_tokens"] = min(int(existing_mct), max_completion_tokens)
        else:
            payload["max_completion_tokens"] = max_completion_tokens
        existing_mt = payload.get("max_tokens")
        if (
            isinstance(existing_mt, (int, float))
            and not isinstance(existing_mt, bool)
            and int(existing_mt) > 0
        ):
            payload["max_tokens"] = min(int(existing_mt), max_completion_tokens)

    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_request_body(handler: http.server.BaseHTTPRequestHandler) -> bytes:
    """Read an HTTP request body from `Content-Length` or `Transfer-Encoding: chunked`."""
    transfer_encoding = str(handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in transfer_encoding:
        chunks: list[bytes] = []
        while True:
            size_line = handler.rfile.readline()
            if not size_line:
                break
            size_token = size_line.split(b";", 1)[0].strip()
            if not size_token:
                continue
            chunk_size = int(size_token, 16)
            if chunk_size == 0:
                while True:
                    trailer = handler.rfile.readline()
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                break
            chunks.append(handler.rfile.read(chunk_size))
            handler.rfile.readline()
        return b"".join(chunks)

    raw_length = handler.headers.get("Content-Length")
    if not raw_length:
        return b""
    try:
        length = max(0, int(raw_length))
    except ValueError:
        return b""
    return handler.rfile.read(length) if length else b""


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def _local_openai_proxy(
    upstream_base_url: str,
    *,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
) -> Iterator[str]:
    """Run a loopback proxy shaping `/chat/completions` payloads, or pass through unchanged."""
    if reasoning_effort is None and max_completion_tokens is None:
        yield upstream_base_url
        return

    parts = urllib.parse.urlsplit(upstream_base_url)
    upstream_origin = f"{parts.scheme}://{parts.netloc}"
    upstream_prefix = parts.path.rstrip("/")
    client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0))

    class _ProxyHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _forward(self, method: str) -> None:
            raw_body = _read_request_body(self) if method in ("POST", "PUT") else b""
            path_only = urllib.parse.urlsplit(self.path).path.rstrip("/")
            if method == "POST" and path_only.endswith("/chat/completions"):
                raw_body = _rewrite_chat_completion_body(
                    raw_body,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=max_completion_tokens,
                )

            fwd_headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in _HOP_BY_HOP_REQ_HEADERS
            }
            fwd_headers["accept-encoding"] = "identity"
            target_url = f"{upstream_origin}{self.path}"

            try:
                with client.stream(
                    method, target_url, headers=fwd_headers, content=raw_body or None
                ) as resp:
                    self.send_response(resp.status_code)
                    for hk, hv in resp.headers.items():
                        if hk.lower() in _HOP_BY_HOP_RESP_HEADERS:
                            continue
                        self.send_header(hk, hv)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError, httpx.HTTPError):
                return

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

        def do_PUT(self) -> None:
            self._forward("PUT")

        def do_DELETE(self) -> None:
            self._forward("DELETE")

        def do_HEAD(self) -> None:
            self._forward("HEAD")

        def do_OPTIONS(self) -> None:
            self._forward("OPTIONS")

    server = _ThreadingHTTPServer(("127.0.0.1", 0), _ProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        yield f"http://127.0.0.1:{port}{upstream_prefix}"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


class AntigravitySDKAdapter:
    """Google Antigravity Python SDK adapter for local Ollama / LM Studio execution."""

    harness = "antigravity_sdk"

    def package_version(self) -> str | None:
        try:
            # find_spec raises (rather than returning None) when the parent
            # ``google`` namespace package itself is not installed.
            if importlib.util.find_spec("google.antigravity") is None:
                return None
        except (ImportError, ValueError):
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
        sandbox: bool = True,
        allow_shell: bool = False,
        skill_dirs: list[Path] | None = None,
        stall_seconds: float | None = None,
        stall_poll_seconds: float = 10.0,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        """Run the local agent in `app_dir`.

        Gives up after `timeout_seconds` overall, or earlier when `stall_seconds` is set and
        no file in the workspace changes for that long (a small model stuck in a loop), so
        the caller can hand the build to a stronger model quickly.

        The agent gets file tools only unless `allow_shell` is set. Its run records (the SDK's
        SQLite trajectory) are deleted when it changes files, and kept, with their folder
        logged, when it fails or writes nothing, so the attempt can be examined afterwards.
        """
        records_dir = tempfile.mkdtemp(prefix="atlas-sdk-")
        keep_records = True
        try:
            build, call = await self._run_workspace_task(
                candidate=candidate,
                prompt_text=prompt_text,
                app_dir=app_dir,
                timeout_seconds=timeout_seconds,
                sandbox=sandbox,
                allow_shell=allow_shell,
                skill_dirs=skill_dirs,
                stall_seconds=stall_seconds,
                stall_poll_seconds=stall_poll_seconds,
                records_dir=records_dir,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_completion_tokens,
            )
            keep_records = not (build.created_files or build.modified_files or build.deleted_files)
            return build, call
        finally:
            records = Path(records_dir)
            if keep_records and records.is_dir() and any(records.iterdir()):
                logger.warning(
                    "[build] Kept the local agent's records of this attempt in %s "
                    "(SQLite trajectory; delete the folder when done).",
                    records_dir,
                )
            else:
                shutil.rmtree(records_dir, ignore_errors=True)

    async def _run_workspace_task(
        self,
        *,
        candidate: LocalModelCandidate,
        prompt_text: str,
        app_dir: Path,
        timeout_seconds: float,
        sandbox: bool,
        allow_shell: bool,
        skill_dirs: list[Path] | None,
        stall_seconds: float | None,
        stall_poll_seconds: float,
        records_dir: str,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
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

        # No `policy.auto()`: with a local model it sends a "policy guardian" prompt to that
        # same model before every shell command (minutes each), and it does not allow file
        # tools, so they fail closed. `workspace_only` + `allow_all` allows file tools inside
        # the product folder only. Shell commands are off unless `allow_shell` is set.
        workspace_rules = hooks.policy.workspace_only([str(resolved_app_dir)])
        if allow_shell:
            require_sandbox = bool(sandbox)
            deny_reason = (
                f"Commands must run sandboxed with Cwd inside '{resolved_app_dir}'."
                if require_sandbox
                else f"Commands must execute with Cwd inside '{resolved_app_dir}'."
            )
            shell_rule = hooks.policy.deny(
                "run_command",
                when=lambda args: _is_command_outside_workspace(
                    resolved_app_dir, args, require_sandbox=require_sandbox
                ),
                name="workspace_cwd",
                reason=deny_reason,
            )
        else:
            shell_rule = hooks.policy.deny(
                "run_command",
                name="no_shell",
                reason=(
                    "Shell commands are turned off for the local builder. Use the file tools; "
                    "Atlas runs the tests after you finish."
                ),
            )
        policies = [*workspace_rules, shell_rule, hooks.policy.allow_all()]
        config_kwargs: dict[str, Any] = {
            "model": candidate.name,
            "base_url": candidate.base_url,
            "workspaces": [str(resolved_app_dir)],
            "policies": policies,
        }
        if skill_dirs:
            config_kwargs["skills_paths"] = [str(p.resolve()) for p in skill_dirs]
        sdk_types = getattr(sdk, "types", None)
        if (
            not allow_shell
            and sdk_types is not None
            and hasattr(sdk_types, "CapabilitiesConfig")
            and hasattr(sdk_types, "BuiltinTools")
        ):
            # Hide the shell tool so the model never spends a turn on it; `.lightweight()`
            # keeps its other minimal tools (view, create, edit file).
            config_kwargs["capabilities"] = sdk_types.CapabilitiesConfig(
                disabled_tools=[sdk_types.BuiltinTools.RUN_COMMAND]
            )
        elif (
            allow_shell
            and sdk_types is not None
            and hasattr(sdk_types, "CapabilitiesConfig")
            and hasattr(sdk_types, "RunCommandConfig")
        ):
            cmd_timeout = min(max(float(timeout_seconds), 5.0), 120.0)
            config_kwargs["capabilities"] = sdk_types.CapabilitiesConfig(
                run_command_config=sdk_types.RunCommandConfig(
                    enable_daemons=False,
                    timeout_seconds=cmd_timeout,
                    enable_sandbox=bool(sandbox),
                )
            )

        started = time.monotonic()
        prev_cwd = Path.cwd()
        response_text = ""
        usage_meta: Any = None
        try:
            os.chdir(resolved_app_dir)
            with (
                _credentials_hidden_from_children(),
                _local_openai_proxy(
                    candidate.base_url,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=max_completion_tokens,
                ) as effective_base_url,
            ):
                config_kwargs["base_url"] = effective_base_url
                config_kwargs["save_dir"] = records_dir
                cfg = LocalOpenAIAgentConfig(**config_kwargs).lightweight()
                async with Agent(cfg) as agent:

                    async def _converse() -> tuple[str, Any]:
                        response = await agent.chat(prompt_text)
                        text = await response.text()
                        return text, getattr(response, "usage_metadata", None)

                    chat_task: asyncio.Future[tuple[str, Any]] = asyncio.ensure_future(
                        _converse()
                    )
                    watched: set[asyncio.Future[Any]] = {chat_task}
                    stall_task: asyncio.Future[float] | None = None
                    if stall_seconds is not None and stall_seconds > 0:
                        stall_task = asyncio.ensure_future(
                            _wait_for_stall(
                                resolved_app_dir, float(stall_seconds), stall_poll_seconds
                            )
                        )
                        watched.add(stall_task)
                    try:
                        done, _ = await asyncio.wait(
                            watched,
                            timeout=timeout_seconds,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        if stall_task is not None:
                            await _cancel_quietly(stall_task, grace_seconds=1.0)
                        if not chat_task.done():
                            await _cancel_quietly(chat_task)
                    if chat_task in done:
                        response_text, usage_meta = chat_task.result()
                    else:
                        if stall_task is not None and stall_task in done:
                            raise HarnessError(
                                f"Antigravity SDK local build stalled: no file changes for "
                                f"{float(stall_seconds or 0):.0f}s (model={candidate.name})."
                            )
                        raise TimeoutError
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
