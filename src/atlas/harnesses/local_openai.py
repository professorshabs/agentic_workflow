"""Lightweight native Python OpenAI tool-loop harness for local model servers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from atlas.local_models import LocalModelCandidate
from atlas.pipeline.verify import credential_free_env
from atlas.schemas import WorkspaceBuildResult

from .antigravity import _cancel_quietly, _wait_for_stall
from .base import (
    _WORKSPACE_IGNORED_DIRS,
    AuthStatus,
    HarnessError,
    HarnessInvocationResult,
    capture_workspace_build_result,
    find_git_changes_outside_app,
    snapshot_file_hashes,
    snapshot_git_status,
)

logger = logging.getLogger("atlas.harnesses.local_openai")

_MAX_READ_CHARS = 28_000
_MAX_CMD_OUTPUT_CHARS = 8_000
_MAX_LIST_ENTRIES = 200

_TINY_JPEG_URI = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP"
    "//////////////////////////////////////////////////////////////////////////////////////"
    "wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)

_SYSTEM_PROMPT_BASE = """\
You are a senior software engineer building a complete, minimal, runnable, and tested
application inside the current project folder.

Follow these steps:
1. First read `BUILD_BRIEF.md` and any pre-existing `tests/test_e2e*` files using `read_file`
   before writing files.
2. Pre-seeded files (`BUILD_BRIEF.md`, `PLAN.md`, `tests/test_e2e*`) are read-only — never
   modify or overwrite them; write the application files so those tests pass.
3. Write concise, complete files directly to disk one at a time (`pyproject.toml`, `README.md`,
   main application files, etc.) keeping each file compact (under ~3,000 tokens). Use
   workspace tools (`list_files`, `read_file`, `write_file`, `edit_file`{shell_hint}) with paths
   relative to the project root. Never paste whole files in chat and never leave `TODO` or `...`.
4. If images are needed without image files or a generator tool, reuse this tiny valid 1x1 JPEG
   data URI instead of inventing base64:
   `{tiny_jpeg_uri}`
   and note it in `README.md`.
5. Once finished, reply briefly with what you built and end with
   `Skills used: <comma-separated names or none>`.
"""


def _resolve_workspace_path(
    resolved_app_dir: Path,
    raw_path: str,
    *,
    allow_skill_symlink: bool = False,
    for_write: bool = False,
) -> tuple[Path, str]:
    """Validate `raw_path` stays in `resolved_app_dir`, returning `(resolved, rel_posix)`."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Path must be a non-empty string.")
    cleaned = raw_path.strip()
    candidate = Path(cleaned)
    joined = candidate if candidate.is_absolute() else (resolved_app_dir / candidate)
    logical = Path(os.path.normpath(str(joined)))
    try:
        rel_logical = logical.relative_to(resolved_app_dir)
    except ValueError as exc:
        raise ValueError(f"Path '{raw_path}' escapes workspace directory.") from exc

    rel_posix = "." if rel_logical == Path(".") else rel_logical.as_posix()
    if for_write:
        if rel_posix == ".":
            raise ValueError("Target path must name a file inside the workspace, not '.'")
        if rel_logical.parts and rel_logical.parts[0] == ".agents":
            raise ValueError("Modifying files inside '.agents/' is not permitted.")

    in_skills = len(rel_logical.parts) >= 2 and rel_logical.parts[:2] == (".agents", "skills")
    resolved = logical.resolve()
    if allow_skill_symlink and in_skills:
        return resolved, rel_posix

    try:
        resolved.relative_to(resolved_app_dir)
    except ValueError as exc:
        raise ValueError(f"Path '{raw_path}' resolves outside workspace directory.") from exc
    return resolved, rel_posix


def _tool_schemas(*, allow_shell: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files inside the workspace folder with sizes in bytes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path (default '.').",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read lines from a UTF-8 file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to the file."},
                        "start_line": {
                            "type": "integer",
                            "description": "1-based first line to read (default 1).",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based inclusive last line to read (default 500).",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or overwrite a file in the workspace with complete content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path."},
                        "content": {"type": "string", "description": "Complete file contents."},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace the first exact occurrence of old_text with new_text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path."},
                        "old_text": {"type": "string", "description": "Exact text to replace."},
                        "new_text": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
    ]
    if allow_shell:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command inside the workspace root directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run."},
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Command timeout in seconds (5-120, default 60).",
                            },
                        },
                        "required": ["command"],
                    },
                },
            }
        )
    return tools


def _list_files_tool(resolved_app_dir: Path, raw_path: str = ".") -> str:
    target, rel_posix = _resolve_workspace_path(
        resolved_app_dir, raw_path or ".", allow_skill_symlink=True
    )
    if not target.exists():
        return f"Directory not found: {rel_posix}"
    if target.is_file():
        return f"{rel_posix} ({target.stat().st_size} bytes)"

    include_agents = rel_posix == ".agents" or rel_posix.startswith(".agents/")
    rows: list[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in _WORKSPACE_IGNORED_DIRS or (include_agents and d == ".agents")
        )
        for fname in sorted(filenames):
            full = Path(dirpath) / fname
            try:
                rel_entry = full.relative_to(target).as_posix()
                display = (
                    rel_entry
                    if rel_posix == "."
                    else posixpath.join(rel_posix.rstrip("/"), rel_entry)
                )
                size = full.stat().st_size
            except OSError:
                continue
            rows.append(f"{display} ({size} B)")
            if len(rows) >= _MAX_LIST_ENTRIES:
                truncated = True
                break
        if truncated:
            break
    if not rows:
        return f"(no files in {rel_posix})"
    if truncated:
        rows.append(f"... (truncated after {_MAX_LIST_ENTRIES} entries)")
    return "\n".join(rows)


def _read_file_tool(
    resolved_app_dir: Path,
    raw_path: str,
    *,
    start_line: int = 1,
    end_line: int = 500,
) -> str:
    target, rel_posix = _resolve_workspace_path(
        resolved_app_dir, raw_path, allow_skill_symlink=True
    )
    if not target.is_file():
        return f"File not found: {rel_posix}"

    start = max(1, int(start_line or 1))
    end = max(start, int(end_line or (start + 499)))
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    if total == 0:
        return f"{rel_posix} is empty (0 lines)."
    if start > total:
        return f"{rel_posix} has only {total} lines (requested start_line={start})."

    slice_end = min(end, total)
    selected = lines[start - 1 : slice_end]
    body = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
    if len(body) > _MAX_READ_CHARS:
        body = body[:_MAX_READ_CHARS].rstrip()
        last_shown = start + body.count("\n")
        body += (
            f"\n... [truncated at {_MAX_READ_CHARS} chars; "
            f"read next range with start_line={last_shown + 1}]"
        )
    elif slice_end < total:
        body += f"\n... [lines {start}-{slice_end} of {total}; next start_line={slice_end + 1}]"
    return body


def _normalize_protected_set(
    before_hashes: dict[str, str],
    protected_files: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    """Return normalized lowercase workspace-relative paths that must stay read-only."""
    protected: set[str] = set()
    for raw in protected_files or ():
        norm = posixpath.normpath(str(raw).replace("\\", "/").lstrip("./"))
        if norm and norm != ".":
            protected.add(norm.lower())
    for rel in before_hashes:
        rel_low = rel.replace("\\", "/").lower()
        base = posixpath.basename(rel_low)
        if rel_low in ("build_brief.md", "plan.md") or rel_low.startswith("tests/test_e2e") or (
            base.startswith("test_e2e")
        ):
            protected.add(rel_low)
    return protected


def _protected_error(rel_posix: str) -> str:
    logger.warning("[build] Blocked modification of read-only pre-seeded file: %s", rel_posix)
    return (
        f"Error: '{rel_posix}' is a pre-seeded read-only file and cannot be modified. "
        "Use read_file to inspect it and write the application files instead."
    )


def _write_file_tool(
    resolved_app_dir: Path,
    raw_path: str,
    content: str,
    *,
    protected_lower: set[str] | None = None,
) -> str:
    if not isinstance(content, str):
        raise ValueError("'content' must be a string.")
    target, rel_posix = _resolve_workspace_path(resolved_app_dir, raw_path, for_write=True)
    if protected_lower and rel_posix.lower() in protected_lower:
        return _protected_error(rel_posix)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    target.write_bytes(encoded)
    line_count = len(content.splitlines())
    logger.info("[build] local write_file: %s (%d bytes)", rel_posix, len(encoded))
    return f"Wrote {rel_posix} ({len(encoded)} bytes, {line_count} lines)."


def _edit_file_tool(
    resolved_app_dir: Path,
    raw_path: str,
    old_text: str,
    new_text: str,
    *,
    protected_lower: set[str] | None = None,
) -> str:
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("'old_text' must be a non-empty string.")
    if not isinstance(new_text, str):
        raise ValueError("'new_text' must be a string.")
    target, rel_posix = _resolve_workspace_path(resolved_app_dir, raw_path, for_write=True)
    if protected_lower and rel_posix.lower() in protected_lower:
        return _protected_error(rel_posix)
    if not target.is_file():
        return f"Error: file not found: {rel_posix}"
    original = target.read_text(encoding="utf-8", errors="replace")
    if old_text not in original:
        return f"Error: old_text not found in {rel_posix}. Read the file to verify exact text."
    updated = original.replace(old_text, new_text, 1)
    target.write_text(updated, encoding="utf-8")
    logger.info("[build] local edit_file: %s (%d bytes)", rel_posix, len(updated.encode("utf-8")))
    return f"Edited {rel_posix} ({len(updated.encode('utf-8'))} bytes)."


async def _run_command_tool(
    resolved_app_dir: Path, command: str, *, timeout_seconds: int = 60
) -> str:
    if not isinstance(command, str) or not command.strip():
        raise ValueError("'command' must be a non-empty string.")
    timeout = float(min(max(int(timeout_seconds or 60), 5), 120))
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        command,
        cwd=str(resolved_app_dir),
        env=credential_free_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return f"Error: command timed out after {timeout:.0f}s."
    out = stdout_b.decode("utf-8", errors="replace")
    err = stderr_b.decode("utf-8", errors="replace")
    combined = f"exit_code={proc.returncode}\n"
    if out:
        combined += f"stdout:\n{out}"
    if err:
        combined += f"\nstderr:\n{err}"
    if len(combined) > _MAX_CMD_OUTPUT_CHARS:
        combined = combined[:_MAX_CMD_OUTPUT_CHARS] + "\n... [output truncated]"
    return combined.strip()


def _extract_usage(usage_obj: Any) -> dict[str, int]:
    if not isinstance(usage_obj, dict):
        return {}
    prompt = int(usage_obj.get("prompt_tokens") or 0)
    completion = int(usage_obj.get("completion_tokens") or 0)
    total = int(usage_obj.get("total_tokens") or (prompt + completion))
    p_details = usage_obj.get("prompt_tokens_details") or {}
    c_details = usage_obj.get("completion_tokens_details") or {}
    cached = int((p_details.get("cached_tokens") if isinstance(p_details, dict) else 0) or 0)
    reasoning = int(
        (c_details.get("reasoning_tokens") if isinstance(c_details, dict) else 0)
        or usage_obj.get("reasoning_tokens")
        or 0
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


def _merge_usage(totals: dict[str, Any], turn_usage: dict[str, int]) -> None:
    if not turn_usage:
        return
    totals["usage_reported"] = True
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        totals[key] = int(totals.get(key) or 0) + int(turn_usage.get(key) or 0)


def _has_repetition_loop(text: str) -> bool:
    """Return True when the tail of a stream is stuck in a repeating >=12-char cycle."""
    if len(text) < 96:
        return False
    tail = text[-1000:]
    max_period = min(80, len(tail) // 8)
    for period in range(12, max_period + 1):
        unit = tail[-period:]
        if not unit.strip() or len(set(unit)) < 3:
            continue
        if tail.endswith(unit * 8):
            return True
    return False


async def _stream_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    turn_index: int,
    on_usage: Callable[[dict[str, int]], None] | None = None,
) -> tuple[str, list[dict[str, Any]], str | None, dict[str, int]]:
    """Send one `/chat/completions` request (SSE streaming with JSON fallback)."""
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    turn_usage: dict[str, int] = {}
    stream_tail = ""
    chars_since_check = 0

    async with client.stream("POST", url, json=payload) as resp:
        if resp.status_code >= 400:
            raw_err = (await resp.aread()).decode("utf-8", errors="replace")
            raise HarnessError(
                f"Local model HTTP {resp.status_code} from {url}: {raw_err[:400]}"
            )
        ctype = (resp.headers.get("content-type") or "").lower()
        if "application/json" in ctype and "text/event-stream" not in ctype:
            raw = await resp.aread()
            data = json.loads(raw.decode("utf-8"))
            turn_usage = _extract_usage(data.get("usage"))
            if turn_usage and on_usage is not None:
                on_usage(turn_usage)
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
            content = str(msg.get("content") or "")
            raw_tcs = msg.get("tool_calls") or []
            parsed_tcs: list[dict[str, Any]] = []
            for idx, tc in enumerate(raw_tcs):
                fn = tc.get("function") or {}
                parsed_tcs.append(
                    {
                        "id": str(tc.get("id") or f"call_{turn_index}_{idx}"),
                        "type": "function",
                        "function": {
                            "name": str(fn.get("name") or ""),
                            "arguments": str(fn.get("arguments") or ""),
                        },
                    }
                )
            return content, parsed_tcs, finish_reason, turn_usage

        broke_loop = False
        async for line in resp.aiter_lines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("data:"):
                continue
            data_str = stripped[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "usage" in chunk and isinstance(chunk["usage"], dict):
                parsed_u = _extract_usage(chunk["usage"])
                if parsed_u:
                    turn_usage = parsed_u
                    if on_usage is not None:
                        on_usage(parsed_u)
            added_parts: list[str] = []
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    content_parts.append(piece)
                    added_parts.append(piece)
                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index")
                    if idx is None:
                        idx = max(len(tool_calls_by_idx) - 1, 0)
                    idx = int(idx)
                    entry = tool_calls_by_idx.setdefault(
                        idx,
                        {
                            "id": f"call_{turn_index}_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if tc_delta.get("id"):
                        entry["id"] = str(tc_delta["id"])
                    fn_delta = tc_delta.get("function") or {}
                    if fn_delta.get("name"):
                        entry["function"]["name"] += str(fn_delta["name"])
                    if fn_delta.get("arguments"):
                        arg_piece = str(fn_delta["arguments"])
                        entry["function"]["arguments"] += arg_piece
                        added_parts.append(arg_piece)
            if added_parts:
                added = "".join(added_parts)
                stream_tail = (stream_tail + added)[-1200:]
                chars_since_check += len(added)
                if chars_since_check >= 256:
                    chars_since_check = 0
                    if _has_repetition_loop(stream_tail):
                        finish_reason = "repetition_loop"
                        broke_loop = True
                        break
        if broke_loop:
            logger.warning(
                "[build] Aborted local turn %d stream after detecting repetition loop.",
                turn_index + 1,
            )

    ordered_tcs = [tool_calls_by_idx[i] for i in sorted(tool_calls_by_idx)]
    return "".join(content_parts), ordered_tcs, finish_reason, turn_usage


class LocalOpenAIAdapter:
    """Lightweight native Python tool-calling loop over local OpenAI-compatible servers."""

    harness = "local_openai"

    def package_version(self) -> str | None:
        return f"httpx/{httpx.__version__}"

    async def check_auth(self) -> AuthStatus:
        version = self.package_version()
        return AuthStatus(
            provider=self.harness,
            available=True,
            authenticated=True,
            auth_mode="local",
            version=version,
            details=f"Native local OpenAI tool-loop ready ({version}).",
        )

    async def run_workspace_task(
        self,
        *,
        candidate: LocalModelCandidate,
        prompt_text: str,
        app_dir: Path,
        timeout_seconds: float = 600.0,
        sandbox: bool = True,  # accepted for parity with AntigravitySDKAdapter
        allow_shell: bool = False,
        skill_dirs: list[Path] | None = None,
        stall_seconds: float | None = None,
        stall_poll_seconds: float = 10.0,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
        max_turns: int = 24,
        protected_files: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[WorkspaceBuildResult, HarnessInvocationResult]:
        """Run a multi-turn OpenAI tool-calling loop inside `app_dir` without `os.chdir`."""
        del sandbox, skill_dirs  # skill paths are indexed in prompt_text & readable under .agents/
        app_dir.mkdir(parents=True, exist_ok=True)
        resolved_app_dir = app_dir.resolve()
        before_hashes = snapshot_file_hashes(resolved_app_dir)
        before_git = snapshot_git_status(resolved_app_dir)
        protected_lower = _normalize_protected_set(before_hashes, protected_files)

        shell_hint = ", `run_command`" if allow_shell else ""
        system_prompt = _SYSTEM_PROMPT_BASE.format(
            shell_hint=shell_hint, tiny_jpeg_uri=_TINY_JPEG_URI
        )
        tools = _tool_schemas(allow_shell=allow_shell)
        url = f"{candidate.base_url.rstrip('/')}/chat/completions"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        totals: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "usage_reported": False,
        }

        async def _dispatch_tool(name: str, args_raw: str) -> str:
            try:
                args = json.loads(args_raw) if args_raw.strip() else {}
            except json.JSONDecodeError as exc:
                return (
                    f"Tool argument JSON error for '{name}': {exc}. "
                    "Retry with complete valid JSON (and smaller file content if truncated)."
                )
            if not isinstance(args, dict):
                return f"Tool error for '{name}': arguments must be a JSON object."

            try:
                if name == "list_files":
                    return _list_files_tool(resolved_app_dir, str(args.get("path") or "."))
                if name == "read_file":
                    return _read_file_tool(
                        resolved_app_dir,
                        str(args.get("path") or ""),
                        start_line=int(args.get("start_line") or 1),
                        end_line=int(args.get("end_line") or 500),
                    )
                if name == "write_file":
                    return _write_file_tool(
                        resolved_app_dir,
                        str(args.get("path") or ""),
                        args.get("content", ""),
                        protected_lower=protected_lower,
                    )
                if name == "edit_file":
                    return _edit_file_tool(
                        resolved_app_dir,
                        str(args.get("path") or ""),
                        args.get("old_text", ""),
                        args.get("new_text", ""),
                        protected_lower=protected_lower,
                    )
                if name == "run_command":
                    if not allow_shell:
                        return "Error: run_command is disabled; use file tools only."
                    return await _run_command_tool(
                        resolved_app_dir,
                        str(args.get("command") or ""),
                        timeout_seconds=int(args.get("timeout_seconds") or 60),
                    )
                return f"Error: unknown tool '{name}'."
            except Exception as exc:
                return f"Error executing '{name}': {exc}"

        async def _run_loop() -> str:
            final_text = ""
            effective_max_turns = max(1, int(max_turns))
            client_timeout = httpx.Timeout(
                connect=15.0,
                read=max(float(timeout_seconds), 60.0),
                write=30.0,
                pool=15.0,
            )
            async with httpx.AsyncClient(timeout=client_timeout) as client:
                for turn in range(effective_max_turns):
                    payload: dict[str, Any] = {
                        "model": candidate.name,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "temperature": 0.2,
                    }
                    if reasoning_effort is not None:
                        payload["reasoning_effort"] = reasoning_effort
                    if max_completion_tokens is not None:
                        payload["max_completion_tokens"] = int(max_completion_tokens)

                    turn_counted = False

                    def _on_usage(u: dict[str, int]) -> None:
                        nonlocal turn_counted
                        if not turn_counted:
                            turn_counted = True
                            _merge_usage(totals, u)

                    content, tool_calls, finish_reason, turn_usage = await _stream_chat_completion(
                        client,
                        url,
                        payload,
                        turn_index=turn,
                        on_usage=_on_usage,
                    )
                    if turn_usage and not turn_counted:
                        _merge_usage(totals, turn_usage)

                    if tool_calls:
                        valid_tcs: list[dict[str, Any]] = []
                        bad_names: list[str] = []
                        for tc in tool_calls:
                            fn = tc.get("function") or {}
                            tname = str(fn.get("name") or "")
                            targs = str(fn.get("arguments") or "")
                            if finish_reason == "repetition_loop":
                                bad_names.append(tname or "tool")
                                continue
                            try:
                                parsed_args = json.loads(targs) if targs.strip() else {}
                                if not isinstance(parsed_args, dict):
                                    raise ValueError("arguments must be a JSON object")
                            except (json.JSONDecodeError, ValueError):
                                bad_names.append(tname or "tool")
                                continue
                            valid_tcs.append(tc)

                        if valid_tcs:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": content or None,
                                    "tool_calls": valid_tcs,
                                }
                            )
                            call_summary: list[str] = []
                            for tc in valid_tcs:
                                fn = tc.get("function") or {}
                                tname = str(fn.get("name") or "")
                                targs = str(fn.get("arguments") or "")
                                tool_output = await _dispatch_tool(tname, targs)
                                target_hint = ""
                                with contextlib.suppress(Exception):
                                    parsed = json.loads(targs) if targs.strip() else {}
                                    if isinstance(parsed, dict) and parsed.get("path"):
                                        target_hint = f"({parsed['path']})"
                                call_summary.append(f"{tname}{target_hint}")
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": str(tc.get("id") or f"call_{turn}"),
                                        "content": tool_output,
                                    }
                                )
                            if bad_names and turn + 1 < effective_max_turns:
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Note: truncated/invalid JSON tool call(s) skipped: "
                                            f"{', '.join(bad_names)}. Retry those cleanly with "
                                            "shorter valid JSON."
                                        ),
                                    }
                                )
                            logger.info(
                                "[build] local turn %d: %s (in=%s out=%s, finish=%s)",
                                turn + 1,
                                ", ".join(call_summary),
                                turn_usage.get("prompt_tokens", "?"),
                                turn_usage.get("completion_tokens", "?"),
                                finish_reason or "stop",
                            )
                            continue

                        # Every tool call had truncated/invalid JSON or hit repetition_loop.
                        # Never append malformed tool_calls JSON into history (crashes LM Studio).
                        messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    content.strip()
                                    or "[Incomplete tool call omitted due to output limit.]"
                                ),
                            }
                        )
                        logger.info(
                            "[build] local turn %d: %s recovered (in=%s out=%s, finish=%s)",
                            turn + 1,
                            ", ".join(bad_names),
                            turn_usage.get("prompt_tokens", "?"),
                            turn_usage.get("completion_tokens", "?"),
                            finish_reason or "error",
                        )
                        if turn + 1 >= effective_max_turns:
                            break
                        if finish_reason == "repetition_loop":
                            recovery_msg = (
                                f"Your `{', '.join(bad_names)}` call entered a repeating output "
                                "loop (e.g. inventing long base64 data) and was stopped. Call "
                                "`write_file` again using the tiny 1x1 JPEG data URI from your "
                                "instructions (`data:image/jpeg;base64,...` under 200 chars), "
                                "never a large base64 blob."
                            )
                        else:
                            recovery_msg = (
                                f"Your `{', '.join(bad_names)}` tool call was truncated before "
                                "producing valid JSON (hit max_completion_tokens). Call "
                                "`write_file` again with shorter, valid JSON content."
                            )
                        messages.append({"role": "user", "content": recovery_msg})
                        continue

                    messages.append({"role": "assistant", "content": content or ""})
                    if finish_reason in ("length", "repetition_loop") and (
                        turn + 1 < effective_max_turns
                    ):
                        logger.info(
                            "[build] local turn %d: text %s (in=%s out=%s)",
                            turn + 1,
                            finish_reason,
                            turn_usage.get("prompt_tokens", "?"),
                            turn_usage.get("completion_tokens", "?"),
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your response stopped before calling a file tool. Call "
                                    "`write_file` directly with concise content and continue."
                                ),
                            }
                        )
                        continue

                    logger.info(
                        "[build] local turn %d: final summary (in=%s out=%s, finish=%s)",
                        turn + 1,
                        turn_usage.get("prompt_tokens", "?"),
                        turn_usage.get("completion_tokens", "?"),
                        finish_reason or "stop",
                    )
                    final_text = content
                    break
            return final_text

        started = time.monotonic()
        loop_task: asyncio.Future[str] = asyncio.ensure_future(_run_loop())
        watched: set[asyncio.Future[Any]] = {loop_task}
        stall_task: asyncio.Future[float] | None = None
        if stall_seconds is not None and stall_seconds > 0:
            stall_task = asyncio.ensure_future(
                _wait_for_stall(resolved_app_dir, float(stall_seconds), stall_poll_seconds)
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
            if not loop_task.done():
                await _cancel_quietly(loop_task)

        if loop_task in done:
            try:
                response_text = loop_task.result()
            except HarnessError as exc:
                if not exc.usage and totals.get("usage_reported"):
                    exc.usage = dict(totals)
                raise
            except (httpx.HTTPError, OSError, ValueError) as exc:
                raise HarnessError(
                    f"Local OpenAI build failed ({candidate.name}): {type(exc).__name__}: {exc}",
                    usage=dict(totals) if totals.get("usage_reported") else None,
                ) from exc
        elif stall_task is not None and stall_task in done:
            raise HarnessError(
                f"Local OpenAI build stalled: no file changes for "
                f"{float(stall_seconds or 0):.0f}s (model={candidate.name}).",
                usage=dict(totals) if totals.get("usage_reported") else None,
            )
        else:
            raise HarnessError(
                f"Local OpenAI build timed out after {timeout_seconds:.0f}s "
                f"(model={candidate.name}).",
                usage=dict(totals) if totals.get("usage_reported") else None,
            )

        escaped_paths = find_git_changes_outside_app(resolved_app_dir, before_git)
        if escaped_paths:
            raise HarnessError(
                f"Local builder modified files outside '{app_dir}': {', '.join(escaped_paths)}",
                usage=dict(totals) if totals.get("usage_reported") else None,
            )

        elapsed_ms = round((time.monotonic() - started) * 1000, 1)
        used_model = f"{candidate.provider}/{candidate.name}"
        build_result = capture_workspace_build_result(
            resolved_app_dir,
            before_hashes,
            summary=response_text.strip() or f"Workspace built locally with {used_model}",
            harness=self.harness,
            harness_version=self.package_version(),
            model=used_model,
            exit_code=0,
        )
        invocation = HarnessInvocationResult(
            text=response_text.strip() or build_result.summary,
            used_model=used_model,
            harness=self.harness,
            harness_version=self.package_version(),
            auth_mode="local",
            elapsed_ms=elapsed_ms,
            prompt_tokens=int(totals["prompt_tokens"]),
            completion_tokens=int(totals["completion_tokens"]),
            cached_tokens=int(totals["cached_tokens"]),
            reasoning_tokens=int(totals["reasoning_tokens"]),
            total_tokens=int(totals["total_tokens"]),
            cost_usd=0.0,
            cost_source="local_free",
            usage_reported=bool(totals["usage_reported"]),
            exit_code=0,
        )
        return build_result, invocation


# Silence unused-type check warning when Callable is only used in helper type hints.
_CallableRef = Callable[[dict[str, int]], None]
_AwaitableRef = Awaitable[str]
