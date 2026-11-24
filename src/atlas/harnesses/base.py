"""Shared harness adapter interfaces, credential scoping, and workspace diffing."""

from __future__ import annotations

import hashlib
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from atlas.pipeline.verify import detect_stack
from atlas.schemas import (
    AuthMode,
    CostSource,
    StrictModel,
    WorkspaceBuildResult,
    _validate_relative_app_path,
)

_WORKSPACE_IGNORED_DIRS = {
    ".agents",
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}
_SAFE_BASE_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "TMPDIR",
    "TEMP",
    "TMP",
    "TERM",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
}
_SAFE_BASE_ENV_PREFIXES = ("LC_", "XDG_")


class HarnessError(RuntimeError):
    """Base error raised by harness adapters."""


class HarnessAuthError(HarnessError):
    """Raised when a harness CLI/SDK is installed but missing valid authentication."""


class HarnessUnavailableError(HarnessError):
    """Raised when a required CLI binary, local daemon, or SDK is unavailable."""


class AuthStatus(StrictModel):
    """Pre-flight binary/SDK and live authentication status for a harness."""

    provider: str
    available: bool
    authenticated: bool
    auth_mode: AuthMode | None = None
    binary_path: str | None = None
    version: str | None = None
    details: str = ""
    login_hint: str = ""


class HarnessInvocationResult(StrictModel):
    """Normalized output and honest telemetry from a harness execution."""

    text: str
    used_model: str
    harness: str
    harness_version: str | None = None
    auth_mode: AuthMode | None = None
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cost_source: CostSource = "unavailable"
    usage_reported: bool = False
    exit_code: int = 0


def scoped_subprocess_env(
    allowed_prefixes: tuple[str, ...] = (),
    allowed_names: set[str] | None = None,
    *,
    drop_names: set[str] | None = None,
) -> dict[str, str]:
    """Build a subprocess env exposing only baseline OS vars and one provider's credentials."""
    extra_names = allowed_names or set()
    blocked = drop_names or set()
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in blocked:
            continue
        if (
            key in _SAFE_BASE_ENV_KEYS
            or key.startswith(_SAFE_BASE_ENV_PREFIXES)
            or key in extra_names
            or (allowed_prefixes and key.startswith(allowed_prefixes))
        ):
            env[key] = value
    return env


def _find_git_root(path: Path) -> Path | None:
    cur = path.resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def snapshot_git_status(path: Path) -> tuple[Path, set[str]] | None:
    """Capture `git status --porcelain` lines for the enclosing git repo, if inside one."""
    root = _find_git_root(path)
    if root is None:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uall"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return root, {line for line in proc.stdout.splitlines() if len(line) > 3}
    except Exception:
        return None


def find_git_changes_outside_app(app_dir: Path, before: tuple[Path, set[str]] | None) -> list[str]:
    """Return newly dirty git status paths outside `app_dir` after a workspace run."""
    if before is None:
        return []
    root, prev_lines = before
    after = snapshot_git_status(root)
    if after is None:
        return []
    _, after_lines = after
    new_lines = sorted(after_lines - prev_lines)
    if not new_lines:
        return []

    app_resolved = app_dir.resolve()
    escaped: list[str] = []
    for line in new_lines:
        raw_path = line[3:].strip()
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1].strip()
        candidate = (root / raw_path.strip('"')).resolve()
        if candidate != app_resolved and app_resolved not in candidate.parents:
            escaped.append(raw_path)
    return escaped


def snapshot_file_hashes(app_dir: Path) -> dict[str, str]:
    """Return `{rel_path: sha256}` for all non-ignored files inside `app_dir`."""
    if not app_dir.is_dir():
        return {}
    root = app_dir.resolve()
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.name == ".gitkeep":
            continue
        if path.is_symlink():
            target = path.resolve()
            if root not in target.parents:
                raise ValueError(f"Symlink escapes workspace directory: {path}")
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _WORKSPACE_IGNORED_DIRS for part in rel.parts):
            continue
        rel_str = _validate_relative_app_path(rel.as_posix())
        hashes[rel_str] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def capture_workspace_build_result(
    app_dir: Path,
    before_hashes: dict[str, str],
    *,
    summary: str,
    harness: str,
    harness_version: str | None,
    model: str,
    exit_code: int = 0,
) -> WorkspaceBuildResult:
    """Compute a path-validated file diff & snapshot after a workspace harness finishes."""
    after_hashes = snapshot_file_hashes(app_dir)
    before_keys = set(before_hashes)
    after_keys = set(after_hashes)

    created = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(k for k in (before_keys & after_keys) if before_hashes[k] != after_hashes[k])
    snapshot = sorted(after_keys)
    stack = detect_stack(app_dir) if app_dir.is_dir() else "other"

    return WorkspaceBuildResult(
        summary=summary.strip() or f"Built application with {harness} ({model})",
        harness=harness,
        harness_version=harness_version,
        model=model,
        app_dir=str(app_dir),
        stack=stack,
        created_files=created,
        modified_files=modified,
        deleted_files=deleted,
        files_snapshot=snapshot,
        exit_code=exit_code,
    )


class HarnessAdapter(ABC):
    """Interface implemented by CLI/SDK agent harnesses."""

    provider: str

    @abstractmethod
    async def check_auth(self, auth: str | None = None) -> AuthStatus:
        """Check binary/SDK availability and live authentication status."""

    @abstractmethod
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
        """Execute a structured, non-mutating call returning JSON or plain text."""

    @abstractmethod
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
        """Execute a workspace build/edit task confined inside `app_dir`."""
