"""Application code generation, materialization, and iterative fix stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from atlas.router import ModelRouter
from atlas.schemas import (
    GeneratedApp,
    GeneratedFile,
    ParsedSpec,
    PatchResult,
    Plan,
    VerificationRound,
)

from .util import complete_json_model, prompt

_SNAPSHOT_IGNORED_DIRS = {
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


def _resolve_inside(app_root: Path, rel_path: str) -> Path:
    target = (app_root / rel_path).resolve()
    if app_root not in target.parents:
        raise ValueError(f"Path escapes app directory: {rel_path}")
    return target


def materialize_files(
    app_dir: Path,
    files: list[GeneratedFile],
    *,
    clean: bool = False,
    delete_files: list[str] | None = None,
) -> list[str]:
    """Write generated files safely into `app_dir`, returning sorted relative paths."""
    app_dir.mkdir(parents=True, exist_ok=True)
    root = app_dir.resolve()

    if clean:
        for child in list(root.iterdir()):
            if child.name == ".gitkeep":
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    for rel_del in delete_files or []:
        target = _resolve_inside(root, rel_del)
        if target.is_file() or target.is_symlink():
            target.unlink()

    written: list[str] = []
    for gen_file in files:
        target = _resolve_inside(root, gen_file.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = gen_file.content
        if content and not content.endswith("\n"):
            content = content + "\n"
        target.write_text(content, encoding="utf-8")
        written.append(gen_file.path)

    return sorted(written)


def snapshot_app_files(app_dir: Path, max_total_chars: int = 60_000) -> dict[str, str]:
    """Collect current text files in `app_dir` up to `max_total_chars` for fix prompts."""
    if not app_dir.is_dir():
        return {}
    root = app_dir.resolve()
    snapshot: dict[str, str] = {}
    remaining = max_total_chars

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(root)
        if any(part in _SNAPSHOT_IGNORED_DIRS for part in rel.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if len(text) > remaining:
            if remaining > 500:
                snapshot[rel.as_posix()] = text[:remaining] + "\n... [truncated]"
            break
        snapshot[rel.as_posix()] = text
        remaining -= len(text)

    return snapshot


def _resolve_model(router: ModelRouter, role: str, fallbacks: tuple[str, ...]) -> str:
    for candidate_role in (role, *fallbacks):
        try:
            return router.models_for(candidate_role)[0]
        except ValueError:
            continue
    raise ValueError(f"No model configured for role '{role}' or fallbacks {fallbacks}")


async def build_app(
    router: ModelRouter,
    spec: ParsedSpec,
    plan: Plan | None = None,
    *,
    raw_spec: str = "",
) -> GeneratedApp:
    """Generate a complete, testable application from a parsed spec and optional plan."""
    model = _resolve_model(router, "builder", ("judge", "parser"))
    parts = [f"PARSED SPECIFICATION:\n{spec.model_dump_json(indent=2)}"]
    if raw_spec.strip():
        parts.append(f"RAW SPECIFICATION:\n{raw_spec.strip()}")
    if plan is not None:
        decisions_payload = [
            {
                "requirement_id": d.requirement_id,
                "pick": d.pick,
                "rationale": d.rationale,
            }
            for d in plan.decisions
        ]
        parts.append(
            "ENGINEERING PLAN DECISIONS & TASKS:\n"
            + json.dumps({"decisions": decisions_payload, "tasks": plan.tasks}, indent=2)
        )

    _, generated = await complete_json_model(
        router,
        model=model,
        role="builder",
        system=prompt("build"),
        user="\n\n".join(parts),
        schema=GeneratedApp,
    )
    return generated


async def fix_app(
    router: ModelRouter,
    spec: ParsedSpec,
    app_dir: Path,
    failed_round: VerificationRound,
    *,
    plan: Plan | None = None,
) -> PatchResult:
    """Diagnose failing verification checks and generate file patches."""
    model = _resolve_model(router, "fixer", ("builder", "judge"))
    failing_checks = [
        {
            "name": c.name,
            "command": c.command,
            "exit_code": c.exit_code,
            "stdout": c.stdout[-4000:],
            "stderr": c.stderr[-4000:],
        }
        for c in failed_round.checks
        if not c.passed
    ]
    files_snapshot = snapshot_app_files(app_dir)

    payload: dict[str, object] = {
        "spec_summary": spec.summary,
        "requirements": [r.model_dump(mode="json") for r in spec.requirements],
        "constraints": spec.constraints,
        "round_number": failed_round.round_number,
        "stack_detected": failed_round.stack_detected,
        "failing_checks": failing_checks,
        "current_files": files_snapshot,
    }
    if plan is not None:
        payload["selected_tools"] = {d.requirement_id: d.pick for d in plan.decisions}

    _, patch = await complete_json_model(
        router,
        model=model,
        role="fixer",
        system=prompt("fix"),
        user=json.dumps(payload, indent=2),
        schema=PatchResult,
    )
    return patch
