"""Application code generation, materialization, and iterative fix stages."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from atlas.harnesses import (
    AntigravitySDKAdapter,
    HarnessError,
    HarnessUnavailableError,
    get_harness_adapter,
)
from atlas.local_models import DEFAULT_LM_STUDIO_URL, DEFAULT_OLLAMA_URL, resolve_local_model
from atlas.router import ModelRouter
from atlas.schemas import (
    GeneratedApp,
    GeneratedFile,
    ParsedSpec,
    PatchResult,
    Plan,
    Review,
    VerificationRound,
    WorkspaceBuildResult,
)
from atlas.skills import load_role_skill_prompt, sync_workspace_skills

from .util import complete_json_model, extract_json_object, prompt

_SNAPSHOT_IGNORED_DIRS = {
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


def spec_slug(spec_path: Path) -> str:
    """Convert a spec filename (e.g. specs/example-app.md) into a safe directory slug."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", spec_path.stem.lower()).strip(".-")
    return slug or "app"


def default_app_dir(spec_path: Path, app_root: Path = Path("app")) -> Path:
    """Return the default per-spec output directory under `app_root`."""
    return app_root / spec_slug(spec_path)


def _resolve_inside(app_root: Path, rel_path: str) -> Path:
    target = (app_root / rel_path).resolve()
    if app_root not in target.parents:
        raise ValueError(f"Path escapes app directory: {rel_path}")
    return target


def _clean_app_root(root: Path) -> None:
    for child in list(root.iterdir()):
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


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
        _clean_app_root(root)

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
    fix_round: int = 1,
    previous_diagnoses: list[str] | None = None,
) -> PatchResult:
    """Diagnose failing verification checks and generate file patches."""
    fixer_backend = router.backend_for("fixer")
    rotate = fixer_backend.get("rotate")
    if isinstance(rotate, list) and rotate:
        model = str(rotate[(max(1, fix_round) - 1) % len(rotate)])
    else:
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
    if previous_diagnoses:
        payload["previous_diagnoses"] = previous_diagnoses
    if plan is not None:
        payload["selected_tools"] = {d.requirement_id: d.pick for d in plan.decisions}

    skill_prompt = load_role_skill_prompt("fixer", router.skills_config())
    system = prompt("fix")
    if skill_prompt:
        system = f"{system.rstrip()}\n\n{skill_prompt}"

    _, patch = await complete_json_model(
        router,
        model=model,
        role="fixer",
        system=system,
        user=json.dumps(payload, indent=2),
        schema=PatchResult,
    )
    return patch


async def supervise_app(
    router: ModelRouter,
    spec: ParsedSpec,
    app_dir: Path,
    verification_round: VerificationRound,
    *,
    plan: Plan | None = None,
) -> Review:
    """Run a strictly read-only Supervisor acceptance review inside `app_dir`."""
    cfg = router.backend_for("supervisor")
    provider = str(cfg.get("provider") or "claude_code")
    model = str(cfg.get("model") or "opus")
    tools = list(cfg.get("tools") or ["Read", "Grep", "Glob"])
    auth = cfg.get("auth") or "subscription"
    max_budget = float(cfg["max_budget_usd"]) if cfg.get("max_budget_usd") is not None else 1.50
    timeout = float(cfg.get("timeout_seconds") or router.timeout_for("supervisor"))

    system = (
        "You are a strictly read-only engineering supervisor reviewing a built product inside its "
        "workspace directory. Inspect the files and verification summary, then return ONLY JSON "
        "matching this schema: {\"approved\": bool, \"summary\": str, \"issues\": [str], "
        "\"requirement_notes\": {req_id: note}}. Approve only if all functional requirements "
        "appear genuinely satisfied cleanly."
    )
    review_req = {
        "spec_summary": spec.summary,
        "requirements": [r.model_dump(mode="json") for r in spec.requirements],
        "checks": [
            {"name": c.name, "passed": c.passed, "command": c.command}
            for c in verification_round.checks
        ],
        "selected_tools": {d.requirement_id: d.pick for d in plan.decisions} if plan else {},
    }
    adapter = get_harness_adapter(provider)
    res = await adapter.run_structured(
        role="supervisor",
        model=model,
        system=system,
        user=json.dumps(review_req, indent=2),
        json_schema=Review.model_json_schema(),
        timeout_seconds=timeout,
        auth=str(auth),
        max_budget_usd=max_budget,
        tools=tools,
        cwd=app_dir.resolve(),
    )
    router.record_harness_call(
        role="supervisor",
        requested_model=model,
        used_model=res.used_model,
        harness=res.harness,
        harness_version=res.harness_version,
        auth_mode=res.auth_mode,
        elapsed_ms=res.elapsed_ms,
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        cached_tokens=res.cached_tokens,
        reasoning_tokens=res.reasoning_tokens,
        total_tokens=res.total_tokens,
        cost_usd=res.cost_usd,
        cost_source=res.cost_source,
        usage_reported=res.usage_reported,
    )
    return Review.model_validate(extract_json_object(res.text))


def _build_workspace_prompt(
    spec: ParsedSpec,
    plan: Plan | None,
    raw_spec: str,
    skill_prompt: str,
) -> str:
    parts = [prompt("build").strip()]
    if skill_prompt:
        parts.append(skill_prompt)
    parts.append(
        "WORKSPACE MODE INSTRUCTIONS:\n"
        "- Create and update application and test files directly in the current workspace.\n"
        "- Keep all files inside the project directory.\n"
        "- Include the requirement ID (for example `# R001` or `// R001`) in tests covering each "
        "functional requirement so `spec-coverage` verification passes."
    )
    parts.append(f"PARSED SPECIFICATION:\n{spec.model_dump_json(indent=2)}")
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
    return "\n\n".join(parts)


async def build_app_workspace(
    router: ModelRouter,
    spec: ParsedSpec,
    app_dir: Path,
    plan: Plan | None = None,
    *,
    raw_spec: str = "",
    clean: bool = False,
    local_model_override: str | None = None,
) -> WorkspaceBuildResult:
    """Build an application directly on disk using a local-first or configured CLI harness."""
    app_dir.mkdir(parents=True, exist_ok=True)
    root = app_dir.resolve()
    if clean:
        _clean_app_root(root)

    backend = router.backend_for("builder")
    backend_type = str(backend.get("type") or "harness")
    timeout_seconds = float(backend.get("timeout_seconds") or router.timeout_for("builder"))
    sandbox = bool(backend.get("sandbox", True))
    skills_cfg = router.skills_config()
    skill_prompt = load_role_skill_prompt("builder", skills_cfg)
    prompt_text = _build_workspace_prompt(spec, plan, raw_spec, skill_prompt)

    created_links = sync_workspace_skills(root, skills_cfg) if skills_cfg else []
    try:
        if backend_type == "local_harness":
            explicit_model = local_model_override or backend.get("local_model")
            providers = backend.get("local_providers") or {
                "ollama": DEFAULT_OLLAMA_URL,
                "lm_studio": DEFAULT_LM_STUDIO_URL,
            }
            resolution = await resolve_local_model(
                explicit_model=str(explicit_model) if explicit_model else None,
                providers=providers,
                preferred=list(backend["preferred"]) if backend.get("preferred") else None,
                require_tools=bool(backend.get("require_tools", True)),
                min_context_tokens=backend.get("min_context_tokens"),
                max_model_size_gb=backend.get("max_model_size_gb"),
                memory_budget_fraction=backend.get("memory_budget_fraction"),
                require_tool_smoke_check=bool(backend.get("require_tool_smoke_check", True)),
            )
            if resolution.selected is None:
                raise HarnessUnavailableError(resolution.reason)

            candidate = resolution.selected
            sdk_adapter = AntigravitySDKAdapter()
            try:
                result, invocation = await sdk_adapter.run_workspace_task(
                    candidate=candidate,
                    prompt_text=prompt_text,
                    app_dir=root,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                router.record_harness_call(
                    role="builder",
                    requested_model=candidate.name,
                    used_model=f"{candidate.provider}/{candidate.name}",
                    harness=sdk_adapter.harness,
                    harness_version=sdk_adapter.package_version(),
                    auth_mode="local",
                    elapsed_ms=0.0,
                    cost_usd=0.0,
                    cost_source="local_free",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            router.record_harness_call(
                role="builder",
                requested_model=candidate.name,
                used_model=invocation.used_model,
                harness=invocation.harness,
                harness_version=invocation.harness_version,
                auth_mode=invocation.auth_mode,
                elapsed_ms=invocation.elapsed_ms,
                prompt_tokens=invocation.prompt_tokens,
                completion_tokens=invocation.completion_tokens,
                cached_tokens=invocation.cached_tokens,
                reasoning_tokens=invocation.reasoning_tokens,
                total_tokens=invocation.total_tokens,
                cost_usd=invocation.cost_usd,
                cost_source=invocation.cost_source,
                usage_reported=invocation.usage_reported,
            )
            return result

        if backend_type == "harness":
            provider = str(backend.get("provider") or "antigravity")
            model = str(backend.get("model") or "gemini-4-hydrogen-high")
        else:
            raise HarnessError(f"Unsupported workspace builder backend type: {backend_type}")

        adapter = get_harness_adapter(provider)
        try:
            result, invocation = await adapter.run_workspace_task(
                role="builder",
                model=model,
                prompt_text=prompt_text,
                app_dir=root,
                timeout_seconds=timeout_seconds,
                sandbox=sandbox,
            )
        except Exception as exc:
            router.record_harness_call(
                role="builder",
                requested_model=model,
                used_model=model,
                harness=provider,
                harness_version=None,
                elapsed_ms=0.0,
                cost_usd=None,
                cost_source="unavailable",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        router.record_harness_call(
            role="builder",
            requested_model=model,
            used_model=invocation.used_model,
            harness=invocation.harness,
            harness_version=invocation.harness_version,
            auth_mode=invocation.auth_mode,
            elapsed_ms=invocation.elapsed_ms,
            prompt_tokens=invocation.prompt_tokens,
            completion_tokens=invocation.completion_tokens,
            cached_tokens=invocation.cached_tokens,
            reasoning_tokens=invocation.reasoning_tokens,
            total_tokens=invocation.total_tokens,
            cost_usd=invocation.cost_usd,
            cost_source=invocation.cost_source,
            usage_reported=invocation.usage_reported,
        )
        return result
    finally:
        for link in created_links:
            if link.is_symlink():
                link.unlink()
        agents_skills = root / ".agents" / "skills"
        if agents_skills.is_dir() and not any(agents_skills.iterdir()):
            agents_skills.rmdir()
            agents_dir = root / ".agents"
            if agents_dir.is_dir() and not any(agents_dir.iterdir()):
                agents_dir.rmdir()
