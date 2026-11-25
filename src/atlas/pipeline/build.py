"""Application code generation, materialization, and iterative fix stages."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from atlas.events import publish
from atlas.harnesses import (
    AntigravitySDKAdapter,
    HarnessError,
    HarnessUnavailableError,
    get_harness_adapter,
)
from atlas.harnesses.base import capture_workspace_build_result, snapshot_file_hashes
from atlas.local_models import (
    DEFAULT_LM_STUDIO_URL,
    DEFAULT_OLLAMA_URL,
    local_resolution_kwargs,
    resolve_local_model,
)
from atlas.router import ModelRouter, TokenBudgetExceededError
from atlas.schemas import (
    ClarificationReport,
    E2ETestSuite,
    GeneratedApp,
    GeneratedFile,
    ParsedSpec,
    PatchResult,
    Plan,
    Review,
    ReviewerVerdict,
    SupervisorOutput,
    VerificationRound,
    WorkspaceBuildResult,
)
from atlas.skills import format_role_skill_index, load_role_skill_prompt, sync_workspace_skills

from .render import render_build_brief
from .util import complete_json_model, extract_json_object, prompt, role_target_label

logger = logging.getLogger("atlas.pipeline.build")

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


def _clean_app_root(root: Path, keep: set[str] | None = None) -> None:
    keep_names = {".gitkeep", *(keep or set())}
    for child in list(root.iterdir()):
        if child.name in keep_names:
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


def _format_e2e_contract(
    e2e_suite: E2ETestSuite,
    *,
    preseeded: bool = False,
    workspace_root: Path | None = None,
) -> str:
    if not e2e_suite.verifiable:
        reason = e2e_suite.unverifiable_reason or "Not verifiable offline"
        return (
            "END-TO-END VERIFICATION NOTE:\n"
            f"- Product interface: {e2e_suite.product_interface}\n"
            f"- Automated offline E2E tests unavailable: {reason}"
        )
    if preseeded:
        test_paths = [
            str((workspace_root / f.path).resolve()) if workspace_root is not None else f.path
            for f in e2e_suite.files
        ]
        payload: dict[str, Any] = {
            "product_interface": e2e_suite.product_interface,
            "scenarios": e2e_suite.scenarios,
            "preseeded_test_files": test_paths,
            "instructions": (
                "These authoritative E2E test files are already on disk. Read them directly "
                "and run pytest / node test against them while building; do NOT delete or "
                "weaken them."
            ),
        }
        return "END-TO-END PRODUCT CONTRACT (TESTS ON DISK):\n" + json.dumps(payload, indent=2)
    payload = {
        "product_interface": e2e_suite.product_interface,
        "scenarios": e2e_suite.scenarios,
        "files": [{"path": f.path, "content": f.content} for f in e2e_suite.files],
    }
    return (
        "END-TO-END PRODUCT TEST CONTRACT "
        "(your product MUST implement this external interface cleanly):\n"
        + json.dumps(payload, indent=2)
    )


def _format_compact_spec(spec: ParsedSpec) -> str:
    payload = {
        "summary": spec.summary,
        "constraints": spec.constraints,
        "requirements": [
            {
                "id": req.id,
                "title": req.title,
                "kind": req.kind,
                "acceptance_criteria": req.acceptance_criteria,
            }
            for req in spec.requirements
        ],
    }
    return "COMPACT SPECIFICATION:\n" + json.dumps(payload, indent=2)


async def build_app(
    router: ModelRouter,
    spec: ParsedSpec,
    plan: Plan | None = None,
    *,
    raw_spec: str = "",
    e2e_suite: E2ETestSuite | None = None,
    model: str | None = None,
) -> GeneratedApp:
    """Generate a complete, testable application from a parsed spec and optional plan."""
    model = model or _resolve_model(router, "builder", ("judge", "parser"))
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
    if e2e_suite is not None:
        parts.append(_format_e2e_contract(e2e_suite))

    _, generated = await complete_json_model(
        router,
        model=model,
        role="builder",
        system=prompt("build"),
        user="\n\n".join(parts),
        schema=GeneratedApp,
    )
    return generated


def fixer_target_for_round(
    router: ModelRouter, fix_round: int
) -> tuple[str, dict[str, Any] | None, str]:
    """Return `(fallback_model, backend_override, display_label)` for `fix_round` (1-based)."""
    bcfg = router.backend_for("fixer")
    rotate = bcfg.get("rotate")
    default_fb = _resolve_model(router, "fixer", ("builder", "judge"))
    if isinstance(rotate, list) and rotate:
        entry = rotate[(max(1, fix_round) - 1) % len(rotate)]
        if isinstance(entry, dict):
            etype = str(entry.get("type") or "openrouter").strip()
            if etype == "harness":
                prov = str(entry.get("provider") or "harness")
                hmodel = str(entry.get("model") or default_fb)
                override = dict(entry)
                if override.get("fallback") is None and bcfg.get("fallback") is not None:
                    override["fallback"] = bcfg["fallback"]
                fb_model = str(entry.get("router_model") or default_fb)
                return fb_model, override, f"{prov}/{hmodel}"
            rmodel = str(entry.get("model") or default_fb)
            return rmodel, dict(entry), rmodel
        return str(entry), None, str(entry)
    return default_fb, None, role_target_label(router, "fixer", fallback_model=default_fb)


def fixer_model_for_round(router: ModelRouter, fix_round: int) -> str:
    """Display target or model string for `fix_round` (1-based)."""
    return fixer_target_for_round(router, fix_round)[2]


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
    model, backend_override, target = fixer_target_for_round(router, fix_round)

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

    logger.info(
        "[fix] Calling model %s (round %d, %d failing check(s), %d workspace file(s))...",
        target,
        fix_round,
        len(failing_checks),
        len(files_snapshot),
    )
    _, patch = await complete_json_model(
        router,
        model=model,
        role="fixer",
        system=system,
        user=json.dumps(payload, indent=2),
        schema=PatchResult,
        backend_override=backend_override,
    )
    return patch


def _merge_tagged_items(pairs: list[tuple[str, str]]) -> list[str]:
    """Deduplicate strings case-insensitively while tagging which reviewer(s) raised each."""
    ordered_keys: list[str] = []
    text_by_key: dict[str, str] = {}
    reviewers_by_key: dict[str, list[str]] = {}
    for reviewer, raw in pairs:
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in text_by_key:
            ordered_keys.append(key)
            text_by_key[key] = cleaned
            reviewers_by_key[key] = [reviewer]
        elif reviewer not in reviewers_by_key[key]:
            reviewers_by_key[key].append(reviewer)
    return [
        f"[{', '.join(reviewers_by_key[k])}] {text_by_key[k]}"
        for k in ordered_keys
    ]


def merge_supervisor_reviews(
    verdicts: list[ReviewerVerdict], *, mode: str = "unanimous"
) -> Review:
    """Merge independent reviewer verdicts into one unified Review + improvement plan."""
    if not verdicts:
        raise ValueError("Cannot merge empty supervisor verdicts list")
    if mode not in ("unanimous", "any_approve"):
        raise ValueError(f"Unsupported supervisor merge mode: {mode}")

    if len(verdicts) == 1:
        only = verdicts[0]
        plan_items = list(only.improvement_plan)
        if not only.approved and not plan_items:
            plan_items = list(only.issues)
        return Review(
            approved=only.approved,
            summary=only.summary,
            issues=list(only.issues),
            improvement_plan=plan_items,
            requirement_notes=dict(only.requirement_notes),
            reviewers=[only],
        )

    approved = (
        all(v.approved for v in verdicts)
        if mode == "unanimous"
        else any(v.approved for v in verdicts)
    )
    summary_parts = [
        f"[{v.reviewer}: {'APPROVED' if v.approved else 'CHANGES REQUESTED'}] {v.summary}"
        for v in verdicts
    ]
    issue_pairs = [
        (v.reviewer, item)
        for v in verdicts
        for item in v.issues
    ]
    plan_pairs: list[tuple[str, str]] = []
    for v in verdicts:
        items = v.improvement_plan or (v.issues if not v.approved else [])
        for item in items:
            plan_pairs.append((v.reviewer, item))

    merged_notes: dict[str, str] = {}
    req_ids: list[str] = []
    for v in verdicts:
        for req_id in v.requirement_notes:
            if req_id not in req_ids:
                req_ids.append(req_id)
    for req_id in req_ids:
        note_parts = [
            f"[{v.reviewer}] {v.requirement_notes[req_id].strip()}"
            for v in verdicts
            if req_id in v.requirement_notes and v.requirement_notes[req_id].strip()
        ]
        if note_parts:
            merged_notes[req_id] = " | ".join(note_parts)

    return Review(
        approved=approved,
        summary=" | ".join(summary_parts),
        issues=_merge_tagged_items(issue_pairs),
        improvement_plan=_merge_tagged_items(plan_pairs),
        requirement_notes=merged_notes,
        reviewers=list(verdicts),
    )


def _supervisor_fallback_model(router: ModelRouter, rcfg: dict[str, Any]) -> str:
    explicit = rcfg.get("router_model")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _resolve_model(router, "supervisor", ("researcher", "judge", "parser"))


async def supervise_app(
    router: ModelRouter,
    spec: ParsedSpec,
    app_dir: Path,
    verification_round: VerificationRound,
    *,
    plan: Plan | None = None,
) -> Review:
    """Run strictly read-only parallel Supervisor acceptance reviews inside `app_dir`."""
    cfg = router.backend_for("supervisor")
    reviewers = router.supervisor_reviewers()
    mode = str(cfg.get("mode") or "unanimous")
    system = prompt("supervisor")
    resolved_app_dir = app_dir.resolve()
    files_snapshot = snapshot_app_files(resolved_app_dir)

    base_req: dict[str, Any] = {
        "spec_summary": spec.summary,
        "requirements": [r.model_dump(mode="json") for r in spec.requirements],
        "constraints": spec.constraints,
        "checks": [
            {"name": c.name, "passed": c.passed, "command": c.command}
            for c in verification_round.checks
        ],
        "selected_tools": {d.requirement_id: d.pick for d in plan.decisions} if plan else {},
        "file_list": sorted(files_snapshot.keys()),
    }
    full_req = {**base_req, "current_files": files_snapshot}

    async def _run_one(rcfg: dict[str, Any]) -> ReviewerVerdict:
        reviewer_name = str(
            rcfg.get("name") or rcfg.get("provider") or rcfg.get("model") or "supervisor"
        ).strip()
        rtype = str(rcfg.get("type") or "harness").strip()
        rev_started = time.monotonic()
        logger.info("[supervisor] Starting reviewer '%s' (%s)...", reviewer_name, rtype)
        publish("reviewer", "supervisor", "running", reviewer=reviewer_name, harness=rtype)

        def _finish_verdict(verdict: ReviewerVerdict) -> ReviewerVerdict:
            publish(
                "reviewer",
                "supervisor",
                "passed" if verdict.approved else "partial",
                reviewer=verdict.reviewer,
                model=verdict.model,
                harness=verdict.harness,
                approved=verdict.approved,
                issues=verdict.issues[:20],
                summary=verdict.summary[:800],
                seconds=round(time.monotonic() - rev_started, 1),
            )
            logger.info(
                "[supervisor] Reviewer '%s' (%s/%s): %s in %.1fs (%d issue(s))",
                verdict.reviewer,
                verdict.harness,
                verdict.model,
                "APPROVED" if verdict.approved else "CHANGES REQUESTED",
                time.monotonic() - rev_started,
                len(verdict.issues),
            )
            return verdict

        if rtype == "harness":
            provider = str(rcfg.get("provider") or "claude_code")
            model = str(
                rcfg.get("model")
                or ("opus" if provider == "claude_code" else "gemini-4-neon-high")
            )
            auth = rcfg.get("auth") or ("subscription" if provider == "claude_code" else None)
            max_budget = (
                float(rcfg["max_budget_usd"])
                if rcfg.get("max_budget_usd") is not None
                else (1.50 if provider == "claude_code" else None)
            )
            tools = (
                list(rcfg.get("tools") or ["Read", "Grep", "Glob"])
                if provider == "claude_code"
                else None
            )
            timeout = float(rcfg.get("timeout_seconds") or router.timeout_for("supervisor"))
            payload = base_req if provider == "claude_code" else full_req
            started = time.monotonic()
            adapter = get_harness_adapter(provider)
            try:
                router._check_budget()
                sup_extra: dict[str, Any] = {}
                if rcfg.get("effort") is not None:
                    sup_extra["effort"] = str(rcfg["effort"])
                res = await adapter.run_structured(
                    role="supervisor",
                    model=model,
                    system=system,
                    user=json.dumps(payload, indent=2),
                    json_schema=SupervisorOutput.model_json_schema(),
                    timeout_seconds=timeout,
                    auth=str(auth) if auth is not None else None,
                    max_budget_usd=max_budget,
                    tools=tools,
                    cwd=resolved_app_dir,
                    **sup_extra,
                )
            except TokenBudgetExceededError:
                raise
            except HarnessError as exc:
                router.record_harness_call(
                    role="supervisor",
                    requested_model=model,
                    used_model=model,
                    harness=provider,
                    harness_version=None,
                    elapsed_ms=round((time.monotonic() - started) * 1000, 1),
                    cost_usd=None,
                    cost_source="unavailable",
                    usage_reported=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if rcfg.get("fallback") == "router":
                    logger.warning(
                        "Supervisor '%s' harness failed (%s); falling back to ModelRouter.",
                        reviewer_name,
                        exc,
                    )
                    fb_model = _supervisor_fallback_model(router, rcfg)
                    used_model, out = await complete_json_model(
                        router,
                        model=fb_model,
                        role="supervisor",
                        system=system,
                        user=json.dumps(full_req, indent=2),
                        schema=SupervisorOutput,
                        use_harness=False,
                    )
                    return _finish_verdict(
                        ReviewerVerdict(
                            reviewer=reviewer_name,
                            model=used_model,
                            harness="openrouter",
                            approved=out.approved,
                            summary=out.summary,
                            issues=out.issues,
                            improvement_plan=out.improvement_plan,
                            requirement_notes=out.requirement_notes,
                        )
                    )
                raise

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
            out = SupervisorOutput.model_validate(extract_json_object(res.text))
            return _finish_verdict(
                ReviewerVerdict(
                    reviewer=reviewer_name,
                    model=res.used_model,
                    harness=res.harness,
                    approved=out.approved,
                    summary=out.summary,
                    issues=out.issues,
                    improvement_plan=out.improvement_plan,
                    requirement_notes=out.requirement_notes,
                )
            )

        target_model = str(rcfg.get("model") or _supervisor_fallback_model(router, rcfg))
        used_model, out = await complete_json_model(
            router,
            model=target_model,
            role="supervisor",
            system=system,
            user=json.dumps(full_req, indent=2),
            schema=SupervisorOutput,
            use_harness=False,
        )
        return _finish_verdict(
            ReviewerVerdict(
                reviewer=reviewer_name,
                model=used_model,
                harness=rtype,
                approved=out.approved,
                summary=out.summary,
                issues=out.issues,
                improvement_plan=out.improvement_plan,
                requirement_notes=out.requirement_notes,
            )
        )

    outcomes = await asyncio.gather(*(_run_one(r) for r in reviewers), return_exceptions=True)
    failed = [
        f"{rcfg.get('name', 'supervisor')}: {type(out).__name__}: {out}"
        for rcfg, out in zip(reviewers, outcomes, strict=True)
        if isinstance(out, BaseException)
    ]
    if failed:
        raise RuntimeError(
            f"Supervisor review failed for {len(failed)} of {len(reviewers)} reviewer(s): "
            + " | ".join(failed)
        )
    verdicts = [out for out in outcomes if isinstance(out, ReviewerVerdict)]
    return merge_supervisor_reviews(verdicts, mode=mode)


_PLAN_FILE = "PLAN.md"
_BRIEF_FILE = "BUILD_BRIEF.md"
_OFFLOAD_PROMPT_CHARS = 40_000


def _build_workspace_prompt(
    spec: ParsedSpec,
    plan: Plan | None,
    raw_spec: str,
    skill_prompt: str,
    e2e_suite: E2ETestSuite | None = None,
    *,
    plan_file: str | None = None,
    brief_file: str | None = None,
    e2e_preseeded: bool = False,
    previous_attempt: bool = False,
    max_prompt_chars: int = 12_000,
    workspace_root: Path | None = None,
    require_absolute_paths: bool = False,
) -> str:
    resolved_root = workspace_root.resolve() if workspace_root is not None else None

    def _path_ref(rel: str) -> str:
        if resolved_root is not None and require_absolute_paths:
            return str((resolved_root / rel).resolve())
        return rel

    parts = [prompt("build_workspace").strip()]
    if skill_prompt:
        parts.append(skill_prompt)
    workspace_notes = [
        "WORKSPACE MODE INSTRUCTIONS:",
        "- Create and update application and test files directly in the current workspace.",
        "- Keep all files inside the project directory.",
        "- Include the requirement ID (for example `# R001` or `// R001`) in tests covering each "
        "functional requirement so `spec-coverage` verification passes.",
    ]
    if resolved_root is not None:
        workspace_notes.append(f"- Workspace root directory: `{resolved_root}`.")
    if require_absolute_paths and resolved_root is not None:
        workspace_notes.append(
            f"- IMPORTANT: Always pass full absolute paths under `{resolved_root}` to every "
            f"file tool (for example `{resolved_root}/BUILD_BRIEF.md`, never relative filenames "
            "like `BUILD_BRIEF.md`)."
        )
    if brief_file:
        workspace_notes.append(
            f"- Read `{_path_ref(brief_file)}` first: it lists the chosen approach for each "
            "requirement, answers to clarifying questions and the implementation tasks."
        )
        if plan_file:
            workspace_notes.append(
                f"- `{_path_ref(plan_file)}` is the long research version of the plan; do not "
                "read it whole (it does not fit your context)."
            )
    elif plan_file:
        workspace_notes.append(
            f"- Read `{_path_ref(plan_file)}` on disk for the full architecture plan, clarified "
            "assumptions, chosen libraries, and implementation tasks before writing code."
        )
    if previous_attempt:
        workspace_notes.append(
            "- A smaller local model attempted this build first and may have left partial "
            "files here. Review them, keep what is correct, and fix or replace the rest."
        )
    parts.append("\n".join(workspace_notes))

    if plan_file or brief_file:
        parts.append(_format_compact_spec(spec))
    else:
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
    if e2e_suite is not None:
        parts.append(
            _format_e2e_contract(
                e2e_suite,
                preseeded=e2e_preseeded,
                workspace_root=resolved_root if require_absolute_paths else None,
            )
        )

    assembled = "\n\n".join(parts)
    if max_prompt_chars > 0 and len(assembled) > max_prompt_chars:
        raw_ref = brief_file or plan_file
        on_disk = _path_ref(raw_ref) if raw_ref else None
        trailer = (
            f"\n\n... [prompt trimmed to fit local context window; read `{on_disk}` on disk]"
            if on_disk
            else "\n\n... [prompt trimmed to fit local context window]"
        )
        keep = max(200, max_prompt_chars - len(trailer))
        assembled = assembled[:keep].rstrip() + trailer
    return assembled


def _record_build_success(
    router: ModelRouter, *, requested_model: str, invocation: Any
) -> None:
    router.record_harness_call(
        role="builder",
        requested_model=requested_model,
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


async def _run_local_build(
    router: ModelRouter,
    backend: dict[str, Any],
    root: Path,
    *,
    prompt_text: str,
    timeout_seconds: float,
    skill_dirs: list[Path] | None,
    local_model_override: str | None,
    emit_passed: bool = True,
) -> WorkspaceBuildResult:
    """Resolve the configured local model and build in `root` with the Antigravity SDK."""
    explicit_model = local_model_override or backend.get("local_model")
    providers = backend.get("local_providers") or {
        "ollama": DEFAULT_OLLAMA_URL,
        "lm_studio": DEFAULT_LM_STUDIO_URL,
    }
    logger.info(
        "[build] Resolving local model (explicit=%s)...",
        explicit_model or "auto",
    )
    publish(
        "attempt",
        "build",
        "running",
        attempt="local",
        phase="resolving",
        provider="antigravity_sdk",
        model=str(explicit_model or "auto"),
    )
    resolution = await resolve_local_model(
        explicit_model=str(explicit_model) if explicit_model else None,
        providers=providers,
        require_tool_smoke_check=bool(backend.get("require_tool_smoke_check", True)),
        **local_resolution_kwargs(backend),
    )
    if resolution.selected is None:
        publish(
            "attempt",
            "build",
            "failed",
            attempt="local",
            provider="antigravity_sdk",
            error=resolution.reason,
        )
        raise HarnessUnavailableError(resolution.reason)

    candidate = resolution.selected
    publish(
        "attempt",
        "build",
        "running",
        attempt="local",
        phase="building",
        provider="antigravity_sdk",
        model=f"{candidate.provider}/{candidate.name}",
        stall_seconds=backend.get("stall_seconds"),
        timeout_seconds=timeout_seconds,
    )
    logger.info(
        "[build] Selected local model %s/%s — starting Antigravity SDK agent...",
        candidate.provider,
        candidate.name,
    )
    sdk_adapter = AntigravitySDKAdapter()
    run_kwargs: dict[str, Any] = {
        "candidate": candidate,
        "prompt_text": prompt_text,
        "app_dir": root,
        "timeout_seconds": timeout_seconds,
        "skill_dirs": skill_dirs,
    }
    if backend.get("stall_seconds"):
        run_kwargs["stall_seconds"] = float(backend["stall_seconds"])
    try:
        result, invocation = await sdk_adapter.run_workspace_task(**run_kwargs)
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
        publish(
            "attempt",
            "build",
            "failed",
            attempt="local",
            provider="antigravity_sdk",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _record_build_success(router, requested_model=candidate.name, invocation=invocation)
    if emit_passed:
        publish(
            "attempt",
            "build",
            "passed",
            attempt="local",
            provider="antigravity_sdk",
            model=invocation.used_model,
            files=len(result.created_files) + len(result.modified_files),
        )
    return result


async def _run_harness_build(
    router: ModelRouter,
    *,
    provider: str,
    model: str,
    prompt_text: str,
    root: Path,
    timeout_seconds: float,
    sandbox: bool,
    auth: str | None = None,
    max_budget_usd: float | None = None,
    effort: str | None = None,
) -> WorkspaceBuildResult:
    """Build in `root` with a CLI workspace harness (`agy` or `claude_code`), recording usage."""
    adapter = get_harness_adapter(provider)
    logger.info("[build] Starting %s workspace harness (model=%s)...", provider, model)
    ws_extra: dict[str, Any] = {"effort": effort} if effort is not None else {}
    try:
        result, invocation = await adapter.run_workspace_task(
            role="builder",
            model=model,
            prompt_text=prompt_text,
            app_dir=root,
            timeout_seconds=timeout_seconds,
            sandbox=sandbox,
            auth=auth,
            max_budget_usd=max_budget_usd,
            **ws_extra,
        )
    except Exception as exc:
        failed_usage: dict[str, Any] = dict(getattr(exc, "usage", None) or {})
        router.record_harness_call(
            role="builder",
            requested_model=model,
            used_model=model,
            harness=provider,
            harness_version=None,
            auth_mode=failed_usage.get("auth_mode"),
            elapsed_ms=0.0,
            prompt_tokens=int(failed_usage.get("prompt_tokens") or 0),
            completion_tokens=int(failed_usage.get("completion_tokens") or 0),
            cached_tokens=int(failed_usage.get("cached_tokens") or 0),
            reasoning_tokens=int(failed_usage.get("reasoning_tokens") or 0),
            total_tokens=int(failed_usage.get("total_tokens") or 0),
            cost_usd=failed_usage.get("cost_usd"),
            cost_source=str(failed_usage.get("cost_source") or "unavailable"),
            usage_reported=bool(failed_usage.get("usage_reported")),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _record_build_success(router, requested_model=model, invocation=invocation)
    return result


async def _run_single_shot_build(
    router: ModelRouter,
    *,
    model: str,
    spec: ParsedSpec,
    plan: Plan | None,
    raw_spec: str,
    e2e_suite: E2ETestSuite | None,
    root: Path,
) -> WorkspaceBuildResult:
    """Build with one structured API call (the router records its usage) and write the files."""
    before_hashes = snapshot_file_hashes(root)
    generated = await build_app(
        router, spec, plan, raw_spec=raw_spec, e2e_suite=e2e_suite, model=model
    )
    materialize_files(root, generated.files, clean=False)
    if e2e_suite is not None and e2e_suite.files:
        # The authoritative E2E tests win over any copy the model returned.
        materialize_files(root, e2e_suite.files, clean=False)
    result = capture_workspace_build_result(
        root,
        before_hashes,
        summary=generated.summary,
        harness=model.split("/", 1)[0] if "/" in model else "litellm",
        harness_version=None,
        model=model,
    )
    return result.model_copy(update={"run_instructions": generated.run_instructions})


def _offload_label(target: dict[str, Any]) -> str:
    return str(
        target.get("name") or target.get("model") or target.get("provider") or "offload"
    )


async def _offload_build(
    router: ModelRouter,
    targets: list[dict[str, Any]],
    *,
    spec: ParsedSpec,
    plan: Plan | None,
    raw_spec: str,
    e2e_suite: E2ETestSuite | None,
    root: Path,
    prompt_text: str,
    sandbox: bool,
    local_error: BaseException,
) -> WorkspaceBuildResult:
    """Hand the build to stronger models in order until one succeeds."""
    failures = [f"local: {type(local_error).__name__}: {local_error}"]
    for idx, target in enumerate(targets):
        label = _offload_label(target)
        target_type = str(target.get("type") or "harness")
        has_next = idx + 1 < len(targets)
        provider = (
            str(target.get("provider") or "antigravity")
            if target_type == "harness"
            else target_type
        )
        default_model = "sonnet" if provider == "claude_code" else "gemini-4-hydrogen-high"
        display_model = str(
            target.get("model") or (default_model if target_type == "harness" else "")
        )
        publish(
            "attempt",
            "build",
            "running",
            attempt=label,
            phase="building",
            provider=provider,
            model=display_model,
        )
        try:
            router._check_budget()
            if target_type == "harness":
                model = str(target.get("model") or default_model)
                max_budget = (
                    float(target["max_budget_usd"])
                    if target.get("max_budget_usd") is not None
                    else None
                )
                logger.info(
                    "[build] Offloading to '%s' (%s workspace harness, model=%s)...",
                    label,
                    provider,
                    model,
                )
                result = await _run_harness_build(
                    router,
                    provider=provider,
                    model=model,
                    prompt_text=prompt_text,
                    root=root,
                    timeout_seconds=float(
                        target.get("timeout_seconds") or router.timeout_for("builder")
                    ),
                    sandbox=bool(target.get("sandbox", sandbox)),
                    auth=str(target["auth"]) if target.get("auth") is not None else None,
                    max_budget_usd=max_budget,
                    effort=str(target["effort"]) if target.get("effort") is not None else None,
                )
            else:
                model = str(target.get("model") or "")
                logger.info(
                    "[build] Offloading to '%s' (%s, model=%s)...", label, target_type, model
                )
                result = await _run_single_shot_build(
                    router,
                    model=model,
                    spec=spec,
                    plan=plan,
                    raw_spec=raw_spec,
                    e2e_suite=e2e_suite,
                    root=root,
                )
            written = len(result.created_files) + len(result.modified_files)
            if has_next and written == 0:
                raise HarnessError(
                    f"Offload target '{label}' ({result.model}) finished without writing any files."
                )
            publish(
                "attempt",
                "build",
                "passed",
                attempt=label,
                provider=provider,
                model=result.model,
                files=written,
            )
            return result
        except TokenBudgetExceededError:
            publish(
                "attempt",
                "build",
                "failed",
                attempt=label,
                provider=provider,
                error="Budget cap reached",
            )
            raise
        except Exception as exc:
            logger.warning(
                "[build] Offload target '%s' failed: %s: %s", label, type(exc).__name__, exc
            )
            if has_next:
                publish(
                    "attempt",
                    "build",
                    "offloaded",
                    attempt=label,
                    provider=provider,
                    next=_offload_label(targets[idx + 1]),
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                publish(
                    "attempt",
                    "build",
                    "failed",
                    attempt=label,
                    provider=provider,
                    error=f"{type(exc).__name__}: {exc}",
                )
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    raise HarnessError("Local build and every offload target failed: " + " | ".join(failures))


async def build_app_workspace(
    router: ModelRouter,
    spec: ParsedSpec,
    app_dir: Path,
    plan: Plan | None = None,
    *,
    raw_spec: str = "",
    clean: bool = False,
    local_model_override: str | None = None,
    e2e_suite: E2ETestSuite | None = None,
    plan_markdown: str | None = None,
    clarifications: ClarificationReport | None = None,
) -> WorkspaceBuildResult:
    """Build an application directly on disk using a local-first or configured CLI harness.

    With a `local_harness` builder, anything the local model cannot finish (no model ready,
    error, timeout, stall, or no files written) goes to the builder's `offload` targets in
    order, when any are configured.
    """
    app_dir.mkdir(parents=True, exist_ok=True)
    root = app_dir.resolve()
    backend = router.backend_for("builder")
    backend_type = str(backend.get("type") or "harness")
    is_local = backend_type == "local_harness"
    if clean:
        _clean_app_root(root)

    def _seed_workspace() -> tuple[str | None, str | None, bool]:
        plan_file: str | None = None
        brief_file: str | None = None
        if plan_markdown and plan_markdown.strip():
            (root / _PLAN_FILE).write_text(plan_markdown.rstrip() + "\n", encoding="utf-8")
            plan_file = _PLAN_FILE
            if is_local and plan is not None:
                brief = render_build_brief(
                    plan,
                    clarifications=clarifications,
                    e2e_suite=e2e_suite,
                    max_chars=int(backend.get("brief_max_chars") or 24_000),
                )
                (root / _BRIEF_FILE).write_text(brief, encoding="utf-8")
                brief_file = _BRIEF_FILE
        preseeded = False
        if e2e_suite is not None and e2e_suite.files:
            materialize_files(root, e2e_suite.files, clean=False)
            preseeded = True
        return plan_file, brief_file, preseeded

    plan_file, brief_file, e2e_preseeded = _seed_workspace()

    timeout_seconds = float(backend.get("timeout_seconds") or router.timeout_for("builder"))
    sandbox = bool(backend.get("sandbox", True))
    skills_cfg = router.skills_config()
    created_links = sync_workspace_skills(root, skills_cfg) if skills_cfg else []
    discovery_mode = str(skills_cfg.get("discovery_mode") or "").strip().lower()
    if discovery_mode == "progressive" and created_links:
        skill_prompt = format_role_skill_index(
            "builder", skills_cfg, workspace_root=root if is_local else None
        )
        offload_skill_prompt = format_role_skill_index("builder", skills_cfg)
    else:
        skill_prompt = load_role_skill_prompt("builder", skills_cfg)
        offload_skill_prompt = skill_prompt

    prompt_text = _build_workspace_prompt(
        spec,
        plan,
        raw_spec,
        skill_prompt,
        e2e_suite,
        plan_file=plan_file,
        brief_file=brief_file,
        e2e_preseeded=e2e_preseeded,
        workspace_root=root,
        require_absolute_paths=is_local,
    )
    skill_dirs = [root / ".agents" / "skills"] if created_links else None
    logger.info(
        "[build] Prepared workspace %s (PLAN.md=%s, brief=%s, e2e_files=%d, skills=%d)",
        root,
        bool(plan_file),
        bool(brief_file),
        len(e2e_suite.files) if e2e_preseeded and e2e_suite else 0,
        len(created_links),
    )
    try:
        if is_local:
            offload = [dict(t) for t in (backend.get("offload") or []) if isinstance(t, dict)]
            try:
                result = await _run_local_build(
                    router,
                    backend,
                    root,
                    prompt_text=prompt_text,
                    timeout_seconds=timeout_seconds,
                    skill_dirs=skill_dirs,
                    local_model_override=local_model_override,
                    emit_passed=False,
                )
                written = len(result.created_files) + len(result.modified_files)
                if offload and written == 0:
                    raise HarnessError(
                        f"Local model {result.model} finished without writing any files."
                    )
                publish(
                    "attempt",
                    "build",
                    "passed",
                    attempt="local",
                    provider="antigravity_sdk",
                    model=result.model,
                    files=written,
                )
                return result
            except TokenBudgetExceededError:
                raise
            except Exception as exc:
                if not offload:
                    raise
                local_error: BaseException = exc

            publish(
                "attempt",
                "build",
                "offloaded",
                attempt="local",
                error=f"{type(local_error).__name__}: {local_error}",
                next=_offload_label(offload[0]),
            )
            logger.warning(
                "[build] Local model could not finish (%s: %s); offloading to %s.",
                type(local_error).__name__,
                local_error,
                " -> ".join(_offload_label(t) for t in offload),
            )
            if clean:
                _clean_app_root(root, keep={".agents"})
                _seed_workspace()
            offload_prompt = _build_workspace_prompt(
                spec,
                plan,
                raw_spec,
                offload_skill_prompt,
                e2e_suite,
                plan_file=plan_file,
                e2e_preseeded=e2e_preseeded,
                previous_attempt=not clean,
                max_prompt_chars=_OFFLOAD_PROMPT_CHARS,
                workspace_root=root,
                require_absolute_paths=False,
            )
            return await _offload_build(
                router,
                offload,
                spec=spec,
                plan=plan,
                raw_spec=raw_spec,
                e2e_suite=e2e_suite,
                root=root,
                prompt_text=offload_prompt,
                sandbox=sandbox,
                local_error=local_error,
            )

        if backend_type == "harness":
            provider = str(backend.get("provider") or "antigravity")
            default_model = "sonnet" if provider == "claude_code" else "gemini-4-hydrogen-high"
            model = str(backend.get("model") or default_model)
        else:
            raise HarnessError(f"Unsupported workspace builder backend type: {backend_type}")

        publish(
            "attempt",
            "build",
            "running",
            attempt=provider,
            phase="building",
            provider=provider,
            model=model,
        )
        try:
            harness_result = await _run_harness_build(
                router,
                provider=provider,
                model=model,
                prompt_text=prompt_text,
                root=root,
                timeout_seconds=timeout_seconds,
                sandbox=sandbox,
                auth=str(backend["auth"]) if backend.get("auth") is not None else None,
                max_budget_usd=(
                    float(backend["max_budget_usd"])
                    if backend.get("max_budget_usd") is not None
                    else None
                ),
                effort=str(backend["effort"]) if backend.get("effort") is not None else None,
            )
        except Exception as exc:
            publish(
                "attempt",
                "build",
                "failed",
                attempt=provider,
                provider=provider,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        publish(
            "attempt",
            "build",
            "passed",
            attempt=provider,
            provider=provider,
            model=harness_result.model,
            files=len(harness_result.created_files) + len(harness_result.modified_files),
        )
        return harness_result
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
