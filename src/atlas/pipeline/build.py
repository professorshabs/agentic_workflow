"""Application code generation, materialization, and iterative fix stages."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from atlas.events import publish, redact
from atlas.harnesses import (
    AntigravitySDKAdapter,
    HarnessError,
    HarnessUnavailableError,
    LocalOpenAIAdapter,
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
from atlas.skills import (
    SkillMetadata,
    detect_used_skills,
    format_role_skill_index,
    load_role_skill_prompt,
    resolve_role_skills,
    snapshot_skill_atimes,
    sync_workspace_skills,
)

from .render import render_build_brief
from .util import complete_json_model, extract_json_object, prompt, role_target_label
from .verify import normalize_rel_path

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
_SNAPSHOT_IGNORED_FILES = {".gitkeep", ".ignore", "PLAN.md", "BUILD_BRIEF.md"}
_SNAPSHOT_SECRET_DIRS = {
    ".aws",
    ".ssh",
    "certs",
    "certificates",
    "credentials",
    "private",
    "secrets",
}
_SNAPSHOT_SECRET_NAMES = {
    ".env",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "aws_credentials",
    "credentials",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
_SNAPSHOT_SAFE_ENV_EXAMPLES = {".env.example", ".env.sample", ".env.template", ".env.dist"}
_SNAPSHOT_SECRET_SUFFIXES = (
    ".cer",
    ".crt",
    ".der",
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pfx",
    ".pem",
    ".tfstate",
    ".token",
)
_SNAPSHOT_TRUNCATION_MARKER = "\n... [truncated]"


def _is_sensitive_snapshot_path(rel: Path) -> bool:
    """Avoid copying common credential/config files into model prompts."""
    parts = {part.casefold() for part in rel.parts}
    name = rel.name.casefold()
    if parts & _SNAPSHOT_SECRET_DIRS:
        return True
    if name in _SNAPSHOT_SAFE_ENV_EXAMPLES:
        return False
    if name in _SNAPSHOT_SECRET_NAMES or name.startswith(".env."):
        return True
    return name.endswith(_SNAPSHOT_SECRET_SUFFIXES)


class FixerRequestError(RuntimeError):
    """A failure while requesting or validating a structured fixer response."""


def _drop_partial_file_replacements(patch: PatchResult, partial_files: list[str]) -> PatchResult:
    """Refuse full replacements for files the fixer was shown cut short or not at all.

    The fixer returns whole files, so rewriting one it only saw in part would silently drop
    the part it never saw (for example the rest of a large single-page `index.html`).
    """
    blocked = {normalize_rel_path(p).casefold() for p in partial_files}
    kept = [f for f in patch.files if normalize_rel_path(f.path).casefold() not in blocked]
    dropped = [f.path for f in patch.files if normalize_rel_path(f.path).casefold() in blocked]
    if not dropped:
        return patch
    logger.warning(
        "[fix] Ignoring full replacement(s) of %s: the fixer was only shown part of %s.",
        ", ".join(dropped),
        "it" if len(dropped) == 1 else "them",
    )
    note = (
        f"\n\n(Atlas did not apply the replacement for {', '.join(dropped)}: the file was too "
        "large to show in full, so rewriting it would have lost the part that was not shown. "
        "Fix it through smaller files instead.)"
    )
    return patch.model_copy(update={"files": kept, "diagnosis": patch.diagnosis + note})


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


def snapshot_app_files(
    app_dir: Path, max_total_chars: int = 100_000, *, partial: list[str] | None = None
) -> dict[str, str]:
    """Collect current text files in `app_dir` up to `max_total_chars` for fix prompts.

    When `partial` is given, it receives the paths the snapshot could not show whole: files
    cut short, files left out once the budget ran out, and files that are not UTF-8 text.
    """
    if not app_dir.is_dir() or max_total_chars <= 0:
        return {}
    root = app_dir.resolve()
    snapshot: dict[str, str] = {}
    remaining = max_total_chars
    single_file_cap = 40_000 if max_total_chars >= 40_000 else max_total_chars

    for path in sorted(root.rglob("*")):
        if remaining <= 0 and partial is None:
            break
        if path.is_symlink() or not path.is_file() or path.name == ".gitkeep":
            continue
        rel = path.relative_to(root)
        if any(part in _SNAPSHOT_IGNORED_DIRS for part in rel.parts):
            continue
        if len(rel.parts) == 1 and rel.name in _SNAPSHOT_IGNORED_FILES:
            continue
        if _is_sensitive_snapshot_path(rel):
            continue
        try:
            resolved = path.resolve()
            resolved_rel = resolved.relative_to(root).as_posix()
            if root not in resolved.parents or resolved_rel != rel.as_posix():
                continue
        except (OSError, ValueError):
            continue
        cap = min(remaining, single_file_cap)
        if cap <= 0:
            if partial is not None and len(partial) < 500:
                partial.append(rel.as_posix())
            continue
        try:
            with path.open("r", encoding="utf-8") as stream:
                text = stream.read(cap + 1)
        except UnicodeDecodeError:
            if partial is not None and len(partial) < 500:
                partial.append(rel.as_posix())
            continue
        except OSError:
            continue
        if len(text) > cap:
            if partial is not None and len(partial) < 500:
                partial.append(rel.as_posix())
            marker_size = min(cap, len(_SNAPSHOT_TRUNCATION_MARKER))
            prefix_size = cap - marker_size
            chunk = text[:prefix_size] + _SNAPSHOT_TRUNCATION_MARKER[:marker_size]
        else:
            chunk = text
        if chunk:
            snapshot[rel.as_posix()] = chunk
            remaining -= len(chunk)

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
                "and, if you can run commands, run pytest / node test against them while "
                "building; do NOT delete or weaken them."
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
    protected_files: list[str] | None = None,
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
    partial_files: list[str] = []
    files_snapshot = snapshot_app_files(app_dir, partial=partial_files)

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
    if protected_files:
        payload["protected_files"] = sorted(protected_files)
    if partial_files:
        payload["partial_files"] = sorted(partial_files)
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
    try:
        _, patch = await complete_json_model(
            router,
            model=model,
            role="fixer",
            system=system,
            user=json.dumps(payload, indent=2),
            schema=PatchResult,
            backend_override=backend_override,
        )
    except TokenBudgetExceededError:
        raise
    except Exception as exc:
        detail = redact(str(exc).strip().splitlines()[0] if str(exc).strip() else "", 300)
        raise FixerRequestError(
            f"Fixer request failed with {type(exc).__name__}" + (f": {detail}" if detail else "")
        ) from exc
    if partial_files:
        patch = _drop_partial_file_replacements(patch, partial_files)
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
            adapter = get_harness_adapter(provider)
            sup_extra: dict[str, Any] = {}
            if rcfg.get("effort") is not None:
                sup_extra["effort"] = str(rcfg["effort"])

            async def _fallback_to_router(reason: BaseException) -> ReviewerVerdict:
                logger.warning(
                    "Supervisor '%s' harness failed (%s); falling back to ModelRouter.",
                    reviewer_name,
                    reason,
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

            repair_attempts = max(
                0,
                int(
                    rcfg.get(
                        "repair_attempts",
                        router.cfg.get("limits", {}).get("repair_attempts", 1),
                    )
                ),
            )
            prompt_user = json.dumps(payload, indent=2)
            last_parse_exc: BaseException | None = None
            for attempt in range(repair_attempts + 1):
                started = time.monotonic()
                try:
                    router._check_budget()
                    res = await adapter.run_structured(
                        role="supervisor",
                        model=model,
                        system=system,
                        user=prompt_user,
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
                    failed_usage: dict[str, Any] = dict(getattr(exc, "usage", None) or {})
                    router.record_harness_call(
                        role="supervisor",
                        requested_model=model,
                        used_model=str(failed_usage.get("used_model") or model),
                        harness=provider,
                        harness_version=None,
                        auth_mode=failed_usage.get("auth_mode"),
                        elapsed_ms=round((time.monotonic() - started) * 1000, 1),
                        prompt_tokens=int(failed_usage.get("prompt_tokens") or 0),
                        completion_tokens=int(failed_usage.get("completion_tokens") or 0),
                        cached_tokens=int(failed_usage.get("cached_tokens") or 0),
                        reasoning_tokens=int(failed_usage.get("reasoning_tokens") or 0),
                        total_tokens=int(failed_usage.get("total_tokens") or 0),
                        cost_usd=failed_usage.get("cost_usd"),
                        cost_source=str(failed_usage.get("cost_source") or "unavailable"),
                        usage_reported=bool(failed_usage.get("usage_reported")),
                        api_cost_usd=failed_usage.get("api_cost_usd"),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if rcfg.get("fallback") == "router":
                        return await _fallback_to_router(exc)
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
                    api_cost_usd=res.api_cost_usd,
                )
                try:
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
                except (ValidationError, ValueError) as exc:
                    last_parse_exc = exc
                    logger.warning(
                        "Supervisor '%s' returned invalid JSON/schema (attempt %d/%d): %s: %s",
                        reviewer_name,
                        attempt + 1,
                        repair_attempts + 1,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt < repair_attempts:
                        prompt_user = (
                            f"{json.dumps(payload, indent=2)}\n\n"
                            "Your previous response did not match the SupervisorOutput JSON "
                            f"schema ({type(exc).__name__}: {exc}). Return ONLY valid JSON "
                            "matching the schema.\n"
                            f"Previous output snippet:\n{res.text[:1500]}"
                        )

            assert last_parse_exc is not None
            if rcfg.get("fallback") == "router":
                return await _fallback_to_router(last_parse_exc)
            raise last_parse_exc

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
    file_tools_only: bool = False,
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
    if file_tools_only:
        workspace_notes.append(
            "- You have file tools only (view, create and edit files); you cannot run shell "
            "commands. Atlas runs the tests after you finish, so read the test files carefully "
            "and write code that passes them."
        )
    if require_absolute_paths and resolved_root is not None:
        workspace_notes.append(
            f"- IMPORTANT: Always pass full absolute paths under `{resolved_root}` to every "
            f"file tool (for example `{resolved_root}/BUILD_BRIEF.md`, never relative filenames "
            "like `BUILD_BRIEF.md`)."
        )
        workspace_notes.append(
            "- Write small files first (such as `pyproject.toml`, `README.md`, or small helper "
            "modules) before large files, and keep reasoning concise so progress lands on disk "
            "quickly."
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
        api_cost_usd=invocation.api_cost_usd,
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
    protected_files: list[str] | None = None,
) -> WorkspaceBuildResult:
    """Resolve the configured local model and build in `root` with the local builder."""
    local_harness_name = str(backend.get("local_harness") or "antigravity_sdk")
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
        provider=local_harness_name,
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
            provider=local_harness_name,
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
        provider=local_harness_name,
        model=f"{candidate.provider}/{candidate.name}",
        stall_seconds=backend.get("stall_seconds"),
        timeout_seconds=timeout_seconds,
    )
    adapter_label = (
        "native local OpenAI loop"
        if local_harness_name == "local_openai"
        else "Antigravity SDK agent"
    )
    logger.info(
        "[build] Selected local model %s/%s — starting %s...",
        candidate.provider,
        candidate.name,
        adapter_label,
    )
    local_adapter: AntigravitySDKAdapter | LocalOpenAIAdapter = (
        LocalOpenAIAdapter() if local_harness_name == "local_openai" else AntigravitySDKAdapter()
    )
    run_kwargs: dict[str, Any] = {
        "candidate": candidate,
        "prompt_text": prompt_text,
        "app_dir": root,
        "timeout_seconds": timeout_seconds,
        "skill_dirs": skill_dirs,
    }
    if backend.get("sandbox") is False:
        run_kwargs["sandbox"] = False
    if backend.get("allow_shell") is True:
        run_kwargs["allow_shell"] = True
    if backend.get("stall_seconds"):
        run_kwargs["stall_seconds"] = float(backend["stall_seconds"])
    if backend.get("reasoning_effort") is not None:
        run_kwargs["reasoning_effort"] = str(backend["reasoning_effort"])
    if backend.get("max_completion_tokens") is not None:
        run_kwargs["max_completion_tokens"] = int(backend["max_completion_tokens"])
    if local_harness_name == "local_openai" and backend.get("max_turns") is not None:
        run_kwargs["max_turns"] = int(backend["max_turns"])
    if local_harness_name == "local_openai" and protected_files:
        run_kwargs["protected_files"] = protected_files
    started = time.monotonic()
    try:
        result, invocation = await local_adapter.run_workspace_task(**run_kwargs)
    except Exception as exc:
        err_usage = getattr(exc, "usage", None)
        u_dict = err_usage if isinstance(err_usage, dict) else {}
        router.record_harness_call(
            role="builder",
            requested_model=candidate.name,
            used_model=f"{candidate.provider}/{candidate.name}",
            harness=local_adapter.harness,
            harness_version=local_adapter.package_version(),
            auth_mode="local",
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            prompt_tokens=int(u_dict.get("prompt_tokens") or 0),
            completion_tokens=int(u_dict.get("completion_tokens") or 0),
            cached_tokens=int(u_dict.get("cached_tokens") or 0),
            reasoning_tokens=int(u_dict.get("reasoning_tokens") or 0),
            total_tokens=int(u_dict.get("total_tokens") or 0),
            cost_usd=0.0,
            cost_source="local_free",
            usage_reported=bool(u_dict.get("usage_reported", False)),
            error=f"{type(exc).__name__}: {exc}",
        )
        publish(
            "attempt",
            "build",
            "failed",
            attempt="local",
            provider=local_harness_name,
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
            provider=local_harness_name,
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
    started = time.monotonic()
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
            elapsed_ms=round((time.monotonic() - started) * 1000, 1),
            prompt_tokens=int(failed_usage.get("prompt_tokens") or 0),
            completion_tokens=int(failed_usage.get("completion_tokens") or 0),
            cached_tokens=int(failed_usage.get("cached_tokens") or 0),
            reasoning_tokens=int(failed_usage.get("reasoning_tokens") or 0),
            total_tokens=int(failed_usage.get("total_tokens") or 0),
            cost_usd=failed_usage.get("cost_usd"),
            cost_source=str(failed_usage.get("cost_source") or "unavailable"),
            usage_reported=bool(failed_usage.get("usage_reported")),
            api_cost_usd=failed_usage.get("api_cost_usd"),
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


def _existing_skill_links(root: Path, skills: list[SkillMetadata]) -> list[Path]:
    """Links in `.agents/skills` that already point at the configured skill folders."""
    found: list[Path] = []
    base = root.resolve() / ".agents" / "skills"
    for meta in skills:
        skill_dir = Path(meta.path).parent.resolve()
        link = base / skill_dir.name
        try:
            if link.is_symlink() and link.resolve() == skill_dir:
                found.append(link)
        except OSError:
            continue
    return found


def _attach_skill_usage(
    result: WorkspaceBuildResult,
    *,
    resolved_skills: list[SkillMetadata],
    skills_mode: Literal["progressive", "injected"] | None,
    before_atimes: dict[str, int] | None,
) -> WorkspaceBuildResult:
    available = [m.name for m in resolved_skills]
    if not available or skills_mode is None:
        return result
    if skills_mode == "injected":
        used = list(available)
    else:
        used = detect_used_skills(
            resolved_skills, summary_text=result.summary, before_atimes=before_atimes
        )
    logger.info(
        "[build] Builder skills used: %s (available: %s; mode=%s)",
        ", ".join(used) if used else "none",
        ", ".join(available),
        skills_mode,
    )
    return result.model_copy(
        update={
            "available_skills": available,
            "used_skills": used,
            "skills_mode": skills_mode,
        }
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
    resolved_skills: list[SkillMetadata] | None = None,
    skills_mode: Literal["progressive", "injected"] | None = None,
) -> WorkspaceBuildResult:
    """Hand the build to stronger models in order until one succeeds."""
    failures = [f"local: {type(local_error).__name__}: {local_error}"]
    skills_list = resolved_skills or []
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
        # Antigravity scans every SKILL.md when it starts, so access times would mark all
        # skills as used; trust only its "Skills used:" line there.
        attempt_atimes = (
            snapshot_skill_atimes(skills_list)
            if target_type == "harness" and provider != "antigravity"
            else None
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
            result = _attach_skill_usage(
                result,
                resolved_skills=skills_list,
                skills_mode=skills_mode if target_type == "harness" else None,
                before_atimes=attempt_atimes,
            )
            publish(
                "attempt",
                "build",
                "passed",
                attempt=label,
                provider=provider,
                model=result.model,
                files=written,
                available_skills=result.available_skills,
                used_skills=result.used_skills,
                skills_mode=result.skills_mode,
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

    def _write_plan_file() -> str | None:
        if not (plan_markdown and plan_markdown.strip()):
            return None
        (root / _PLAN_FILE).write_text(plan_markdown.rstrip() + "\n", encoding="utf-8")
        return _PLAN_FILE

    def _seed_workspace(*, for_local_model: bool = False) -> tuple[str | None, str | None, bool]:
        plan_file: str | None = None
        brief_file: str | None = None
        if plan_markdown and plan_markdown.strip():
            if is_local and plan is not None:
                brief = render_build_brief(
                    plan,
                    clarifications=clarifications,
                    e2e_suite=e2e_suite,
                    max_chars=int(backend.get("brief_max_chars") or 24_000),
                )
                (root / _BRIEF_FILE).write_text(brief, encoding="utf-8")
                brief_file = _BRIEF_FILE
            if for_local_model and brief_file:
                # The full plan (~30K tokens) would fill most of the local model's context,
                # and it reads the file when it can even when told not to. It gets the brief;
                # PLAN.md is written back once its attempt ends.
                (root / _PLAN_FILE).unlink(missing_ok=True)
            else:
                plan_file = _write_plan_file()
        preseeded = False
        if e2e_suite is not None and e2e_suite.files:
            materialize_files(root, e2e_suite.files, clean=False)
            preseeded = True
        return plan_file, brief_file, preseeded

    plan_file, brief_file, e2e_preseeded = _seed_workspace(for_local_model=is_local)

    timeout_seconds = float(backend.get("timeout_seconds") or router.timeout_for("builder"))
    sandbox = bool(backend.get("sandbox", True))
    skills_cfg = router.skills_config()
    resolved_skills = resolve_role_skills("builder", skills_cfg) if skills_cfg else []
    agents_dir = root / ".agents"
    agents_skills = agents_dir / "skills"
    agents_dir_preexisting = agents_dir.exists() or agents_dir.is_symlink()
    agents_skills_preexisting = agents_skills.exists() or agents_skills.is_symlink()
    created_links = (
        sync_workspace_skills(root, skills_cfg, role="builder") if skills_cfg else []
    )
    # Links left by a run that was killed before its cleanup still expose the skills.
    linked_skills = created_links or _existing_skill_links(root, resolved_skills)
    discovery_mode = str(skills_cfg.get("discovery_mode") or "").strip().lower()
    skills_mode: Literal["progressive", "injected"] | None = None
    local_harness_name = str(backend.get("local_harness") or "antigravity_sdk")
    needs_abs_paths = is_local and local_harness_name == "antigravity_sdk"
    if discovery_mode == "progressive" and linked_skills:
        skill_prompt = format_role_skill_index(
            "builder", skills_cfg, workspace_root=root if needs_abs_paths else None
        )
        offload_skill_prompt = format_role_skill_index("builder", skills_cfg)
        if resolved_skills:
            skills_mode = "progressive"
    else:
        skill_prompt = load_role_skill_prompt("builder", skills_cfg)
        offload_skill_prompt = skill_prompt
        if resolved_skills and skill_prompt:
            skills_mode = "injected"

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
        require_absolute_paths=needs_abs_paths,
        file_tools_only=is_local and backend.get("allow_shell") is not True,
    )
    skill_dirs = [root / ".agents" / "skills"] if linked_skills else None
    avail_names = [m.name for m in resolved_skills]
    logger.info(
        "[build] Prepared workspace %s (PLAN.md=%s, brief=%s, e2e_files=%d, skills=%s, mode=%s)",
        root,
        bool(plan_file),
        bool(brief_file),
        len(e2e_suite.files) if e2e_preseeded and e2e_suite else 0,
        ", ".join(avail_names) if avail_names else "none",
        skills_mode or "none",
    )
    try:
        if is_local:
            offload = [dict(t) for t in (backend.get("offload") or []) if isinstance(t, dict)]
            try:
                local_protected: list[str] = []
                if e2e_preseeded and e2e_suite is not None:
                    local_protected.extend(normalize_rel_path(f.path) for f in e2e_suite.files)
                for extra_doc in (brief_file, plan_file):
                    if extra_doc:
                        local_protected.append(extra_doc)
                result = await _run_local_build(
                    router,
                    backend,
                    root,
                    prompt_text=prompt_text,
                    timeout_seconds=timeout_seconds,
                    skill_dirs=skill_dirs,
                    local_model_override=local_model_override,
                    emit_passed=False,
                    protected_files=local_protected or None,
                )
                written = len(result.created_files) + len(result.modified_files)
                if offload and written == 0:
                    raise HarnessError(
                        f"Local model {result.model} finished without writing any files."
                    )
                # The Antigravity SDK reads every SKILL.md under skill_dirs during startup
                # discovery, bumping OS st_atime_ns even when the model never opens the skill.
                result = _attach_skill_usage(
                    result,
                    resolved_skills=resolved_skills,
                    skills_mode=skills_mode,
                    before_atimes=None,
                )
                publish(
                    "attempt",
                    "build",
                    "passed",
                    attempt="local",
                    provider=local_harness_name,
                    model=result.model,
                    files=written,
                    available_skills=result.available_skills,
                    used_skills=result.used_skills,
                    skills_mode=result.skills_mode,
                )
                return result
            except TokenBudgetExceededError:
                raise
            except Exception as exc:
                if not offload:
                    raise
                local_error: BaseException = exc
            finally:
                # PLAN.md was kept away from the local model; put it back for the offload
                # builders and for the product folder.
                plan_file = _write_plan_file() or plan_file

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
                resolved_skills=resolved_skills,
                skills_mode=skills_mode,
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
        harness_atimes = (
            snapshot_skill_atimes(resolved_skills) if provider != "antigravity" else None
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
        harness_result = _attach_skill_usage(
            harness_result,
            resolved_skills=resolved_skills,
            skills_mode=skills_mode,
            before_atimes=harness_atimes,
        )
        publish(
            "attempt",
            "build",
            "passed",
            attempt=provider,
            provider=provider,
            model=harness_result.model,
            files=len(harness_result.created_files) + len(harness_result.modified_files),
            available_skills=harness_result.available_skills,
            used_skills=harness_result.used_skills,
            skills_mode=harness_result.skills_mode,
        )
        return harness_result
    finally:
        skill_link_targets = {
            Path(meta.path).parent.name: Path(meta.path).parent.resolve()
            for meta in resolved_skills
        }
        for link in created_links:
            if link.is_symlink():
                try:
                    if link.resolve() == skill_link_targets.get(link.name):
                        link.unlink()
                except OSError:
                    continue
        if (
            not agents_skills_preexisting
            and not agents_skills.is_symlink()
            and agents_skills.is_dir()
            and not any(agents_skills.iterdir())
        ):
            agents_skills.rmdir()
        if (
            not agents_dir_preexisting
            and not agents_dir.is_symlink()
            and agents_dir.is_dir()
            and not any(agents_dir.iterdir())
        ):
            agents_dir.rmdir()
