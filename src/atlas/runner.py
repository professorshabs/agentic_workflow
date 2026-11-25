"""Core pipeline orchestration and approval recording shared by CLI and API."""

import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from atlas.pipeline.build import (
    build_app,
    build_app_workspace,
    fix_app,
    materialize_files,
    supervise_app,
)
from atlas.pipeline.clarify import (
    apply_clarifications_to_spec,
    clarify_with_user,
    collect_clarifying_questions,
)
from atlas.pipeline.e2e import generate_e2e_tests
from atlas.pipeline.judge import judge
from atlas.pipeline.parse import parse_spec
from atlas.pipeline.render import render_plan, render_run_report
from atlas.pipeline.research import research
from atlas.pipeline.util import role_target_label
from atlas.pipeline.verify import (
    check_e2e_tests,
    check_product_contract,
    check_requirement_coverage,
    verify_app,
)
from atlas.router import ModelRouter
from atlas.schemas import (
    ApprovalRecord,
    CheckResult,
    ClarificationReport,
    E2ETestSuite,
    GeneratedApp,
    ParsedSpec,
    Plan,
    ResearchReport,
    Review,
    VerificationReport,
    VerificationRound,
    WorkspaceBuildResult,
)
from atlas.telemetry import append_ledger, ledger_entry, summarize_usage

logger = logging.getLogger("atlas.runner")

VerifyFn = Callable[[Path], Awaitable[tuple[str, list[CheckResult]]]]


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _fmt_elapsed(started: float) -> str:
    return f"{time.monotonic() - started:.1f}s"


def _fmt_usage_brief(summary: Any) -> str:
    cost_part = f", ${summary.cost_usd:.4f}" if summary.cost_usd is not None else ""
    return f"{summary.total_calls} calls, {summary.total_tokens:,} tokens{cost_part}"


def _log_verification_round(round_obj: VerificationRound, elapsed_s: float) -> None:
    passed_count = sum(1 for c in round_obj.checks if c.passed)
    logger.info(
        "[verify] Round %d %s in %.1fs (stack=%s, %d/%d checks passed)",
        round_obj.round_number,
        "PASSED" if round_obj.passed else "FAILED",
        elapsed_s,
        round_obj.stack_detected,
        passed_count,
        len(round_obj.checks),
    )
    for chk in round_obj.checks:
        logger.info(
            "[verify]   [%s] %s (exit=%d, %.1fs)",
            "PASS" if chk.passed else "FAIL",
            chk.name,
            chk.exit_code,
            chk.duration_ms / 1000.0,
        )


def _record_in_ledger(ledger_path: Path | None, manifest: dict[str, Any], summary: Any) -> None:
    """Append this run to the project ledger. Never hides the run's own result or error."""
    if ledger_path is None:
        return
    try:
        append_ledger(
            ledger_path,
            ledger_entry(
                run_id=str(manifest["run_id"]),
                mode=str(manifest.get("mode", "plan")),
                spec_path=str(manifest.get("spec_path", "")),
                status=str(manifest.get("status", "unknown")),
                summary=summary,
                created_at=manifest.get("created_at"),
                completed_at=manifest.get("completed_at") or datetime.now(UTC).isoformat(),
                local_model=manifest.get("local_model"),
                builder_backend=manifest.get("builder_backend"),
                fix_rounds=int(manifest.get("fix_rounds") or 0),
            ),
        )
    except OSError as exc:
        logger.warning("Could not append to token ledger %s: %s", ledger_path, exc)


def _run_clarifications(
    out: Path,
    manifest: dict[str, Any],
    spec: ParsedSpec,
    report: ResearchReport | None,
    *,
    interactive_clarify: bool,
    ask_fn: Callable[[str], str] | None,
    clarification_answers: dict[str, str] | None,
) -> tuple[ParsedSpec, ClarificationReport | None]:
    questions = collect_clarifying_questions(spec, report)
    if not questions:
        logger.info("[clarify] No clarifying questions needed")
        return spec, None
    will_prompt = ask_fn is not None or (interactive_clarify and sys.stdin.isatty())
    if clarification_answers:
        answered_keys = {
            str(k).strip().lower()
            for k, v in clarification_answers.items()
            if str(v).strip()
        }
        all_prefilled = all(
            (clarification_answers.get(q.question) or "").strip()
            or q.question.strip().lower() in answered_keys
            or str(idx) in answered_keys
            for idx, q in enumerate(questions, 1)
        )
        mode = "prefilled" if (all_prefilled or not will_prompt) else "prefilled + interactive"
    else:
        mode = "interactive" if will_prompt else "non-interactive"
    logger.info("[clarify] %d clarifying question(s) (%s)", len(questions), mode)
    clarifications = clarify_with_user(
        questions,
        ask_fn=ask_fn,
        prefilled_answers=clarification_answers,
        interactive=interactive_clarify,
    )
    (out / "clarifications.json").write_text(
        clarifications.model_dump_json(indent=2), encoding="utf-8"
    )
    answered_count = sum(1 for item in clarifications.items if item.answer.strip())
    manifest["clarifications_count"] = len(clarifications.items)
    manifest["clarifications_answered"] = answered_count
    manifest["stages"]["clarify"] = "completed"
    if answered_count > 0:
        spec = apply_clarifications_to_spec(spec, clarifications)
        (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    _write_manifest(out, manifest)
    logger.info("[clarify] Done — %d/%d answered", answered_count, len(clarifications.items))
    return spec, clarifications


async def run_plan(
    spec_path: Path,
    out_root: Path,
    config_path: Path | None = None,
    router: ModelRouter | None = None,
    ledger_path: Path | None = None,
    *,
    interactive_clarify: bool = False,
    ask_fn: Callable[[str], str] | None = None,
    clarification_answers: dict[str, str] | None = None,
    generate_e2e: bool | None = None,
) -> Path:
    if not spec_path.is_file():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    raw_spec = spec_path.read_text(encoding="utf-8").strip()
    if not raw_spec:
        raise ValueError("Specification file is empty")

    active_router = router if router is not None else ModelRouter(config_path)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    # Preserve input spec immediately so failed runs remain auditable.
    (out / "spec.md").write_text(raw_spec + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "spec_path": str(spec_path),
        "stages": {},
    }
    _write_manifest(out, manifest)
    run_started = time.monotonic()
    failed = False
    logger.info("Atlas plan started: run_id=%s spec=%s out=%s", run_id, spec_path, out)

    try:
        stage_started = time.monotonic()
        logger.info(
            "[parse] Parsing specification (%s)...",
            role_target_label(active_router, "parser"),
        )
        spec = await parse_spec(active_router, raw_spec)
        manifest["stages"]["parse"] = "completed"
        (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        _write_manifest(out, manifest)
        functional_count = sum(1 for r in spec.requirements if r.kind == "functional")
        logger.info(
            "[parse] Done in %s — %d requirement(s) (%d functional), %d open question(s)",
            _fmt_elapsed(stage_started),
            len(spec.requirements),
            functional_count,
            len(spec.open_questions),
        )

        stage_started = time.monotonic()
        logger.info(
            "[research] Researching %d requirement(s) (%s)...",
            len(spec.requirements),
            role_target_label(active_router, "researcher"),
        )
        report = await research(active_router, spec)
        manifest["stages"]["research"] = "completed"
        manifest["research_notes_count"] = len(report.notes)
        manifest["research_failures_count"] = len(report.failures)
        (out / "research.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _write_manifest(out, manifest)
        logger.info(
            "[research] Done in %s — %d note(s), %d failure(s)",
            _fmt_elapsed(stage_started),
            len(report.notes),
            len(report.failures),
        )

        spec, clarifications = _run_clarifications(
            out,
            manifest,
            spec,
            report,
            interactive_clarify=interactive_clarify,
            ask_fn=ask_fn,
            clarification_answers=clarification_answers,
        )

        stage_started = time.monotonic()
        logger.info(
            "[judge] Judging architecture decisions for %d requirement(s) (%s)...",
            len(spec.requirements),
            role_target_label(active_router, "judge"),
        )
        plan = await judge(active_router, spec, report)
        plan_hash = plan.content_sha256()
        manifest["stages"]["judge"] = "completed"
        manifest["plan_sha256"] = plan_hash
        (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info(
            "[judge] Done in %s — %d decision(s), %d task(s), %d risk(s)",
            _fmt_elapsed(stage_started),
            len(plan.decisions),
            len(plan.tasks),
            len(plan.risks),
        )

        e2e_suite: E2ETestSuite | None = None
        use_e2e = (
            generate_e2e
            if generate_e2e is not None
            else active_router.require_e2e_tests
        )
        if use_e2e:
            stage_started = time.monotonic()
            logger.info("[e2e] Generating authoritative E2E product tests...")
            e2e_suite = await generate_e2e_tests(
                active_router,
                spec,
                report,
                plan,
                clarifications,
                raw_spec=raw_spec,
            )
            (out / "e2e_tests.json").write_text(
                e2e_suite.model_dump_json(indent=2), encoding="utf-8"
            )
            manifest["stages"]["e2e_tests"] = "completed"
            manifest["e2e_verifiable"] = e2e_suite.verifiable
            manifest["e2e_test_files"] = [f.path for f in e2e_suite.files]
            _write_manifest(out, manifest)
            if e2e_suite.verifiable:
                e2e_desc = ", ".join(f.path for f in e2e_suite.files) or "no files"
                logger.info(
                    "[e2e] Done in %s — verifiable (%s)", _fmt_elapsed(stage_started), e2e_desc
                )
            else:
                logger.info(
                    "[e2e] Done in %s — marked unverifiable (%s)",
                    _fmt_elapsed(stage_started),
                    e2e_suite.unverifiable_reason or "unspecified",
                )

        rendered = render_plan(
            plan,
            run_id=run_id,
            plan_sha256=plan_hash,
            research_failures=report.failures,
            clarifications=clarifications,
            e2e_suite=e2e_suite,
        )
        (out / "plan.md").write_text(rendered, encoding="utf-8")
        manifest["stages"]["render"] = "completed"
        logger.info("[plan] Rendered %s", out / "plan.md")

        manifest["status"] = "plan_ready"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as exc:
        failed = True
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[failed] Plan %s failed after %s: %s: %s",
            run_id,
            _fmt_elapsed(run_started),
            type(exc).__name__,
            exc,
        )
        raise
    finally:
        summary = summarize_usage(active_router.call_log)
        (out / "tokens.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        manifest["token_usage"] = summary.model_dump(mode="json", exclude={"calls"})
        manifest["model_calls"] = active_router.call_log
        _write_manifest(out, manifest)
        _record_in_ledger(ledger_path, manifest, summary)
        if not failed:
            logger.info(
                "[done] Plan ready in %s — %s (%s)",
                _fmt_elapsed(run_started),
                out / "plan.md",
                _fmt_usage_brief(summary),
            )

    return out


def _load_resumed_artifacts(
    from_run: Path,
) -> tuple[
    ParsedSpec,
    ResearchReport | None,
    ClarificationReport | None,
    Plan | None,
    E2ETestSuite | None,
]:
    if not from_run.is_dir():
        raise FileNotFoundError(f"Run directory not found: {from_run}")
    spec_file = from_run / "spec.json"
    if not spec_file.is_file():
        raise FileNotFoundError(f"Cannot resume from {from_run}: spec.json not found")
    spec = ParsedSpec.model_validate_json(spec_file.read_text(encoding="utf-8"))
    research_file = from_run / "research.json"
    report_obj = (
        ResearchReport.model_validate_json(research_file.read_text(encoding="utf-8"))
        if research_file.is_file()
        else None
    )
    clar_file = from_run / "clarifications.json"
    clarifications = (
        ClarificationReport.model_validate_json(clar_file.read_text(encoding="utf-8"))
        if clar_file.is_file()
        else None
    )
    plan_file = from_run / "plan.json"
    plan = (
        Plan.model_validate_json(plan_file.read_text(encoding="utf-8"))
        if plan_file.is_file()
        else None
    )
    e2e_file = from_run / "e2e_tests.json"
    e2e_suite = (
        E2ETestSuite.model_validate_json(e2e_file.read_text(encoding="utf-8"))
        if e2e_file.is_file()
        else None
    )
    return spec, report_obj, clarifications, plan, e2e_suite


async def run_build(
    spec_path: Path,
    out_root: Path,
    app_dir: Path,
    *,
    config_path: Path | None = None,
    router: ModelRouter | None = None,
    skip_research: bool = False,
    clean_app_dir: bool = False,
    max_fix_rounds: int | None = None,
    verify_fn: VerifyFn | None = None,
    report_out: Path | None = None,
    ledger_path: Path | None = None,
    local_model: str | None = None,
    interactive_clarify: bool = False,
    ask_fn: Callable[[str], str] | None = None,
    clarification_answers: dict[str, str] | None = None,
    generate_e2e: bool | None = None,
    from_run: Path | None = None,
) -> Path:
    """Run spec -> (optional research/plan) -> build -> verify/fix loop with token tracking."""
    if not spec_path.is_file():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    raw_spec = spec_path.read_text(encoding="utf-8").strip()
    if not raw_spec:
        raise ValueError("Specification file is empty")

    resumed = _load_resumed_artifacts(from_run) if from_run is not None else None

    active_router = router if router is not None else ModelRouter(config_path)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    (out / "spec.md").write_text(raw_spec + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "mode": "build",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "spec_path": str(spec_path),
        "app_dir": str(app_dir),
        "stages": {},
    }
    if from_run is not None:
        manifest["resumed_from"] = str(from_run)
    _write_manifest(out, manifest)
    run_started = time.monotonic()
    logger.info("Atlas build started: run_id=%s spec=%s app_dir=%s", run_id, spec_path, app_dir)

    spec: ParsedSpec | None = None
    plan: Plan | None = None
    report_obj: ResearchReport | None = None
    clarifications: ClarificationReport | None = None
    e2e_suite: E2ETestSuite | None = None
    rendered_plan: str | None = None
    generated_app: GeneratedApp | WorkspaceBuildResult | None = None
    verification: VerificationReport | None = None
    latest_review: Review | None = None
    written_files: list[str] = []
    tracked_files: set[str] = set()
    error_text: str | None = None

    try:
        use_e2e = (
            generate_e2e
            if generate_e2e is not None
            else active_router.require_e2e_tests
        )

        if resumed is not None:
            spec, report_obj, clarifications, plan, e2e_suite = resumed
            manifest["stages"]["parse"] = "reused"
            (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

            if report_obj is not None:
                manifest["stages"]["research"] = "reused"
                manifest["research_notes_count"] = len(report_obj.notes)
                manifest["research_failures_count"] = len(report_obj.failures)
                (out / "research.json").write_text(
                    report_obj.model_dump_json(indent=2), encoding="utf-8"
                )
            else:
                manifest["stages"]["research"] = "skipped"

            if clarifications is not None:
                manifest["stages"]["clarify"] = "reused"
                manifest["clarifications_count"] = len(clarifications.items)
                manifest["clarifications_answered"] = sum(
                    1 for item in clarifications.items if item.answer.strip()
                )
                (out / "clarifications.json").write_text(
                    clarifications.model_dump_json(indent=2), encoding="utf-8"
                )

            if plan is not None:
                plan_hash = plan.content_sha256()
                manifest["stages"]["judge"] = "reused"
                manifest["plan_sha256"] = plan_hash
                (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            else:
                manifest["stages"]["judge"] = "skipped"

            if not use_e2e:
                e2e_suite = None
            elif e2e_suite is not None:
                manifest["stages"]["e2e_tests"] = "reused"
                manifest["e2e_verifiable"] = e2e_suite.verifiable
                manifest["e2e_test_files"] = [f.path for f in e2e_suite.files]
                (out / "e2e_tests.json").write_text(
                    e2e_suite.model_dump_json(indent=2), encoding="utf-8"
                )
            else:
                stage_started = time.monotonic()
                logger.info("[e2e] Generating authoritative E2E product tests...")
                e2e_suite = await generate_e2e_tests(
                    active_router,
                    spec,
                    report_obj,
                    plan,
                    clarifications,
                    raw_spec=raw_spec,
                )
                (out / "e2e_tests.json").write_text(
                    e2e_suite.model_dump_json(indent=2), encoding="utf-8"
                )
                manifest["stages"]["e2e_tests"] = "completed"
                manifest["e2e_verifiable"] = e2e_suite.verifiable
                manifest["e2e_test_files"] = [f.path for f in e2e_suite.files]
                if e2e_suite.verifiable:
                    e2e_desc = ", ".join(f.path for f in e2e_suite.files) or "no files"
                    logger.info(
                        "[e2e] Done in %s — verifiable (%s)",
                        _fmt_elapsed(stage_started),
                        e2e_desc,
                    )
                else:
                    logger.info(
                        "[e2e] Done in %s — marked unverifiable (%s)",
                        _fmt_elapsed(stage_started),
                        e2e_suite.unverifiable_reason or "unspecified",
                    )

            if plan is not None:
                rendered_plan = render_plan(
                    plan,
                    run_id=run_id,
                    plan_sha256=plan.content_sha256(),
                    research_failures=report_obj.failures if report_obj else [],
                    clarifications=clarifications,
                    e2e_suite=e2e_suite,
                )
                (out / "plan.md").write_text(rendered_plan, encoding="utf-8")

            _write_manifest(out, manifest)
            logger.info(
                "[resume] Reused planning artifacts from %s "
                "(%d requirement(s), plan=%s, e2e=%s)",
                from_run,
                len(spec.requirements),
                bool(plan),
                bool(e2e_suite),
            )
        else:
            stage_started = time.monotonic()
            logger.info(
                "[parse] Parsing specification (%s)...",
                role_target_label(active_router, "parser"),
            )
            spec = await parse_spec(active_router, raw_spec)
            manifest["stages"]["parse"] = "completed"
            (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
            _write_manifest(out, manifest)
            functional_count = sum(1 for r in spec.requirements if r.kind == "functional")
            logger.info(
                "[parse] Done in %s — %d requirement(s) (%d functional), %d open question(s)",
                _fmt_elapsed(stage_started),
                len(spec.requirements),
                functional_count,
                len(spec.open_questions),
            )

            if skip_research:
                manifest["stages"]["research"] = "skipped"
                manifest["stages"]["judge"] = "skipped"
                logger.info("[research] Skipped (--skip-research)")
                logger.info("[judge] Skipped (--skip-research)")
                spec, clarifications = _run_clarifications(
                    out,
                    manifest,
                    spec,
                    None,
                    interactive_clarify=interactive_clarify,
                    ask_fn=ask_fn,
                    clarification_answers=clarification_answers,
                )
                if use_e2e:
                    stage_started = time.monotonic()
                    logger.info("[e2e] Generating authoritative E2E product tests...")
                    e2e_suite = await generate_e2e_tests(
                        active_router,
                        spec,
                        None,
                        None,
                        clarifications,
                        raw_spec=raw_spec,
                    )
                    (out / "e2e_tests.json").write_text(
                        e2e_suite.model_dump_json(indent=2), encoding="utf-8"
                    )
                    manifest["stages"]["e2e_tests"] = "completed"
                    manifest["e2e_verifiable"] = e2e_suite.verifiable
                    manifest["e2e_test_files"] = [f.path for f in e2e_suite.files]
                    _write_manifest(out, manifest)
                    if e2e_suite.verifiable:
                        e2e_desc = ", ".join(f.path for f in e2e_suite.files) or "no files"
                        logger.info(
                            "[e2e] Done in %s — verifiable (%s)",
                            _fmt_elapsed(stage_started),
                            e2e_desc,
                        )
                    else:
                        logger.info(
                            "[e2e] Done in %s — marked unverifiable (%s)",
                            _fmt_elapsed(stage_started),
                            e2e_suite.unverifiable_reason or "unspecified",
                        )
            else:
                stage_started = time.monotonic()
                logger.info(
                    "[research] Researching %d requirement(s) (%s)...",
                    len(spec.requirements),
                    role_target_label(active_router, "researcher"),
                )
                report_obj = await research(active_router, spec)
                manifest["stages"]["research"] = "completed"
                manifest["research_notes_count"] = len(report_obj.notes)
                manifest["research_failures_count"] = len(report_obj.failures)
                (out / "research.json").write_text(
                    report_obj.model_dump_json(indent=2), encoding="utf-8"
                )
                _write_manifest(out, manifest)
                logger.info(
                    "[research] Done in %s — %d note(s), %d failure(s)",
                    _fmt_elapsed(stage_started),
                    len(report_obj.notes),
                    len(report_obj.failures),
                )

                spec, clarifications = _run_clarifications(
                    out,
                    manifest,
                    spec,
                    report_obj,
                    interactive_clarify=interactive_clarify,
                    ask_fn=ask_fn,
                    clarification_answers=clarification_answers,
                )

                stage_started = time.monotonic()
                logger.info(
                    "[judge] Judging architecture decisions for %d requirement(s) (%s)...",
                    len(spec.requirements),
                    role_target_label(active_router, "judge"),
                )
                plan = await judge(active_router, spec, report_obj)
                plan_hash = plan.content_sha256()
                manifest["stages"]["judge"] = "completed"
                manifest["plan_sha256"] = plan_hash
                (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                logger.info(
                    "[judge] Done in %s — %d decision(s), %d task(s), %d risk(s)",
                    _fmt_elapsed(stage_started),
                    len(plan.decisions),
                    len(plan.tasks),
                    len(plan.risks),
                )

                if use_e2e:
                    stage_started = time.monotonic()
                    logger.info("[e2e] Generating authoritative E2E product tests...")
                    e2e_suite = await generate_e2e_tests(
                        active_router,
                        spec,
                        report_obj,
                        plan,
                        clarifications,
                        raw_spec=raw_spec,
                    )
                    (out / "e2e_tests.json").write_text(
                        e2e_suite.model_dump_json(indent=2), encoding="utf-8"
                    )
                    manifest["stages"]["e2e_tests"] = "completed"
                    manifest["e2e_verifiable"] = e2e_suite.verifiable
                    manifest["e2e_test_files"] = [f.path for f in e2e_suite.files]
                    if e2e_suite.verifiable:
                        e2e_desc = ", ".join(f.path for f in e2e_suite.files) or "no files"
                        logger.info(
                            "[e2e] Done in %s — verifiable (%s)",
                            _fmt_elapsed(stage_started),
                            e2e_desc,
                        )
                    else:
                        logger.info(
                            "[e2e] Done in %s — marked unverifiable (%s)",
                            _fmt_elapsed(stage_started),
                            e2e_suite.unverifiable_reason or "unspecified",
                        )

                rendered_plan = render_plan(
                    plan,
                    run_id=run_id,
                    plan_sha256=plan_hash,
                    research_failures=report_obj.failures,
                    clarifications=clarifications,
                    e2e_suite=e2e_suite,
                )
                (out / "plan.md").write_text(rendered_plan, encoding="utf-8")
                _write_manifest(out, manifest)
                logger.info("[plan] Rendered %s", out / "plan.md")

        builder_backend = active_router.backend_for("builder")
        builder_type = str(builder_backend.get("type") or "openrouter")
        manifest["build_mode"] = builder_type
        manifest["builder_backend"] = builder_type
        stage_started = time.monotonic()
        target_label = (
            f"local/{local_model}"
            if (builder_type == "local_harness" and local_model)
            else role_target_label(active_router, "builder")
        )
        logger.info(
            "[build] Building product in %s (backend=%s, target=%s)...",
            app_dir,
            builder_type,
            target_label,
        )
        if builder_type in ("local_harness", "harness"):
            generated_app = await build_app_workspace(
                active_router,
                spec,
                app_dir,
                plan,
                raw_spec=raw_spec,
                clean=clean_app_dir,
                local_model_override=local_model,
                e2e_suite=e2e_suite,
                plan_markdown=rendered_plan,
                clarifications=clarifications,
            )
            written_files = list(generated_app.files_snapshot)
            if builder_type == "local_harness":
                if generated_app.harness == "antigravity_sdk":
                    manifest["local_model"] = generated_app.model
                else:
                    # The local model could not finish; a cloud model built the product.
                    manifest["build_offloaded_to"] = generated_app.model
                    manifest["build_offload_harness"] = generated_app.harness
        else:
            generated_app = await build_app(
                active_router,
                spec,
                plan,
                raw_spec=raw_spec,
                e2e_suite=e2e_suite,
            )
            written_files = materialize_files(app_dir, generated_app.files, clean=clean_app_dir)

        if e2e_suite is not None and e2e_suite.files:
            e2e_written = materialize_files(app_dir, e2e_suite.files, clean=False)
            tracked_files.update(e2e_written)

        manifest["stages"]["build"] = "completed"
        (out / "build.json").write_text(generated_app.model_dump_json(indent=2), encoding="utf-8")
        tracked_files.update(written_files)
        _write_manifest(out, manifest)
        model_note = (
            f", model={generated_app.model}"
            if isinstance(generated_app, WorkspaceBuildResult)
            else ""
        )
        logger.info(
            "[build] Done in %s — %d file(s) written%s",
            _fmt_elapsed(stage_started),
            len(written_files),
            model_note,
        )

        rounds_limit = (
            max_fix_rounds if max_fix_rounds is not None else active_router.max_fix_rounds
        )
        rounds: list[VerificationRound] = []

        functional_ids = [r.id for r in spec.requirements if r.kind == "functional"]
        check_coverage = active_router.require_requirement_tests and bool(functional_ids)
        check_contract = active_router.require_product_contract
        e2e_paths = [f.path for f in e2e_suite.files] if e2e_suite is not None else []
        e2e_paths_set = set(e2e_paths)

        async def _run_verify() -> tuple[str, list[CheckResult]]:
            if verify_fn is not None:
                stack, found_checks = await verify_fn(app_dir)
            else:
                stack, found_checks = await verify_app(
                    app_dir,
                    timeout_seconds=active_router.verify_timeout_seconds,
                )
            if check_coverage:
                found_checks = [*found_checks, check_requirement_coverage(app_dir, functional_ids)]
            if check_contract:
                found_checks = [*found_checks, check_product_contract(app_dir)]
            if e2e_suite is not None:
                found_checks = [
                    *found_checks,
                    check_e2e_tests(
                        app_dir,
                        e2e_paths,
                        verifiable=e2e_suite.verifiable,
                        unverifiable_reason=e2e_suite.unverifiable_reason,
                    ),
                ]
            return stack, found_checks

        round_num = 1
        verify_started = time.monotonic()
        logger.info("[verify] Round %d: running verification checks in %s...", round_num, app_dir)
        stack_detected, checks = await _run_verify()
        current_round = VerificationRound(
            round_number=round_num,
            passed=bool(checks) and all(c.passed for c in checks),
            stack_detected=stack_detected,
            checks=checks,
        )
        rounds.append(current_round)
        _log_verification_round(current_round, time.monotonic() - verify_started)

        fix_attempt = 0
        previous_diagnoses: list[str] = []
        supervisor_enabled = bool(active_router.backend_for("supervisor").get("enabled", False))
        while fix_attempt <= rounds_limit:
            if current_round.passed and supervisor_enabled:
                sup_started = time.monotonic()
                rnames = ", ".join(
                    str(r.get("name") or r.get("provider") or "reviewer")
                    for r in active_router.supervisor_reviewers()
                )
                logger.info("[supervisor] Running acceptance review (%s)...", rnames)
                review = await supervise_app(
                    active_router, spec, app_dir, current_round, plan=plan
                )
                latest_review = review
                (out / "supervisor_review.json").write_text(
                    review.model_dump_json(indent=2), encoding="utf-8"
                )
                manifest["supervisor_approved"] = review.approved
                manifest["supervisor_reviewers"] = [v.reviewer for v in review.reviewers]
                logger.info(
                    "[supervisor] Verdict in %s: %s (%d issue(s))",
                    _fmt_elapsed(sup_started),
                    "APPROVED" if review.approved else "CHANGES REQUESTED",
                    len(review.issues),
                )
                if review.approved:
                    break
                reviewer_names = ", ".join(v.reviewer for v in review.reviewers)
                sup_cmd = (
                    f"supervisor acceptance review ({reviewer_names})"
                    if reviewer_names
                    else "supervisor acceptance review"
                )
                stderr_sections: list[str] = []
                if review.issues:
                    stderr_sections.append(
                        "Issues:\n" + "\n".join(f"- {item}" for item in review.issues)
                    )
                if review.improvement_plan:
                    stderr_sections.append(
                        "Improvement plan:\n"
                        + "\n".join(
                            f"{idx}. {step}"
                            for idx, step in enumerate(review.improvement_plan, 1)
                        )
                    )
                if review.requirement_notes:
                    stderr_sections.append(
                        "Requirement notes:\n"
                        + "\n".join(
                            f"- {req_id}: {note}"
                            for req_id, note in review.requirement_notes.items()
                        )
                    )
                sup_stderr = "\n\n".join(stderr_sections) or review.summary
                sup_check = CheckResult(
                    name="supervisor-review",
                    command=sup_cmd,
                    passed=False,
                    exit_code=1,
                    stdout=review.summary,
                    stderr=sup_stderr,
                )
                current_round.checks = [*current_round.checks, sup_check]
                current_round.passed = False

            if current_round.passed or fix_attempt >= rounds_limit:
                break

            fix_attempt += 1
            fix_started = time.monotonic()
            failing_names = ", ".join(c.name for c in current_round.checks if not c.passed)
            logger.info(
                "[fix] Round %d/%d: fixing [%s]...", fix_attempt, rounds_limit, failing_names
            )
            patch = await fix_app(
                active_router,
                spec,
                app_dir,
                current_round,
                plan=plan,
                fix_round=fix_attempt,
                previous_diagnoses=list(previous_diagnoses),
            )
            current_round.diagnosis = patch.diagnosis
            previous_diagnoses.append(patch.diagnosis)
            safe_deletes = [p for p in patch.delete_files if p not in e2e_paths_set]
            updated = materialize_files(
                app_dir,
                patch.files,
                clean=False,
                delete_files=safe_deletes,
            )
            tracked_files.update(updated)
            tracked_files.difference_update(safe_deletes)
            diag_line = patch.diagnosis.splitlines()[0][:120] if patch.diagnosis else ""
            logger.info(
                "[fix] Round %d/%d done in %s — updated %d file(s)%s",
                fix_attempt,
                rounds_limit,
                _fmt_elapsed(fix_started),
                len(updated),
                f" — {diag_line}" if diag_line else "",
            )

            round_num += 1
            verify_started = time.monotonic()
            logger.info(
                "[verify] Round %d: running verification checks in %s...", round_num, app_dir
            )
            stack_detected, checks = await _run_verify()
            current_round = VerificationRound(
                round_number=round_num,
                passed=bool(checks) and all(c.passed for c in checks),
                stack_detected=stack_detected,
                checks=checks,
            )
            rounds.append(current_round)
            _log_verification_round(current_round, time.monotonic() - verify_started)

        manifest["fix_rounds"] = fix_attempt

        verification = VerificationReport(
            passed=current_round.passed,
            stack_detected=current_round.stack_detected,
            rounds=rounds,
            final_checks=current_round.checks,
        )
        (out / "verification.json").write_text(
            verification.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest["stages"]["verify"] = "completed"
        manifest["verification_passed"] = verification.passed
        manifest["verification_rounds"] = len(rounds)
        manifest["status"] = "verified" if verification.passed else "verification_failed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as exc:
        manifest["status"] = "failed"
        error_text = f"{type(exc).__name__}: {exc}"
        manifest["error"] = error_text
        logger.error(
            "[failed] Build %s failed after %s: %s",
            run_id,
            _fmt_elapsed(run_started),
            error_text,
        )
        raise
    finally:
        token_summary = summarize_usage(active_router.call_log)
        (out / "tokens.json").write_text(
            token_summary.model_dump_json(indent=2), encoding="utf-8"
        )
        manifest["token_usage"] = token_summary.model_dump(mode="json", exclude={"calls"})
        manifest["model_calls"] = active_router.call_log
        sorted_files = sorted(tracked_files)
        manifest["generated_files"] = sorted_files
        if verification is not None:
            manifest["verification_passed"] = verification.passed
            manifest["verification_rounds"] = len(verification.rounds)
        report_md = render_run_report(
            run_id=run_id,
            spec_path=str(spec_path),
            status=str(manifest.get("status", "unknown")),
            spec=spec,
            generated_app=generated_app,
            verification=verification,
            token_summary=token_summary,
            written_files=sorted_files,
            app_dir=str(app_dir),
            error=error_text,
            supervisor_review=latest_review,
            clarifications=clarifications,
            e2e_suite=e2e_suite,
        )
        (out / "report.md").write_text(report_md, encoding="utf-8")
        if report_out is not None:
            report_out.parent.mkdir(parents=True, exist_ok=True)
            report_out.write_text(report_md, encoding="utf-8")
        _write_manifest(out, manifest)
        _record_in_ledger(ledger_path, manifest, token_summary)
        if error_text is None:
            logger.info(
                "[done] Build %s in %s — report=%s (%s)",
                manifest["status"],
                _fmt_elapsed(run_started),
                out / "report.md",
                _fmt_usage_brief(token_summary),
            )

    return out


def record_approval(
    run_dir: Path,
    *,
    plan_sha256: str,
    approved_by: str,
    source: Literal["cli", "api", "telegram", "github"] = "api",
    notes: str | None = None,
) -> ApprovalRecord:
    """Record an explicit human approval bound to the run's exact plan SHA-256."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in ("plan_ready", "approved"):
        raise ValueError(
            f"Run '{manifest.get('run_id')}' is not ready for approval "
            f"(status={manifest.get('status')})"
        )

    expected_hash = manifest.get("plan_sha256")
    if not expected_hash or expected_hash != plan_sha256:
        raise ValueError("Provided plan_sha256 does not match the run's generated plan")

    record = ApprovalRecord(
        run_id=manifest["run_id"],
        plan_sha256=plan_sha256,
        approved_by=approved_by,
        approved_at=datetime.now(UTC).isoformat(),
        source=source,
        notes=notes,
    )
    (run_dir / "approval.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    manifest["status"] = "approved"
    manifest["approved_at"] = record.approved_at
    manifest["approved_by"] = record.approved_by
    _write_manifest(run_dir, manifest)
    return record
