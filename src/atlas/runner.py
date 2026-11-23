"""Core pipeline orchestration and approval recording shared by CLI and API."""

import json
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from atlas.pipeline.build import build_app, fix_app, materialize_files
from atlas.pipeline.judge import judge
from atlas.pipeline.parse import parse_spec
from atlas.pipeline.render import render_plan, render_run_report
from atlas.pipeline.research import research
from atlas.pipeline.verify import verify_app
from atlas.router import ModelRouter
from atlas.schemas import (
    ApprovalRecord,
    CheckResult,
    GeneratedApp,
    ParsedSpec,
    Plan,
    VerificationReport,
    VerificationRound,
)
from atlas.telemetry import summarize_usage

VerifyFn = Callable[[Path], Awaitable[tuple[str, list[CheckResult]]]]


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


async def run_plan(
    spec_path: Path,
    out_root: Path,
    config_path: Path | None = None,
    router: ModelRouter | None = None,
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

    try:
        spec = await parse_spec(active_router, raw_spec)
        manifest["stages"]["parse"] = "completed"
        (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        _write_manifest(out, manifest)

        report = await research(active_router, spec)
        manifest["stages"]["research"] = "completed"
        manifest["research_notes_count"] = len(report.notes)
        manifest["research_failures_count"] = len(report.failures)
        (out / "research.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        _write_manifest(out, manifest)

        plan = await judge(active_router, spec, report)
        plan_hash = plan.content_sha256()
        manifest["stages"]["judge"] = "completed"
        manifest["plan_sha256"] = plan_hash
        (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        rendered = render_plan(
            plan,
            run_id=run_id,
            plan_sha256=plan_hash,
            research_failures=report.failures,
        )
        (out / "plan.md").write_text(rendered, encoding="utf-8")
        manifest["stages"]["render"] = "completed"

        manifest["status"] = "plan_ready"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = summarize_usage(active_router.call_log)
        (out / "tokens.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        manifest["token_usage"] = summary.model_dump(mode="json", exclude={"calls"})
        manifest["model_calls"] = active_router.call_log
        _write_manifest(out, manifest)

    return out


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
) -> Path:
    """Run spec -> (optional research/plan) -> build -> verify/fix loop with token tracking."""
    if not spec_path.is_file():
        raise FileNotFoundError(f"Specification file not found: {spec_path}")
    raw_spec = spec_path.read_text(encoding="utf-8").strip()
    if not raw_spec:
        raise ValueError("Specification file is empty")

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
    _write_manifest(out, manifest)

    spec: ParsedSpec | None = None
    plan: Plan | None = None
    generated_app: GeneratedApp | None = None
    verification: VerificationReport | None = None
    written_files: list[str] = []
    tracked_files: set[str] = set()
    error_text: str | None = None

    try:
        spec = await parse_spec(active_router, raw_spec)
        manifest["stages"]["parse"] = "completed"
        (out / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        _write_manifest(out, manifest)

        if skip_research:
            manifest["stages"]["research"] = "skipped"
            manifest["stages"]["judge"] = "skipped"
        else:
            report = await research(active_router, spec)
            manifest["stages"]["research"] = "completed"
            manifest["research_notes_count"] = len(report.notes)
            manifest["research_failures_count"] = len(report.failures)
            (out / "research.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            _write_manifest(out, manifest)

            plan = await judge(active_router, spec, report)
            plan_hash = plan.content_sha256()
            manifest["stages"]["judge"] = "completed"
            manifest["plan_sha256"] = plan_hash
            (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            rendered_plan = render_plan(
                plan,
                run_id=run_id,
                plan_sha256=plan_hash,
                research_failures=report.failures,
            )
            (out / "plan.md").write_text(rendered_plan, encoding="utf-8")
            _write_manifest(out, manifest)

        generated_app = await build_app(active_router, spec, plan, raw_spec=raw_spec)
        manifest["stages"]["build"] = "completed"
        (out / "build.json").write_text(generated_app.model_dump_json(indent=2), encoding="utf-8")
        written_files = materialize_files(app_dir, generated_app.files, clean=clean_app_dir)
        tracked_files.update(written_files)
        _write_manifest(out, manifest)

        rounds_limit = (
            max_fix_rounds if max_fix_rounds is not None else active_router.max_fix_rounds
        )
        rounds: list[VerificationRound] = []

        async def _run_verify() -> tuple[str, list[CheckResult]]:
            if verify_fn is not None:
                return await verify_fn(app_dir)
            return await verify_app(
                app_dir,
                timeout_seconds=active_router.verify_timeout_seconds,
            )

        stack_detected, checks = await _run_verify()
        round_num = 1
        current_round = VerificationRound(
            round_number=round_num,
            passed=bool(checks) and all(c.passed for c in checks),
            stack_detected=stack_detected,
            checks=checks,
        )
        rounds.append(current_round)

        fix_attempt = 0
        while not current_round.passed and fix_attempt < rounds_limit:
            fix_attempt += 1
            patch = await fix_app(active_router, spec, app_dir, current_round, plan=plan)
            current_round.diagnosis = patch.diagnosis
            updated = materialize_files(
                app_dir,
                patch.files,
                clean=False,
                delete_files=patch.delete_files,
            )
            tracked_files.update(updated)
            tracked_files.difference_update(patch.delete_files)

            round_num += 1
            stack_detected, checks = await _run_verify()
            current_round = VerificationRound(
                round_number=round_num,
                passed=bool(checks) and all(c.passed for c in checks),
                stack_detected=stack_detected,
                checks=checks,
            )
            rounds.append(current_round)

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
            error=error_text,
        )
        (out / "report.md").write_text(report_md, encoding="utf-8")
        _write_manifest(out, manifest)

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
