"""Structured run events for the live workflow UI.

Pipeline stages call :func:`publish`. It does nothing unless a :class:`RunEventBus` is attached
(the CLI attaches one when the UI is on), so library callers and tests see no change. The bus
numbers events, appends them to ``runs/<run-id>/events.jsonl`` and lets the UI server stream
them to the browser. :func:`load_run_events` reads that file back for ``atlas ui`` replay, or
rebuilds an approximate timeline from the artifacts of runs made before events existed.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger("atlas.events")

EVENT_SCHEMA_VERSION = 1
STAGES = (
    "run",
    "parse",
    "research",
    "clarify",
    "judge",
    "e2e_tests",
    "build",
    "verify",
    "fix",
    "supervisor",
    "report",
)
_MAX_TEXT = 4000
_SECRET_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{12,}|sk-or-[A-Za-z0-9_\-]{12,}|sk-[A-Za-z0-9_\-]{20,}"
    r"|AIza[0-9A-Za-z_\-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|(?i:bearer)\s+[A-Za-z0-9._\-]{16,})"
)
_LOG_PREFIX_STAGE = {
    "parse": "parse",
    "research": "research",
    "clarify": "clarify",
    "judge": "judge",
    "e2e": "e2e_tests",
    "plan": "judge",
    "build": "build",
    "verify": "verify",
    "fix": "fix",
    "supervisor": "supervisor",
    "done": "report",
    "failed": "report",
    "resume": "run",
    "ui": "run",
}
_LOG_PREFIX_RE = re.compile(r"^\[([a-z_0-9]+)\]")


def redact(text: str, limit: int = _MAX_TEXT) -> str:
    """Mask API-key-shaped strings and cap the length of text sent to the browser."""
    cleaned = _SECRET_RE.sub("[redacted]", text)
    if len(cleaned) > limit:
        cleaned = cleaned[:limit] + " ...[truncated]"
    return cleaned


def _clean(value: Any, depth: int = 0) -> Any:
    """JSON-safe copy of an event payload with strings redacted and sizes capped."""
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth > 5:
        return redact(str(value), 300)
    if isinstance(value, dict):
        return {str(k): _clean(v, depth + 1) for k, v in list(value.items())[:300]}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v, depth + 1) for v in list(value)[:500]]
    return redact(str(value))


class RunEventBus:
    """Thread-safe, ordered event log for one run, persisted as JSON lines.

    The first ``run`` event with status ``running`` binds the bus to its run: it carries
    ``run_id``, ``out_dir`` and ``app_dir``, and opens ``<out_dir>/events.jsonl``.
    """

    def __init__(self, *, max_events: int = 50_000, persist: bool = True) -> None:
        self._cond = threading.Condition()
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._max_events = max_events
        self._persist = persist
        self._file: IO[str] | None = None
        self.run_id: str | None = None
        self.out_dir: Path | None = None
        self.app_dir: Path | None = None
        self.finished = False
        self.closed = False

    # -- writing ---------------------------------------------------------------------

    def add(
        self,
        kind: str,
        stage: str = "run",
        status: str | None = None,
        message: str = "",
        data: dict[str, Any] | None = None,
        *,
        ts: float | None = None,
    ) -> dict[str, Any]:
        """Append one event; returns it with its sequence number."""
        payload = _clean(data or {})
        with self._cond:
            if kind == "run" and status == "running" and self.run_id is None:
                self._bind(payload)
            self._seq += 1
            event: dict[str, Any] = {
                "v": EVENT_SCHEMA_VERSION,
                "seq": self._seq,
                "ts": round(ts if ts is not None else time.time(), 3),
                "run_id": self.run_id,
                "kind": kind,
                "stage": stage,
                "status": status,
                "message": redact(message, 2000) if message else "",
                "data": payload,
            }
            # Logs past the cap still reach events.jsonl; the browser keeps the structure.
            if kind != "log" or len(self._events) < self._max_events:
                self._events.append(event)
            self._write(event)
            if kind == "run" and status not in (None, "running"):
                self.finished = True
            self._cond.notify_all()
            return event

    def extend(self, events: list[dict[str, Any]]) -> None:
        """Load already-numbered events (replay of a finished run)."""
        with self._cond:
            for event in events:
                self._events.append(event)
                self._seq = max(self._seq, int(event.get("seq") or 0))
                if event.get("kind") == "run":
                    data = event.get("data") or {}
                    if event.get("status") == "running":
                        self.run_id = self.run_id or event.get("run_id") or data.get("run_id")
                        if data.get("out_dir"):
                            self.out_dir = Path(str(data["out_dir"]))
                        if data.get("app_dir"):
                            self.app_dir = Path(str(data["app_dir"]))
                    elif event.get("status") is not None:
                        self.finished = True
            self._cond.notify_all()

    def _bind(self, data: dict[str, Any]) -> None:
        self.run_id = str(data.get("run_id") or "") or None
        if data.get("out_dir"):
            self.out_dir = Path(str(data["out_dir"]))
        if data.get("app_dir"):
            self.app_dir = Path(str(data["app_dir"]))
        if self._persist and self.out_dir is not None and self.out_dir.is_dir():
            try:
                self._file = (self.out_dir / "events.jsonl").open("a", encoding="utf-8")
                for earlier in self._events:
                    earlier["run_id"] = self.run_id
                    self._file.write(json.dumps(earlier, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("Could not write events.jsonl: %s", exc)
                self._file = None
            else:
                self._file.flush()

    def _write(self, event: dict[str, Any]) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._file.flush()
        except OSError as exc:
            logger.warning("Stopped writing events.jsonl: %s", exc)
            self._file = None

    def close(self) -> None:
        with self._cond:
            self.closed = True
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
            self._cond.notify_all()

    # -- reading ---------------------------------------------------------------------

    def events_after(self, seq: int, timeout: float = 0.0) -> list[dict[str, Any]]:
        """Events with a sequence number above ``seq``, waiting up to ``timeout`` for one."""
        with self._cond:
            if timeout > 0 and self._seq <= seq and not self.closed:
                self._cond.wait(timeout)
            start = bisect.bisect_right(self._events, seq, key=lambda e: int(e["seq"]))
            return list(self._events[start:])

    @property
    def last_seq(self) -> int:
        with self._cond:
            return self._seq


_ACTIVE: list[RunEventBus] = []
_ACTIVE_LOCK = threading.Lock()


def attach(bus: RunEventBus) -> None:
    with _ACTIVE_LOCK:
        if bus not in _ACTIVE:
            _ACTIVE.append(bus)


def detach(bus: RunEventBus) -> None:
    with _ACTIVE_LOCK:
        if bus in _ACTIVE:
            _ACTIVE.remove(bus)


def publishing() -> bool:
    """True while a UI is listening (lets callers skip building large payloads)."""
    return bool(_ACTIVE)


def publish(
    kind: str,
    stage: str = "run",
    status: str | None = None,
    message: str = "",
    **data: Any,
) -> None:
    """Send one event to every attached bus. Never raises into the pipeline."""
    if not _ACTIVE:
        return
    with _ACTIVE_LOCK:
        buses = list(_ACTIVE)
    for bus in buses:
        try:
            bus.add(kind, stage, status, message, data)
        except Exception:  # pragma: no cover - the UI must never break a build
            pass


def stage_of_log_text(text: str) -> str:
    """Stage a progress log line belongs to, from its `[stage]` prefix (for log filters)."""
    match = _LOG_PREFIX_RE.match(text)
    return _LOG_PREFIX_STAGE.get(match.group(1), "run") if match else "run"


class UILogHandler(logging.Handler):
    """Mirror Atlas log records into the bus as `log` events (for the log panel only)."""

    def __init__(self, bus: RunEventBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.exc_info and record.levelno >= logging.ERROR:
                text = f"{text}\n{self.format(record)}" if self.formatter else text
            self.bus.add(
                "log",
                stage_of_log_text(text),
                None,
                "",
                {"level": record.levelname, "logger": record.name, "text": text},
                ts=record.created,
            )
        except Exception:  # pragma: no cover
            self.handleError(record)


# -- replay ------------------------------------------------------------------------------

_STAGE_STATUS = {"completed": "passed", "reused": "reused", "skipped": "skipped"}


def _iso_ts(value: Any, default: float) -> float:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return default
    return default


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_run_events_with_offset(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Load replay events for a run folder plus the byte offset of `events.jsonl`."""
    path = run_dir / "events.jsonl"
    if path.is_file():
        raw = path.read_bytes()
        events: list[dict[str, Any]] = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and isinstance(item.get("seq"), int):
                events.append(item)
        events.sort(key=lambda e: e["seq"])
        _backfill_token_split(run_dir, events)
        return events, len(raw)
    return synthesize_events(run_dir), 0


def _backfill_token_split(run_dir: Path, events: list[dict[str, Any]]) -> None:
    """Add input/output token counts to replays of runs recorded before usage events had them.

    `tokens.json` (or the manifest's `token_usage`) keeps prompt/completion totals and, per call,
    the same calls in the same order as the `usage` events, so both can be filled in.
    """
    usage_events = [
        e for e in events if e.get("kind") == "usage" and isinstance(e.get("data"), dict)
    ]
    finals = [
        e
        for e in events
        if e.get("kind") == "run"
        and e.get("status") not in (None, "running")
        and isinstance(e.get("data"), dict)
        and isinstance(e["data"].get("usage"), dict)
    ]
    calls_missing = any("input_tokens" not in e["data"] for e in usage_events)
    final_missing = any("input_tokens" not in e["data"]["usage"] for e in finals)
    if not (calls_missing or final_missing):
        return
    doc = _read_json(run_dir / "tokens.json")
    if not isinstance(doc, dict):
        manifest = _read_json(run_dir / "manifest.json")
        doc = manifest.get("token_usage") if isinstance(manifest, dict) else None
    if not isinstance(doc, dict):
        return
    if final_missing and doc.get("prompt_tokens") is not None:
        for e in finals:
            usage = e["data"]["usage"]
            usage.setdefault("input_tokens", int(doc.get("prompt_tokens") or 0))
            usage.setdefault("output_tokens", int(doc.get("completion_tokens") or 0))
    calls = doc.get("calls")
    if not (calls_missing and isinstance(calls, list) and len(calls) == len(usage_events)):
        return
    # Only trust the pairing when every call lines up with its event by role.
    for event, call in zip(usage_events, calls, strict=True):
        if not isinstance(call, dict) or str(call.get("role") or "") != str(
            event["data"].get("role") or ""
        ):
            return
    for event, call in zip(usage_events, calls, strict=True):
        event["data"].setdefault("input_tokens", int(call.get("prompt_tokens") or 0))
        event["data"].setdefault("output_tokens", int(call.get("completion_tokens") or 0))


def load_run_events(run_dir: Path) -> list[dict[str, Any]]:
    """Events for a past run: `events.jsonl` when present, else rebuilt from its artifacts."""
    events, _ = load_run_events_with_offset(run_dir)
    return events


def synthesize_events(run_dir: Path) -> list[dict[str, Any]]:
    """Approximate timeline for runs made before `events.jsonl` (stage states, checks, verdicts)."""
    manifest = _read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict):
        raise FileNotFoundError(f"No manifest.json or events.jsonl in {run_dir}")
    stages: dict[str, str] = dict(manifest.get("stages") or {})
    started = _iso_ts(manifest.get("created_at"), (run_dir / "manifest.json").stat().st_mtime)
    finished_at = _iso_ts(manifest.get("completed_at"), started)
    bus = RunEventBus(persist=False)

    def add(kind: str, stage: str, status: str | None, message: str = "", **data: Any) -> None:
        bus.add(kind, stage, status, message, data, ts=started)

    spec_doc = _read_json(run_dir / "spec.json")
    clar_doc = _read_json(run_dir / "clarifications.json")
    plan_doc = _read_json(run_dir / "plan.json")
    e2e_doc = _read_json(run_dir / "e2e_tests.json")
    build_doc = _read_json(run_dir / "build.json")
    tokens_doc = _read_json(run_dir / "tokens.json")

    run_id = str(manifest.get("run_id") or run_dir.name)
    builder_type = manifest.get("builder_backend") or manifest.get("build_mode")
    reviewers = list(manifest.get("supervisor_reviewers") or [])
    offloaded_to = manifest.get("build_offloaded_to")
    attempts: list[dict[str, str]] = []
    if manifest.get("local_model"):
        attempts.append({
            "name": "local",
            "label": str(manifest["local_model"]),
            "kind": "local",
            "provider": "antigravity_sdk",
        })
    elif offloaded_to:
        attempts.append({
            "name": "local",
            "label": "local model",
            "kind": "local",
            "provider": "antigravity_sdk",
        })
        attempts.append({
            "name": str(manifest.get("build_offload_harness") or "offload"),
            "label": str(offloaded_to),
            "kind": "harness",
            "provider": str(manifest.get("build_offload_harness") or ""),
        })
    elif builder_type:
        attempts.append({
            "name": str(builder_type),
            "label": str(manifest.get("build_model") or builder_type),
            "kind": str(builder_type),
        })

    b_avail = (
        list(build_doc.get("available_skills") or [])
        if isinstance(build_doc, dict)
        else []
    ) or list(manifest.get("builder_available_skills") or [])
    add(
        "run",
        "run",
        "running",
        run_id=run_id,
        mode=manifest.get("mode", "build"),
        spec_path=manifest.get("spec_path", ""),
        app_dir=manifest.get("app_dir", ""),
        out_dir=str(run_dir),
        from_run=manifest.get("resumed_from"),
        replayed_from_artifacts=True,
        flow={
            "skip_research": stages.get("research") == "skipped",
            "e2e": "e2e_tests" in stages,
            "supervisor_enabled": bool(reviewers),
            "supervisor_reviewers": reviewers,
            "builder": {"type": builder_type, "attempts": attempts, "skills": b_avail},
        },
    )
    add("stage", "run", "passed", spec_path=manifest.get("spec_path", ""), run_id=run_id)
    for stage in ("parse", "research", "clarify", "judge", "e2e_tests"):
        state = stages.get(stage)
        if state is None:
            continue
        status = _STAGE_STATUS.get(state, "passed")
        data: dict[str, Any] = {}
        if stage == "parse" and isinstance(spec_doc, dict):
            reqs = [r for r in (spec_doc.get("requirements") or []) if isinstance(r, dict)]
            data = {
                "requirements": len(reqs),
                "functional": sum(1 for r in reqs if r.get("kind") == "functional"),
                "open_questions": len(spec_doc.get("open_questions") or []),
                "summary": spec_doc.get("summary") or "",
                "requirement_titles": {
                    str(r.get("id")): str(r.get("title") or "")
                    for r in reqs
                    if r.get("id")
                },
            }
        elif stage == "research":
            data = {
                "notes": manifest.get("research_notes_count"),
                "failures": manifest.get("research_failures_count"),
            }
            if status == "passed" and data.get("failures"):
                status = "partial"
        elif stage == "clarify":
            items = (
                [
                    {
                        "question": i.get("question"),
                        "answer": i.get("answer"),
                        "requirement_id": i.get("requirement_id"),
                    }
                    for i in (clar_doc.get("items") or [])
                    if isinstance(i, dict)
                ][:30]
                if isinstance(clar_doc, dict)
                else []
            )
            data = {
                "answered": manifest.get("clarifications_answered"),
                "total": manifest.get("clarifications_count"),
                "items": items,
            }
        elif stage == "judge" and isinstance(plan_doc, dict):
            decisions = [d for d in (plan_doc.get("decisions") or []) if isinstance(d, dict)]
            data = {
                "decisions": len(decisions),
                "tasks": len(plan_doc.get("tasks") or []),
                "risks": len(plan_doc.get("risks") or []),
                "picks": {
                    str(d.get("requirement_id")): str(d.get("pick") or "")
                    for d in decisions
                    if d.get("requirement_id")
                },
            }
        elif stage == "e2e_tests":
            verifiable = manifest.get("e2e_verifiable")
            if isinstance(e2e_doc, dict) and verifiable is None:
                verifiable = e2e_doc.get("verifiable")
            e2e_target = ""
            if isinstance(tokens_doc, dict) and isinstance(tokens_doc.get("calls"), list):
                for call in tokens_doc["calls"]:
                    if isinstance(call, dict) and call.get("role") == "e2e":
                        hname = str(call.get("harness") or "").strip()
                        mname = str(
                            call.get("used_model") or call.get("requested_model") or ""
                        ).strip()
                        e2e_target = (
                            f"{hname}/{mname}"
                            if hname and mname and "/" not in mname
                            else (mname or hname)
                        )
                        break
            data = {
                **({"target": e2e_target} if e2e_target else {}),
                "verifiable": verifiable,
                "test_files": manifest.get("e2e_test_files") or [],
                "reason": (
                    e2e_doc.get("unverifiable_reason") or "" if isinstance(e2e_doc, dict) else ""
                ),
                "interface": (
                    e2e_doc.get("product_interface") or "" if isinstance(e2e_doc, dict) else ""
                ),
            }
            if verifiable is False:
                status = "unverifiable"
        add("stage", stage, status, **data)

    error = manifest.get("error")
    run_status = str(manifest.get("status") or "unknown")
    if "build" in stages:
        bfiles = build_doc.get("files_snapshot") if isinstance(build_doc, dict) else None
        if not isinstance(bfiles, list) and isinstance(build_doc, dict):
            bfiles = [
                f.get("path")
                for f in (build_doc.get("files") or [])
                if isinstance(f, dict) and f.get("path")
            ]
        file_list = [str(p) for p in (bfiles or [])]
        build_summary = str(build_doc.get("summary") or "") if isinstance(build_doc, dict) else ""
        b_model = (
            manifest.get("local_model")
            or offloaded_to
            or (build_doc.get("model") if isinstance(build_doc, dict) else None)
        )
        b_harness = (
            manifest.get("build_offload_harness")
            or (build_doc.get("harness") if isinstance(build_doc, dict) else None)
        )
        if manifest.get("local_model"):
            add(
                "attempt",
                "build",
                "passed",
                attempt="local",
                provider="antigravity_sdk",
                model=manifest["local_model"],
                files=len(file_list) or len(manifest.get("generated_files") or []),
            )
        elif offloaded_to:
            target_name = str(manifest.get("build_offload_harness") or "offload")
            add(
                "attempt",
                "build",
                "offloaded",
                attempt="local",
                provider="antigravity_sdk",
                next=target_name,
                error="Local model could not finish (replayed from manifest)",
            )
            add(
                "attempt",
                "build",
                "passed",
                attempt=target_name,
                provider=str(manifest.get("build_offload_harness") or ""),
                model=str(offloaded_to),
                files=len(file_list) or len(manifest.get("generated_files") or []),
            )
        if file_list:
            add(
                "workspace",
                "build",
                None,
                "",
                files=[[p, None] for p in file_list[:400]],
                count=len(file_list),
                truncated=len(file_list) > 400,
            )
        b_used = (
            list(build_doc.get("used_skills") or [])
            if isinstance(build_doc, dict)
            else []
        ) or list(manifest.get("builder_used_skills") or [])
        b_skills_mode = (
            build_doc.get("skills_mode") if isinstance(build_doc, dict) else None
        ) or manifest.get("builder_skills_mode")
        add(
            "stage",
            "build",
            "passed",
            model=b_model,
            harness=b_harness,
            offloaded_to=offloaded_to,
            files=len(file_list) or len(manifest.get("generated_files") or []),
            summary=build_summary[:600],
            available_skills=b_avail,
            used_skills=b_used,
            skills_mode=b_skills_mode,
        )
    elif error and manifest.get("mode", "build") == "build":
        planned = [s for s in ("parse", "research", "judge") if s not in stages]
        failed_stage = planned[0] if planned else "build"
        add("stage", failed_stage, "failed", error=error)

    verification = _read_json(run_dir / "verification.json")
    if isinstance(verification, dict):
        rounds = list(verification.get("rounds") or [])
        for idx, rnd in enumerate(rounds):
            number = int(rnd.get("round_number") or idx + 1)
            if idx > 0:
                prev = rounds[idx - 1]
                add(
                    "stage",
                    "fix",
                    "passed",
                    round=idx,
                    failing=[
                        c.get("name") for c in prev.get("checks", []) if not c.get("passed")
                    ],
                    diagnosis=str(prev.get("diagnosis") or "")[:600],
                )
            add("stage", "verify", "running", round=number)
            for chk in rnd.get("checks") or []:
                add(
                    "check",
                    "verify",
                    "passed" if chk.get("passed") else "failed",
                    round=number,
                    name=chk.get("name"),
                    exit_code=chk.get("exit_code"),
                    seconds=round(float(chk.get("duration_ms") or 0) / 1000.0, 1),
                    command=chk.get("command"),
                    output="" if chk.get("passed") else _tail(chk),
                )
            add(
                "round",
                "verify",
                "passed" if rnd.get("passed") else "failed",
                round=number,
                stack=rnd.get("stack_detected"),
            )
        add("stage", "verify", "passed" if verification.get("passed") else "failed")

    review = _read_json(run_dir / "supervisor_review.json")
    if isinstance(review, dict):
        for verdict in review.get("reviewers") or []:
            add(
                "reviewer",
                "supervisor",
                "passed" if verdict.get("approved") else "partial",
                reviewer=verdict.get("reviewer"),
                model=verdict.get("model"),
                harness=verdict.get("harness"),
                approved=bool(verdict.get("approved")),
                issues=list(verdict.get("issues") or [])[:20],
                summary=str(verdict.get("summary") or "")[:800],
            )
        add(
            "stage",
            "supervisor",
            "passed" if review.get("approved") else "partial",
            approved=bool(review.get("approved")),
            issues=list(review.get("issues") or [])[:30],
            improvement_plan=list(review.get("improvement_plan") or [])[:30],
        )

    calls = (
        tokens_doc.get("calls")
        if isinstance(tokens_doc, dict)
        else manifest.get("model_calls")
    )
    if isinstance(calls, list):
        role_stage = {
            "parser": "parse",
            "researcher": "research",
            "judge": "judge",
            "e2e": "e2e_tests",
            "builder": "build",
            "fixer": "fix",
            "supervisor": "supervisor",
        }
        for c in calls:
            if not isinstance(c, dict):
                continue
            role = str(c.get("role") or "")
            add(
                "usage",
                role_stage.get(role, "run"),
                "failed" if c.get("error") else "passed",
                role=role,
                model=c.get("used_model") or c.get("requested_model"),
                harness=c.get("harness"),
                tokens=int(c.get("total_tokens") or 0),
                input_tokens=int(c.get("prompt_tokens") or 0),
                output_tokens=int(c.get("completion_tokens") or 0),
                cost_usd=c.get("cost_usd"),
                cost_source=c.get("cost_source"),
                api_cost_usd=c.get("api_cost_usd"),
                error=str(c.get("error") or "")[:300],
            )

    usage = manifest.get("token_usage") or {}
    report_artifact = (
        run_dir / "plan.md" if manifest.get("mode") == "plan" else run_dir / "report.md"
    )
    if report_artifact.is_file():
        add("stage", "report", "passed", report=str(report_artifact))
    bus.add(
        "run",
        "run",
        "passed" if run_status in ("verified", "planned", "plan_ready") else "failed",
        "",
        {
            "result": run_status,
            "error": error,
            "usage": {
                "calls": usage.get("total_calls"),
                "tokens": usage.get("total_tokens"),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
                "unpriced_calls": usage.get("unpriced_calls"),
                "api_cost_usd": usage.get("api_cost_usd"),
                "api_unestimated_calls": usage.get("api_unestimated_calls"),
            },
        },
        ts=finished_at,
    )
    return bus.events_after(0)


def _tail(check: dict[str, Any], limit: int = 1500) -> str:
    text = str(check.get("stderr") or check.get("stdout") or "")
    return text[-limit:]
