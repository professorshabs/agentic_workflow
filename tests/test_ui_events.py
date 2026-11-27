"""Run events for the workflow UI: the bus, persistence, log mirroring, replay and the
events a real `run_build` publishes. No model is called."""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from atlas.events import (
    RunEventBus,
    UILogHandler,
    attach,
    detach,
    load_run_events,
    load_run_events_with_offset,
    publish,
    publishing,
    redact,
    stage_of_log_text,
)


@pytest.fixture
def bus():
    b = RunEventBus()
    attach(b)
    yield b
    detach(b)
    b.close()


def test_publish_is_a_no_op_without_a_listener():
    assert publishing() is False
    publish("stage", "parse", "running", target="x")  # must not raise


def test_bus_binds_to_the_run_and_writes_events_jsonl(bus, tmp_path):
    out = tmp_path / "runs" / "r1"
    out.mkdir(parents=True)
    publish("stage", "run", "pending")  # before the run is known
    publish("run", "run", "running", run_id="r1", out_dir=str(out), app_dir=str(tmp_path / "app"))
    publish("stage", "parse", "passed", requirements=3)

    events = bus.events_after(0)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert {e["run_id"] for e in events} == {"r1"}
    assert bus.out_dir == out and bus.app_dir == tmp_path / "app"
    lines = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    assert [e["seq"] for e in lines] == [1, 2, 3]
    assert lines[2]["data"] == {"requirements": 3}


def test_payloads_are_redacted_and_capped(bus):
    secret = "sk-or-v1-abcdefghijklmnopqrstu"
    publish("stage", "build", "failed", error=f"key {secret} leaked", blob="x" * 9000)
    data = bus.events_after(0)[0]["data"]
    assert "sk-or-" not in data["error"] and "[redacted]" in data["error"]
    assert len(data["blob"]) < 4100
    assert redact("AIza" + "b" * 30) == "[redacted]"


def test_events_after_waits_for_the_next_event(bus):
    threading.Timer(0.1, lambda: publish("stage", "parse", "running")).start()
    started = time.monotonic()
    got = bus.events_after(0, timeout=2.0)
    assert got and got[0]["stage"] == "parse"
    assert time.monotonic() - started < 1.5


def test_finished_run_marks_the_bus_finished(bus):
    publish("run", "run", "running", run_id="r")
    assert bus.finished is False
    publish("run", "run", "failed", error="boom")
    assert bus.finished is True


def test_log_handler_mirrors_atlas_logs_with_their_stage(bus):
    log = logging.getLogger("atlas.test_ui_events")
    handler = UILogHandler(bus)
    logging.getLogger("atlas").addHandler(handler)
    old = logging.getLogger("atlas").level
    logging.getLogger("atlas").setLevel(logging.INFO)
    try:
        log.info("[research] %d/%d done: %s", 1, 2, "R001")
        log.warning("Something odd")
        log.debug("not mirrored")
    finally:
        logging.getLogger("atlas").removeHandler(handler)
        logging.getLogger("atlas").setLevel(old)
    logs = [e for e in bus.events_after(0) if e["kind"] == "log"]
    assert [(e["stage"], e["data"]["level"], e["data"]["text"]) for e in logs] == [
        ("research", "INFO", "[research] 1/2 done: R001"),
        ("run", "WARNING", "Something odd"),
    ]


@pytest.mark.parametrize(
    ("text", "stage"),
    [("[e2e] Done", "e2e_tests"), ("[done] Build verified", "report"), ("[fix] Round 1", "fix"),
     ("[unknown] x", "run"), ("plain", "run")],
)
def test_stage_of_log_text(text, stage):
    assert stage_of_log_text(text) == stage


def test_load_run_events_prefers_events_jsonl(tmp_path):
    lines = [{"seq": 2, "kind": "stage"}, {"seq": 1, "kind": "run"}]
    content = "\n".join(json.dumps(x) for x in lines) + "\nnot json\n"
    (tmp_path / "events.jsonl").write_text(content, encoding="utf-8")
    events, offset = load_run_events_with_offset(tmp_path)
    assert [e["seq"] for e in events] == [1, 2]
    assert offset == len(content.encode("utf-8"))
    assert [e["seq"] for e in load_run_events(tmp_path)] == [1, 2]


def test_old_runs_are_rebuilt_from_their_artifacts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run_id": "old",
        "mode": "build",
        "status": "verification_failed",
        "created_at": "2026-11-23T12:00:00+00:00",
        "completed_at": "2026-11-23T12:10:00+00:00",
        "spec_path": "specs/a.md",
        "app_dir": "app/a",
        "stages": {"parse": "completed", "research": "completed", "clarify": "completed",
                   "judge": "completed", "e2e_tests": "completed", "build": "completed",
                   "verify": "completed"},
        "research_failures_count": 1,
        "clarifications_count": 1,
        "clarifications_answered": 1,
        "e2e_verifiable": False,
        "build_offloaded_to": "sonnet",
        "build_offload_harness": "claude_code",
        "supervisor_reviewers": ["opus"],
        "token_usage": {"total_calls": 9, "total_tokens": 1000, "cost_usd": 0.5},
    }), encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps({
        "summary": "Demo app",
        "requirements": [{"id": "R1", "title": "Greet", "kind": "functional"}],
        "open_questions": ["Port?"],
    }), encoding="utf-8")
    (tmp_path / "clarifications.json").write_text(json.dumps({
        "items": [{"requirement_id": "R1", "question": "Port?", "answer": "8000"}],
    }), encoding="utf-8")
    (tmp_path / "plan.json").write_text(json.dumps({
        "decisions": [{"requirement_id": "R1", "pick": "FastAPI"}],
        "tasks": ["Write app.py"],
        "risks": [],
    }), encoding="utf-8")
    (tmp_path / "build.json").write_text(json.dumps({
        "summary": "Built app.py",
        "files_snapshot": ["app.py", "tests/test_app.py"],
        "available_skills": ["developing-fullstack-apps", "writing-clean-code"],
        "used_skills": ["writing-clean-code"],
        "skills_mode": "progressive",
    }), encoding="utf-8")
    (tmp_path / "tokens.json").write_text(json.dumps({
        "calls": [
            {"role": "parser", "used_model": "m1", "total_tokens": 400, "cost_usd": 0.2},
            {"role": "builder", "used_model": "sonnet", "harness": "claude_code",
             "total_tokens": 600, "cost_usd": None, "cost_source": "subscription"},
        ]
    }), encoding="utf-8")
    (tmp_path / "verification.json").write_text(json.dumps({
        "passed": False,
        "rounds": [
            {"round_number": 1, "passed": False, "stack_detected": "python", "diagnosis": "d1",
             "checks": [{"name": "pytest", "passed": False, "exit_code": 1, "stderr": "E bad"}]},
            {"round_number": 2, "passed": False, "stack_detected": "python",
             "checks": [{"name": "pytest", "passed": False, "exit_code": 1}]},
        ],
    }), encoding="utf-8")
    (tmp_path / "report.md").write_text("# r", encoding="utf-8")

    events = load_run_events(tmp_path)
    by = [(e["kind"], e["stage"], e["status"]) for e in events]
    assert by[0] == ("run", "run", "running")
    assert ("stage", "research", "partial") in by
    assert ("stage", "e2e_tests", "unverifiable") in by
    assert ("attempt", "build", "offloaded") in by
    assert ("attempt", "build", "passed") in by
    build_stage = next(e for e in events if e["kind"] == "stage" and e["stage"] == "build")
    assert build_stage["data"]["available_skills"] == [
        "developing-fullstack-apps",
        "writing-clean-code",
    ]
    assert build_stage["data"]["used_skills"] == ["writing-clean-code"]
    assert build_stage["data"]["skills_mode"] == "progressive"
    assert events[0]["data"]["flow"]["builder"]["skills"] == [
        "developing-fullstack-apps",
        "writing-clean-code",
    ]
    assert ("workspace", "build", None) in by
    assert ("usage", "parse", "passed") in by
    assert ("usage", "build", "passed") in by
    assert ("stage", "fix", "passed") in by  # round 2 implies one fix round
    assert ("check", "verify", "failed") in by
    assert by[-1] == ("run", "run", "failed")
    assert events[0]["data"]["replayed_from_artifacts"] is True
    assert events[-1]["data"]["usage"]["calls"] == 9
    assert events[-1]["data"]["usage"]["input_tokens"] is None or isinstance(
        events[-1]["data"]["usage"]["input_tokens"], int
    )
    usage_events = [e for e in events if e["kind"] == "usage"]
    assert all("input_tokens" in e["data"] and "output_tokens" in e["data"] for e in usage_events)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_missing_run_folder_contents_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_run_events(tmp_path)


async def test_a_real_build_publishes_the_whole_flow(tmp_path, monkeypatch, bus):
    from types import SimpleNamespace

    from atlas.router import ModelRouter
    from atlas.runner import run_build
    from atlas.schemas import CheckResult

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
roles:
  parser: [openrouter/m-parse]
  researcher: [openrouter/m-res]
  judge: [openrouter/m-judge]
  builder: [openrouter/m-build]
limits:
  max_parallel_calls: 2
  max_tokens: 2000
  timeout_seconds: 30
  attempts_per_model: 1
  repair_attempts: 0
  max_fix_rounds: 0
""",
        encoding="utf-8",
    )
    spec = tmp_path / "demo.md"
    spec.write_text("# Demo\nSay hello.\n", encoding="utf-8")

    responses = {
        "openrouter/m-parse": json.dumps({
            "summary": "Greeter",
            "requirements": [
                {"id": "R1", "title": "Greet", "description": "Print hi", "kind": "functional"},
                {"id": "R2", "title": "Exit", "description": "Exit 0", "kind": "non_functional"},
            ],
        }),
        "openrouter/m-res": json.dumps({
            "candidates": [{"name": "stdlib"}],
            "findings": ["stdlib is enough"],
            "uncertainties": [],
        }),
        "openrouter/m-judge": json.dumps({
            "decisions": [
                {"requirement_id": "R1", "options": [{"name": "stdlib"}, {"name": "click"}],
                 "pick": "stdlib", "rationale": "Simple"},
            ],
            "risks": [],
            "tasks": ["Write main.py"],
        }),
        "openrouter/m-build": json.dumps({
            "summary": "Built Greeter",
            "stack": "python",
            "files": [{"path": "main.py", "content": "print('hi')\n", "purpose": "entry"}],
            "run_instructions": "python main.py",
        }),
    }

    async def fake_acompletion(*, model, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=responses[model]))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30,
                                cost=0.001),
        )

    async def fake_verify(app_dir):
        return "python", [
            CheckResult(name="python-pytest", command="pytest", passed=True, exit_code=0)
        ]

    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_acompletion)
    router = ModelRouter(cfg)
    out = await run_build(
        spec,
        tmp_path / "runs",
        tmp_path / "app",
        router=router,
        verify_fn=fake_verify,
        generate_e2e=False,
    )

    events = bus.events_after(0)
    flow = [(e["kind"], e["stage"], e["status"]) for e in events if e["kind"] != "log"]
    assert flow[0] == ("run", "run", "running")
    expected_order = [
        ("stage", "parse", "running"),
        ("stage", "parse", "passed"),
        ("stage", "research", "running"),
        ("stage", "judge", "running"),
        ("stage", "judge", "passed"),
        ("stage", "build", "running"),
        ("attempt", "build", "running"),
        ("attempt", "build", "passed"),
        ("stage", "build", "passed"),
        ("stage", "verify", "running"),
        ("round", "verify", "passed"),
        ("stage", "verify", "passed"),
        ("stage", "report", "passed"),
        ("run", "run", "passed"),
    ]
    positions = [flow.index(step) for step in expected_order]
    assert positions == sorted(positions)
    assert sum(1 for k, s, _ in flow if (k, s) == ("item", "research")) == 2
    assert sum(1 for k, s, _ in flow if (k, s) == ("item", "judge")) == 2
    assert any(k == "usage" and s == "build" for k, s, _ in flow)
    usage_ev = next(e for e in events if e["kind"] == "usage" and e["stage"] == "build")
    assert usage_ev["data"]["input_tokens"] > 0 and usage_ev["data"]["output_tokens"] > 0
    final_usage = events[-1]["data"]["usage"]
    assert final_usage["input_tokens"] > 0 and final_usage["output_tokens"] > 0
    assert bus.out_dir == out
    persisted = (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(persisted) == len(events)


async def test_run_plan_publishes_events(tmp_path, monkeypatch, bus):
    from types import SimpleNamespace

    from atlas.router import ModelRouter
    from atlas.runner import run_plan

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        """
roles:
  parser: [openrouter/m-parse]
  researcher: [openrouter/m-res]
  judge: [openrouter/m-judge]
limits:
  max_parallel_calls: 2
  max_tokens: 2000
  timeout_seconds: 30
  attempts_per_model: 1
  repair_attempts: 0
""",
        encoding="utf-8",
    )
    spec = tmp_path / "demo.md"
    spec.write_text("# Demo\nPlan a CLI.\n", encoding="utf-8")

    responses = {
        "openrouter/m-parse": json.dumps({
            "summary": "CLI",
            "requirements": [{"id": "R1", "title": "CLI", "description": "Run CLI"}],
        }),
        "openrouter/m-res": json.dumps({"candidates": [{"name": "argparse"}], "findings": ["ok"]}),
        "openrouter/m-judge": json.dumps({
            "decisions": [{
                "requirement_id": "R1",
                "options": [{"name": "argparse"}, {"name": "typer"}],
                "pick": "argparse",
                "rationale": "Built-in",
            }],
            "tasks": ["Write cli.py"],
        }),
    }

    async def fake_acompletion(*, model, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=responses[model]))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                                cost=0.001),
        )

    monkeypatch.setattr("atlas.router.litellm.acompletion", fake_acompletion)
    out = await run_plan(spec, tmp_path / "runs", router=ModelRouter(cfg), generate_e2e=False)
    events = bus.events_after(0)
    assert events[0]["data"]["mode"] == "plan"
    assert ("stage", "e2e_tests", "skipped") in [
        (e["kind"], e["stage"], e["status"]) for e in events
    ]
    assert events[-1]["kind"] == "run" and events[-1]["status"] == "passed"
    assert (out / "events.jsonl").is_file()


def test_replay_of_an_older_run_fills_in_input_and_output_tokens(tmp_path):
    """Runs recorded before usage events carried the split get it from tokens.json."""
    lines = [
        {"v": 1, "seq": 1, "ts": 1.0, "run_id": "r", "kind": "run", "stage": "run",
         "status": "running", "message": "", "data": {"run_id": "r"}},
        {"v": 1, "seq": 2, "ts": 2.0, "run_id": "r", "kind": "usage", "stage": "parse",
         "status": "passed", "message": "", "data": {"role": "parser", "tokens": 30}},
        {"v": 1, "seq": 3, "ts": 3.0, "run_id": "r", "kind": "usage", "stage": "build",
         "status": "passed", "message": "", "data": {"role": "builder", "tokens": 70}},
        {"v": 1, "seq": 4, "ts": 4.0, "run_id": "r", "kind": "run", "stage": "run",
         "status": "passed", "message": "",
         "data": {"result": "verified", "usage": {"calls": 2, "tokens": 100}}},
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    (tmp_path / "tokens.json").write_text(
        json.dumps({
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "calls": [
                {"role": "parser", "prompt_tokens": 25, "completion_tokens": 5},
                {"role": "builder", "prompt_tokens": 55, "completion_tokens": 15},
            ],
        }),
        encoding="utf-8",
    )
    events = load_run_events(tmp_path)
    usage = [e["data"] for e in events if e["kind"] == "usage"]
    assert [(u["input_tokens"], u["output_tokens"]) for u in usage] == [(25, 5), (55, 15)]
    final = events[-1]["data"]["usage"]
    assert (final["input_tokens"], final["output_tokens"]) == (80, 20)


def test_token_split_is_not_guessed_when_calls_do_not_line_up(tmp_path):
    lines = [
        {"v": 1, "seq": 1, "ts": 1.0, "run_id": "r", "kind": "usage", "stage": "parse",
         "status": "passed", "message": "", "data": {"role": "parser", "tokens": 30}},
    ]
    (tmp_path / "events.jsonl").write_text(json.dumps(lines[0]) + "\n", encoding="utf-8")
    (tmp_path / "tokens.json").write_text(
        json.dumps({"calls": [{"role": "judge", "prompt_tokens": 1, "completion_tokens": 2}]}),
        encoding="utf-8",
    )
    (event,) = load_run_events(tmp_path)
    assert "input_tokens" not in event["data"]


def test_bus_stores_absolute_app_dir_across_chdir(bus, tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    app = project_root / "app" / "example-app"
    out = project_root / "runs" / "r1"
    app.mkdir(parents=True)
    out.mkdir(parents=True)
    monkeypatch.chdir(project_root)

    publish("run", "run", "running", run_id="r1", out_dir="runs/r1", app_dir="app/example-app")
    assert bus.app_dir == app
    assert bus.out_dir == out

    monkeypatch.chdir(app)
    assert bus.app_dir is not None and bus.app_dir.is_dir() and bus.app_dir == app


def test_ui_stall_handoff_copy_uses_idle_duration_words():
    from pathlib import Path

    html = Path("src/atlas/ui/static/index.html").read_text(encoding="utf-8")
    assert "function fmtDurWords" in html
    assert "hands off after ${fmtDurWords(a.stall_seconds)} idle" in html
    assert "handed on at" not in html
