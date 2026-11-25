"""In-process tests for `atlas` CLI commands (subprocess tests elsewhere are not traced by
coverage). Pipeline entry points are replaced with fakes; no model is called."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from atlas import cli
from atlas.local_models import LocalModelCandidate, LocalResolutionResult
from atlas.schemas import CheckResult

REPO = Path(__file__).resolve().parents[1]
HARNESS_CFG = REPO / "config" / "harness.yaml"

CALLS = [
    {
        "role": "parser",
        "requested_model": "m",
        "used_model": "m",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost_usd": 0.01,
        "cost_source": "reported",
    }
]


@pytest.fixture
def run_cli(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: False)  # never read the real .env

    def _run(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["atlas", *argv])
        code = 0
        try:
            cli.main()
        except SystemExit as exc:
            code = int(exc.code or 0)
        out, err = capsys.readouterr()
        return code, out, err

    return _run


def _write_run(run_dir: Path, status: str = "verified") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": status, "model_calls": CALLS}), encoding="utf-8"
    )
    return run_dir


def test_version_and_app_dir(run_cli):
    code, out, _ = run_cli("--version")
    assert code == 0 and out.startswith("atlas ")
    code, out, _ = run_cli("app-dir", "specs/My Cool App.md", "--app-root", "products")
    assert code == 0 and out.strip().startswith("products/")


def test_plan_passes_options_and_prints_tokens(run_cli, monkeypatch, tmp_path):
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"1": "Postgres"}), encoding="utf-8")
    seen: dict[str, Any] = {}

    async def fake_run_plan(spec, out, config, **kwargs):
        seen.update(kwargs, spec=spec)
        return _write_run(tmp_path / "runs" / "r1")

    monkeypatch.setattr(cli, "run_plan", fake_run_plan)
    code, out, _ = run_cli(
        "plan", "specs/x.md", "--no-clarify", "--answers", str(answers), "--e2e-tests",
        "--no-ledger",
    )

    assert code == 0
    assert "Plan written to" in out and "Tokens: 120 total" in out
    assert seen["interactive_clarify"] is False
    assert seen["clarification_answers"] == {"1": "Postgres"}
    assert seen["generate_e2e"] is True
    assert seen["ledger_path"] is None


@pytest.mark.parametrize(("status", "expected_code"), [("verified", 0), ("failed", 1)])
def test_build_exit_code_follows_verification(run_cli, monkeypatch, tmp_path, status,
                                              expected_code):
    seen: dict[str, Any] = {}

    async def fake_run_build(spec, out, app_dir, **kwargs):
        seen.update(kwargs, app_dir=app_dir)
        return _write_run(tmp_path / "runs" / "r2", status=status)

    monkeypatch.setattr(cli, "run_build", fake_run_build)
    code, out, _ = run_cli(
        "run", "specs/example-app.md", "--clarify", "--local-model", "gemma4:12b",
        "--max-fix-rounds", "2", "--skip-research", "--clean", "--from-run", "runs/prev",
    )

    assert code == expected_code
    assert f"Build status: {status}" in out
    assert seen["app_dir"] == Path("app/example-app")
    assert seen["local_model"] == "gemma4:12b"
    assert seen["max_fix_rounds"] == 2
    assert seen["skip_research"] is True and seen["clean_app_dir"] is True
    assert seen["interactive_clarify"] is True
    assert seen["from_run"] == Path("runs/prev")


def test_build_reports_missing_answers_file_as_usage_error(run_cli, tmp_path):
    code, _, err = run_cli("build", "specs/x.md", "--answers", str(tmp_path / "nope.json"))
    assert code == 2 and "Answers file not found" in err


def test_verify_text_json_and_failure(run_cli, monkeypatch):
    checks = [
        CheckResult(name="pytest", command="pytest -q", passed=True, exit_code=0),
        CheckResult(name="ruff", command="ruff check", passed=False, exit_code=1,
                    stderr="E501 line too long"),
    ]

    async def fake_verify(app_dir, timeout_seconds):
        return "python", checks

    monkeypatch.setattr(cli, "verify_app", fake_verify)
    code, out, _ = run_cli("verify", "--app-dir", "app/x")
    assert code == 1
    assert "[PASS] pytest" in out and "[FAIL] ruff" in out and "E501" in out

    code, out, _ = run_cli("verify", "--app-dir", "app/x", "--json")
    payload = json.loads(out)
    assert code == 1 and payload["stack"] == "python" and payload["passed"] is False

    async def passing(app_dir, timeout_seconds):
        return "python", checks[:1]

    monkeypatch.setattr(cli, "verify_app", passing)
    assert run_cli("verify")[0] == 0


def test_tokens_latest_run_explicit_run_and_ledger(run_cli, tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs / "20260101T000000Z-a")
    latest = _write_run(runs / "20260102T000000Z-b")
    (runs / "not-a-run").mkdir()

    code, out, _ = run_cli("tokens", "--runs-root", str(runs))
    assert code == 0 and "120" in out

    code, out, _ = run_cli("tokens", str(latest), "--json")
    assert code == 0 and json.loads(out)["total_tokens"] == 120

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"run_id": "a", "spec_path": "specs/x.md", "status": "verified",
                    "calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
                    "total_tokens": 120, "cost_usd": 0.01}) + "\n",
        encoding="utf-8",
    )
    code, out, _ = run_cli("tokens", "--all", "--ledger", str(ledger), "--json")
    assert code == 0 and json.loads(out)["runs"] == 1
    code, out, _ = run_cli("tokens", "--all", "--ledger", str(ledger))
    assert code == 0 and "Project token usage" in out


def test_tokens_errors_are_usage_errors(run_cli, tmp_path):
    code, _, err = run_cli("tokens", "--runs-root", str(tmp_path / "missing"))
    assert code == 2 and "Runs directory not found" in err
    (tmp_path / "empty").mkdir()
    code, _, err = run_cli("tokens", "--runs-root", str(tmp_path / "empty"))
    assert code == 2 and "No run directories" in err
    (tmp_path / "bare").mkdir()
    code, _, err = run_cli("tokens", str(tmp_path / "bare"))
    assert code == 2 and "Neither tokens.json nor manifest.json" in err


def test_changed_specs_and_validate_config(run_cli, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "changed_specs", lambda repo, before, after: ["specs/a.md"])
    code, out, _ = run_cli("changed-specs", "--before", "abc123")
    assert code == 0 and out.strip() == "specs/a.md"

    code, out, _ = run_cli("validate-config", "--config", str(HARNESS_CFG))
    assert code == 0 and "Config valid" in out

    bad = tmp_path / "bad.yaml"
    bad.write_text("roles: {}\nlimits: {}\nbackends:\n  judge:\n    type: nope\n",
                   encoding="utf-8")
    code, _, err = run_cli("validate-config", "--config", str(bad))
    assert code == 2 and err.startswith("Error:")


def test_doctor_text_json_and_unhealthy(run_cli, monkeypatch):
    report = {
        "healthy": False,
        "checks": [
            {"provider": "claude_code", "available": True, "authenticated": True,
             "auth_mode": "subscription", "version": "2.1.331", "details": "signed in"},
            {"provider": "antigravity", "available": True, "authenticated": False,
             "auth_mode": "cached_login", "details": "authentication required",
             "login_hint": "Run `agy` interactively to sign in"},
            {"provider": "local_model_daemon", "available": True, "authenticated": True,
             "ready": True, "details": "selected gemma4:12b"},
        ],
        "skills": {"configured": True, "resolved": ["writing-clean-code"]},
    }
    seen: dict[str, Any] = {}

    async def fake_doctor(router, *, live=False):
        seen["live"] = live
        return report

    monkeypatch.setattr(cli, "run_doctor", fake_doctor)
    code, out, _ = run_cli("doctor", "--config", str(HARNESS_CFG), "--live")
    assert code == 1 and seen["live"] is True
    assert "[OK] claude_code (2.1.331) [auth=subscription]" in out
    assert "[FAIL] antigravity" in out and "hint: Run `agy` interactively" in out
    assert "[OK] local_model_daemon" in out
    assert "Skills allowlist resolved: writing-clean-code" in out

    report["healthy"] = True
    code, out, _ = run_cli("doctor", "--config", str(HARNESS_CFG), "--json")
    assert code == 0 and json.loads(out)["healthy"] is True


def test_detect_local_lists_candidates_and_fails_without_selection(run_cli, monkeypatch):
    chosen = LocalModelCandidate(provider="ollama", name="gemma4:12b",
                                 base_url="http://localhost:11434/v1", parameter_size_b=12,
                                 loaded=True, tool_smoke_passed=True)
    other = LocalModelCandidate(provider="lm_studio", name="tiny", base_url="http://x/v1",
                                tool_smoke_passed=False)
    seen: dict[str, Any] = {}

    async def fake_resolve(**kwargs):
        seen.update(kwargs)
        return LocalResolutionResult(selected=chosen, candidates=[chosen, other],
                                     rejected={"tiny": "tool smoke check failed"},
                                     reason="Selected gemma4:12b")

    monkeypatch.setattr(cli, "resolve_local_model", fake_resolve)
    code, out, _ = run_cli("detect-local", "--config", str(HARNESS_CFG), "--local-model",
                           "gemma4:12b")
    assert code == 0
    assert "ollama: gemma4:12b (12B) [loaded, smoke=ok, SELECTED]" in out
    assert "lm_studio: tiny (?B) [smoke=fail]" in out
    assert "rejected tiny: tool smoke check failed" in out
    assert seen["explicit_model"] == "gemma4:12b"
    assert seen["require_tool_smoke_check"] is True

    async def none_ready(**kwargs):
        seen.update(kwargs)
        return LocalResolutionResult(reason="No local model is ready")

    monkeypatch.setattr(cli, "resolve_local_model", none_ready)
    code, out, _ = run_cli("detect-local", "--config", str(HARNESS_CFG), "--no-smoke-check",
                           "--json")
    assert code == 1 and json.loads(out)["selected"] is None
    assert seen["require_tool_smoke_check"] is False


def test_runtime_errors_exit_one(run_cli, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("token budget exceeded")

    monkeypatch.setattr(cli, "run_plan", boom)
    code, _, err = run_cli("plan", "specs/x.md", "--no-clarify")
    assert code == 1 and "token budget exceeded" in err


def test_build_starts_the_live_ui_only_when_asked_or_interactive(run_cli, monkeypatch, tmp_path):
    import contextlib

    import atlas.ui

    started: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def fake_ui(**kwargs):
        started.append(kwargs)
        yield None

    async def fake_run_build(spec, out, app_dir, **kwargs):
        return _write_run(tmp_path / "runs" / "r3")

    monkeypatch.setattr(atlas.ui, "LiveWorkflowUI", fake_ui)
    monkeypatch.setattr(cli, "run_build", fake_run_build)

    code, _, _ = run_cli("build", "specs/example-app.md")  # not a terminal under pytest
    assert code == 0 and started == []

    code, _, _ = run_cli(
        "build", "specs/example-app.md", "--ui", "--no-open", "--ui-port", "8765", "--ui-keep-open"
    )
    assert code == 0 and started == [
        {"port": 8765, "open_browser": False, "keep_open": True}
    ]

    monkeypatch.setenv("CI", "")
    tty = type("Tty", (), {"isatty": lambda self: True})()
    monkeypatch.setattr("atlas.ui.server.sys.stderr", tty)
    code, _, _ = run_cli("build", "specs/example-app.md", "--no-ui")
    assert code == 0 and len(started) == 1


def test_plan_ui_is_opt_in_only(run_cli, monkeypatch, tmp_path):
    import contextlib

    import atlas.ui

    started: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def fake_ui(**kwargs):
        started.append(kwargs)
        yield None

    async def fake_run_plan(spec, out, config, **kwargs):
        return _write_run(tmp_path / "runs" / "p1")

    monkeypatch.setattr(atlas.ui, "LiveWorkflowUI", fake_ui)
    monkeypatch.setattr(cli, "run_plan", fake_run_plan)
    monkeypatch.setenv("CI", "")
    tty = type("Tty", (), {"isatty": lambda self: True})()
    monkeypatch.setattr("atlas.ui.server.sys.stderr", tty)

    code, _, _ = run_cli("plan", "specs/example-app.md", "--no-clarify")
    assert code == 0 and started == []

    code, _, _ = run_cli("plan", "specs/example-app.md", "--no-clarify", "--ui", "--no-open")
    assert code == 0 and started == [{"port": 0, "open_browser": False, "keep_open": False}]


def test_quiet_build_with_ui_keeps_info_logs_for_the_browser(run_cli, monkeypatch, tmp_path):
    import contextlib
    import logging

    import atlas.ui

    levels: list[int] = []

    @contextlib.contextmanager
    def fake_ui(**kwargs):
        levels.append(logging.getLogger("atlas").getEffectiveLevel())
        yield None

    async def fake_run_build(spec, out, app_dir, **kwargs):
        return _write_run(tmp_path / "runs" / "r4")

    monkeypatch.setattr(atlas.ui, "LiveWorkflowUI", fake_ui)
    monkeypatch.setattr(cli, "run_build", fake_run_build)
    code, _, _ = run_cli("build", "specs/example-app.md", "--ui", "-q")
    assert code == 0 and levels == [logging.INFO]


def test_ui_command_replays_newest_or_given_run(run_cli, monkeypatch, tmp_path):
    import atlas.ui

    seen: list[tuple[Path, int, bool]] = []
    monkeypatch.setattr(
        atlas.ui,
        "serve_replay",
        lambda run_dir, port=0, open_browser=True: seen.append((run_dir, port, open_browser)),
    )
    _write_run(tmp_path / "runs" / "20261101T000000Z-a")
    _write_run(tmp_path / "runs" / "20261102T000000Z-b")

    code, _, _ = run_cli("ui", "--runs-root", str(tmp_path / "runs"), "--no-open")
    assert code == 0 and seen[-1] == (tmp_path / "runs" / "20261102T000000Z-b", 0, False)

    code, _, _ = run_cli("ui", str(tmp_path / "runs" / "20261101T000000Z-a"), "--port", "9000")
    assert code == 0 and seen[-1] == (tmp_path / "runs" / "20261101T000000Z-a", 9000, True)

    code, _, err = run_cli("ui", str(tmp_path / "missing"))
    assert code == 2 and "Run folder not found" in err


def test_build_summary_prints_before_the_ui_is_kept_open(run_cli, monkeypatch, tmp_path, capsys):
    import contextlib

    import atlas.ui

    printed_before_exit: list[bool] = []

    @contextlib.contextmanager
    def fake_ui(**kwargs):
        yield None
        printed_before_exit.append("Build status: verified" in capsys.readouterr().out)

    async def fake_run_build(spec, out, app_dir, **kwargs):
        return _write_run(tmp_path / "runs" / "r5")

    monkeypatch.setattr(atlas.ui, "LiveWorkflowUI", fake_ui)
    monkeypatch.setattr(cli, "run_build", fake_run_build)
    code, _, _ = run_cli("build", "specs/example-app.md", "--ui", "--no-open", "--ui-keep-open")
    assert code == 0 and printed_before_exit == [True]
