"""Per-role limits, billed-but-unusable responses, the project ledger, and requirement coverage."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas.pipeline.verify import check_requirement_coverage
from atlas.router import ModelRouter
from atlas.runner import run_build
from atlas.telemetry import (
    append_ledger,
    ledger_entry,
    ledger_totals,
    read_ledger,
    render_ledger_report_md,
    render_token_report_md,
    summarize_usage,
)

CFG = """
roles:
  builder: [m-1]
  parser: [m-1]
role_fallbacks:
  builder: [m-2]
limits:
  max_parallel_calls: 2
  max_tokens: 100
  max_tokens_by_role:
    builder: 32000
  timeout_seconds: 10
  timeout_seconds_by_role:
    builder: 600
  attempts_per_model: 2
"""


def _response(content, finish_reason="stop", prompt=10, completion=5):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


@pytest.fixture
def router(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    return ModelRouter(cfg)


async def test_role_limits_are_sent_to_the_provider(router, monkeypatch):
    fake = AsyncMock(return_value=_response('{"ok": true}'))
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake)

    await router.complete("m-1", "sys", "usr", role="builder")
    assert fake.call_args.kwargs["max_tokens"] == 32000
    assert fake.call_args.kwargs["timeout"] == 600

    await router.complete("m-1", "sys", "usr", role="parser")
    assert fake.call_args.kwargs["max_tokens"] == 100
    assert fake.call_args.kwargs["timeout"] == 10


async def test_truncated_output_is_counted_and_fallbacks_are_not_paid_for(router, monkeypatch):
    from atlas.router import OutputTruncatedError

    calls = []

    async def fake(*, model, **kwargs):
        calls.append(model)
        return _response('{"files": [', finish_reason="length", prompt=50, completion=32000)

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake))
    with pytest.raises(OutputTruncatedError, match="max_tokens_by_role.builder"):
        await router.complete("m-1", "sys", "usr", role="builder")

    # The fallback m-2 has the same limit and gets the same prompt: in a real run three judge
    # models were each cut off at 16,000 tokens and all three were billed. Stop at the first.
    assert calls == ["m-1"]
    summary = summarize_usage(router.call_log)
    assert summary.total_calls == 1
    assert summary.failed_calls == 1
    assert summary.completion_tokens == 32000  # the cut-off reply was billed, so it counts
    assert "unusable response" in render_token_report_md(summary)


async def test_truncation_error_is_not_retried_as_invalid_json(tmp_path, monkeypatch):
    from atlas.pipeline.util import complete_json_model
    from atlas.router import OutputTruncatedError
    from atlas.schemas import ParsedSpec

    cfg = tmp_path / "models.yaml"
    cfg.write_text(CFG + "  repair_attempts: 2\n", encoding="utf-8")
    router = ModelRouter(cfg)
    fake = AsyncMock(return_value=_response('{"summary": "cut', finish_reason="length"))
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake)
    with pytest.raises(OutputTruncatedError):
        await complete_json_model(router, model="m-1", role="parser", system="s", user="u",
                                  schema=ParsedSpec)
    assert fake.await_count == 1  # no "repair" call that would be cut off the same way


async def test_empty_reply_with_usage_is_still_counted(router, monkeypatch):
    replies = [_response("   ", prompt=70, completion=3), _response('{"ok": 1}')]
    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=replies))
    await router.complete("m-1", "sys", "usr", role="parser")
    summary = summarize_usage(router.call_log)
    assert summary.total_calls == 2
    assert summary.failed_calls == 1
    assert summary.prompt_tokens == 80


async def test_spending_cap_holds_for_parallel_calls(tmp_path, monkeypatch):
    import asyncio

    from atlas.router import TokenBudgetExceededError

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "roles:\n  researcher: [m-1]\nrole_fallbacks:\n  researcher: [m-2]\nlimits:\n"
        "  max_parallel_calls: 1\n  max_tokens: 100\n  timeout_seconds: 10\n"
        "  attempts_per_model: 1\n  max_total_tokens: 30\n",
        encoding="utf-8",
    )
    router = ModelRouter(cfg)

    async def slow_reply(**kwargs):
        await asyncio.sleep(0.01)  # like a real network call, so the six calls overlap
        return _response('{"ok": 1}', prompt=10, completion=5)

    fake = AsyncMock(side_effect=slow_reply)
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake)

    # Six calls queued at once, like the research fan-out. Each uses 15 tokens; the cap is 30.
    results = await asyncio.gather(
        *(router.complete("m-1", "s", "u", role="researcher") for _ in range(6)),
        return_exceptions=True,
    )
    assert fake.await_count == 2  # before: all 6 passed the check while spend was still 0
    assert sum(isinstance(r, TokenBudgetExceededError) for r in results) == 4
    assert {c["used_model"] for c in router.call_log} == {"m-1"}  # cap did not try fallbacks


@pytest.mark.parametrize(
    "extra, match",
    [
        ("  max_tokens_by_role: [1]\n", "max_tokens_by_role must be a mapping"),
        ("  max_tokens_by_role:\n    builder: 0\n", r"max_tokens_by_role\['builder'\]"),
        ("  timeout_seconds_by_role:\n    builder: 0\n", r"timeout_seconds_by_role\['builder'\]"),
    ],
)
def test_role_limit_validation(tmp_path, extra, match):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "roles:\n  p: [m]\nlimits:\n  max_parallel_calls: 1\n  max_tokens: 10\n"
        "  timeout_seconds: 5\n" + extra,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        ModelRouter(cfg)


def test_shipped_config_enables_role_limits_and_coverage():
    router = ModelRouter(Path(__file__).resolve().parents[1] / "config" / "models.yaml")
    assert router.max_tokens_for("builder") > router.max_tokens_for("parser")
    assert router.timeout_for("builder") > router.timeout_for("parser")
    assert router.require_requirement_tests is True


# ---------------------------------------------------------------- ledger


def _summary(prompt, completion, cost, role="builder", model="m-1"):
    return summarize_usage(
        [
            {
                "role": role,
                "used_model": model,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cost_usd": cost,
            }
        ]
    )


def test_ledger_append_read_totals_and_report(tmp_path):
    path = tmp_path / "usage" / "ledger.jsonl"
    append_ledger(path, ledger_entry(run_id="r1", mode="build", spec_path="specs/a.md",
                                     status="verified", summary=_summary(100, 50, 0.01)))
    append_ledger(path, ledger_entry(run_id="r2", mode="plan", spec_path="specs/a.md",
                                     status="plan_ready",
                                     summary=_summary(10, 5, None, role="judge", model="m-9")))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{broken line from a bad merge\n\n")

    entries = read_ledger(path)
    assert [e["run_id"] for e in entries] == ["r1", "r2"]
    totals = ledger_totals(entries)
    assert totals["total_tokens"] == 165
    assert totals["cost_usd"] == 0.01
    assert totals["by_spec"]["specs/a.md"]["runs"] == 2
    assert set(totals["by_model"]) == {"m-1", "m-9"}
    report = render_ledger_report_md(entries)
    assert "**165 tokens** across 2 run(s)" in report
    assert "| `r2` | plan | `specs/a.md` | plan_ready | 15 |" in report
    assert "No runs recorded" in render_ledger_report_md([])


def test_cli_tokens_all_reads_the_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_ledger(path, ledger_entry(run_id="r1", mode="build", spec_path="specs/a.md",
                                     status="verified", summary=_summary(1000, 234, 0.5)))
    res = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "tokens", "--all", "--ledger", str(path), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["total_tokens"] == 1234


# ---------------------------------------------------------------- requirement coverage


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_requirement_coverage_check(tmp_path):
    _write(tmp_path, "tests/test_core.py", "def test_r001_add():\n    pass\n# R10\n")
    _write(tmp_path, "app/latest.py", "# R002 is mentioned here, but this is not a test file\n")
    _write(tmp_path, "web/sum.test.js", "test('R003 sums', () => {})\n")
    _write(tmp_path, "node_modules/x/test_x.py", "# R002 inside a dependency\n")

    ok = check_requirement_coverage(tmp_path, ["R001", "R003", "R10"])
    assert ok.passed and ok.name == "spec-coverage"
    assert "R003: web/sum.test.js" in ok.stdout

    bad = check_requirement_coverage(tmp_path, ["R001", "R002", "R1"])
    assert not bad.passed
    assert "R002, R1" in bad.stderr  # R1 must not match R10 or R001

    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No test files" in check_requirement_coverage(empty, ["R001"]).stderr


class ScriptedRouter(ModelRouter):
    """Returns canned JSON per role and records token usage, without any network."""

    def __init__(self, replies: dict[str, list[str]], coverage: bool = True):
        self.path = Path("fake")
        self.cfg = {
            "roles": {"parser": ["m-p"], "builder": ["m-b"], "fixer": ["m-f"]},
            "limits": {"max_parallel_calls": 1, "max_tokens": 100, "timeout_seconds": 5,
                       "repair_attempts": 0, "max_fix_rounds": 2, "verify_timeout_seconds": 60},
            "verification": {"require_requirement_tests": coverage},
        }
        self._semaphores = {}
        self.call_log = []
        self.replies = {k: list(v) for k, v in replies.items()}
        self.prompts: dict[str, list[str]] = {}

    async def complete(self, model, system, user, role=None):
        self.prompts.setdefault(role, []).append(user)
        self.call_log.append({"role": role, "used_model": model, "prompt_tokens": 10,
                              "completion_tokens": 10, "cost_usd": 0.001})
        return model, self.replies[role].pop(0)


async def test_missing_requirement_test_goes_to_fixer_and_run_is_ledgered(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Adder\nadd(a, b) and sub(a, b) with tests.\n", encoding="utf-8")
    parsed = {"summary": "Math", "requirements": [
        {"id": "R001", "title": "add", "description": "add(a, b) returns a + b"},
        {"id": "R002", "title": "sub", "description": "sub(a, b) returns a - b"},
        {"id": "R003", "title": "fast", "description": "under 1s", "kind": "non_functional"},
    ]}
    code = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
    built = {"summary": "Math lib", "stack": "python", "files": [
        {"path": "mathlib.py", "content": code},
        {"path": "tests/test_mathlib.py",
         "content": "from mathlib import add\n\n\n"
                    "def test_r001_add():\n    assert add(2, 3) == 5\n"},
    ]}
    patch = {"diagnosis": "R002 had no test.", "files": [
        {"path": "tests/test_sub.py",
         "content": "from mathlib import sub\n\n\ndef test_sub():  # R002\n"
                    "    assert sub(5, 3) == 2\n"},
    ]}
    router = ScriptedRouter({"parser": [json.dumps(parsed)], "builder": [json.dumps(built)],
                             "fixer": [json.dumps(patch)]})
    ledger = tmp_path / "usage" / "ledger.jsonl"

    out = await run_build(spec, tmp_path / "runs", tmp_path / "app", router=router,
                          skip_research=True, clean_app_dir=True, ledger_path=ledger)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    first = {c["name"]: c for c in verification["rounds"][0]["checks"]}
    assert first["python-pytest"]["passed"] is True  # the tests themselves were fine ...
    assert first["spec-coverage"]["passed"] is False  # ... but R002 had none
    assert "R002" in first["spec-coverage"]["stderr"]
    assert "R003" not in first["spec-coverage"]["command"]  # non-functional is not required
    assert "spec-coverage" in router.prompts["fixer"][0]  # the fixer was told why

    entries = read_ledger(ledger)
    assert len(entries) == 1
    assert entries[0]["status"] == "verified"
    assert entries[0]["mode"] == "build"
    assert entries[0]["total_tokens"] == 60  # parser + builder + fixer, 20 tokens each
    assert entries[0]["by_stage"]["fixer"]["calls"] == 1


async def test_coverage_check_is_off_unless_configured(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Adder\n", encoding="utf-8")
    parsed = {"summary": "Add", "requirements": [{"id": "R001", "title": "add",
                                                  "description": "add"}]}
    built = {"summary": "Add", "stack": "python", "files": [
        {"path": "tests/test_ok.py", "content": "def test_ok():\n    assert 1 + 1 == 2\n"}]}
    router = ScriptedRouter({"parser": [json.dumps(parsed)], "builder": [json.dumps(built)]},
                            coverage=False)
    out = await run_build(spec, tmp_path / "runs", tmp_path / "app", router=router,
                          skip_research=True)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert not (tmp_path / "usage").exists()  # no ledger unless a path is given


# ---------------------------------------------------------------- provider failures

DOWN_CFG = """
roles:
  parser: [openrouter/a]
  researcher: [openrouter/r1, openrouter/r2, openrouter/r3]
role_fallbacks:
  parser: [openrouter/b, gemini/g]
  researcher: [gemini/g]
limits:
  max_parallel_calls: 2
  max_tokens: 100
  timeout_seconds: 10
  attempts_per_model: 2
"""


class _HTTPError(Exception):
    def __init__(self, status_code, message="error"):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def down_router(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY_FALLBACK", raising=False)
    cfg = tmp_path / "models.yaml"
    cfg.write_text(DOWN_CFG, encoding="utf-8")
    return ModelRouter(cfg)


async def test_no_credit_skips_the_provider_for_the_rest_of_the_run(down_router, monkeypatch):
    calls = []

    async def fake(*, model, **kwargs):
        calls.append(model)
        if model.startswith("openrouter/"):
            raise _HTTPError(402, "This request requires more credits")
        return _response('{"ok": true}')

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake))
    monkeypatch.setattr("atlas.router.asyncio.sleep", AsyncMock())

    used, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used == "gemini/g"
    # one try at OpenRouter (no retry of a 402), then openrouter/b is skipped without a call
    assert calls == ["openrouter/a", "gemini/g"]
    assert down_router.unavailable_reason("openrouter/anything") is not None

    calls.clear()
    await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert calls == ["gemini/g"]  # later calls go straight to the working provider


async def test_unknown_model_is_skipped_but_its_provider_is_not(down_router, monkeypatch):
    calls = []

    async def fake(*, model, **kwargs):
        calls.append(model)
        if model == "openrouter/a":
            raise _HTTPError(404, "model not found")
        return _response('{"ok": true}')

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake))
    used, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used == "openrouter/b"
    assert calls == ["openrouter/a", "openrouter/b"]
    assert down_router.unavailable_reason("openrouter/b") is None


async def test_rate_limits_are_still_retried(down_router, monkeypatch):
    replies = [_HTTPError(429, "slow down"), _response('{"ok": true}')]
    fake = AsyncMock(side_effect=replies)
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake)
    monkeypatch.setattr("atlas.router.asyncio.sleep", AsyncMock())
    used, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used == "openrouter/a"
    assert fake.await_count == 2
    assert down_router.unavailable_reason("openrouter/a") is None


def test_credit_error_without_status_code_is_recognised():
    from atlas.router import _status_code

    assert _status_code(RuntimeError('{"error":{"code":402,"message":"..."}}')) == 402
    assert _status_code(RuntimeError("This request requires more credits")) == 402
    assert _status_code(RuntimeError("something else")) is None


async def test_research_does_not_ask_the_same_fallback_model_three_times(
    down_router, monkeypatch
):
    from atlas.pipeline.research import research
    from atlas.schemas import ParsedSpec

    down_router._mark_unavailable("openrouter/r1", 402, RuntimeError("no credit"))
    fake = AsyncMock(return_value=_response('{"findings": ["use the standard library"]}'))
    monkeypatch.setattr("atlas.router.litellm.acompletion", fake)
    spec = ParsedSpec.model_validate({"summary": "s", "requirements": [
        {"id": "R001", "title": "a", "description": "a"},
        {"id": "R002", "title": "b", "description": "b"},
    ]})

    report = await research(down_router, spec)

    assert fake.await_count == 2  # one call per requirement, not three
    assert {n.model for n in report.notes} == {"gemini/g"}


async def test_provider_switches_to_fallback_api_key_before_disabling_provider(
    down_router, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
    monkeypatch.setenv("OPENROUTER_API_KEY_FALLBACK", "backup-key")

    seen_calls: list[tuple[str, str | None]] = []

    async def fake(*, model, **kwargs):
        api_key = kwargs.get("api_key")
        seen_calls.append((model, api_key))
        if model.startswith("openrouter/") and api_key != "backup-key":
            raise _HTTPError(402, "This request requires more credits")
        if model.startswith("openrouter/") and len(seen_calls) >= 4:
            raise _HTTPError(402, "Backup key also out of credits")
        return _response('{"ok": true}')

    monkeypatch.setattr("atlas.router.litellm.acompletion", AsyncMock(side_effect=fake))
    monkeypatch.setattr("atlas.router.asyncio.sleep", AsyncMock())

    used1, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used1 == "openrouter/a"
    assert seen_calls == [("openrouter/a", None), ("openrouter/a", "backup-key")]

    used2, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used2 == "openrouter/a"
    assert seen_calls[-1] == ("openrouter/a", "backup-key")

    used3, _ = await down_router.complete("openrouter/a", "s", "u", role="parser")
    assert used3 == "gemini/g"
    assert seen_calls[-2:] == [("openrouter/a", "backup-key"), ("gemini/g", None)]
    assert down_router.unavailable_reason("openrouter/anything") is not None


# ---------------------------------------------------------------- judge per requirement


def _judge_spec_and_report():
    from atlas.schemas import CandidateTool, ParsedSpec, ResearchNote, ResearchReport

    spec = ParsedSpec.model_validate({"summary": "s", "requirements": [
        {"id": "R001", "title": "a", "description": "a"},
        {"id": "R002", "title": "b", "description": "b"},
        {"id": "R003", "title": "c", "description": "c"},
    ]})
    notes = [
        ResearchNote(requirement_id=rid, model="m", candidates=[
            CandidateTool(name="x", pros=["fast"]), CandidateTool(name="y")])
        for rid in ("R001", "R002", "R003")
    ]
    return spec, ResearchReport(notes=notes)


class JudgeRouter(ModelRouter):
    def __init__(self, reply_for):
        self.path = Path("fake")
        self.cfg = {"roles": {"judge": ["m-j"]},
                    "limits": {"max_parallel_calls": 3, "max_tokens": 100,
                               "timeout_seconds": 5, "repair_attempts": 1}}
        self._semaphores = {}
        self.call_log = []
        self.reply_for = reply_for
        self.users: list[str] = []

    async def complete(self, model, system, user, role=None):
        import asyncio

        self.users.append(user)
        req_id = next(r for r in ("R001", "R002", "R003") if f"RESEARCH FOR {r}" in user)
        await asyncio.sleep(0.05 if req_id == "R003" else 0)  # R003 finishes last
        reply = self.reply_for(req_id)
        if isinstance(reply, Exception):
            raise reply
        self.call_log.append({"role": role, "used_model": model, "prompt_tokens": 10,
                              "completion_tokens": 10})
        return model, reply


def _decision(req_id_as_written):
    return json.dumps({"decisions": [{
        "requirement_id": req_id_as_written,
        "options": [{"name": "x"}, {"name": "y"}], "pick": "x", "rationale": "fits"}]})


async def test_judge_fixes_a_mistyped_requirement_id_without_a_repair_call():
    from atlas.pipeline.judge import judge

    spec, report = _judge_spec_and_report()
    written = {"R001": "r001", "R002": "R2", "R003": "R003"}
    router = JudgeRouter(lambda rid: _decision(written[rid]))
    plan = await judge(router, spec, report)
    assert [d.requirement_id for d in plan.decisions] == ["R001", "R002", "R003"]
    assert len(router.users) == 3  # no repair calls
    assert '"pros":["fast"]' in router.users[0]  # compact JSON: indentation is billed
    assert "\n  " not in router.users[0]


async def test_judge_records_every_call_before_reporting_a_failure():
    from atlas.pipeline.judge import judge

    spec, report = _judge_spec_and_report()

    def reply(rid):
        return RuntimeError("provider down") if rid == "R001" else _decision(rid)

    router = JudgeRouter(reply)
    with pytest.raises(RuntimeError, match=r"Judge failed for 1 of 3 requirement\(s\): R001"):
        await judge(router, spec, report)
    # R003 was still running when R001 failed; its tokens must still be in the log.
    assert len(router.call_log) == 2
