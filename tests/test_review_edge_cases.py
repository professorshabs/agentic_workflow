"""Edge cases found in the 2026-11-26 review: fixer guards, E2E hashes, skills, doctor, CLI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from atlas.pipeline import build as build_mod
from atlas.pipeline.build import (
    FixerRequestError,
    _drop_partial_file_replacements,
    _existing_skill_links,
    materialize_files,
    snapshot_app_files,
)
from atlas.pipeline.verify import check_e2e_tests, e2e_file_hashes
from atlas.router import ModelRouter
from atlas.schemas import (
    CheckResult,
    GeneratedFile,
    ParsedSpec,
    PatchResult,
    Requirement,
    VerificationRound,
)
from atlas.skills import SkillMetadata, detect_used_skills

# --- fixer only sees part of a large file ------------------------------------------------


class _NoSkillsRouter:
    """Just enough router for `fix_app` once the model call is stubbed out."""

    def skills_config(self) -> dict[str, object]:
        return {}


def test_snapshot_reports_files_it_could_not_show_whole(tmp_path):
    (tmp_path / "a_small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b_big.html").write_text("<p>" + "y" * 500 + "</p>\n", encoding="utf-8")
    (tmp_path / "c_after_budget.py").write_text("z = 3\n", encoding="utf-8")
    (tmp_path / "hero.webp").write_bytes(b"RIFF\x00\xff\xfe\x00WEBPVP8 ")
    partial: list[str] = []

    snap = snapshot_app_files(tmp_path, max_total_chars=200, partial=partial)

    assert snap["a_small.py"] == "x = 1\n"
    assert snap["b_big.html"].endswith("[truncated]")
    assert "c_after_budget.py" not in snap
    assert sorted(partial) == ["b_big.html", "c_after_budget.py", "hero.webp"]


def test_snapshot_without_partial_keeps_its_old_behaviour(tmp_path):
    (tmp_path / "a.py").write_text("a" * 50, encoding="utf-8")
    (tmp_path / "b.py").write_text("b" * 50, encoding="utf-8")
    assert list(snapshot_app_files(tmp_path, max_total_chars=40)) == ["a.py"]


def test_full_rewrites_of_partly_seen_files_are_dropped():
    patch = PatchResult(
        diagnosis="Fix the hero",
        files=[
            GeneratedFile(path="Index.html", content="<html>short</html>\n"),
            GeneratedFile(path="styles.css", content="p { color: red }\n"),
        ],
    )
    kept = _drop_partial_file_replacements(patch, ["index.html"])
    assert [f.path for f in kept.files] == ["styles.css"]
    assert "did not apply the replacement for Index.html" in kept.diagnosis
    assert _drop_partial_file_replacements(patch, ["other.py"]) is patch


@pytest.mark.asyncio
async def test_fix_app_tells_the_fixer_which_files_it_only_saw_in_part(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    big = "<html>" + "x" * 45_000 + "</html>\n"
    (app / "index.html").write_text(big, encoding="utf-8")
    (app / "notes.txt").write_text("ok\n", encoding="utf-8")
    seen: dict[str, object] = {}

    async def fake_complete(router, *, model, role, system, user, schema, **kwargs):
        seen.update(json.loads(user))
        return model, PatchResult(
            diagnosis="Rewrote the page",
            files=[
                GeneratedFile(path="index.html", content="<html>cut</html>\n"),
                GeneratedFile(path="notes.txt", content="fixed\n"),
            ],
        )

    monkeypatch.setattr(build_mod, "complete_json_model", fake_complete)
    monkeypatch.setattr(
        build_mod, "fixer_target_for_round", lambda router, rnd: ("openrouter/m", None, "m")
    )
    spec = ParsedSpec(
        summary="s", requirements=[Requirement(id="R001", title="t", description="d")]
    )
    failed = VerificationRound(
        round_number=1,
        stack_detected="python",
        passed=False,
        checks=[CheckResult(name="pytest", command="pytest", passed=False, exit_code=1)],
    )

    patch = await build_mod.fix_app(_NoSkillsRouter(), spec, app, failed)  # type: ignore[arg-type]

    assert seen["partial_files"] == ["index.html"]
    assert [f.path for f in patch.files] == ["notes.txt"]
    materialize_files(app, patch.files)
    assert (app / "index.html").read_text(encoding="utf-8") == big


@pytest.mark.asyncio
async def test_fixer_errors_keep_the_underlying_reason(tmp_path, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("model stopped at the output limit\nsecond line")

    monkeypatch.setattr(build_mod, "complete_json_model", boom)
    monkeypatch.setattr(
        build_mod, "fixer_target_for_round", lambda router, rnd: ("openrouter/m", None, "m")
    )
    spec = ParsedSpec(
        summary="s", requirements=[Requirement(id="R001", title="t", description="d")]
    )
    failed = VerificationRound(round_number=1, stack_detected="python", passed=False, checks=[])
    with pytest.raises(FixerRequestError) as info:
        await build_mod.fix_app(_NoSkillsRouter(), spec, tmp_path, failed)  # type: ignore[arg-type]
    assert str(info.value) == (
        "Fixer request failed with RuntimeError: model stopped at the output limit"
    )


# --- a fixer round that fails with rounds left does not end the run ----------------------


@pytest.mark.asyncio
async def test_failed_fixer_round_retries_the_same_fixer_while_rounds_remain(
    tmp_path, monkeypatch
):
    from atlas import runner as runner_mod
    from atlas.runner import run_build
    from atlas.schemas import GeneratedApp, Plan, ToolDecision

    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nA small app\n", encoding="utf-8")
    spec = ParsedSpec(
        summary="Demo app",
        requirements=[Requirement(id="R001", title="Run", description="Runs cleanly")],
    )
    plan = Plan(
        spec=spec,
        decisions=[ToolDecision(requirement_id="R001", options=[{"name": "a"}, {"name": "b"}],
                                pick="a", rationale="r")],
        risks=[],
        tasks=["Write app.py"],
    )
    from_run = tmp_path / "prev_run"
    from_run.mkdir()
    (from_run / "spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    (from_run / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "roles:\n  parser: [openrouter/m]\n  researcher: [openrouter/m]\n"
        "  judge: [openrouter/m]\n  builder: [openrouter/m]\n  fixer: [openrouter/m]\n"
        "limits:\n  max_parallel_calls: 2\n  max_tokens: 1000\n  timeout_seconds: 10\n"
        "  max_fix_rounds: 3\n"
        "verification:\n  require_requirement_tests: false\n"
        "  require_product_contract: false\n  require_e2e_tests: false\n",
        encoding="utf-8",
    )

    async def fake_build(*args, **kwargs):
        return GeneratedApp(summary="Built", stack="python",
                            files=[GeneratedFile(path="app.py", content="x = 1\n")])

    state = {"fixed": False, "fix_calls": 0}

    async def fake_verify(app_dir: Path):
        return "python", [CheckResult(name="pytest", command="pytest", passed=state["fixed"],
                                      exit_code=0 if state["fixed"] else 1)]

    diagnoses_seen: list[list[str]] = []

    async def fake_fix(*args, previous_diagnoses=None, **kwargs):
        state["fix_calls"] += 1
        diagnoses_seen.append(list(previous_diagnoses or []))
        if state["fix_calls"] == 1:
            raise FixerRequestError("Fixer request failed with OutputTruncatedError: too long")
        state["fixed"] = True
        return PatchResult(diagnosis="Fixed", files=[GeneratedFile(path="app.py", content="x=2\n")])

    monkeypatch.setattr(runner_mod, "build_app", fake_build)
    monkeypatch.setattr(runner_mod, "fix_app", fake_fix)

    await run_build(
        spec_file,
        tmp_path / "runs",
        tmp_path / "app" / "demo",
        config_path=cfg,
        from_run=from_run,
        verify_fn=fake_verify,
        ledger_path=tmp_path / "usage" / "ledger.jsonl",
    )

    assert state["fix_calls"] == 2
    assert "OutputTruncatedError: too long" in diagnoses_seen[1][0]
    assert "keep each one short" in diagnoses_seen[1][0]
    [out_dir] = list((tmp_path / "runs").iterdir())
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["verification_passed"] is True


# --- E2E integrity hash ------------------------------------------------------------------


def test_e2e_hash_matches_a_crlf_test_file_written_by_atlas(tmp_path):
    files = [GeneratedFile(path="tests/test_e2e.py",
                           content="def test_r001():\r\n    assert 1 == 1\r\n")]
    materialize_files(tmp_path, files)
    result = check_e2e_tests(tmp_path, ["tests/test_e2e.py"],
                             expected_hashes=e2e_file_hashes(files))
    assert result.passed, result.stderr


# --- skills ------------------------------------------------------------------------------


def _skill(tmp_path: Path, key: str, name: str) -> SkillMetadata:
    d = tmp_path / "skills" / key
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\nBody\n", "utf-8")
    return SkillMetadata(
        name=name, description="x", path=str(d / "SKILL.md"), source_root=str(d.parent)
    )


def test_skill_names_containing_and_are_not_split(tmp_path):
    skills = [_skill(tmp_path, "build-and-test", "build-and-test"),
              _skill(tmp_path, "writing-clean-code", "writing-clean-code")]
    used = detect_used_skills(
        skills, summary_text="Done.\nSkills used: build-and-test and writing-clean-code"
    )
    assert used == ["build-and-test", "writing-clean-code"]


def test_links_left_by_a_killed_run_still_count_as_linked(tmp_path):
    meta = _skill(tmp_path, "writing-clean-code", "writing-clean-code")
    root = tmp_path / "app"
    links = root / ".agents" / "skills"
    links.mkdir(parents=True)
    (links / "writing-clean-code").symlink_to(Path(meta.path).parent, target_is_directory=True)
    (links / "elsewhere").symlink_to(tmp_path, target_is_directory=True)
    assert _existing_skill_links(root, [meta]) == [
        root.resolve() / ".agents" / "skills" / "writing-clean-code"
    ]


# --- doctor ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doctor_does_not_ask_for_a_key_for_local_ollama_models(tmp_path, monkeypatch):
    from atlas.harnesses import run_doctor

    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "roles:\n  parser: [openrouter/m]\n  researcher: [openrouter/m]\n"
        "  judge: [openrouter/m]\n  builder: [ollama_chat/gemma4:12b]\n  fixer: [openrouter/m]\n"
        "limits:\n  max_parallel_calls: 2\n  max_tokens: 1000\n  timeout_seconds: 10\n"
        "backends:\n  builder:\n    type: litellm\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OLLAMA_CHAT_API_KEY", raising=False)
    report = await run_doctor(ModelRouter(cfg))
    by_provider = {c["provider"]: c for c in report["checks"]}
    assert by_provider["ollama_chat"]["authenticated"] is True
    assert by_provider["ollama_chat"]["auth_mode"] == "local"
    assert "OLLAMA_CHAT_API_KEY" not in json.dumps(report)


# --- CLI ---------------------------------------------------------------------------------


def test_verify_from_run_checks_the_product_folder_the_run_built(tmp_path, monkeypatch, capsys):
    from atlas import cli

    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"app_dir": "app/example-app"}), "utf-8")
    spec = ParsedSpec(
        summary="s",
        requirements=[
            Requirement(id="R001", title="t", description="d", kind="non_functional")
        ],
    )
    (run_dir / "spec.json").write_text(spec.model_dump_json(), "utf-8")
    seen: list[Path] = []

    async def fake_verify(app_path, timeout_seconds, **kwargs):
        seen.append(Path(app_path))
        return "python", [CheckResult(name="pytest", command="pytest", passed=True, exit_code=0)]

    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(cli, "verify_app", fake_verify)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["atlas", "verify", "--from-run", str(run_dir)])
    try:
        cli.main()
    except SystemExit as exc:
        assert not exc.code
    assert seen == [Path("app/example-app")]
    assert "App folder: app/example-app" in capsys.readouterr().out
    assert os.getcwd() == str(tmp_path)
