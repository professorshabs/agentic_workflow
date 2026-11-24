import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.pipeline.build import default_app_dir, materialize_files, snapshot_app_files, spec_slug
from atlas.pipeline.verify import (
    _run_subprocess,
    _verification_env,
    check_product_contract,
    detect_stack,
    verify_app,
)
from atlas.router import ModelRouter
from atlas.runner import run_build
from atlas.schemas import GeneratedFile


class MockBuildRouter(ModelRouter):
    def __init__(self, responses: dict[tuple[str, str], list[dict[str, object]]]):
        self.path = Path("fake")
        self.cfg = {
            "roles": {
                "parser": ["m-parse"],
                "researcher": ["m-res"],
                "judge": ["m-judge"],
                "builder": ["m-build"],
                "fixer": ["m-fix"],
            },
            "role_fallbacks": {},
            "fallbacks": [],
            "limits": {
                "max_parallel_calls": 4,
                "max_tokens": 2000,
                "timeout_seconds": 30,
                "attempts_per_model": 1,
                "repair_attempts": 1,
                "max_fix_rounds": 3,
                "verify_timeout_seconds": 60,
            },
        }
        self._semaphores = {}
        self.call_log = []
        self._responses = {k: list(v) for k, v in responses.items()}

    async def complete(self, model: str, system: str, user: str, role: str | None = None):
        item = self._responses[(role or "", model)].pop(0)
        prompt_tokens = int(item["prompt_tokens"])
        completion_tokens = int(item["completion_tokens"])
        cost_usd = float(item["cost_usd"])
        self.call_log.append(
            {
                "role": role,
                "requested_model": model,
                "used_model": model,
                "attempts": 1,
                "elapsed_ms": 12.5,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost_usd,
                "usage_reported": True,
            }
        )
        return model, str(item["text"])


def test_verification_environment_removes_credentials(monkeypatch):
    monkeypatch.setenv("PATH", "/tools/bin")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY_FALLBACK", "openrouter-backup-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", "/tmp/summary")
    monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "runtime-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:password@example.test/simple")
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", "registry-auth")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/google-creds.json")

    env = _verification_env()

    assert env["PATH"] == "/tools/bin"
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env
    assert "OPENROUTER_API_KEY_FALLBACK" not in env
    assert "GEMINI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "GITHUB_STEP_SUMMARY" not in env
    assert "ACTIONS_RUNTIME_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "PIP_INDEX_URL" not in env
    assert "DOCKER_AUTH_CONFIG" not in env
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env


def test_generated_file_rejects_unsafe_paths_and_preserves_indentation():
    for bad in ("../escape.py", "/etc/passwd", ".git/config", ".github/workflows/ci.yml", "  "):
        with pytest.raises(ValidationError):
            GeneratedFile(path=bad, content="x = 1\n")

    gf = GeneratedFile(path="./src/app.py", content="    def f():\n        return 1\n")
    assert gf.path == "src/app.py"
    assert gf.content.startswith("    def f():")


async def test_run_build_autofix_loop_and_token_accounting(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Adder\nBuild add(a, b) with pytest tests.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "Adder library",
            "requirements": [
                {
                    "id": "R001",
                    "title": "Add two integers",
                    "description": "Provide add(a, b) returning a + b.",
                }
            ],
        }
    )

    buggy_app = json.dumps(
        {
            "summary": "Minimal Python adder package",
            "stack": "python",
            "files": [
                {
                    "path": "adder.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                    "purpose": "Adder implementation",
                },
                {
                    "path": "tests/test_adder.py",
                    "content": (
                        "from adder import add\n\n\n"
                        "def test_add():\n"
                        "    assert add(2, 3) == 5\n"
                    ),
                    "purpose": "Unit tests",
                },
            ],
            "run_instructions": "pytest",
        }
    )

    fix_patch = json.dumps(
        {
            "diagnosis": "add() subtracted b instead of adding b.",
            "files": [
                {
                    "path": "adder.py",
                    "content": "def add(a: int, b: int) -> int:\n    return a + b\n",
                    "purpose": "Fix addition operator",
                }
            ],
            "delete_files": [],
        }
    )

    router = MockBuildRouter(
        {
            ("parser", "m-parse"): [
                {
                    "text": parsed_json,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost_usd": 0.001,
                }
            ],
            ("builder", "m-build"): [
                {
                    "text": buggy_app,
                    "prompt_tokens": 400,
                    "completion_tokens": 250,
                    "cost_usd": 0.004,
                }
            ],
            ("fixer", "m-fix"): [
                {
                    "text": fix_patch,
                    "prompt_tokens": 300,
                    "completion_tokens": 100,
                    "cost_usd": 0.002,
                }
            ],
        }
    )

    app_dir = tmp_path / "app"
    runs_dir = tmp_path / "runs"
    out_dir = await run_build(
        spec_file,
        runs_dir,
        app_dir,
        router=router,
        skip_research=True,
        clean_app_dir=True,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["verification_passed"] is True
    assert manifest["verification_rounds"] == 2
    assert manifest["token_usage"]["total_calls"] == 3
    assert manifest["token_usage"]["prompt_tokens"] == 800
    assert manifest["token_usage"]["completion_tokens"] == 400
    assert manifest["token_usage"]["total_tokens"] == 1200

    verification = json.loads((out_dir / "verification.json").read_text(encoding="utf-8"))
    assert len(verification["rounds"]) == 2
    assert verification["rounds"][0]["passed"] is False
    assert "subtracted" in verification["rounds"][0]["diagnosis"]
    assert verification["rounds"][1]["passed"] is True

    tokens_data = json.loads((out_dir / "tokens.json").read_text(encoding="utf-8"))
    assert tokens_data["total_tokens"] == 1200
    assert tokens_data["cost_usd"] == 0.007
    assert [b["name"] for b in tokens_data["by_stage"]] == ["parser", "builder", "fixer"]

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Token Usage & Cost" in report_md
    assert "1,200" in report_md
    assert "add() subtracted b instead of adding b." in report_md


async def test_verify_app_supports_nodejs_zero_dependency(tmp_path):
    app_dir = tmp_path / "node_app"
    app_dir.mkdir()
    (app_dir / "package.json").write_text(
        json.dumps({"name": "demo", "version": "1.0.0", "scripts": {"test": "node --test"}}),
        encoding="utf-8",
    )
    (app_dir / "index.mjs").write_text(
        "export function greet(name) { return `Hello, ${name}!`; }\n",
        encoding="utf-8",
    )
    (app_dir / "index.test.mjs").write_text(
        (
            "import test from 'node:test';\n"
            "import assert from 'node:assert/strict';\n"
            "import { greet } from './index.mjs';\n"
            "test('greet', () => { assert.equal(greet('Atlas'), 'Hello, Atlas!'); });\n"
        ),
        encoding="utf-8",
    )

    stack, checks = await verify_app(app_dir)
    assert stack == "node"
    assert checks
    assert all(c.passed for c in checks)


async def test_run_build_exhausted_fix_rounds_still_writes_tokens_and_report(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Broken\nAlways fails tests.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "Broken app",
            "requirements": [{"id": "R001", "title": "Fail", "description": "Fails"}],
        }
    )
    broken_app = json.dumps(
        {
            "summary": "Broken python app",
            "stack": "python",
            "files": [
                {
                    "path": "tests/test_fail.py",
                    "content": "def test_always_fails():\n    assert 1 == 2\n",
                }
            ],
        }
    )
    still_broken_patch = json.dumps(
        {
            "diagnosis": "Attempted fix but assertion still fails.",
            "files": [
                {
                    "path": "tests/test_fail.py",
                    "content": "def test_always_fails():\n    assert 1 == 3\n",
                }
            ],
        }
    )

    router = MockBuildRouter(
        {
            ("parser", "m-parse"): [
                {
                    "text": parsed_json,
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "cost_usd": 0.001,
                }
            ],
            ("builder", "m-build"): [
                {
                    "text": broken_app,
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "cost_usd": 0.002,
                }
            ],
            ("fixer", "m-fix"): [
                {
                    "text": still_broken_patch,
                    "prompt_tokens": 80,
                    "completion_tokens": 30,
                    "cost_usd": 0.001,
                }
            ],
        }
    )

    out_dir = await run_build(
        spec_file,
        tmp_path / "runs",
        tmp_path / "app",
        router=router,
        skip_research=True,
        max_fix_rounds=1,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verification_failed"
    assert manifest["verification_rounds"] == 2
    assert (out_dir / "tokens.json").is_file()
    assert (out_dir / "report.md").is_file()
    assert manifest["token_usage"]["total_tokens"] == 320


def test_cli_verify_tokens_and_validate_config_smoke(tmp_path):
    app_dir = tmp_path / "py_app"
    (app_dir / "tests").mkdir(parents=True)
    (app_dir / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")

    verify_proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "verify", "--app-dir", str(app_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_proc.returncode == 0, verify_proc.stderr
    payload = json.loads(verify_proc.stdout)
    assert payload["stack"] == "python"
    assert payload["passed"] is True

    run_dir = tmp_path / "runs" / "20261123T120000Z-deadbeef"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20261123T120000Z-deadbeef",
                "model_calls": [
                    {
                        "role": "builder",
                        "used_model": "test-model",
                        "prompt_tokens": 11,
                        "completion_tokens": 9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tokens_proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "tokens", "--runs-root", str(tmp_path / "runs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tokens_proc.returncode == 0, tokens_proc.stderr
    assert "## Token Usage & Cost" in tokens_proc.stdout
    assert "20" in tokens_proc.stdout

    val_proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "validate-config"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert val_proc.returncode == 0, val_proc.stderr
    assert "Config valid:" in val_proc.stdout


def test_materialize_files_clean_delete_and_snapshot(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / ".gitkeep").write_text("", encoding="utf-8")
    (app_dir / "stale.py").write_text("old\n", encoding="utf-8")
    (app_dir / "old_dir").mkdir()
    (app_dir / "old_dir" / "junk.txt").write_text("junk", encoding="utf-8")

    written = materialize_files(
        app_dir,
        [GeneratedFile(path="main.py", content="print('ok')")],
        clean=True,
    )
    assert written == ["main.py"]
    assert (app_dir / ".gitkeep").is_file()
    assert not (app_dir / "stale.py").exists()
    assert not (app_dir / "old_dir").exists()
    assert (app_dir / "main.py").read_text(encoding="utf-8") == "print('ok')\n"

    materialize_files(
        app_dir,
        [GeneratedFile(path="extra.py", content="x = 1\n")],
        delete_files=["main.py"],
    )
    assert not (app_dir / "main.py").exists()
    assert (app_dir / "extra.py").is_file()

    (app_dir / "__pycache__").mkdir()
    (app_dir / "__pycache__" / "ignored.pyc").write_bytes(b"\x00\x01")
    (app_dir / "z_big.txt").write_text("A" * 800, encoding="utf-8")

    snap = snapshot_app_files(app_dir, max_total_chars=600)
    assert "extra.py" in snap
    assert ".gitkeep" not in snap
    assert "__pycache__/ignored.pyc" not in snap
    assert snap["z_big.txt"].endswith("... [truncated]")


async def test_detect_stack_and_verify_app_edge_cases(tmp_path):
    missing_stack, missing_checks = await verify_app(tmp_path / "does_not_exist")
    assert missing_stack == "other"
    assert missing_checks[0].name == "app-dir-exists"
    assert missing_checks[0].passed is False

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").write_text("", encoding="utf-8")
    (empty_dir / ".venv").mkdir()
    (empty_dir / ".venv" / "ignored.py").write_text("x = 1\n", encoding="utf-8")
    assert detect_stack(empty_dir) == "other"
    empty_stack, empty_checks = await verify_app(empty_dir)
    assert empty_stack == "other"
    assert empty_checks[0].name == "stack-detection"
    assert empty_checks[0].passed is False

    py_no_tests = tmp_path / "py_no_tests"
    py_no_tests.mkdir()
    (py_no_tests / "app.py").write_text("x = 1\n", encoding="utf-8")
    assert detect_stack(py_no_tests) == "python"
    stack, checks = await verify_app(py_no_tests)
    assert stack == "python"
    pytest_chk = next(c for c in checks if c.name == "python-pytest")
    assert pytest_chk.passed is False
    assert "0 tests" in pytest_chk.stderr

    py_syntax_err = tmp_path / "py_syntax_err"
    py_syntax_err.mkdir()
    (py_syntax_err / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    _, syn_checks = await verify_app(py_syntax_err)
    assert next(c for c in syn_checks if c.name == "python-compile").passed is False

    node_bad_json = tmp_path / "node_bad"
    node_bad_json.mkdir()
    (node_bad_json / "package.json").write_text("{bad json", encoding="utf-8")
    _, bad_node_checks = await verify_app(node_bad_json)
    assert bad_node_checks[0].name == "node-package-json"
    assert bad_node_checks[0].passed is False

    node_no_tests = tmp_path / "node_no_tests"
    node_no_tests.mkdir()
    (node_no_tests / "package.json").write_text(
        json.dumps({"name": "no-tests", "scripts": {"test": "echo Error: no test specified"}}),
        encoding="utf-8",
    )
    _, no_test_checks = await verify_app(node_no_tests)
    assert no_test_checks[0].name == "node-tests-present"
    assert no_test_checks[0].passed is False

    node_scripts = tmp_path / "node_scripts"
    node_scripts.mkdir()
    (node_scripts / "package.json").write_text(
        json.dumps(
            {
                "name": "scripts-demo",
                "scripts": {
                    "typecheck": "node -e \"console.log('typecheck ok')\"",
                    "build": "node -e \"console.log('build ok')\"",
                    "lint": "node -e \"console.log('lint ok')\"",
                },
            }
        ),
        encoding="utf-8",
    )
    (node_scripts / "basic.test.mjs").write_text(
        "import test from 'node:test';\nimport assert from 'node:assert/strict';\n"
        "test('ok', () => assert.equal(1, 1));\n",
        encoding="utf-8",
    )
    _, script_checks = await verify_app(node_scripts)
    assert {c.name for c in script_checks} == {
        "node-test",
        "node-typecheck",
        "node-build",
        "node-lint",
    }
    assert all(c.passed for c in script_checks)

    mixed_dir = tmp_path / "mixed"
    (mixed_dir / "tests").mkdir(parents=True)
    (mixed_dir / "tests" / "test_mix.py").write_text("def test_mix():\n    assert 1 == 1\n")
    (mixed_dir / "package.json").write_text(
        json.dumps({"name": "mix", "scripts": {"test": "node --test"}}),
        encoding="utf-8",
    )
    (mixed_dir / "mix.test.mjs").write_text(
        "import test from 'node:test';\ntest('mix', () => {});\n",
        encoding="utf-8",
    )
    assert detect_stack(mixed_dir) == "mixed"
    mix_stack, mix_checks = await verify_app(mixed_dir)
    assert mix_stack == "mixed"
    assert any(c.name == "python-pytest" and c.passed for c in mix_checks)
    assert any(c.name == "node-test" and c.passed for c in mix_checks)

    timeout_res = await _run_subprocess(
        "slow",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        timeout=0.1,
    )
    assert timeout_res.passed is False
    assert timeout_res.exit_code == 124
    assert "timed out" in timeout_res.stderr.lower()


async def test_run_build_patch_delete_and_failure_writes_report_out(tmp_path):
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("# Demo\nBuild and test.\n", encoding="utf-8")

    parsed_json = json.dumps(
        {
            "summary": "Demo",
            "requirements": [{"id": "R1", "title": "T", "description": "D"}],
        }
    )
    app_with_bad_file = json.dumps(
        {
            "summary": "Demo",
            "stack": "python",
            "files": [
                {"path": "tests/test_good.py", "content": "def test_ok():\n    assert True\n"},
                {"path": "tests/test_bad.py", "content": "def test_bad():\n    assert False\n"},
            ],
        }
    )
    delete_patch = json.dumps(
        {
            "diagnosis": "Remove obsolete broken test file.",
            "files": [],
            "delete_files": ["tests/test_bad.py"],
        }
    )
    router = MockBuildRouter(
        {
            ("parser", "m-parse"): [
                {
                    "text": parsed_json,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cost_usd": 0.001,
                }
            ],
            ("builder", "m-build"): [
                {
                    "text": app_with_bad_file,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "cost_usd": 0.002,
                }
            ],
            ("fixer", "m-fix"): [
                {
                    "text": delete_patch,
                    "prompt_tokens": 15,
                    "completion_tokens": 5,
                    "cost_usd": 0.001,
                }
            ],
        }
    )
    app_dir = tmp_path / "app"
    report_out = tmp_path / "ci" / "run-report.md"
    out_dir = await run_build(
        spec_file,
        tmp_path / "runs",
        app_dir,
        router=router,
        skip_research=True,
        report_out=report_out,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["generated_files"] == ["tests/test_good.py"]
    assert not (app_dir / "tests/test_bad.py").exists()
    assert report_out.is_file()

    # Now verify report_out + tokens.json are still written if run_build raises mid-run
    class FailingBuildRouter(MockBuildRouter):
        async def complete(self, model: str, system: str, user: str, role: str | None = None):
            if role == "builder":
                raise RuntimeError("Missing API key")
            return await super().complete(model, system, user, role=role)

    fail_router = FailingBuildRouter(
        {
            ("parser", "m-parse"): [
                {
                    "text": parsed_json,
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "cost_usd": 0.001,
                }
            ]
        }
    )
    fail_report_out = tmp_path / "ci" / "fail-report.md"
    fail_runs = tmp_path / "fail_runs"
    with pytest.raises(RuntimeError, match="Missing API key"):
        await run_build(
            spec_file,
            fail_runs,
            tmp_path / "fail_app",
            router=fail_router,
            skip_research=True,
            report_out=fail_report_out,
        )
    assert fail_report_out.is_file()
    fail_text = fail_report_out.read_text(encoding="utf-8")
    assert "Missing API key" in fail_text
    assert "## Token Usage & Cost" in fail_text


def test_cli_error_codes_version_and_tokens_json(tmp_path):
    ver_proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ver_proc.returncode == 0
    assert "atlas" in ver_proc.stdout

    missing_verify = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "verify", "--app-dir", str(tmp_path / "nope")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_verify.returncode == 1

    missing_spec = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "build", str(tmp_path / "missing.md")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_spec.returncode == 2

    missing_runs = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "tokens", "--runs-root", str(tmp_path / "no_runs")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_runs.returncode == 2

    run_dir = tmp_path / "runs" / "20261123T120000Z-12345678"
    run_dir.mkdir(parents=True)
    (run_dir / "tokens.json").write_text(
        json.dumps(
            {
                "total_calls": 1,
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 12,
            }
        ),
        encoding="utf-8",
    )
    tok_json = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "tokens", str(run_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tok_json.returncode == 0
    assert json.loads(tok_json.stdout)["total_tokens"] == 12


def test_per_spec_app_dir_and_gitignore_rules(tmp_path):
    assert spec_slug(Path("specs/example-app.md")) == "example-app"
    assert spec_slug(Path("specs/My Cool App!.md")) == "my-cool-app"
    assert default_app_dir(Path("specs/001-api.md")) == Path("app/001-api")

    cli_proc = subprocess.run(
        [sys.executable, "-m", "atlas.cli", "app-dir", "specs/example-app.md"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cli_proc.stdout.strip() == "app/example-app"

    repo_root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "app/example-app/main.py"],
        cwd=repo_root,
        check=False,
    )
    assert ignored.returncode == 0

    kept = subprocess.run(
        ["git", "check-ignore", "-q", "app/.gitkeep"],
        cwd=repo_root,
        check=False,
    )
    assert kept.returncode != 0


def test_check_product_contract_enforces_standalone_app(tmp_path):
    prod = tmp_path / "product"
    prod.mkdir()
    (prod / "main.py").write_text("from atlas.router import ModelRouter\n", encoding="utf-8")

    bad = check_product_contract(prod)
    assert bad.name == "product-contract"
    assert bad.passed is False
    assert "Missing README" in bad.stderr
    assert "Missing dependency file" in bad.stderr
    assert "No test files found" in bad.stderr
    assert "main.py" in bad.stderr

    (prod / "README.md").write_text("# Product\n", encoding="utf-8")
    (prod / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (prod / "main.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    (prod / "test_main.py").write_text("# R001\nfrom main import hello\n", encoding="utf-8")

    good = check_product_contract(prod)
    assert good.passed is True
    assert good.exit_code == 0
