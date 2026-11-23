import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.pipeline.verify import _verification_env, verify_app
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
