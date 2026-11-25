"""`run_build --skip-research` with clarifications and E2E tests (no researcher/judge stage)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.router import ModelRouter
from atlas.runner import run_build

PARSED = {
    "summary": "CLI greeter.",
    "requirements": [
        {
            "id": "R001",
            "title": "Greeting",
            "description": "python -m greeter <name> prints a greeting.",
            "acceptance_criteria": ["python -m greeter Ada prints the agreed greeting"],
        }
    ],
    "open_questions": ["What greeting prefix should be printed?"],
}
E2E_TEST = (
    "import subprocess\nimport sys\nfrom pathlib import Path\n\n\n"
    "def test_cli_greets_end_to_end():\n"
    "    # R001\n"
    "    root = Path(__file__).resolve().parents[1]\n"
    "    proc = subprocess.run([sys.executable, '-m', 'greeter', 'Ada'], cwd=root,\n"
    "                          capture_output=True, text=True, check=False)\n"
    "    assert proc.returncode == 0\n"
    "    assert proc.stdout.strip() == 'HELLO: Ada'\n"
)
E2E = {
    "summary": "Subprocess E2E check.",
    "product_interface": "python -m greeter <name> -> stdout 'HELLO: <name>'",
    "stack": "python",
    "scenarios": ["Run python -m greeter Ada"],
    "files": [{"path": "tests/test_e2e_product.py", "content": E2E_TEST,
               "purpose": "Black-box E2E"}],
}
APP = {
    "summary": "Greeter",
    "stack": "python",
    "files": [
        {"path": "README.md", "content": "# Greeter\nRun `python -m greeter Ada`.\n"},
        {"path": "requirements.txt", "content": "pytest\n"},
        {"path": "greeter/__init__.py", "content": ""},
        {"path": "greeter/__main__.py",
         "content": "import sys\n\nprint(f\"HELLO: {sys.argv[1]}\")\n"},
    ],
}


class _Router(ModelRouter):
    def __init__(self) -> None:
        self.path = Path("fake")
        self.cfg = {
            "roles": {r: [f"m-{r}"] for r in ("parser", "researcher", "judge", "builder", "fixer")},
            "role_fallbacks": {},
            "fallbacks": [],
            "limits": {"max_parallel_calls": 2, "max_tokens": 4000, "timeout_seconds": 30,
                       "attempts_per_model": 1, "repair_attempts": 0, "max_fix_rounds": 0,
                       "verify_timeout_seconds": 60},
            "verification": {"require_requirement_tests": True,
                             "require_product_contract": True, "require_e2e_tests": True},
        }
        self._semaphores = {}
        self.call_log = []
        self.roles_called: list[str] = []
        self.prompts: dict[str, str] = {}

    async def complete(self, model, system, user, role=None):
        key = role or ""
        if key == "researcher" and "black-box end-to-end" in system.lower():
            key = "e2e"
        self.roles_called.append(key)
        self.prompts[key] = user
        self.call_log.append({"role": role, "requested_model": model, "used_model": model,
                              "prompt_tokens": 10, "completion_tokens": 5})
        payload = {"parser": PARSED, "e2e": E2E, "builder": APP}[key]
        return model, json.dumps(payload)


async def test_skip_research_still_clarifies_and_generates_e2e_tests(tmp_path):
    spec = tmp_path / "greeter.md"
    spec.write_text("# Greeter\nBuild python -m greeter <name>.\n", encoding="utf-8")
    router = _Router()

    out = await run_build(
        spec, tmp_path / "runs", tmp_path / "app", router=router, skip_research=True,
        clarification_answers={"1": "HELLO:"}, generate_e2e=True,
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert router.roles_called == ["parser", "e2e", "builder"]  # no researcher, no judge
    assert manifest["stages"]["research"] == "skipped"
    assert manifest["stages"]["judge"] == "skipped"
    assert manifest["stages"]["e2e_tests"] == "completed"
    assert manifest["e2e_test_files"] == ["tests/test_e2e_product.py"]
    assert "HELLO:" in router.prompts["e2e"]  # the clarified answer reached the E2E author
    assert manifest["status"] == "verified", manifest.get("error")


async def test_run_build_from_run_reuses_plan_and_e2e_without_replanning(tmp_path):
    spec = tmp_path / "greeter.md"
    spec.write_text("# Greeter\nBuild python -m greeter <name>.\n", encoding="utf-8")

    prev_dir = tmp_path / "prev_run"
    prev_dir.mkdir(parents=True)
    (prev_dir / "spec.json").write_text(json.dumps(PARSED), encoding="utf-8")
    (prev_dir / "research.json").write_text(
        json.dumps(
            {
                "notes": [
                    {
                        "requirement_id": "R001",
                        "model": "m-res",
                        "candidates": [{"name": "stdlib"}],
                        "findings": ["Use sys.argv"],
                    }
                ],
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    (prev_dir / "clarifications.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "question": "What greeting prefix should be printed?",
                        "answer": "HELLO:",
                        "source": "spec",
                    }
                ],
                "interactive": False,
            }
        ),
        encoding="utf-8",
    )
    (prev_dir / "plan.json").write_text(
        json.dumps(
            {
                "spec": PARSED,
                "decisions": [
                    {
                        "requirement_id": "R001",
                        "options": [{"name": "stdlib"}, {"name": "click"}],
                        "pick": "stdlib",
                        "rationale": "Zero dependencies",
                    }
                ],
                "risks": [],
                "tasks": ["Implement greeter/__main__.py"],
            }
        ),
        encoding="utf-8",
    )
    (prev_dir / "e2e_tests.json").write_text(json.dumps(E2E), encoding="utf-8")

    router = _Router()
    out = await run_build(
        spec,
        tmp_path / "runs",
        tmp_path / "app",
        router=router,
        from_run=prev_dir,
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert router.roles_called == ["builder"]
    assert manifest["resumed_from"] == str(prev_dir)
    assert manifest["stages"]["parse"] == "reused"
    assert manifest["stages"]["research"] == "reused"
    assert manifest["stages"]["clarify"] == "reused"
    assert manifest["stages"]["judge"] == "reused"
    assert manifest["stages"]["e2e_tests"] == "reused"
    assert manifest["status"] == "verified", manifest.get("error")

    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        await run_build(
            spec,
            tmp_path / "runs2",
            tmp_path / "app2",
            router=router,
            from_run=tmp_path / "missing_run",
        )
