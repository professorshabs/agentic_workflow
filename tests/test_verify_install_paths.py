"""Product verification install/test paths, with subprocesses faked (no network, no npm)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.pipeline import verify
from atlas.schemas import CheckResult


class Recorder:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.fail = fail or set()

    async def __call__(self, name, argv, cwd, timeout, env=None):
        self.calls.append((name, list(argv)))
        ok = name not in self.fail
        return CheckResult(name=name, command=" ".join(argv), passed=ok, exit_code=0 if ok else 1)

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]


def _python_app(tmp_path: Path, *, pyproject: str | None = None) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "requirements.txt").write_text("# deps\nfastapi>=0.110\npytest\n", encoding="utf-8")
    if pyproject is not None:
        (app / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return app


@pytest.mark.parametrize("has_uv", [True, False])
async def test_python_app_with_dependencies_is_verified_in_an_isolated_venv(
    monkeypatch, tmp_path, has_uv
):
    app = _python_app(tmp_path, pyproject='[project]\nname = "x"\ndependencies = []\n')
    rec = Recorder()
    monkeypatch.setattr(verify, "_run_subprocess", rec)
    monkeypatch.setattr(
        verify, "_find_tool", lambda name: "/usr/bin/uv" if (name == "uv" and has_uv) else None
    )

    checks = await verify._verify_python(app, 30)

    assert rec.names() == ["python-venv", "python-install", "python-compile", "python-pytest"]
    venv_cmd, install_cmd = rec.calls[0][1], rec.calls[1][1]
    if has_uv:
        assert venv_cmd[:2] == ["/usr/bin/uv", "venv"]
        assert install_cmd[:3] == ["/usr/bin/uv", "pip", "install"]
    else:
        assert venv_cmd[1:3] == ["-m", "venv"]
        assert install_cmd[1:4] == ["-m", "pip", "install"]
    assert "-r" in install_cmd and "-e" in install_cmd and "pytest" in install_cmd
    venv_python = rec.calls[2][1][0]
    assert venv_python.endswith("bin/python") and "atlas-verify-" in venv_python
    assert [c.name for c in checks][0] == "python-install"


@pytest.mark.parametrize("failing", ["python-venv", "python-install"])
async def test_isolated_venv_stops_at_the_first_failed_setup_step(monkeypatch, tmp_path, failing):
    app = _python_app(tmp_path)
    rec = Recorder(fail={failing})
    monkeypatch.setattr(verify, "_run_subprocess", rec)
    monkeypatch.setattr(verify, "_find_tool", lambda name: None)

    checks = await verify._verify_python(app, 30)

    assert [c.name for c in checks] == [failing]
    assert checks[0].passed is False
    assert "python-pytest" not in rec.names()


def test_unparseable_pyproject_counts_as_having_dependencies(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "pyproject.toml").write_text("[project\nbroken", encoding="utf-8")
    assert verify._has_external_python_deps(app) is True
    (app / "pyproject.toml").write_text('[project]\ndependencies = ["pytest>=8"]\n')
    assert verify._has_external_python_deps(app) is False


def _node_app(tmp_path: Path, pkg: dict | str) -> Path:
    app = tmp_path / "web"
    app.mkdir()
    text = pkg if isinstance(pkg, str) else json.dumps(pkg)
    (app / "package.json").write_text(text, encoding="utf-8")
    return app


async def test_invalid_package_json_fails_fast(tmp_path):
    checks = await verify._verify_node(_node_app(tmp_path, "{not json"), 30)
    assert [c.name for c in checks] == ["node-package-json"] and not checks[0].passed


async def test_node_dependencies_without_npm_fail_install(monkeypatch, tmp_path):
    app = _node_app(tmp_path, {"dependencies": {"express": "^5"}, "scripts": {"test": "vitest"}})
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    checks = await verify._verify_node(app, 30)
    assert [c.name for c in checks] == ["node-install"] and checks[0].exit_code == 127


async def test_node_install_failure_stops_verification(monkeypatch, tmp_path):
    app = _node_app(tmp_path, {"devDependencies": {"vitest": "^3"}, "scripts": {"test": "vitest"}})
    rec = Recorder(fail={"node-install"})
    monkeypatch.setattr(verify, "_run_subprocess", rec)
    monkeypatch.setattr(verify.shutil, "which", lambda name: f"/usr/bin/{name}")
    checks = await verify._verify_node(app, 30)
    assert [c.name for c in checks] == ["node-install"]
    assert rec.calls[0][1] == ["/usr/bin/npm", "install", "--no-audit", "--no-fund"]


async def test_node_runs_test_then_typecheck_build_lint_scripts(monkeypatch, tmp_path):
    app = _node_app(
        tmp_path,
        {"scripts": {"test": "vitest run", "typecheck": "tsc", "build": "vite build",
                     "lint": "eslint .", "dev": "vite"}},
    )
    rec = Recorder()
    monkeypatch.setattr(verify, "_run_subprocess", rec)
    monkeypatch.setattr(verify.shutil, "which", lambda name: f"/usr/bin/{name}")
    await verify._verify_node(app, 30)
    assert rec.names() == ["node-test", "node-typecheck", "node-build", "node-lint"]


async def test_node_test_script_without_npm_is_reported(monkeypatch, tmp_path):
    app = _node_app(tmp_path, {"scripts": {"test": "vitest run"}})
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    checks = await verify._verify_node(app, 30)
    assert checks[0].name == "node-test" and checks[0].exit_code == 127


async def test_node_test_files_need_node(monkeypatch, tmp_path):
    app = _node_app(tmp_path, {"scripts": {"test": 'echo "Error: no test specified"'}})
    (app / "sum.test.js").write_text("// R1\n", encoding="utf-8")
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    checks = await verify._verify_node(app, 30)
    assert checks[0].command == "node --test" and checks[0].exit_code == 127

    rec = Recorder()
    monkeypatch.setattr(verify, "_run_subprocess", rec)
    monkeypatch.setattr(verify.shutil, "which", lambda name: f"/usr/bin/{name}")
    await verify._verify_node(app, 30)
    assert rec.calls[0][1] == ["/usr/bin/node", "--test", "sum.test.js"]


async def test_node_app_without_any_tests_fails(monkeypatch, tmp_path):
    app = _node_app(tmp_path, {"name": "x"})
    monkeypatch.setattr(verify.shutil, "which", lambda name: None)
    checks = await verify._verify_node(app, 30)
    assert [c.name for c in checks] == ["node-tests-present"] and not checks[0].passed
