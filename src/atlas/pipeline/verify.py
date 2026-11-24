"""Stack-agnostic verification runner for generated applications."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Literal

from atlas.schemas import CheckResult

_IGNORED_DIRS = {
    ".agents",
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}
_MAX_OUTPUT_CHARS = 8000
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "ACCESS_TOKEN",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "INDEX_URL",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "AUTHORIZATION",
)
_SENSITIVE_ENV_NAMES = {
    "AWS_CONFIG_FILE",
    "AWS_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "DATABASE_URL",
    "DOCKER_CONFIG",
    "DOCKER_AUTH_CONFIG",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "KUBECONFIG",
    "MONGO_URL",
    "MONGODB_URI",
    "NPM_CONFIG_USERCONFIG",
    "REDIS_URL",
    "SSH_AUTH_SOCK",
}
_SENSITIVE_ENV_PREFIXES = ("ACTIONS_", "GITHUB_", "RUNNER_")
# tests/, __tests__/, test_x.py, x_test.go, x.test.ts, x.spec.js ... (but not "latest.py")
_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|__tests__|specs?)/|(^|/)test_[^/]+$|_test\.[^/]+$|\.(test|spec)\.[^/]+$",
    flags=re.IGNORECASE,
)


def _tail(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return "... [truncated]\n" + text[-limit:]


def _find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    sibling = Path(sys.executable).parent / name
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return None


def _verification_env() -> dict[str, str]:
    """Build a subprocess environment without credentials from the Atlas run."""
    env = os.environ.copy()
    for key in list(env):
        normalized = key.upper()
        if (
            normalized in _SENSITIVE_ENV_NAMES
            or normalized.startswith(_SENSITIVE_ENV_PREFIXES)
            or any(marker in normalized for marker in _SENSITIVE_ENV_MARKERS)
        ):
            env.pop(key, None)
    return env


def _iter_source_files(app_dir: Path) -> list[Path]:
    if not app_dir.is_dir():
        return []
    root = app_dir.resolve()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _IGNORED_DIRS for part in rel.parts):
            continue
        files.append(path)
    return files


def detect_stack(app_dir: Path) -> Literal["python", "node", "mixed", "other"]:
    """Detect whether `app_dir` is a Python, Node.js, mixed, or other project."""
    files = _iter_source_files(app_dir)
    names = {f.name for f in files}
    has_python = (
        "pyproject.toml" in names
        or "requirements.txt" in names
        or any(f.suffix == ".py" for f in files)
    )
    has_node = "package.json" in names or any(
        f.suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"} for f in files
    )
    if has_python and has_node:
        return "mixed"
    if has_python:
        return "python"
    if has_node:
        return "node"
    return "other"


async def _run_subprocess(
    name: str,
    argv: list[str],
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> CheckResult:
    started = time.monotonic()
    cmd_str = " ".join(shlex.quote(part) for part in argv)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env if env is not None else _verification_env(),
        )
    except OSError as exc:
        return CheckResult(
            name=name,
            command=cmd_str,
            passed=False,
            exit_code=127,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            stdout="",
            stderr=str(exc),
        )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return CheckResult(
            name=name,
            command=cmd_str,
            passed=False,
            exit_code=124,
            duration_ms=round((time.monotonic() - started) * 1000, 1),
            stdout="",
            stderr=f"Command timed out after {timeout:.0f}s",
        )

    exit_code = proc.returncode if proc.returncode is not None else 1
    stdout = _tail(stdout_b.decode("utf-8", errors="replace").strip())
    stderr = _tail(stderr_b.decode("utf-8", errors="replace").strip())
    return CheckResult(
        name=name,
        command=cmd_str,
        passed=(exit_code == 0),
        exit_code=exit_code,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
        stdout=stdout,
        stderr=stderr,
    )


def _has_external_python_deps(app_dir: Path) -> bool:
    req_file = app_dir / "requirements.txt"
    if req_file.is_file():
        lines = [
            ln.strip()
            for ln in req_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if any(not ln.lower().startswith("pytest") for ln in lines):
            return True

    pyproject = app_dir / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            return True
        project = data.get("project", {})
        if isinstance(project, dict):
            deps = project.get("dependencies", [])
            if isinstance(deps, list) and any(
                isinstance(d, str) and not d.lower().startswith("pytest") for d in deps
            ):
                return True
    return False


def _python_env(app_dir: Path) -> dict[str, str]:
    env = _verification_env()
    paths = [str(app_dir.resolve())]
    src_dir = (app_dir / "src").resolve()
    if src_dir.is_dir():
        paths.append(str(src_dir))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def _run_python_checks(
    app_dir: Path, python_exe: str, timeout: float, env: dict[str, str]
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    compile_res = await _run_subprocess(
        "python-compile",
        [python_exe, "-m", "compileall", "-f", "-q", "."],
        app_dir,
        timeout,
        env=env,
    )
    checks.append(compile_res)

    pytest_res = await _run_subprocess(
        "python-pytest",
        [python_exe, "-B", "-m", "pytest"],
        app_dir,
        timeout,
        env=env,
    )
    if pytest_res.exit_code == 5:
        note = "Pytest collected 0 tests (exit code 5)."
        pytest_res.stderr = f"{note}\n{pytest_res.stderr}".strip() if pytest_res.stderr else note
    checks.append(pytest_res)

    ruff_exe = _find_tool("ruff")
    if ruff_exe:
        ruff_res = await _run_subprocess(
            "python-ruff",
            [ruff_exe, "check", "--isolated", "--no-cache", "."],
            app_dir,
            timeout,
            env=env,
        )
        checks.append(ruff_res)

    return checks


async def _verify_python(app_dir: Path, timeout: float) -> list[CheckResult]:
    env = _python_env(app_dir)
    pytest_available = importlib.util.find_spec("pytest") is not None
    needs_isolated_install = _has_external_python_deps(app_dir) or not pytest_available

    if not needs_isolated_install:
        return await _run_python_checks(app_dir, sys.executable, timeout, env)

    with tempfile.TemporaryDirectory(prefix="atlas-verify-") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        uv_exe = _find_tool("uv")
        if uv_exe:
            venv_res = await _run_subprocess(
                "python-venv",
                [uv_exe, "venv", "--python", sys.executable, str(venv_dir)],
                app_dir,
                timeout,
            )
        else:
            venv_res = await _run_subprocess(
                "python-venv", [sys.executable, "-m", "venv", str(venv_dir)], app_dir, timeout
            )
        if not venv_res.passed:
            return [venv_res]

        venv_python = str(
            venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        install_targets: list[str] = ["pytest"]
        req_file = app_dir / "requirements.txt"
        if req_file.is_file():
            install_targets.extend(["-r", str(req_file)])
        if (app_dir / "pyproject.toml").is_file():
            install_targets.extend(["-e", str(app_dir)])

        if uv_exe:
            install_cmd = [uv_exe, "pip", "install", "--python", venv_python, *install_targets]
        else:
            install_cmd = [venv_python, "-m", "pip", "install", *install_targets]

        install_res = await _run_subprocess("python-install", install_cmd, app_dir, timeout)
        if not install_res.passed:
            return [install_res]

        checks = [install_res]
        checks.extend(await _run_python_checks(app_dir, venv_python, timeout, env))
        return checks


async def _verify_node(app_dir: Path, timeout: float) -> list[CheckResult]:
    checks: list[CheckResult] = []
    pkg_path = app_dir / "package.json"
    scripts: dict[str, object] = {}
    has_deps = False

    if pkg_path.is_file():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return [
                CheckResult(
                    name="node-package-json",
                    command="parse package.json",
                    passed=False,
                    exit_code=1,
                    stderr=f"Invalid package.json: {exc}",
                )
            ]
        raw_scripts = pkg.get("scripts", {})
        if isinstance(raw_scripts, dict):
            scripts = raw_scripts
        deps = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})
        has_deps = bool(deps) or bool(dev_deps)

    npm_exe = shutil.which("npm")
    node_exe = shutil.which("node")

    if has_deps and not (app_dir / "node_modules").exists():
        if not npm_exe:
            return [
                CheckResult(
                    name="node-install",
                    command="npm install",
                    passed=False,
                    exit_code=127,
                    stderr="npm executable not found on PATH",
                )
            ]
        install_res = await _run_subprocess(
            "node-install",
            [npm_exe, "install", "--no-audit", "--no-fund"],
            app_dir,
            timeout,
        )
        checks.append(install_res)
        if not install_res.passed:
            return checks

    test_script = scripts.get("test")
    has_real_test_script = (
        isinstance(test_script, str)
        and bool(test_script.strip())
        and "no test specified" not in test_script.lower()
    )
    node_test_files = [
        f
        for f in _iter_source_files(app_dir)
        if f.name.endswith((".test.js", ".test.mjs", ".test.cjs"))
    ]

    if has_real_test_script:
        if not npm_exe:
            checks.append(
                CheckResult(
                    name="node-test",
                    command="npm test",
                    passed=False,
                    exit_code=127,
                    stderr="npm executable not found on PATH",
                )
            )
        else:
            checks.append(await _run_subprocess("node-test", [npm_exe, "test"], app_dir, timeout))
    elif node_test_files:
        if not node_exe:
            checks.append(
                CheckResult(
                    name="node-test",
                    command="node --test",
                    passed=False,
                    exit_code=127,
                    stderr="node executable not found on PATH",
                )
            )
        else:
            rel_tests = [str(f.relative_to(app_dir.resolve())) for f in sorted(node_test_files)]
            checks.append(
                await _run_subprocess(
                    "node-test", [node_exe, "--test", *rel_tests], app_dir, timeout
                )
            )
    else:
        checks.append(
            CheckResult(
                name="node-tests-present",
                command="npm test",
                passed=False,
                exit_code=1,
                stderr="No test script in package.json and no *.test.{js,mjs,cjs} files found.",
            )
        )

    if npm_exe:
        for script_name in ("typecheck", "build", "lint"):
            if isinstance(scripts.get(script_name), str) and str(scripts[script_name]).strip():
                checks.append(
                    await _run_subprocess(
                        f"node-{script_name}",
                        [npm_exe, "run", script_name],
                        app_dir,
                        timeout,
                    )
                )

    return checks


def check_requirement_coverage(app_dir: Path, requirement_ids: list[str]) -> CheckResult:
    """Pass only if every requirement ID is mentioned in at least one test file.

    Plain text search, no model call. It catches generated suites that skip a requirement;
    it cannot prove a test is meaningful, so reviewing the PR still matters.
    """
    started = time.monotonic()
    root = app_dir.resolve() if app_dir.is_dir() else app_dir
    test_texts: dict[str, str] = {}
    for path in _iter_source_files(app_dir):
        rel = path.relative_to(root).as_posix()
        if _TEST_PATH_RE.search(rel):
            try:
                test_texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

    found: dict[str, list[str]] = {}
    missing: list[str] = []
    for req_id in requirement_ids:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(req_id)}(?![0-9])", flags=re.IGNORECASE
        )
        files = sorted(rel for rel, text in test_texts.items() if pattern.search(text))
        if files:
            found[req_id] = files
        else:
            missing.append(req_id)

    stdout = "\n".join(f"{rid}: {', '.join(files)}" for rid, files in found.items())
    stderr = ""
    if not test_texts:
        stderr = "No test files found."
    elif missing:
        stderr = (
            f"No test mentions requirement(s): {', '.join(missing)}. Add tests for them and put "
            f"the requirement ID in the test name or a comment next to it (e.g. `# {missing[0]}`)."
        )
    passed = bool(test_texts) and not missing
    return CheckResult(
        name="spec-coverage",
        command=f"every requirement has a test mentioning its ID ({', '.join(requirement_ids)})",
        passed=passed,
        exit_code=0 if passed else 1,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
        stdout=stdout,
        stderr=stderr,
    )


_PRODUCT_ATLAS_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+atlas(?:\.|\b)|import\s+atlas(?:\.|\b))", re.M
)
_README_NAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}
_DEP_FILES = {"pyproject.toml", "requirements.txt", "package.json"}


def check_product_contract(app_dir: Path) -> CheckResult:
    """Verify the built product is self-contained (README, deps, tests, no `import atlas`)."""
    started = time.monotonic()
    root = app_dir.resolve() if app_dir.is_dir() else app_dir
    top_names = {p.name.lower() for p in root.iterdir()} if root.is_dir() else set()

    has_readme = bool(top_names & _README_NAMES)
    has_deps = bool(top_names & _DEP_FILES)

    has_tests = False
    atlas_imports: list[str] = []
    for path in _iter_source_files(app_dir):
        rel = path.relative_to(root).as_posix()
        if _TEST_PATH_RE.search(rel):
            has_tests = True
        if path.suffix == ".py":
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if _PRODUCT_ATLAS_IMPORT_RE.search(text):
                atlas_imports.append(rel)

    issues: list[str] = []
    if not has_readme:
        issues.append("Missing README file at product root.")
    if not has_deps:
        issues.append(
            "Missing dependency file (pyproject.toml, requirements.txt, or package.json)."
        )
    if not has_tests:
        issues.append("No test files found inside the product directory.")
    if atlas_imports:
        issues.append(
            "Product must stand on its own without importing Atlas: "
            + ", ".join(sorted(atlas_imports))
        )

    passed = not issues
    stdout_parts = [
        f"readme={has_readme}",
        f"deps={has_deps}",
        f"tests={has_tests}",
        f"atlas_imports={len(atlas_imports)}",
    ]
    return CheckResult(
        name="product-contract",
        command="self-contained product check (README, deps, tests, no import atlas)",
        passed=passed,
        exit_code=0 if passed else 1,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
        stdout=", ".join(stdout_parts),
        stderr="\n".join(issues),
    )


async def verify_app(
    app_dir: Path,
    *,
    timeout_seconds: float = 180.0,
    extra_commands: list[str] | None = None,
    allow_custom_commands: bool = False,
) -> tuple[str, list[CheckResult]]:
    """Detect the project stack and run automated build, lint, and test checks."""
    if not app_dir.is_dir():
        return (
            "other",
            [
                CheckResult(
                    name="app-dir-exists",
                    command=f"test -d {app_dir}",
                    passed=False,
                    exit_code=1,
                    stderr=f"Application directory not found: {app_dir}",
                )
            ],
        )

    stack = detect_stack(app_dir)
    checks: list[CheckResult] = []

    if stack in ("python", "mixed"):
        checks.extend(await _verify_python(app_dir, timeout_seconds))
    if stack in ("node", "mixed"):
        checks.extend(await _verify_node(app_dir, timeout_seconds))

    if stack == "other":
        if allow_custom_commands and extra_commands:
            for idx, cmd in enumerate(extra_commands, 1):
                argv = shlex.split(cmd)
                if argv:
                    checks.append(
                        await _run_subprocess(f"custom-{idx}", argv, app_dir, timeout_seconds)
                    )
        else:
            checks.append(
                CheckResult(
                    name="stack-detection",
                    command="detect_stack",
                    passed=False,
                    exit_code=1,
                    stderr="No supported Python or Node.js project files found in app directory.",
                )
            )

    return stack, checks
