import re
import subprocess
from pathlib import Path

import pytest
import yaml

from atlas.pipeline.util import prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def test_prompts_and_template_files_exist_and_load():
    for name in ("parse", "research", "judge", "build", "fix"):
        text = prompt(name)
        assert len(text.strip()) > 40

    assert (REPO_ROOT / "specs" / "TEMPLATE.md").is_file()
    assert (REPO_ROOT / "specs" / "example-app.md").is_file()
    assert (REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file()
    assert (REPO_ROOT / "ROADMAP.md").is_file()
    assert (REPO_ROOT / "app" / ".gitkeep").is_file()


def test_github_workflows_yaml_and_shell_syntax():
    workflow_files = sorted(WORKFLOW_DIR.glob("*.yml"))
    assert {p.name for p in workflow_files} == {"ci.yml", "spec-to-app.yml"}

    for wf_path in workflow_files:
        data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        for job_name, job in data.get("jobs", {}).items():
            for idx, step in enumerate(job.get("steps", [])):
                run_script = step.get("run")
                if not run_script:
                    continue
                sanitized = re.sub(r"\$\{\{[^}]+\}\}", "placeholder", run_script)
                res = subprocess.run(
                    ["bash", "-n"],
                    input=sanitized,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert res.returncode == 0, (
                    f"{wf_path.name} job={job_name} step={idx} bash -n failed: {res.stderr}"
                )


def _extract_spec_validator_code() -> str:
    wf = yaml.safe_load((WORKFLOW_DIR / "spec-to-app.yml").read_text(encoding="utf-8"))
    run_text = wf["jobs"]["build-and-verify"]["steps"][4]["run"]
    match = re.search(r"python - \"\$SPEC_PATH\" <<'PY'\n(.*?)\nPY", run_text, flags=re.DOTALL)
    assert match is not None
    return match.group(1)


def test_spec_to_app_git_detection_and_path_validation(tmp_path):
    repo = tmp_path / "repo"
    (repo / "specs").mkdir(parents=True)
    (repo / "specs" / "TEMPLATE.md").write_text("# Template\n", encoding="utf-8")
    (repo / "specs" / "example-app.md").write_text("# Example\n", encoding="utf-8")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "Tester")
    git("config", "user.email", "tester@example.com")
    git("add", ".")
    git("commit", "-m", "initial")

    (repo / "specs" / "001-demo.md").write_text("# Demo\nBuild demo.\n", encoding="utf-8")
    git("add", "specs/001-demo.md")
    git("commit", "-m", "add spec 001")

    detected = git(
        "diff",
        "--name-only",
        "--diff-filter=AM",
        "HEAD~1",
        "HEAD",
        "--",
        "specs/*.md",
        "specs/**/*.md",
        ":!specs/TEMPLATE.md",
    ).splitlines()[0]
    assert detected == "specs/001-demo.md"

    validator_code = _extract_spec_validator_code()
    gh_out = tmp_path / "github_output.txt"
    ok_res = subprocess.run(
        ["python3", "-c", validator_code, detected],
        cwd=repo,
        env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(gh_out)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_res.returncode == 0, ok_res.stderr
    assert "path=specs/001-demo.md" in gh_out.read_text(encoding="utf-8")

    for bad_spec in ("../README.md", "specs/missing.md", "specs/bad;cmd.md", "specs/TEMPLATE.md"):
        if bad_spec == "specs/TEMPLATE.md":
            # TEMPLATE.md exists; verify the pathspec exclusion keeps it ignored on git push
            (repo / "specs" / "TEMPLATE.md").write_text("# Updated Template\n", encoding="utf-8")
            git("add", "specs/TEMPLATE.md")
            git("commit", "-m", "edit template")
            changed = git(
                "diff",
                "--name-only",
                "--diff-filter=AM",
                "HEAD~1",
                "HEAD",
                "--",
                "specs/*.md",
                "specs/**/*.md",
                ":!specs/TEMPLATE.md",
            )
            assert changed == ""
            continue
        bad_res = subprocess.run(
            ["python3", "-c", validator_code, bad_spec],
            cwd=repo,
            env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(gh_out)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert bad_res.returncode != 0


@pytest.mark.parametrize("with_code,expected", [(False, "SKIP"), (True, "RUN")])
def test_ci_verify_app_guard_behavior(tmp_path, with_code, expected):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / ".gitkeep").write_text("", encoding="utf-8")
    if with_code:
        (app_dir / "main.py").write_text("print('hi')\n", encoding="utf-8")

    cmd = (
        "if [ -d app ] && find app -mindepth 1 ! -name '.gitkeep' | grep -q .; "
        "then echo RUN; else echo SKIP; fi"
    )
    res = subprocess.run(
        ["bash", "-c", cmd],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == expected
