"""Find which spec files a git push added or changed (used by the Spec to App workflow)."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

IGNORED_SPECS = {"specs/TEMPLATE.md"}


def _is_zero_sha(sha: str) -> bool:
    return not sha or set(sha) == {"0"}


def changed_specs(repo: Path, before: str, after: str) -> list[str]:
    """Markdown specs under specs/ added or modified between two commits of one push.

    Returns nothing when `before` is all zeros (a new branch, or the first commit of a
    repository created from the template), so creating a repo never starts a paid build.
    Compares the whole push, not only its last commit.
    """
    if _is_zero_sha(before):
        return []
    proc = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", before, after, "--", "specs/"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git diff {before}..{after} failed: {proc.stderr.strip()}")
    result = []
    for line in proc.stdout.splitlines():
        path = PurePosixPath(line.strip())
        if path.suffix == ".md" and path.parts[:1] == ("specs",) and str(path) not in IGNORED_SPECS:
            result.append(str(path))
    return sorted(set(result))
