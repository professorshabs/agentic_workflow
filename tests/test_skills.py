from pathlib import Path

from atlas.skills import (
    detect_used_skills,
    discover_skills,
    format_role_skill_index,
    load_role_skill_prompt,
    resolve_role_skills,
    snapshot_skill_atimes,
    sync_workspace_skills,
)


def _seed_skill(
    root: Path, folder: str, *, name: str | None = None, body: str = "# Body\n"
) -> Path:
    d = root / folder
    d.mkdir(parents=True)
    skill_name = name or folder
    (d / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: Desc for {skill_name}\nmetadata:\n"
        f"  version: '2.0'\n---\n\n{body}",
        encoding="utf-8",
    )
    return d


def test_discover_skills_enforces_allowlist_and_role_prompts(tmp_path):
    skills_src = tmp_path / "skills"
    _seed_skill(skills_src, "developing-fullstack-apps", body="Use FastAPI + tests.")
    _seed_skill(skills_src, "writing-clean-code", body="Keep functions small.")
    _seed_skill(skills_src, "writing-ee-blogin-posts", body="Draft technical posts.")
    _seed_skill(skills_src, "using-dbt-for-analytics-engineering", body="dbt SQL.")
    _seed_skill(skills_src, "adding-dbt-unit-test", body="dbt test.")
    _seed_skill(skills_src, "structuring-project-folders", body="Lungisi-EE layout.")

    allowlist = ["developing-fullstack-apps", "writing-clean-code", "writing-ee-blogin-posts"]
    found = discover_skills([skills_src], allowlist=allowlist)
    assert sorted(found.keys()) == sorted(allowlist)
    assert "using-dbt-for-analytics-engineering" not in found
    assert "adding-dbt-unit-test" not in found
    assert "structuring-project-folders" not in found

    cfg = {
        "sources": [str(skills_src)],
        "allowlist": allowlist,
        "by_role": {
            "builder": ["developing-fullstack-apps", "writing-clean-code"],
            "fixer": ["writing-clean-code"],
        },
    }
    fixer_prompt = load_role_skill_prompt(
        "fixer", cfg, extra_skills=["structuring-project-folders"]
    )
    assert "Keep functions small." in fixer_prompt
    assert "Lungisi-EE layout." not in fixer_prompt
    assert "Use FastAPI + tests." not in fixer_prompt

    app_dir = tmp_path / "app" / "demo"
    app_dir.mkdir(parents=True)
    links = sync_workspace_skills(app_dir, cfg)
    assert sorted(p.name for p in links) == sorted(allowlist)
    for p in links:
        assert p.is_symlink()
        assert (p / "SKILL.md").is_file()


def test_detect_used_skills_from_summary_citations_and_atime(monkeypatch, tmp_path):
    skills_src = tmp_path / "skills"
    _seed_skill(skills_src, "developing-fullstack-apps", body="Use FastAPI + tests.")
    _seed_skill(skills_src, "writing-clean-code", body="Keep functions small.")
    cfg = {
        "sources": [str(skills_src)],
        "allowlist": ["developing-fullstack-apps", "writing-clean-code"],
        "by_role": {"builder": ["developing-fullstack-apps", "writing-clean-code"]},
    }
    resolved = resolve_role_skills("builder", cfg)

    assert (
        detect_used_skills(
            resolved,
            summary_text="Built app.\nSkills used: writing-clean-code, nonexistent-skill",
        )
        == ["writing-clean-code"]
    )
    assert detect_used_skills(resolved, summary_text="Skills used: none") == []
    assert (
        detect_used_skills(
            resolved,
            summary_text="Read .agents/skills/developing-fullstack-apps/SKILL.md before coding.",
        )
        == ["developing-fullstack-apps"]
    )

    before = snapshot_skill_atimes(resolved)
    clean_md = next(Path(m.path) for m in resolved if m.name == "writing-clean-code")
    real_stat = Path.stat

    class _FakeStat:
        def __init__(self, st_atime_ns: int) -> None:
            self.st_atime_ns = st_atime_ns

    monkeypatch.setattr(
        Path,
        "stat",
        lambda self, **kw: _FakeStat(before["writing-clean-code"] + 1_000_000)
        if self == clean_md
        else real_stat(self, **kw),
    )
    assert detect_used_skills(resolved, before_atimes=before) == ["writing-clean-code"]


def test_sync_workspace_skills_role_scoping_prunes_disallowed_symlinks(tmp_path):
    skills_src = tmp_path / "skills"
    _seed_skill(skills_src, "developing-fullstack-apps", body="Fullstack.")
    _seed_skill(skills_src, "writing-clean-code", body="Clean.")
    disallowed = _seed_skill(skills_src, "writing-ee-blogin-posts", body="Blog.")

    cfg = {
        "sources": [str(skills_src)],
        "allowlist": ["developing-fullstack-apps", "writing-clean-code", "writing-ee-blogin-posts"],
        "by_role": {"builder": ["developing-fullstack-apps", "writing-clean-code"]},
    }
    app_dir = tmp_path / "app" / "demo"
    target_skills = app_dir / ".agents" / "skills"
    target_skills.mkdir(parents=True)
    # Seed a stale disallowed symlink and a real directory; only the disallowed symlink is pruned.
    stale_link = target_skills / "writing-ee-blogin-posts"
    stale_link.symlink_to(disallowed, target_is_directory=True)
    real_dir = target_skills / "custom-local-skill"
    real_dir.mkdir()

    links = sync_workspace_skills(app_dir, cfg, role="builder")
    assert sorted(p.name for p in links) == ["developing-fullstack-apps", "writing-clean-code"]
    assert not stale_link.exists() and not stale_link.is_symlink()
    assert real_dir.is_dir()


def test_sync_workspace_skills_preserves_external_symlink_and_uses_folder_keys(tmp_path):
    skills_src = tmp_path / "skills"
    skill = _seed_skill(
        skills_src,
        "safe-folder",
        name="../../outside-target",
        body="Use the safe skill.",
    )
    cfg = {
        "sources": [str(skills_src)],
        "allowlist": ["safe-folder"],
        "by_role": {"builder": ["safe-folder"]},
    }

    external = tmp_path / "outside-skills"
    external.mkdir()
    sentinel = external / "user-data.txt"
    sentinel.write_text("keep", encoding="utf-8")
    app_dir = tmp_path / "app"
    (app_dir / ".agents").mkdir(parents=True)
    (app_dir / ".agents" / "skills").symlink_to(external, target_is_directory=True)

    assert sync_workspace_skills(app_dir, cfg, role="builder") == []
    assert sentinel.read_text(encoding="utf-8") == "keep"

    other_app = tmp_path / "app-with-valid-link"
    links = sync_workspace_skills(other_app, cfg, role="builder")
    assert [link.name for link in links] == ["safe-folder"]
    assert links[0].resolve() == skill.resolve()
    assert ".agents/skills/safe-folder/SKILL.md" in format_role_skill_index(
        "builder", cfg
    )


def test_sync_workspace_skills_does_not_claim_or_remove_existing_valid_link(tmp_path):
    skills_src = tmp_path / "skills"
    skill = _seed_skill(skills_src, "writing-clean-code")
    cfg = {
        "sources": [str(skills_src)],
        "allowlist": ["writing-clean-code"],
        "by_role": {"builder": ["writing-clean-code"]},
    }
    app_dir = tmp_path / "app"
    existing = app_dir / ".agents" / "skills" / "writing-clean-code"
    existing.parent.mkdir(parents=True)
    existing.symlink_to(skill.parent, target_is_directory=True)

    links = sync_workspace_skills(app_dir, cfg, role="builder")
    assert links == []
    assert existing.is_symlink()
    assert existing.resolve() == skill.parent.resolve()
