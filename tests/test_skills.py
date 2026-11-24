from pathlib import Path

from atlas.skills import discover_skills, load_role_skill_prompt, sync_workspace_skills


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
