"""Skill discovery / injection edge cases (frontmatter parsing, precedence, workspace links)."""

from __future__ import annotations

import logging
from pathlib import Path

from atlas.skills import (
    discover_skills,
    format_role_skill_index,
    load_role_skill_prompt,
    parse_skill_markdown,
    resolve_role_skills,
    sync_workspace_skills,
)


def _skill(root: Path, folder: str, text: str) -> Path:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    return d


def test_frontmatter_variants(tmp_path):
    plain = _skill(tmp_path, "plain", "# Just markdown\n") / "SKILL.md"
    assert parse_skill_markdown(plain) == ({}, "# Just markdown")
    open_ended = _skill(tmp_path, "open", "---\nname: x\n# never closed\n") / "SKILL.md"
    assert parse_skill_markdown(open_ended)[0] == {}
    listy = _skill(tmp_path, "listy", "---\n- a\n- b\n---\nBody\n") / "SKILL.md"
    assert parse_skill_markdown(listy) == ({}, "Body")


def test_discovery_precedence_metadata_and_noise(tmp_path):
    first, second = tmp_path / "project", tmp_path / "user"
    _skill(first, "clean", "---\nname: clean\nmetadata:\n  priority: high\n  "
                           "mcpmarket-version: 3\n---\nProject copy\n")
    _skill(second, "clean", "---\nname: clean\n---\nUser copy\n")
    _skill(second, "other", "---\nname: other\n---\nOther\n")
    (second / "notes.txt").write_text("not a skill", encoding="utf-8")
    (second / "empty-dir").mkdir()

    found = discover_skills([first, second, tmp_path / "missing"])

    assert set(found) == {"clean", "other"}
    assert found["clean"].source_root == str(first.resolve())  # earlier source wins
    assert found["clean"].priority == "high" and found["clean"].version == "3"


def test_allowlist_matches_frontmatter_name_when_folder_differs(tmp_path):
    _skill(tmp_path, "folder-name", "---\nname: real-name\n---\nBody\n")
    found = discover_skills([tmp_path], allowlist=["real-name"])
    assert list(found) == ["real-name"]


def test_role_resolution_skips_unknown_and_non_allowlisted(tmp_path, caplog):
    _skill(tmp_path, "clean", "---\nname: clean\ndescription: Clean code\n---\nSmall functions\n")
    _skill(tmp_path, "blank", "---\nname: blank\n---\n")
    cfg = {"sources": [str(tmp_path)], "allowlist": ["clean", "blank", "ghost"],
           "by_role": {"builder": ["clean", "blank", "ghost"]}}

    assert resolve_role_skills("builder", {"by_role": {"builder": ["clean"]}}) == []
    assert resolve_role_skills("judge", cfg) == []
    with caplog.at_level(logging.WARNING, logger="atlas.skills"):
        names = [m.name for m in resolve_role_skills("builder", cfg, extra_skills=["secret"])]
    assert names == ["clean", "blank"]
    assert "non-allowlisted skill 'secret'" in caplog.text
    assert "'ghost' was not found" in caplog.text

    prompt = load_role_skill_prompt("builder", cfg)
    assert "### Skill: clean" in prompt and "> Clean code" in prompt
    assert "Skill: blank" not in prompt  # empty bodies are not injected
    assert load_role_skill_prompt("judge", cfg) == ""

    index = format_role_skill_index("builder", cfg)
    assert "`clean`: Clean code — read `.agents/skills/clean/SKILL.md`" in index
    assert format_role_skill_index("judge", cfg) == ""


def test_workspace_links_reuse_replace_and_respect_real_dirs(tmp_path):
    src = tmp_path / "skills"
    _skill(src, "a", "---\nname: a\n---\nA\n")
    _skill(src, "b", "---\nname: b\n---\nB\n")
    _skill(src, "c", "---\nname: c\n---\nC\n")
    ws = tmp_path / "app"
    links_dir = ws / ".agents" / "skills"
    links_dir.mkdir(parents=True)
    (links_dir / "a").symlink_to(src / "a", target_is_directory=True)  # already right
    (links_dir / "b").symlink_to(tmp_path, target_is_directory=True)  # stale target
    (links_dir / "c").mkdir()  # the product's own folder: never replaced

    links = sync_workspace_skills(ws, {"sources": [str(src)]})

    assert sorted(p.name for p in links) == ["a", "b"]
    assert (links_dir / "b").resolve() == (src / "b").resolve()
    assert not (links_dir / "c").is_symlink()
    assert sync_workspace_skills(ws, {"sources": []}) == []
    assert sync_workspace_skills(ws, {"sources": [str(tmp_path / "none")]}) == []
