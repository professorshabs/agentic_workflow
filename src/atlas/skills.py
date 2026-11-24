"""Skill discovery, strict allowlist filtering, and role-prompt loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from atlas.schemas import StrictModel

logger = logging.getLogger("atlas.skills")


class SkillMetadata(StrictModel):
    """Metadata parsed from a skill directory's `SKILL.md`."""

    name: str
    description: str = ""
    path: str
    source_root: str
    version: str | None = None
    priority: str | None = None


def parse_skill_markdown(skill_md: Path) -> tuple[dict[str, Any], str]:
    """Split optional YAML frontmatter (`--- ... ---`) from the Markdown body of a skill."""
    raw = skill_md.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw.strip()

    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break

    if end_idx is None:
        return {}, raw.strip()

    frontmatter_text = "\n".join(lines[1:end_idx])
    parsed = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    body = "\n".join(lines[end_idx + 1 :]).strip()
    return parsed, body


def discover_skills(
    sources: list[str | Path],
    *,
    allowlist: list[str] | None = None,
) -> dict[str, SkillMetadata]:
    """Scan `sources` for `<skill>/SKILL.md` in priority order, filtering by `allowlist`."""
    allowed = set(allowlist) if allowlist is not None else None
    discovered: dict[str, SkillMetadata] = {}

    for raw_src in sources:
        src_dir = Path(raw_src).expanduser().resolve()
        if not src_dir.is_dir():
            continue
        for entry in sorted(src_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            frontmatter, _ = parse_skill_markdown(skill_md)
            skill_name = str(frontmatter.get("name") or entry.name).strip()
            if allowed is not None and skill_name not in allowed and entry.name not in allowed:
                continue
            canonical_key = entry.name if allowed is None or entry.name in allowed else skill_name
            if canonical_key in discovered:
                continue
            meta_dict = frontmatter.get("metadata")
            version: str | None = None
            priority: str | None = None
            if isinstance(meta_dict, dict):
                raw_ver = meta_dict.get("version") or meta_dict.get("mcpmarket-version")
                if raw_ver is not None:
                    version = str(raw_ver)
                if meta_dict.get("priority") is not None:
                    priority = str(meta_dict["priority"])
            discovered[canonical_key] = SkillMetadata(
                name=skill_name,
                description=str(frontmatter.get("description") or "").strip(),
                path=str(skill_md),
                source_root=str(src_dir),
                version=version,
                priority=priority,
            )

    return discovered


def resolve_role_skills(
    role: str,
    skills_cfg: dict[str, Any],
    *,
    extra_skills: list[str] | None = None,
) -> list[SkillMetadata]:
    """Return resolved `SkillMetadata` items assigned to `role` respecting the allowlist."""
    sources = list(skills_cfg.get("sources") or [])
    if not sources:
        return []

    raw_allowlist = skills_cfg.get("allowlist")
    allowlist = list(raw_allowlist) if raw_allowlist is not None else None

    role_names = list((skills_cfg.get("by_role") or {}).get(role) or [])
    if extra_skills:
        for item in extra_skills:
            if item not in role_names:
                role_names.append(item)

    if not role_names:
        return []

    available = discover_skills(sources, allowlist=allowlist)
    resolved: list[SkillMetadata] = []
    for name in role_names:
        if allowlist is not None and name not in allowlist:
            logger.warning("Ignoring non-allowlisted skill '%s' for role '%s'.", name, role)
            continue
        meta = available.get(name)
        if meta is None:
            logger.warning("Configured skill '%s' was not found in sources %s.", name, sources)
            continue
        resolved.append(meta)
    return resolved


def load_role_skill_prompt(
    role: str,
    skills_cfg: dict[str, Any],
    *,
    extra_skills: list[str] | None = None,
) -> str:
    """Load allowlisted skill Markdown bodies assigned to `role` for system prompt injection."""
    resolved = resolve_role_skills(role, skills_cfg, extra_skills=extra_skills)
    if not resolved:
        return ""

    sections = ["## Active Skills", ""]
    for meta in resolved:
        _, body = parse_skill_markdown(Path(meta.path))
        if not body:
            continue
        sections.append(f"### Skill: {meta.name}")
        if meta.description:
            sections.append(f"> {meta.description}")
        sections.append("")
        sections.append(body)
        sections.append("")
    return "\n".join(sections).strip()


def sync_workspace_skills(workspace_root: Path, skills_cfg: dict[str, Any]) -> list[Path]:
    """Expose allowlisted skills into `<workspace_root>/.agents/skills/` via symlinks."""
    sources = list(skills_cfg.get("sources") or [])
    if not sources:
        return []
    raw_allowlist = skills_cfg.get("allowlist")
    allowlist = list(raw_allowlist) if raw_allowlist is not None else None
    discovered = discover_skills(sources, allowlist=allowlist)
    if not discovered:
        return []

    target_root = (workspace_root / ".agents" / "skills").resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    created_links: list[Path] = []

    for key, meta in discovered.items():
        skill_dir = Path(meta.path).parent.resolve()
        if skill_dir == target_root or target_root in skill_dir.parents:
            continue
        link_path = target_root / key
        if link_path.exists() and not link_path.is_symlink():
            logger.debug("Skipping existing non-symlink workspace skill directory: %s", link_path)
            continue
        if link_path.is_symlink():
            if link_path.resolve() == skill_dir:
                created_links.append(link_path)
                continue
            link_path.unlink()
        link_path.symlink_to(skill_dir, target_is_directory=True)
        created_links.append(link_path)

    return created_links
