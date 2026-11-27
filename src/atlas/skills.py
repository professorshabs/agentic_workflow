"""Skill discovery, strict allowlist filtering, and role-prompt loading."""

from __future__ import annotations

import logging
import re
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


def format_role_skill_index(
    role: str,
    skills_cfg: dict[str, Any],
    *,
    extra_skills: list[str] | None = None,
    workspace_root: Path | None = None,
) -> str:
    """Return a compact progressive-disclosure skill index pointing to `.agents/skills/`."""
    resolved = resolve_role_skills(role, skills_cfg, extra_skills=extra_skills)
    if not resolved:
        return ""

    if workspace_root is not None:
        skill_root = (workspace_root / ".agents" / "skills").resolve()
        lines = [
            f"## Active Skills (read on demand from `{skill_root}`)",
            "",
        ]
        for meta in resolved:
            skill_key = Path(meta.path).parent.name
            desc = f": {meta.description}" if meta.description else ""
            lines.append(f"- `{meta.name}`{desc} — read `{skill_root / skill_key / 'SKILL.md'}`")
        return "\n".join(lines)

    lines = [
        "## Active Skills (read on demand from `.agents/skills/`)",
        "",
    ]
    for meta in resolved:
        skill_key = Path(meta.path).parent.name
        desc = f": {meta.description}" if meta.description else ""
        lines.append(f"- `{meta.name}`{desc} — read `.agents/skills/{skill_key}/SKILL.md`")
    return "\n".join(lines)


def sync_workspace_skills(
    workspace_root: Path,
    skills_cfg: dict[str, Any],
    *,
    role: str | None = None,
) -> list[Path]:
    """Expose configured skills into `<workspace_root>/.agents/skills/` via symlinks."""
    sources = list(skills_cfg.get("sources") or [])
    if not sources:
        return []
    if role is not None:
        skill_metadata = resolve_role_skills(role, skills_cfg)
    else:
        raw_allowlist = skills_cfg.get("allowlist")
        allowlist = list(raw_allowlist) if raw_allowlist is not None else None
        skill_metadata = list(discover_skills(sources, allowlist=allowlist).values())
    # Directory names are the link keys used by the skill index; frontmatter is untrusted.
    discovered = {Path(meta.path).parent.name: meta for meta in skill_metadata}

    root = workspace_root.resolve()
    agents_dir = root / ".agents"
    target_root = agents_dir / "skills"
    if agents_dir.is_symlink() or (agents_dir.exists() and not agents_dir.is_dir()):
        logger.warning("Skipping skill sync because workspace `.agents` is not a real directory.")
        return []
    if target_root.is_symlink() or (target_root.exists() and not target_root.is_dir()):
        logger.warning(
            "Skipping skill sync because workspace `.agents/skills` is not a real directory."
        )
        return []

    if role is not None and target_root.is_dir():
        raw_allowlist = skills_cfg.get("allowlist")
        allowlist = list(raw_allowlist) if raw_allowlist is not None else None
        configured = discover_skills(sources, allowlist=allowlist)
        configured_by_key = {
            Path(meta.path).parent.name: Path(meta.path).parent.resolve()
            for meta in configured.values()
        }
        for existing in list(target_root.iterdir()):
            expected_target = configured_by_key.get(existing.name)
            if (
                existing.is_symlink()
                and existing.name not in discovered
                and expected_target is not None
            ):
                try:
                    if existing.resolve() == expected_target:
                        existing.unlink()
                except OSError:
                    continue

    if not discovered:
        return []
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
            try:
                if link_path.resolve() == skill_dir:
                    if role is None:
                        # Preserve the legacy non-role API, whose caller owns cleanup.
                        created_links.append(link_path)
                    continue
            except OSError:
                pass
            if role is None:
                link_path.unlink()
            else:
                logger.warning(
                    "Skipping existing workspace skill symlink with a different target: %s",
                    link_path,
                )
                continue
        link_path.symlink_to(skill_dir, target_is_directory=True)
        created_links.append(link_path)

    return created_links


_SKILLS_USED_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?skills?\s+used(?:\*\*)?\s*:\s*(.+)",
    re.IGNORECASE,
)


def snapshot_skill_atimes(skills: list[SkillMetadata]) -> dict[str, int]:
    """Capture `st_atime_ns` per skill `SKILL.md` to detect reads when the OS records atimes."""
    atimes: dict[str, int] = {}
    for meta in skills:
        try:
            atimes[meta.name] = Path(meta.path).stat().st_atime_ns
        except OSError:
            atimes[meta.name] = 0
    return atimes


def detect_used_skills(
    skills: list[SkillMetadata],
    *,
    summary_text: str = "",
    before_atimes: dict[str, int] | None = None,
) -> list[str]:
    """Determine which allowlisted skills were read or reported used by a workspace builder."""
    if not skills:
        return []

    lookup: dict[str, str] = {}
    for meta in skills:
        lookup[meta.name.lower()] = meta.name
        lookup[Path(meta.path).parent.name.lower()] = meta.name

    used: set[str] = set()

    if before_atimes:
        for meta in skills:
            prev = before_atimes.get(meta.name, 0)
            if prev <= 0:
                continue
            try:
                if Path(meta.path).stat().st_atime_ns > prev:
                    used.add(meta.name)
            except OSError:
                continue

    if summary_text:
        for match in _SKILLS_USED_LINE_RE.finditer(summary_text):
            raw_line = match.group(1).strip()
            # Split on a spaced "and" only: skill names like `build-and-test` contain one.
            for part in re.split(r"[,;|]|\s+and\s+", raw_line, flags=re.IGNORECASE):
                token = part.strip().strip("`*'\".()[]{}").lower()
                if not token or token in {"none", "n/a", "na", "no skills", "no skill"}:
                    continue
                canonical = lookup.get(token)
                if canonical is not None:
                    used.add(canonical)

        lowered = summary_text.lower()
        for meta in skills:
            key = Path(meta.path).parent.name.lower()
            if f"{key}/skill.md" in lowered:
                used.add(meta.name)

    return [meta.name for meta in skills if meta.name in used]
