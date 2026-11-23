"""Check that repo skills in .agents/skills are valid and mirrored by .claude/skills symlinks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if match is None:
        return {}
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _link_target(link: Path) -> str | None:
    """Return a link's target, also for Git checkouts with core.symlinks=false that store it as a plain file."""
    if link.is_symlink():
        return link.readlink().as_posix()
    if link.is_file():
        return link.read_text(encoding="utf-8")
    return None


def find_problems(root: Path) -> list[str]:
    skills_dir = root / ".agents" / "skills"
    links_dir = root / ".claude" / "skills"
    problems = []
    names = sorted(path.name for path in skills_dir.iterdir() if path.is_dir())
    for name in names:
        skill_md = skills_dir / name / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"{skill_md.relative_to(root)}: missing")
            continue
        fields = _frontmatter(skill_md.read_text(encoding="utf-8"))
        if fields.get("name") != name:
            problems.append(f"{skill_md.relative_to(root)}: name must equal directory name {name!r}")
        if len(name) > MAX_NAME_CHARS or NAME_PATTERN.fullmatch(name) is None:
            problems.append(f"{name}: name must be lowercase-hyphen and at most {MAX_NAME_CHARS} characters")
        description = fields.get("description", "")
        if not description or len(description) > MAX_DESCRIPTION_CHARS:
            problems.append(f"{name}: description must be 1-{MAX_DESCRIPTION_CHARS} characters, got {len(description)}")

        link = links_dir / name
        expected = f"../../.agents/skills/{name}"
        if _link_target(link) != expected:
            problems.append(f"{link.relative_to(root)}: must be a symlink to {expected}")

    if links_dir.is_dir():
        problems.extend(
            f"{entry.relative_to(root)}: no matching skill in .agents/skills"
            for entry in sorted(links_dir.iterdir())
            if entry.name not in names
        )
    return problems


def main() -> int:
    problems = find_problems(REPO_ROOT)
    if problems:
        sys.stderr.write("Agent skill problems:\n" + "\n".join(problems) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
