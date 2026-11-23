"""Check that repo skills in .agents/skills are valid and mirrored by .claude/skills symlinks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _frontmatter(text: str) -> dict[str, Any]:
    match = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.DOTALL)
    if match is None:
        return {}
    try:
        fields = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}
    return fields if isinstance(fields, dict) else {}


def _link_target(link: Path) -> str | None:
    """Return a link's target, also for Git checkouts with core.symlinks=false that store it as a plain file."""
    if link.is_symlink():
        return link.readlink().as_posix()
    if link.is_file():
        return link.read_text(encoding="utf-8").rstrip("\r\n")
    return None


def _eval_problems(path: Path, root: Path) -> list[str]:
    """Validate an optional evals.json: a non-empty list of cases, each with an id and a prompt."""
    if not path.exists():
        return []
    label = path.relative_to(root).as_posix()
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"{label}: invalid JSON ({error.msg})"]
    if not isinstance(cases, list) or not cases:
        return [f"{label}: must be a non-empty list of cases"]
    return [
        f"{label}: case {index} needs string 'id' and 'prompt'"
        for index, case in enumerate(cases)
        if not isinstance(case, dict) or not all(isinstance(case.get(key), str) for key in ("id", "prompt"))
    ]


def find_problems(root: Path) -> list[str]:
    skills_dir = root / ".agents" / "skills"
    links_dir = root / ".claude" / "skills"
    if not skills_dir.is_dir():
        return [f"{skills_dir.relative_to(root).as_posix()}: directory is missing"]
    problems = []
    names = sorted(path.name for path in skills_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    for name in names:
        skill_md = skills_dir / name / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"{skill_md.relative_to(root).as_posix()}: missing")
            continue
        fields = _frontmatter(skill_md.read_text(encoding="utf-8"))
        if fields.get("name") != name:
            problems.append(f"{skill_md.relative_to(root).as_posix()}: name must equal directory name {name!r}")
        if len(name) > MAX_NAME_CHARS or NAME_PATTERN.fullmatch(name) is None:
            problems.append(f"{name}: name must be lowercase-hyphen and at most {MAX_NAME_CHARS} characters")
        description = fields.get("description")
        if not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION_CHARS:
            size = len(description) if isinstance(description, str) else 0
            problems.append(f"{name}: description must be a 1-{MAX_DESCRIPTION_CHARS} character string, got {size}")
        problems.extend(_eval_problems(skills_dir / name / "evals" / "evals.json", root))

        link = links_dir / name
        expected = f"../../.agents/skills/{name}"
        if _link_target(link) != expected:
            problems.append(f"{link.relative_to(root).as_posix()}: must be a symlink to {expected}")

    if links_dir.is_dir():
        problems.extend(
            f"{entry.relative_to(root).as_posix()}: no matching skill in .agents/skills"
            for entry in sorted(links_dir.iterdir())
            if entry.name not in names and not entry.name.startswith(".")
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
