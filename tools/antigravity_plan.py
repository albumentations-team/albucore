"""Select Antigravity review for pull requests with review-relevant paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable

ANTIGRAVITY_DOMAINS: Final = frozenset({"runtime", "tests", "workflows", "legal", "self_ci", "unknown"})
LEGAL_PATHS: Final = frozenset(
    {
        "CLA.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "LICENSE_HISTORY.md",
        "LICENSING.md",
        "MANIFEST.in",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/maintaining/license-provenance.md",
        "pyproject.toml",
        "tests/test_legal_integrity.py",
        "tools/verify_legal_integrity.py",
    },
)
DEPENDENCY_PATHS: Final = frozenset({"pyproject.toml", "requirements-dev.txt", "uv.lock"})
QUALITY_CONFIG_PATHS: Final = frozenset({".pre-commit-config.yaml", "pyproject.toml"})
SELF_CI_PATHS: Final = frozenset(
    {
        ".github/workflows/antigravity-pr-checks.yml",
        "tools/antigravity_plan.py",
        "tools/antigravity_review.py",
    },
)
KNOWN_REPOSITORY_FILES: Final = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        ".python-version",
        "AGENTS.md",
        "codecov.yaml",
    },
)


def _is_documentation(path: str) -> bool:
    return path.endswith(".md") or path.startswith(("docs/", ".codex/")) or path in {"AGENTS.md", "CONTRIBUTING.md"}


def _is_legal(path: str) -> bool:
    return path in LEGAL_PATHS or path.startswith(("legal/", "THIRD_PARTY_LICENSES/"))


DOMAIN_RULES: Final[tuple[tuple[str, Callable[[str], bool]], ...]] = (
    ("docs", _is_documentation),
    ("runtime", lambda path: path.startswith("albucore/")),
    ("tests", lambda path: path.startswith("tests/")),
    ("benchmarks", lambda path: path.startswith(("benchmarks/", "benchmark/"))),
    ("legal", _is_legal),
    ("dependencies", DEPENDENCY_PATHS.__contains__),
    ("quality_config", QUALITY_CONFIG_PATHS.__contains__),
    ("workflows", lambda path: path.startswith(".github/")),
    ("self_ci", SELF_CI_PATHS.__contains__),
    ("ci_tooling", lambda path: path.startswith("tools/") and path.endswith(".py")),
    ("repository_config", KNOWN_REPOSITORY_FILES.__contains__),
)


def _normalise_path(raw_path: str) -> str:
    path = raw_path.removeprefix("./").replace("\\", "/")
    candidate = PurePosixPath(path)
    has_control_character = any(ord(character) < 32 or ord(character) == 127 for character in path)
    if not path or has_control_character or candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return candidate.as_posix()


def classify_path(raw_path: str) -> frozenset[str]:
    """Classify one changed path into additive Antigravity risk domains."""
    path = _normalise_path(raw_path)
    if not path:
        return frozenset({"unknown"})

    domains = {domain for domain, matches in DOMAIN_RULES if matches(path)}
    return frozenset(domains or {"unknown"})


def select_antigravity(paths: list[str] | tuple[str, ...]) -> bool:
    """Return whether changed paths should receive an Antigravity review."""
    domains = set().union(*(classify_path(path) for path in paths)) if paths else {"unknown"}
    return bool(domains & ANTIGRAVITY_DOMAINS)


def read_github_files(path: Path) -> list[str]:
    """Read filenames from the nested JSON emitted by ``gh api --slurp``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    paths: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            filename = value.get("filename")
            if isinstance(filename, str):
                paths.append(filename)
            else:
                for item in value.values():
                    visit(item)

    visit(data)
    return paths


def write_github_output(path: Path, *, selected: bool) -> None:
    """Append the Antigravity decision to a GitHub Actions output file."""
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(f"antigravity={str(selected).lower()}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Antigravity path-selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-files-json", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Select the review and write the GitHub Actions output."""
    args = parse_args(argv)
    selected = select_antigravity(read_github_files(args.github_files_json))
    write_github_output(args.github_output, selected=selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
