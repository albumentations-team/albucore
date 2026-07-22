from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER = REPO_ROOT / "tools" / "classify_ci_changes.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_metadata(
    repo: Path,
    version: str,
    numpy_requirement: str = "numpy>=1.24.4",
    numpy_lock_version: str = "2.2.6",
) -> None:
    (repo / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "albucore"',
                f'version = "{version}"',
                f'dependencies = ["{numpy_requirement}"]',
                "",
            ),
        ),
    )
    (repo / "uv.lock").write_text(
        "\n".join(
            (
                "version = 1",
                "",
                "[[package]]",
                'name = "albucore"',
                f'version = "{version}"',
                'source = { editable = "." }',
                'dependencies = [{ name = "numpy" }]',
                "",
                "[[package]]",
                'name = "numpy"',
                f'version = "{numpy_lock_version}"',
                "",
            ),
        ),
    )


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _version_change(repo: Path, old_version: str = "0.2.4", new_version: str = "0.2.5") -> tuple[str, str]:
    _git(repo, "init")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "CI Test")
    _write_metadata(repo, old_version)
    base = _commit(repo, "base")
    _write_metadata(repo, new_version)
    head = _commit(repo, "bump version")
    return base, head


def _classify(repo: Path, base: str, head: str) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, str(CLASSIFIER), "--repo", str(repo), base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", maxsplit=1) for line in result.stdout.splitlines())


def test_version_only_change_skips_tests(tmp_path: Path) -> None:
    base, head = _version_change(tmp_path)

    assert _classify(tmp_path, base, head) == {"run_tests": "false"}


def test_dependency_constraint_change_runs_tests(tmp_path: Path) -> None:
    base, _ = _version_change(tmp_path)
    _write_metadata(tmp_path, "0.2.5", numpy_requirement="numpy>=2.0")
    head = _commit(tmp_path, "change dependency constraint")

    assert _classify(tmp_path, base, head) == {"run_tests": "true"}


def test_locked_dependency_change_runs_tests(tmp_path: Path) -> None:
    base, _ = _version_change(tmp_path)
    _write_metadata(tmp_path, "0.2.5", numpy_lock_version="2.3.0")
    head = _commit(tmp_path, "change locked dependency")

    assert _classify(tmp_path, base, head) == {"run_tests": "true"}


def test_code_change_runs_tests(tmp_path: Path) -> None:
    base, _ = _version_change(tmp_path)
    (tmp_path / "albucore.py").write_text("VALUE = 1\n")
    head = _commit(tmp_path, "change code")

    assert _classify(tmp_path, base, head) == {"run_tests": "true"}
