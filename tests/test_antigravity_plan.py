"""Tests for Antigravity pull-request path selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.antigravity_plan import classify_path, main, read_github_files, select_antigravity


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("albucore/arithmetic.py", {"runtime"}),
        ("tests/test_add.py", {"tests"}),
        (".github/workflows/ci.yml", {"workflows"}),
        ("LICENSE", {"legal"}),
        ("docs/maintaining/license-provenance.md", {"docs", "legal"}),
        ("tools/antigravity_plan.py", {"ci_tooling", "self_ci"}),
        ("tools/ci_matrix.py", {"ci_tooling", "self_ci"}),
        ("docs/image-conventions.md", {"docs"}),
        ("benchmarks/benchmark_router_synthetic.py", {"benchmarks"}),
        ("uv.lock", {"dependencies"}),
        ("new-root/file.dat", {"unknown"}),
    ],
)
def test_classify_path(path: str, expected: set[str]) -> None:
    assert set(classify_path(path)) == expected


@pytest.mark.parametrize(
    "path",
    [
        "albucore/arithmetic.py",
        "tests/test_add.py",
        ".github/workflows/ci.yml",
        "LICENSE",
        "tools/antigravity_review.py",
        "tools/ci_matrix.py",
        "new-root/file.dat",
    ],
)
def test_review_relevant_path_selects_antigravity(path: str) -> None:
    assert select_antigravity([path])


@pytest.mark.parametrize(
    "path",
    [
        "docs/image-conventions.md",
        "benchmarks/benchmark_router_synthetic.py",
        "requirements-dev.txt",
        "uv.lock",
        ".gitignore",
    ],
)
def test_low_risk_path_skips_antigravity(path: str) -> None:
    assert not select_antigravity([path])


def test_mixed_paths_select_antigravity() -> None:
    assert select_antigravity(["docs/image-conventions.md", "albucore/utils.py"])


def test_empty_or_invalid_paths_fail_closed() -> None:
    assert select_antigravity([])
    assert select_antigravity(["docs/line\nbreak.md"])
    assert select_antigravity(["../outside.py"])


def test_github_file_reader_preserves_untrusted_filename_boundaries(tmp_path: Path) -> None:
    files_json = tmp_path / "files.json"
    files_json.write_text(
        '[[{"filename":"docs/normal.md"}], [{"filename":"docs/line\\nbreak.md"}]]',
        encoding="utf-8",
    )

    paths = read_github_files(files_json)

    assert paths == ["docs/normal.md", "docs/line\nbreak.md"]
    assert select_antigravity(paths)


def test_github_file_reader_selects_renamed_file_source_and_destination(tmp_path: Path) -> None:
    files_json = tmp_path / "files.json"
    files_json.write_text(
        '[[{"filename":"docs/old.md","previous_filename":"albucore/old.py","status":"renamed"}]]',
        encoding="utf-8",
    )

    paths = read_github_files(files_json)

    assert paths == ["docs/old.md", "albucore/old.py"]
    assert select_antigravity(paths)


def test_main_writes_github_output(tmp_path: Path) -> None:
    files_json = tmp_path / "files.json"
    github_output = tmp_path / "github-output.txt"
    files_json.write_text('[[{"filename":"albucore/utils.py"}]]', encoding="utf-8")

    assert (
        main(
            [
                "--github-files-json",
                str(files_json),
                "--github-output",
                str(github_output),
            ],
        )
        == 0
    )
    assert github_output.read_text(encoding="utf-8") == "antigravity=true\n"
