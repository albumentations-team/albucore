"""Tests for Albucore license, CLA, and artifact integrity."""

from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path

from tools import verify_legal_integrity as legal_integrity
from tools.verify_legal_integrity import (
    CLA_ARCHIVE_PATH,
    REPO_ROOT,
    REQUIRED_SOURCE_FILES,
    collect_artifact_errors,
    collect_source_errors,
    main,
)


def _distribution_metadata(
    *,
    license_expression: str = "MIT",
    license_files: tuple[str, ...] = ("LICENSE",),
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        "Name: albucore",
        "Version: 0.2.4",
        f"License-Expression: {license_expression}",
        *(f"License-File: {relative_path}" for relative_path in license_files),
        "",
    ]
    return "\n".join(lines).encode()


def _write_wheel(
    path: Path,
    *,
    license_bytes: bytes | None = None,
    metadata: bytes | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "albucore-0.2.4.dist-info/licenses/LICENSE",
            license_bytes if license_bytes is not None else (REPO_ROOT / "LICENSE").read_bytes(),
        )
        archive.writestr("albucore-0.2.4.dist-info/METADATA", metadata or _distribution_metadata())
        for relative_path, content in (extra_files or {}).items():
            archive.writestr(relative_path, content)


def _write_sdist(
    path: Path,
    *,
    license_bytes: bytes | None = None,
    metadata: bytes | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    root = "albucore-0.2.4"
    files = {
        "LICENSE": license_bytes if license_bytes is not None else (REPO_ROOT / "LICENSE").read_bytes(),
        "PKG-INFO": metadata or _distribution_metadata(),
        **(extra_files or {}),
    }
    with tarfile.open(path, "w:gz") as archive:
        for relative_path, content in files.items():
            info = tarfile.TarInfo(f"{root}/{relative_path}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _copy_source_inputs(destination: Path) -> None:
    for relative_path in REQUIRED_SOURCE_FILES:
        source = REPO_ROOT / relative_path
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def test_source_legal_integrity() -> None:
    assert collect_source_errors() == []


def test_source_legal_integrity_with_windows_default_encoding(monkeypatch) -> None:
    original_read_text = Path.read_text

    def read_text_with_windows_default(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        return original_read_text(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text_with_windows_default)

    assert collect_source_errors() == []


def test_pre_commit_legal_integrity_hook_always_runs() -> None:
    config_lines = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8").splitlines()
    hook_start = config_lines.index("      - id: check-legal-integrity")
    hook_lines: list[str] = []
    for line in config_lines[hook_start + 1 :]:
        if not line.startswith("        "):
            break
        hook_lines.append(line)

    assert "        always_run: true" in hook_lines


def test_cli_reports_missing_required_file_without_artifacts(tmp_path, monkeypatch, capsys) -> None:
    missing_file = "LICENSE"
    for relative_path in REQUIRED_SOURCE_FILES:
        if relative_path == missing_file:
            continue
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"placeholder")

    monkeypatch.setattr(legal_integrity, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_legal_integrity.py"])

    assert main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"ERROR: missing required legal-integrity file: {missing_file}\n"


def test_source_rejects_root_and_archive_cla_mismatch(tmp_path) -> None:
    _copy_source_inputs(tmp_path)
    (tmp_path / CLA_ARCHIVE_PATH).write_text("changed archived agreement\n")

    errors = collect_source_errors(tmp_path)

    assert "root CLA.md is not byte-identical to the archived Version 1.0 text" in errors


def test_source_rejects_private_signer_export_anywhere_in_repository(tmp_path) -> None:
    _copy_source_inputs(tmp_path)
    (tmp_path / "signatures.json").write_text('{"signatures": []}\n')

    errors = collect_source_errors(tmp_path)

    assert "private CLA acceptance material must not be committed: signatures.json" in errors


def test_source_rejects_private_acceptance_directories_with_underscores(tmp_path) -> None:
    _copy_source_inputs(tmp_path)
    entity_record = tmp_path / "entity_acceptance" / "acme-corp.eml"
    entity_record.parent.mkdir()
    entity_record.write_text("synthetic entity acceptance\n")
    assistant_export = tmp_path / "cla_assistant" / "export.json"
    assistant_export.parent.mkdir()
    assistant_export.write_text('{"signatures": []}\n')

    errors = collect_source_errors(tmp_path)

    assert "private CLA acceptance material must not be committed: entity_acceptance/acme-corp.eml" in errors
    assert "private CLA acceptance material must not be committed: cla_assistant/export.json" in errors


def test_wheel_and_sdist_accept_mit_metadata_and_exact_license(tmp_path) -> None:
    wheel = tmp_path / "albucore-0.2.4-py3-none-any.whl"
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    assert collect_artifact_errors(wheel, (REPO_ROOT / "LICENSE").read_bytes()) == []
    assert collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes()) == []


def test_artifact_rejects_wrong_license_expression(tmp_path) -> None:
    wheel = tmp_path / "albucore-0.2.4-py3-none-any.whl"
    _write_wheel(wheel, metadata=_distribution_metadata(license_expression="BSD-3-Clause"))

    errors = collect_artifact_errors(wheel, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [f"{wheel.name}: wheel METADATA License-Expression must be 'MIT', found 'BSD-3-Clause'"]


def test_artifact_rejects_changed_license_bytes(tmp_path) -> None:
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_sdist(sdist, license_bytes=b"changed license")

    errors = collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [f"{sdist.name}: LICENSE is not byte-identical to the source file"]


def test_artifact_rejects_cla_and_archive(tmp_path) -> None:
    wheel = tmp_path / "albucore-0.2.4-py3-none-any.whl"
    _write_wheel(
        wheel,
        extra_files={
            "CLA.md": b"inbound agreement",
            "legal/cla/archive/CLA-v1.0-2026-07-14.md": b"inbound agreement",
        },
    )

    errors = collect_artifact_errors(wheel, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [
        f"{wheel.name}: inbound CLA material leaked into artifact as CLA.md",
        f"{wheel.name}: inbound CLA material leaked into artifact as legal/cla/archive/CLA-v1.0-2026-07-14.md",
    ]


def test_artifact_rejects_internal_material(tmp_path) -> None:
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_sdist(sdist, extra_files={"_internal/private-plan.md": b"private"})

    errors = collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [
        f"{sdist.name}: private _internal material leaked into artifact as albucore-0.2.4/_internal/private-plan.md",
    ]


def test_artifact_rejects_private_signer_export(tmp_path) -> None:
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_sdist(sdist, extra_files={"signatures.json": b'{"signatures": []}'})

    errors = collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [
        f"{sdist.name}: private CLA acceptance material leaked into artifact as albucore-0.2.4/signatures.json",
    ]


def test_artifact_rejects_private_acceptance_directories_with_underscores(tmp_path) -> None:
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_sdist(
        sdist,
        extra_files={
            "entity_acceptance/acme-corp.eml": b"synthetic entity acceptance",
            "cla_assistant/export.json": b'{"signatures": []}',
        },
    )

    errors = collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [
        f"{sdist.name}: private CLA acceptance material leaked into artifact as "
        "albucore-0.2.4/entity_acceptance/acme-corp.eml",
        f"{sdist.name}: private CLA acceptance material leaked into artifact as "
        "albucore-0.2.4/cla_assistant/export.json",
    ]


def test_sdist_rejects_nested_distribution_artifact(tmp_path) -> None:
    sdist = tmp_path / "albucore-0.2.4.tar.gz"
    _write_sdist(sdist, extra_files={"unexpected/albucore-0.2.3-py3-none-any.whl": b"old wheel"})

    errors = collect_artifact_errors(sdist, (REPO_ROOT / "LICENSE").read_bytes())

    assert errors == [
        f"{sdist.name}: nested distribution artifact leaked into sdist as "
        "albucore-0.2.4/unexpected/albucore-0.2.3-py3-none-any.whl",
    ]
