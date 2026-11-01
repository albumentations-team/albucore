# ruff: noqa: S101
"""Tests for the dependency license registry verifier."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools.verify_dependency_licenses import check_license_evidence, check_requirements, enrich_sbom, load_registry

if TYPE_CHECKING:
    from pathlib import Path


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_sources": {"test": "test evidence"},
                "components": [
                    {
                        "name": "example-package",
                        "reviewed_versions": ["1.0"],
                        "license_expression": "MIT",
                        "evidence_source": "test",
                        "decision": "accepted",
                        "notice": "none",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    return path


def _version_specific_registry(tmp_path: Path) -> Path:
    path = _registry(tmp_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["components"][0]["reviewed_versions"].append("2.0")
    registry["components"][0]["version_license_expressions"] = {"2.0": "MIT AND BSD-3-Clause"}
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_requirements_require_reviewed_versions(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.1\nnew-package==2.0\n", encoding="utf-8")

    assert check_requirements(registry, [requirements]) == [
        "example-package==1.1 is not a reviewed version in the dependency registry",
        "new-package==2.0 is absent from the reviewed dependency registry",
    ]


def test_requirements_reject_unsupported_lines(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package == 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"requirements\.txt:1: unsupported requirements line"):
        check_requirements(registry, [requirements])


def test_enrich_sbom_writes_the_reviewed_expression(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps({"components": [{"name": "Example_Package", "version": "1.0"}]}),
        encoding="utf-8",
    )

    assert enrich_sbom(registry, sbom) == []
    licenses = json.loads(sbom.read_text(encoding="utf-8"))["components"][0]["licenses"]
    assert licenses == [{"acknowledgement": "declared", "expression": "MIT"}]


def test_enrich_sbom_rejects_an_unreviewed_version(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps({"components": [{"name": "Example_Package", "version": "1.1"}]}),
        encoding="utf-8",
    )

    assert enrich_sbom(registry, sbom) == [
        f"{sbom}: example-package==1.1 is not a reviewed version in the dependency registry",
    ]


def test_enrich_sbom_uses_version_specific_expression(tmp_path: Path) -> None:
    registry = load_registry(_version_specific_registry(tmp_path))
    sbom = tmp_path / "sbom.json"
    sbom.write_text(
        json.dumps({"components": [{"name": "example-package", "version": "2.0"}]}),
        encoding="utf-8",
    )

    assert enrich_sbom(registry, sbom) == []
    licenses = json.loads(sbom.read_text(encoding="utf-8"))["components"][0]["licenses"]
    assert licenses == [{"acknowledgement": "declared", "expression": "MIT AND BSD-3-Clause"}]


def test_license_evidence_matches_version_specific_expression(tmp_path: Path) -> None:
    registry = load_registry(_version_specific_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==2.0\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "example-package",
                        "version": "2.0",
                        "licenses": [{"expression": "MIT AND BSD-3-Clause"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert check_license_evidence(registry, [requirements], evidence) == []


def test_version_specific_expression_must_match_reviewed_version(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    registry["components"][0]["version_license_expressions"] = {"2.0": "MIT AND BSD-3-Clause"}
    path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="version_license_expressions.*reviewed_versions"):
        load_registry(path)


def test_license_evidence_requires_every_active_locked_component(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        "example-package==1.0 ; sys_platform != 'never' # active\nmissing-package==1.0 ; sys_platform == 'never'\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"components": []}), encoding="utf-8")

    assert check_license_evidence(registry, [requirements], evidence) == [
        f"{evidence}: example-package==1.0 is absent from installed dependency license evidence",
    ]


def test_license_evidence_rejects_an_unreviewed_version(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.1\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "example-package",
                        "version": "1.1",
                        "licenses": [{"expression": "MIT"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert check_license_evidence(registry, [requirements], evidence) == [
        f"{evidence}: example-package==1.1 is not a reviewed version in the dependency registry",
    ]


def test_license_evidence_normalizes_identifier_case(tmp_path: Path) -> None:
    registry = load_registry(_registry(tmp_path))
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package==1.0\n", encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "name": "example-package",
                        "version": "1.0",
                        "licenses": [{"expression": "mit"}],
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert check_license_evidence(registry, [requirements], evidence) == []
