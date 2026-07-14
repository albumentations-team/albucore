"""Verify Albucore's MIT license, CLA archive, and distribution artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the project dependency
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

SPDX_LICENSE = "MIT"
MIT_LICENSE_SHA256 = "645b7bab67908fe69ff10077947e6459d3e03d3cb21939e08c95592d0e9e3516"
CLA_V1_SHA256 = "1543cc0df9dc3184a5428cff3539247769fd5e3ae4e65b276c4c63815442c2dc"
CLA_ARCHIVE_PATH = "legal/cla/archive/CLA-v1.0-2026-07-14.md"
CLA_MANIFEST_PATH = "legal/cla/archive/MANIFEST.md"
CLA_GIST_URL = "https://gist.github.com/ternaus/8c80c573845e5453fb4ddb8dc23b442d"
CLA_GIST_REVISION = "4be4e1aaed8363e1413fcc9c82dc70da71b784ce"
CLA_GIST_METADATA_SHA256 = "0a15f455a0489b677b9681cb648219ba8090a587ee7d9c127ccdf38c5eff654a"

PRIVATE_ACCEPTANCE_MARKERS = (
    "signature",
    "signer",
    "acceptance-record",
    "entity-acceptance",
    "cla-assistant",
)
PRIVATE_ACCEPTANCE_DIRECTORIES = {
    "private",
    "-private",
    "signatures",
    "signers",
    "acceptance-records",
    "entity-acceptances",
    "cla-assistant-exports",
}
SOURCE_SCAN_EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy-cache",
    ".pytest-cache",
    ".ruff-cache",
    ".tox",
    ".venv",
    "--pycache--",
    "build",
    "dist",
}

REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "CLA.md",
    CLA_ARCHIVE_PATH,
    CLA_MANIFEST_PATH,
    "CONTRIBUTING.md",
    "README.md",
    "docs/maintaining/license-provenance.md",
    "pyproject.toml",
)


def sha256(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for data."""
    return hashlib.sha256(data).hexdigest()


def _normalized_exclude(exclude: str) -> str:
    return exclude.replace("\\", "/").strip("/").removesuffix("/**").rstrip("/")


def _check_project_metadata(repo_root: Path) -> list[str]:
    pyproject_path = repo_root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [f"pyproject.toml is not valid TOML: {exc}"]

    project = pyproject.get("project", {})
    errors: list[str] = []
    if project.get("license") != SPDX_LICENSE:
        errors.append(f"pyproject project.license must be {SPDX_LICENSE!r}")

    if project.get("license-files") != ["LICENSE"]:
        errors.append("pyproject project.license-files must be exactly ['LICENSE']")

    classifiers = project.get("classifiers", [])
    if "License :: OSI Approved :: MIT License" not in classifiers:
        errors.append("pyproject classifiers must include the MIT License classifier")

    hatch_build = pyproject.get("tool", {}).get("hatch", {}).get("build", {})
    configured_exclude_values = [
        *hatch_build.get("exclude", []),
        *hatch_build.get("targets", {}).get("sdist", {}).get("exclude", []),
    ]
    configured_excludes = {_normalized_exclude(value) for value in configured_exclude_values if isinstance(value, str)}
    required_excludes = (
        "CLA.md",
        "legal",
        "_internal",
        "private",
        "data/private",
        "**/*signature*",
        "**/*signer*",
        "**/*acceptance*record*",
        "**/*entity*acceptance*",
        "**/*cla*assistant*",
    )
    errors.extend(
        f"pyproject hatch build exclusions must contain {required_exclude!r}"
        for required_exclude in required_excludes
        if required_exclude not in configured_excludes
    )
    return errors


def _check_license(repo_root: Path) -> list[str]:
    if sha256((repo_root / "LICENSE").read_bytes()) != MIT_LICENSE_SHA256:
        return ["LICENSE does not match the reviewed MIT text"]
    return []


def _check_cla_archive(repo_root: Path) -> list[str]:
    root_cla = (repo_root / "CLA.md").read_bytes()
    archived_cla = (repo_root / CLA_ARCHIVE_PATH).read_bytes()
    errors: list[str] = []

    if sha256(root_cla) != CLA_V1_SHA256:
        errors.append("root CLA.md does not match its immutable Version 1.0 SHA-256 identifier")
    if sha256(archived_cla) != CLA_V1_SHA256:
        errors.append("archived CLA Version 1.0 does not match its immutable SHA-256 identifier")
    if root_cla != archived_cla:
        errors.append("root CLA.md is not byte-identical to the archived Version 1.0 text")

    archive_dir = repo_root / "legal/cla/archive"
    archived_versions = {
        path.relative_to(repo_root).as_posix() for path in archive_dir.glob("CLA-v*.md") if path.is_file()
    }
    if archived_versions != {CLA_ARCHIVE_PATH}:
        unexpected = sorted(archived_versions - {CLA_ARCHIVE_PATH})
        missing = sorted({CLA_ARCHIVE_PATH} - archived_versions)
        if unexpected:
            errors.append(f"unregistered CLA archive files found: {', '.join(unexpected)}")
        if missing:
            errors.append(f"registered CLA archive files are missing: {', '.join(missing)}")

    manifest = (repo_root / CLA_MANIFEST_PATH).read_text(encoding="utf-8")
    required_manifest_values = (
        CLA_ARCHIVE_PATH,
        CLA_V1_SHA256,
        CLA_GIST_URL,
        CLA_GIST_REVISION,
        CLA_GIST_METADATA_SHA256,
        "Version 1.0",
        "July 14, 2026",
    )
    errors.extend(
        f"CLA manifest is missing {required_value!r}"
        for required_value in required_manifest_values
        if required_value not in manifest
    )

    normalized_cla = " ".join(root_cla.decode("utf-8").split())
    required_cla_phrases = (
        "Albumentations, LLC",
        "license or sublicense it under open-source, source-available, commercial, proprietary, or other terms",
        "first or simultaneously make the accepted version of that Covered Contribution available",
        "I have read and agree to the Albucore CLA Version 1.0 (July 14, 2026) as an individual.",
    )
    errors.extend(
        f"CLA Version 1.0 is missing {required_phrase!r}"
        for required_phrase in required_cla_phrases
        if required_phrase not in normalized_cla
    )
    return errors


def _check_public_copy(repo_root: Path) -> list[str]:
    contributing = " ".join((repo_root / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    required_contributing_phrases = (
        "Albucore remains publicly available under the [MIT License](LICENSE)",
        "does not change the repository's license",
        "Version 1.0 Acceptance Record",
    )
    errors = [
        f"CONTRIBUTING.md is missing {required_phrase!r}"
        for required_phrase in required_contributing_phrases
        if required_phrase not in contributing
    ]

    readme = " ".join((repo_root / "README.md").read_text(encoding="utf-8").split())
    if "License-MIT" not in readme:
        errors.append("README.md must identify the public project license as MIT")

    provenance = " ".join(
        (repo_root / "docs/maintaining/license-provenance.md").read_text(encoding="utf-8").split(),
    )
    required_provenance_phrases = (
        "The CLA does not replace or narrow MIT permissions",
        "Do not commit them to this repository or include them in release artifacts",
        "License, CLA, and package notices",
    )
    errors.extend(
        f"license-provenance.md is missing {required_phrase!r}"
        for required_phrase in required_provenance_phrases
        if required_phrase not in provenance
    )
    return errors


def _normalized_path_part(part: str) -> str:
    return part.lower().replace("_", "-").replace(" ", "-")


def _is_private_acceptance_path(path: PurePosixPath) -> bool:
    normalized_parts = tuple(_normalized_path_part(part) for part in path.parts)
    if any(part in PRIVATE_ACCEPTANCE_DIRECTORIES for part in normalized_parts):
        return True
    return any(marker in part for part in normalized_parts for marker in PRIVATE_ACCEPTANCE_MARKERS)


def _check_private_acceptance_records(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(repo_root)
        normalized_parts = {_normalized_path_part(part) for part in relative_path.parts[:-1]}
        if normalized_parts & SOURCE_SCAN_EXCLUDED_DIRECTORIES:
            continue
        relative_posix_path = PurePosixPath(relative_path.as_posix())
        if _is_private_acceptance_path(relative_posix_path):
            errors.append(f"private CLA acceptance material must not be committed: {relative_posix_path}")
    return errors


def collect_source_errors(repo_root: Path = REPO_ROOT) -> list[str]:
    """Collect all source-tree legal-integrity violations."""
    errors = [
        f"missing required legal-integrity file: {relative_path}"
        for relative_path in REQUIRED_SOURCE_FILES
        if not (repo_root / relative_path).is_file()
    ]
    if errors:
        return errors

    errors.extend(_check_project_metadata(repo_root))
    errors.extend(_check_license(repo_root))
    errors.extend(_check_cla_archive(repo_root))
    errors.extend(_check_public_copy(repo_root))
    errors.extend(_check_private_acceptance_records(repo_root))
    return errors


def _zip_members(artifact: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(artifact) as archive:
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}


def _tar_members(artifact: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(artifact, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                members[member.name] = extracted.read()
    return members


def read_artifact_members(artifact: Path) -> dict[str, bytes]:
    """Read regular files from a wheel, zip file, or source tarball."""
    if zipfile.is_zipfile(artifact):
        return _zip_members(artifact)
    if tarfile.is_tarfile(artifact):
        return _tar_members(artifact)
    raise ValueError(f"unsupported distribution artifact: {artifact}")


def _metadata_errors(artifact: Path, metadata: bytes, label: str) -> list[str]:
    message = BytesParser().parsebytes(metadata, headersonly=True)
    errors: list[str] = []

    license_expressions = message.get_all("License-Expression", [])
    if license_expressions != [SPDX_LICENSE]:
        expression_summary = ", ".join(license_expressions) if license_expressions else "missing"
        errors.append(
            f"{artifact.name}: {label} License-Expression must be {SPDX_LICENSE!r}, found {expression_summary!r}",
        )

    license_files = message.get_all("License-File", [])
    if license_files != ["LICENSE"]:
        file_summary = ", ".join(license_files) if license_files else "missing"
        errors.append(f"{artifact.name}: {label} License-File must be exactly 'LICENSE', found {file_summary!r}")
    return errors


def _wheel_metadata_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    metadata_members = [
        name
        for name in members
        if PurePosixPath(name).name == "METADATA" and PurePosixPath(name).parent.name.endswith(".dist-info")
    ]
    if len(metadata_members) != 1:
        return [f"{artifact.name}: expected exactly one wheel .dist-info/METADATA file, found {len(metadata_members)}"]
    return _metadata_errors(artifact, members[metadata_members[0]], "wheel METADATA")


def _sdist_metadata_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    metadata_members = [
        name for name in members if PurePosixPath(name).name == "PKG-INFO" and len(PurePosixPath(name).parts) == 2
    ]
    if len(metadata_members) != 1:
        return [f"{artifact.name}: expected exactly one root sdist PKG-INFO file, found {len(metadata_members)}"]
    return _metadata_errors(artifact, members[metadata_members[0]], "sdist PKG-INFO")


def _contains_contiguous_parts(path: PurePosixPath, expected_parts: tuple[str, ...]) -> bool:
    path_parts = path.parts
    expected_length = len(expected_parts)
    return any(
        path_parts[index : index + expected_length] == expected_parts
        for index in range(len(path_parts) - expected_length + 1)
    )


def _forbidden_artifact_errors(artifact: Path, members: Mapping[str, bytes]) -> list[str]:
    errors: list[str] = []
    for member_name in members:
        member_path = PurePosixPath(member_name)
        if member_path.name == "CLA.md" or _contains_contiguous_parts(member_path, ("legal", "cla")):
            errors.append(f"{artifact.name}: inbound CLA material leaked into artifact as {member_name}")
        if "_internal" in member_path.parts:
            errors.append(f"{artifact.name}: private _internal material leaked into artifact as {member_name}")
        if _is_private_acceptance_path(member_path):
            errors.append(f"{artifact.name}: private CLA acceptance material leaked into artifact as {member_name}")
        if artifact.suffix != ".whl" and member_name.endswith((".whl", ".tar.gz")):
            errors.append(f"{artifact.name}: nested distribution artifact leaked into sdist as {member_name}")
    return errors


def collect_artifact_errors(artifact: Path, expected_license: bytes) -> list[str]:
    """Collect package metadata, license-content, and inbound-material errors."""
    members = read_artifact_members(artifact)
    if artifact.suffix == ".whl":
        errors = _wheel_metadata_errors(artifact, members)
    else:
        errors = _sdist_metadata_errors(artifact, members)

    license_members = [name for name in members if PurePosixPath(name).name == "LICENSE"]
    if len(license_members) != 1:
        errors.append(f"{artifact.name}: expected exactly one LICENSE file, found {len(license_members)}")
    elif members[license_members[0]] != expected_license:
        errors.append(f"{artifact.name}: LICENSE is not byte-identical to the source file")

    errors.extend(_forbidden_artifact_errors(artifact, members))
    return errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        default=(),
        help="Built wheel and sdist paths to verify after the source-tree checks.",
    )
    return parser.parse_args()


def main() -> int:
    """Run source and optional artifact checks."""
    args = parse_args()
    errors = collect_source_errors(REPO_ROOT)
    if not errors and args.artifacts:
        expected_license = (REPO_ROOT / "LICENSE").read_bytes()
        for artifact in args.artifacts:
            if not artifact.is_file():
                errors.append(f"artifact does not exist: {artifact}")
                continue
            try:
                errors.extend(collect_artifact_errors(artifact, expected_license))
            except (OSError, tarfile.TarError, ValueError, zipfile.BadZipFile) as exc:
                errors.append(f"{artifact.name}: could not inspect artifact: {exc}")

    if errors:
        for error in errors:
            sys.stderr.write(f"ERROR: {error}\n")
        return 1

    artifact_count = len(args.artifacts)
    sys.stdout.write(f"Legal integrity verified: source tree and {artifact_count} artifact(s).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
