"""Verify release-candidate artifacts before and after PyPI publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Callable

PROJECT_NAME: Final = "albucore"
PYPI_VERSION_JSON_URL: Final = "https://pypi.org/pypi/{project}/{version}/json"


@dataclass(frozen=True)
class PyPIWaitConfig:
    """Settings for polling PyPI after publication."""

    attempts: int = 12
    sleep_seconds: float = 10.0
    timeout: float = 15.0
    urlopen: Callable[..., Any] | None = None
    sleep: Callable[[float], None] = time.sleep


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    return payload


def verify_candidate_run(run: dict[str, Any], expected_sha: str) -> None:
    """Validate that publish is using a successful Release Candidate run."""
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        msg = f"Candidate run did not succeed: {run}."
        raise ValueError(msg)
    if run.get("workflowName") != "Release Candidate":
        msg = f"Candidate run used workflow {run.get('workflowName')!r}, expected 'Release Candidate'."
        raise ValueError(msg)
    if run.get("headSha") != expected_sha:
        msg = f"Candidate run SHA {run.get('headSha')} does not match {expected_sha}."
        raise ValueError(msg)


def verify_release_metadata(metadata: dict[str, Any], expected_version: str, expected_sha: str) -> None:
    """Validate release-candidate metadata against requested publish inputs."""
    if metadata.get("version") != expected_version:
        msg = f"Candidate metadata version {metadata.get('version')} does not match {expected_version}."
        raise ValueError(msg)
    if metadata.get("commit_sha") != expected_sha:
        msg = f"Candidate metadata SHA {metadata.get('commit_sha')} does not match {expected_sha}."
        raise ValueError(msg)


def expected_distribution_files(dist_dir: Path, version: str) -> tuple[Path, Path]:
    """Return the expected sdist and universal wheel paths for an Albucore release."""
    return (
        dist_dir / f"{PROJECT_NAME}-{version}.tar.gz",
        dist_dir / f"{PROJECT_NAME}-{version}-py3-none-any.whl",
    )


def verify_distribution_files(dist_dir: Path, version: str) -> None:
    """Fail if the expected wheel or sdist is missing."""
    missing = [str(path) for path in expected_distribution_files(dist_dir, version) if not path.exists()]
    if missing:
        msg = f"Missing expected distribution artifacts: {missing}."
        raise FileNotFoundError(msg)


def copy_distribution_files(dist_dir: Path, output_dir: Path, version: str) -> None:
    """Copy only publishable wheel and sdist files to output_dir."""
    verify_distribution_files(dist_dir, version)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in expected_distribution_files(dist_dir, version):
        shutil.copy2(source, output_dir / source.name)


def _checksum_path(dist_dir: Path, filename: str) -> Path:
    clean_name = filename.removeprefix("./")
    relative_path = Path(clean_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        msg = f"Checksum entry references unsafe path {filename!r}."
        raise ValueError(msg)
    return dist_dir / relative_path


def verify_checksums(dist_dir: Path) -> None:
    """Verify every file listed in SHA256SUMS.txt."""
    checksums = dist_dir / "SHA256SUMS.txt"
    if not checksums.exists():
        msg = "Missing SHA256SUMS.txt."
        raise FileNotFoundError(msg)

    for line_number, line in enumerate(checksums.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, filename = line.split(maxsplit=1)
        except ValueError as exc:
            msg = f"Malformed checksum line {line_number}: {line!r}."
            raise ValueError(msg) from exc
        if len(digest) != 64:
            msg = f"Malformed sha256 digest on line {line_number}: {digest!r}."
            raise ValueError(msg)
        path = _checksum_path(dist_dir, filename)
        if not path.exists():
            msg = f"Checksum entry references missing file {path}."
            raise FileNotFoundError(msg)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            msg = f"Checksum mismatch for {path}: {actual} != {digest}."
            raise ValueError(msg)


def pypi_version_exists(
    project: str,
    version: str,
    *,
    timeout: float = 15.0,
    urlopen: Callable[..., Any] | None = None,
) -> bool:
    """Return True when PyPI already has project/version metadata."""
    opener = urllib.request.urlopen if urlopen is None else urlopen
    url = PYPI_VERSION_JSON_URL.format(project=project, version=version)
    try:
        with opener(url, timeout=timeout) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def verify_pypi_absent(
    project: str,
    version: str,
    *,
    urlopen: Callable[..., Any] | None = None,
) -> None:
    """Fail if PyPI already has the version being published."""
    if pypi_version_exists(project, version, urlopen=urlopen):
        msg = f"PyPI already has {project} {version}; refusing to republish."
        raise ValueError(msg)


def verify_prepublish_artifacts(dist_dir: Path, candidate_run_json: Path, version: str, commit_sha: str) -> None:
    """Validate downloaded release-candidate artifacts before publishing."""
    verify_candidate_run(_load_json_object(candidate_run_json), commit_sha)
    verify_release_metadata(_load_json_object(dist_dir / "release-candidate-metadata.json"), version, commit_sha)
    verify_distribution_files(dist_dir, version)
    verify_checksums(dist_dir)


def verify_direct_release_artifacts(dist_dir: Path, version: str, commit_sha: str) -> None:
    """Validate artifacts built by the release-published publish job before uploading."""
    verify_release_metadata(_load_json_object(dist_dir / "release-candidate-metadata.json"), version, commit_sha)
    verify_distribution_files(dist_dir, version)
    verify_checksums(dist_dir)


def pypi_file_count(payload: dict[str, Any]) -> int:
    """Return the number of files listed in a PyPI version JSON payload."""
    files = payload.get("urls", [])
    if not isinstance(files, list):
        msg = "PyPI version payload field 'urls' must be a list."
        raise TypeError(msg)
    return len(files)


def verify_pypi_publication(project: str, version: str, config: PyPIWaitConfig | None = None) -> int:
    """Poll PyPI until the published version has at least one file."""
    wait_config = PyPIWaitConfig() if config is None else config
    opener = urllib.request.urlopen if wait_config.urlopen is None else wait_config.urlopen
    url = PYPI_VERSION_JSON_URL.format(project=project, version=version)

    for attempt in range(wait_config.attempts):
        try:
            with opener(url, timeout=wait_config.timeout) as response:
                payload = json.load(response)
        except OSError:
            if attempt == wait_config.attempts - 1:
                raise
            wait_config.sleep(wait_config.sleep_seconds)
            continue

        file_count = pypi_file_count(payload)
        if file_count > 0:
            return file_count
        if attempt == wait_config.attempts - 1:
            msg = f"PyPI has {project} {version} metadata but no files."
            raise ValueError(msg)
        wait_config.sleep(wait_config.sleep_seconds)

    msg = f"PyPI publication check exhausted attempts for {project} {version}."
    raise ValueError(msg)


def _prepublish_command(args: argparse.Namespace) -> None:
    verify_prepublish_artifacts(args.dist_dir, args.candidate_run_json, args.version, args.commit_sha)
    verify_pypi_absent(args.project, args.version)


def _direct_release_command(args: argparse.Namespace) -> None:
    verify_direct_release_artifacts(args.dist_dir, args.version, args.commit_sha)
    verify_pypi_absent(args.project, args.version)


def _prepare_pypi_dist_command(args: argparse.Namespace) -> None:
    copy_distribution_files(args.dist_dir, args.output_dir, args.version)


def _publication_command(args: argparse.Namespace) -> None:
    file_count = verify_pypi_publication(
        args.project,
        args.version,
        PyPIWaitConfig(
            attempts=args.attempts,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
        ),
    )
    sys.stdout.write(f"PyPI has {args.project} {args.version} with {file_count} files.\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepublish = subparsers.add_parser("prepublish")
    prepublish.add_argument("--dist-dir", type=Path, default=Path("dist"))
    prepublish.add_argument("--candidate-run-json", type=Path, required=True)
    prepublish.add_argument("--version", required=True)
    prepublish.add_argument("--commit-sha", required=True)
    prepublish.add_argument("--project", default=PROJECT_NAME)
    prepublish.set_defaults(func=_prepublish_command)

    direct_release = subparsers.add_parser("direct-release")
    direct_release.add_argument("--dist-dir", type=Path, default=Path("dist"))
    direct_release.add_argument("--version", required=True)
    direct_release.add_argument("--commit-sha", required=True)
    direct_release.add_argument("--project", default=PROJECT_NAME)
    direct_release.set_defaults(func=_direct_release_command)

    prepare = subparsers.add_parser("prepare-pypi-dist")
    prepare.add_argument("--dist-dir", type=Path, default=Path("dist"))
    prepare.add_argument("--output-dir", type=Path, default=Path("pypi-dist"))
    prepare.add_argument("--version", required=True)
    prepare.set_defaults(func=_prepare_pypi_dist_command)

    publication = subparsers.add_parser("publication")
    publication.add_argument("--version", required=True)
    publication.add_argument("--project", default=PROJECT_NAME)
    publication.add_argument("--attempts", type=int, default=12)
    publication.add_argument("--sleep-seconds", type=float, default=10.0)
    publication.add_argument("--timeout", type=float, default=15.0)
    publication.set_defaults(func=_publication_command)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Publish artifact verification failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
