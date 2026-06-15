"""Collect CI/test environment metadata as JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGES = (
    "albucore",
    "numpy",
    "numkong",
    "stringzilla",
    "opencv-python-headless",
    "opencv-python",
    "opencv-contrib-python-headless",
    "opencv-contrib-python",
    "pytest",
)


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_environment(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Collect local test environment metadata."""
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "command": sys.argv,
        "github": {
            "sha": os.environ.get("GITHUB_SHA"),
            "ref": os.environ.get("GITHUB_REF"),
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "job": os.environ.get("GITHUB_JOB"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        },
        "packages": {package: _package_version(package) for package in packages},
        "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    payload = json.dumps(collect_environment(), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
