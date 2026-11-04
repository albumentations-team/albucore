"""Resumed benchmark rows retain their measured source and execution profile."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.timing import _check_resume_profile


_OPENCV_PACKAGES = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


@pytest.fixture
def profile() -> dict[str, Any]:
    return {
        "candidate_source_sha256": "source",
        "conversion_source_sha256": "conversion",
        "benchmark_source_sha256": "benchmark",
        "baseline_commit": "commit",
        "baseline": "method",
        "core_source": "/checkout/albucore/__init__.py",
        "cpu": "CPU",
        "platform": "platform",
        "python": "3.11.15",
        "capabilities": {"neon": True},
        "native_x86_cast_route": False,
        "threads": {"torch": 1},
        "packages": {"numpy": "2.4.6"},
        "samples": 21,
        "minimum_count": 1,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_source_sha256", "other-source"),
        ("conversion_source_sha256", "other-conversion"),
        ("benchmark_source_sha256", "other-benchmark"),
        ("baseline_commit", "other-commit"),
        ("baseline", "other-method"),
        ("core_source", "/other-checkout/albucore/__init__.py"),
        ("cpu", "other-CPU"),
        ("platform", "other-platform"),
        ("python", "3.12.0"),
        ("capabilities", {"neon": False}),
        ("native_x86_cast_route", True),
        ("threads", {"torch": 2}),
        ("packages", {"numpy": "2.5.4"}),
        ("samples", 101),
        ("minimum_count", 10),
    ],
)
def test_resume_rejects_changed_profile(profile: dict[str, Any], field: str, value: Any) -> None:
    previous = {**profile, field: value}

    with pytest.raises(RuntimeError, match=field):
        _check_resume_profile(previous, profile)


@pytest.mark.parametrize("missing_from", ["previous", "current"])
def test_resume_rejects_missing_profile_field(profile: dict[str, Any], missing_from: str) -> None:
    previous = dict(profile)
    current = dict(profile)
    (previous if missing_from == "previous" else current).pop("benchmark_source_sha256")

    with pytest.raises(RuntimeError, match="benchmark_source_sha256"):
        _check_resume_profile(previous, current)


@pytest.mark.parametrize("previous_cases,current_cases", [(None, None), (["guard"], ["out"]), (None, ["out"])])
def test_resume_accepts_matching_profile_and_different_subsets(
    profile: dict[str, Any], previous_cases: list[str] | None, current_cases: list[str] | None
) -> None:
    previous = dict(profile)
    current = dict(profile)
    if previous_cases is not None:
        previous["case_list"] = previous_cases
    if current_cases is not None:
        current["case_list"] = current_cases

    _check_resume_profile(previous, current)


def test_profile_collection_preserves_thread_settings() -> None:
    variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    code = (
        "import os\n"
        "import tests.test_cast_benchmark_profile\n"
        f"assert all(os.environ[name] == '3' for name in {variables!r})\n"
    )

    subprocess.run([sys.executable, "-c", code], env={**os.environ, **dict.fromkeys(variables, "3")}, check=True)


@pytest.mark.parametrize("opencv_package", _OPENCV_PACKAGES)
def test_benchmark_accepts_opencv_distribution(opencv_package: str, tmp_path: Path) -> None:
    code = """
import importlib.metadata as metadata
import runpy
import sys
from types import SimpleNamespace

benchmark = runpy.run_path('benchmarks/benchmark_float_cast_guard.py')
opencv_package = sys.argv[1]
installed = [dist for dist in metadata.distributions() if not dist.metadata['Name'].startswith('opencv-')]
installed.append(SimpleNamespace(metadata={'Name': opencv_package}, version='test-version'))
original_version = metadata.version

def version(name):
    if name.startswith('opencv-') and name != opencv_package:
        raise metadata.PackageNotFoundError(name)
    return original_version(name)

metadata.version = version
metadata.distributions = lambda: installed
sys.argv = [sys.argv[0], '--memory-worker', 'guard', '--output', sys.argv[2]]
benchmark['main']()
"""
    output = tmp_path / "metadata.json"

    subprocess.run([sys.executable, "-c", code, opencv_package, str(output)], check=True)

    packages = json.loads(output.read_text())["metadata"]["packages"]
    assert set(packages) & set(_OPENCV_PACKAGES) == {opencv_package}
    assert packages[opencv_package] == "test-version"
