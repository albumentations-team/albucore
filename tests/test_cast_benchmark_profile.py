"""Resumed benchmark rows retain their measured source and execution profile."""

from typing import Any

import pytest

from benchmarks.benchmark_float_cast_guard import _check_resume_profile


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
