"""Benchmark CLIs reject missing inputs and output paths that could pollute Git."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.benchmark_output import REPO_ROOT, output_path


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_benchmark_output_accepts_external_and_ignored_paths(tmp_path: Path) -> None:
    external = tmp_path / "custom.csv"
    assert output_path(str(external)) == external
    ignored = REPO_ROOT / "benchmarks/results/report.json"
    assert output_path(str(ignored)) == ignored


@pytest.mark.parametrize("name", ["result.json", "docs/report.json", "benchmarks/results/../../result.json"])
def test_benchmark_output_rejects_other_checkout_paths(name: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="outside the checkout"):
        output_path(str(REPO_ROOT / name))


@pytest.mark.parametrize(
    "script,args",
    [
        ("benchmark_router_synthetic.py", []),
        ("benchmark_warp_affine3d_xy_nearest.py", ["--baseline", "missing.py"]),
    ],
)
def test_benchmark_cli_rejects_arbitrary_checkout_output(script: str, args: list[str]) -> None:
    flag = "--output-json" if script == "benchmark_router_synthetic.py" else "--output"
    result = _run_script(script, *args, flag, str(REPO_ROOT / "result.json"))
    assert result.returncode == 2
    assert "outside the checkout or under benchmarks/results/" in result.stderr
    assert "Traceback" not in result.stderr


def test_filtering_parent_requires_output_before_loading_baseline() -> None:
    result = _run_script("benchmark_filtering_cleanup.py", "--baseline-source", "missing.py")
    assert result.returncode == 2
    assert "--output is required unless --rss-worker is set" in result.stderr
    assert "Traceback" not in result.stderr
