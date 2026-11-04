"""Benchmark CLIs reject missing inputs and output paths that could pollute Git."""

from __future__ import annotations

import argparse
import importlib
import json
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


@pytest.mark.parametrize("name", ["result.json", "timings.csv", "checkpoint.tmp"])
def test_benchmark_output_accepts_external_and_ignored_paths(tmp_path: Path, name: str) -> None:
    assert output_path(str(tmp_path / name)) == tmp_path / name
    ignored = REPO_ROOT / "benchmarks/results" / name
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


@pytest.mark.parametrize("provided", [None, "output", "timing-results"])
def test_memory_parent_requires_output_and_timings(tmp_path: Path, provided: str | None) -> None:
    args = ["--baseline-ax", "missing"]
    if provided:
        args.extend((f"--{provided}", str(tmp_path / "data.json")))
    result = _run_script("benchmark_ax_image_kernels_memory.py", *args)
    assert result.returncode == 2
    assert "--output and --timing-results are required for parent runs" in result.stderr
    assert "Traceback" not in result.stderr


def test_memory_worker_does_not_require_parent_output_paths(monkeypatch, capsys) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "benchmarks"))
    memory = importlib.import_module("benchmark_ax_image_kernels_memory")
    monkeypatch.setattr(memory, "worker", lambda args: {"case": args.case})
    monkeypatch.setattr(sys, "argv", ["benchmark", "--baseline-ax", "missing", "--case", "convolve|core|uint8"])
    memory.main()
    assert json.loads(capsys.readouterr().out) == {"case": "convolve|core|uint8"}
