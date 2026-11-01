from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numkong as nk
import pytest


def test_router_benchmark_refuses_restricted_neon_by_default() -> None:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        pytest.skip("The NumKong sysctl restriction applies to macOS ARM64.")

    if nk.get_capabilities().get("neon", False):
        pytest.skip("Native NumKong NEON detection succeeds in this process.")

    repo_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "benchmark_router_synthetic.py"),
            "--quick",
            "--threads",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=30,
    )

    if result.returncode != 2 or "sysctl" not in result.stderr:
        pytest.fail(f"Expected restricted-NEON preflight to refuse timing; got {result.returncode}: {result.stderr}")


def test_operand_workloads_use_their_public_router(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    output = tmp_path / "router.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NO_ALBUMENTATIONS_UPDATE": "1",
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "benchmark_router_synthetic.py"),
            "--quick",
            "--threads",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--allow-restricted-isa",
            "--output-json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        timeout=60,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    metadata = result["meta"]
    rows = result["rows"]
    assert metadata["allow_restricted_isa"] is True
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        expected_isa_mode = "native" if nk.get_capabilities().get("neon", False) else "restricted-diagnostic"
    else:
        expected_isa_mode = "not-apple-arm64"
    assert metadata["numkong_isa_mode"] == expected_isa_mode
    expected = {"add_channel", "add_field", "multiply_channel", "multiply_field"}
    failures = [
        (row["op"], row["layout"], row["shape"], row["dtype"], row["detail"])
        for row in rows
        if row["op"] in expected and row["status"] != "ok"
    ]
    observed = {row["op"] for row in rows if row["op"] in expected and row["status"] == "ok"}

    assert observed == expected, f"Missing operand workloads: {expected - observed}; failures: {failures}"
    assert not failures, f"Operand workloads failed: {failures}"
