"""Benchmark CLIs reject missing inputs and output paths that could pollute Git."""

from __future__ import annotations

import argparse
import importlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from benchmarks import benchmark_output
from benchmarks.benchmark_output import REPO_ROOT, output_path, write_json_atomic


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmarks" / name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_benchmark_output_accepts_external_and_ignored_paths(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(benchmark_output, "REPO_ROOT", checkout)
    external = tmp_path / "custom.csv"
    assert output_path(str(external)) == external
    ignored = checkout / "benchmarks/results/report.json"
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


@pytest.mark.parametrize("flag", ["--operations", "--rss-operation"])
def test_filtering_rejects_unknown_operations_before_loading_baseline(flag: str) -> None:
    result = _run_script("benchmark_filtering_cleanup.py", "--baseline-source", "missing.py", flag, "box_blur")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "box_blur" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("symlink", [False, True])
def test_atomic_report_preserves_unrelated_temporary_file(tmp_path: Path, symlink: bool) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep this data")
    output = tmp_path / "report.json"
    fixed_temporary = output.with_suffix(".tmp")
    if symlink:
        try:
            fixed_temporary.symlink_to(unrelated)
        except (OSError, NotImplementedError):
            pytest.skip("Creating file symlinks is unavailable")
    else:
        fixed_temporary.write_text("keep this temporary file")

    write_json_atomic(output, {"rows": [1, 2]})

    assert json.loads(output.read_text()) == {"rows": [1, 2]}
    assert not output.is_symlink()
    assert unrelated.read_text() == "keep this data"
    assert fixed_temporary.read_text() == ("keep this data" if symlink else "keep this temporary file")
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_atomic_report_preserves_previous_result_on_serialization_failure(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text('{"previous": true}\n')

    with pytest.raises(TypeError):
        write_json_atomic(output, {"invalid": object()})

    assert output.read_text() == '{"previous": true}\n'
    assert not list(tmp_path.glob(".report.json.*.tmp"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX access modes")
def test_atomic_report_preserves_existing_permissions(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}\n")
    output.chmod(0o640)

    write_json_atomic(output, {"rows": [1]})

    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert json.loads(output.read_text()) == {"rows": [1]}


def test_xy_benchmark_rejects_symlinked_csv_before_loading_baseline(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep this data")
    results = REPO_ROOT / "benchmarks/results"
    results.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=results, prefix="csv-path-test-") as directory:
        output = Path(directory) / "report.json"
        try:
            output.with_suffix(".csv").symlink_to(unrelated)
        except (OSError, NotImplementedError):
            pytest.skip("Creating file symlinks is unavailable")
        result = _run_script("benchmark_warp_affine3d_xy_nearest.py", "--baseline", "missing.py", "--output", str(output))

    assert result.returncode == 2
    assert "Timings CSV path must not be a symlink" in result.stderr
    assert "Traceback" not in result.stderr
    assert unrelated.read_text() == "keep this data"


@pytest.mark.parametrize("suffix", [".csv", ".json"])
def test_xy_report_writer_does_not_follow_late_symlinks(tmp_path: Path, monkeypatch, suffix: str) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "benchmarks"))
    benchmark = importlib.import_module("benchmark_warp_affine3d_xy_nearest")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep this data")
    output = tmp_path / "report.json"
    try:
        output.with_suffix(suffix).symlink_to(unrelated)
    except (OSError, NotImplementedError):
        pytest.skip("Creating file symlinks is unavailable")

    benchmark._write_results(output, {}, [])

    assert unrelated.read_text() == "keep this data"
    assert not output.is_symlink()
    assert not output.with_suffix(".csv").is_symlink()
    assert json.loads(output.read_text())["timings"] == "report.csv"
    assert output.with_suffix(".csv").read_text().startswith("phase,run,")


@pytest.mark.parametrize("suffix", [".csv", ".CSV"])
def test_xy_benchmark_rejects_colliding_output_paths(tmp_path: Path, suffix: str) -> None:
    output = REPO_ROOT / "benchmarks/results" / tmp_path.name / f"metadata{suffix}"
    result = _run_script("benchmark_warp_affine3d_xy_nearest.py", "--baseline", "missing.py", "--output", str(output))
    assert result.returncode == 2
    assert "Metadata JSON and timings CSV need different output paths" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("destination", ["contiguous", "strided"])
@pytest.mark.parametrize("partial_write", [False, True])
def test_lut_benchmark_rejects_incomplete_output(tmp_path: Path, destination: str, partial_write: bool) -> None:
    code = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path('benchmarks').resolve()))
import benchmark_lut_allocation as benchmark

destination, partial_write, output = sys.argv[1:]
original = benchmark.core.apply_uint8_lut

def incomplete(image, table, *, out=None):
    if out is None:
        return original(image, table)
    if partial_write == 'True':
        out.flat[0] = table[image.flat[0]]
    return out

benchmark.core.apply_uint8_lut = incomplete
benchmark.cases = lambda quick: [((5, 7, 3), 'readonly', destination)]
sys.argv = [sys.argv[0], '--baseline-source', 'albucore/lut.py', '--output', output, '--skip-memory']
benchmark.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, destination, str(partial_write), str(tmp_path / "report.json")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "AssertionError" in result.stderr
    assert "Arrays are not equal" in result.stderr
