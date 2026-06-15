"""Check router benchmark JSON artifacts for performance regressions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tests.router_contracts import ROUTER_CONTRACTS

Mode = Literal["advisory", "release"]

RELEASE_BLOCKING_OPS = frozenset(
    benchmark_name
    for contract in ROUTER_CONTRACTS.values()
    if contract.release_blocking_performance
    for benchmark_name in contract.benchmark_names
)


@dataclass(frozen=True)
class BenchCell:
    """One successful benchmark timing cell."""

    op: str
    layout: str
    shape: tuple[int, ...]
    dtype: str
    median_ms: float

    @property
    def key(self) -> tuple[str, str, tuple[int, ...], str]:
        """Stable comparison key for matching baseline and current rows."""
        return self.op, self.layout, self.shape, self.dtype


@dataclass(frozen=True)
class Regression:
    """One benchmark cell that slowed down enough to report."""

    cell: BenchCell
    baseline_ms: float
    current_ms: float
    slowdown: float
    status: str

    @property
    def slowdown_pct(self) -> float:
        """Slowdown as a percentage."""
        return self.slowdown * 100.0


def _load_cells(path: Path) -> dict[tuple[str, str, tuple[int, ...], str], BenchCell]:
    data = json.loads(path.read_text())
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        msg = f"{path} does not contain benchmark rows"
        raise TypeError(msg)

    cells: dict[tuple[str, str, tuple[int, ...], str], BenchCell] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        median = row.get("ms_median")
        shape = row.get("shape")
        if not isinstance(median, (int, float)) or not isinstance(shape, list):
            continue
        cell = BenchCell(
            op=str(row.get("op", "unknown")),
            layout=str(row.get("layout", "unknown")),
            shape=tuple(int(part) for part in shape),
            dtype=str(row.get("dtype", "unknown")),
            median_ms=float(median),
        )
        cells[cell.key] = cell
    return cells


def _status_for_slowdown(op: str, slowdown: float, mode: Mode) -> str:
    status = "ok"
    if slowdown > 0 and op not in RELEASE_BLOCKING_OPS:
        status = "report"
    elif mode == "release" and slowdown > 0.10:
        status = "blocking"
    elif mode == "release" and slowdown > 0.05:
        status = "review"
    elif mode == "advisory" and slowdown > 0.15:
        status = "warning"
    return status


def _regressions(
    baseline: dict[tuple[str, str, tuple[int, ...], str], BenchCell],
    current: dict[tuple[str, str, tuple[int, ...], str], BenchCell],
    mode: Mode,
) -> list[Regression]:
    regressions: list[Regression] = []
    for key in sorted(set(baseline) & set(current)):
        base_cell = baseline[key]
        current_cell = current[key]
        if base_cell.median_ms <= 0:
            continue
        slowdown = current_cell.median_ms / base_cell.median_ms - 1.0
        status = _status_for_slowdown(current_cell.op, slowdown, mode)
        if status != "ok":
            regressions.append(
                Regression(
                    cell=current_cell,
                    baseline_ms=base_cell.median_ms,
                    current_ms=current_cell.median_ms,
                    slowdown=slowdown,
                    status=status,
                ),
            )
    return sorted(regressions, key=lambda item: item.slowdown, reverse=True)


def _family_warnings(
    baseline: dict[tuple[str, str, tuple[int, ...], str], BenchCell],
    current: dict[tuple[str, str, tuple[int, ...], str], BenchCell],
) -> dict[str, float]:
    slowdowns_by_op: dict[str, list[float]] = defaultdict(list)
    for key in set(baseline) & set(current):
        base_cell = baseline[key]
        current_cell = current[key]
        if base_cell.median_ms <= 0:
            continue
        slowdowns_by_op[current_cell.op].append(current_cell.median_ms / base_cell.median_ms - 1.0)

    return {
        op: median_slowdown
        for op, slowdowns in slowdowns_by_op.items()
        if op in RELEASE_BLOCKING_OPS and (median_slowdown := statistics.median(slowdowns)) > 0.10
    }


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in shape)


def _markdown_report(
    baseline_path: Path,
    current_path: Path,
    regressions: list[Regression],
    family_warnings: dict[str, float],
    max_rows: int,
) -> str:
    lines = [
        "# Benchmark Regression Check",
        "",
        f"- Baseline: `{baseline_path}`",
        f"- Current: `{current_path}`",
        f"- Regressions reported: {len(regressions)}",
        "",
        "## Cell Regressions",
        "",
        "| Status | Operation | Layout | Shape | Dtype | Baseline ms | Current ms | Slowdown |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for regression in regressions[:max_rows]:
        cell = regression.cell
        lines.append(
            f"| `{regression.status}` | `{cell.op}` | `{cell.layout}` | `{_shape_text(cell.shape)}` | "
            f"`{cell.dtype}` | {regression.baseline_ms:.6g} | {regression.current_ms:.6g} | "
            f"{regression.slowdown_pct:.2f}% |",
        )

    lines.extend(["", "## Family Warnings", "", "| Operation | Median slowdown |", "| --- | ---: |"])
    for op, slowdown in sorted(family_warnings.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| `{op}` | {slowdown * 100.0:.2f}% |")
    if not family_warnings:
        lines.append("| none | 0.00% |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--mode", choices=("advisory", "release"), default="advisory")
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=25)
    args = parser.parse_args()

    mode = args.mode
    baseline = _load_cells(args.baseline)
    current = _load_cells(args.current)
    regressions = _regressions(baseline, current, mode)
    family_warnings = _family_warnings(baseline, current)
    report = _markdown_report(args.baseline, args.current, regressions, family_warnings, args.max_rows)

    if args.output_md is not None:
        args.output_md.write_text(report)
    else:
        sys.stdout.write(report)

    if mode == "release" and any(regression.status == "blocking" for regression in regressions):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
