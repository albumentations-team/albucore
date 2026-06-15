"""Summarize Albucore benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _shape_text(row: dict[str, Any]) -> str:
    shape = row.get("shape", [])
    if not isinstance(shape, list):
        return "?"
    return "x".join(str(part) for part in shape)


def _summarize(path: Path) -> str:
    data = _load_json(path)
    meta = data.get("meta", {})
    rows = data.get("rows", [])
    if not isinstance(meta, dict) or not isinstance(rows, list):
        msg = f"{path} is not a benchmark_router_synthetic JSON artifact"
        raise TypeError(msg)

    status_counts: Counter[str] = Counter()
    op_counts: dict[str, Counter[str]] = defaultdict(Counter)
    slowest: list[tuple[float, dict[str, Any]]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "unknown"))
        op = str(row.get("op", "unknown"))
        status_counts[status] += 1
        op_counts[op][status] += 1
        median = row.get("ms_median")
        if status == "ok" and isinstance(median, (int, float)):
            slowest.append((float(median), row))

    lines = [
        f"# Benchmark Summary: `{path}`",
        "",
        "## Metadata",
        "",
        f"- Albucore: `{meta.get('albucore_version', 'unknown')}`",
        f"- Distribution: `{meta.get('distribution_version', 'unknown')}`",
        f"- Python: `{meta.get('python', 'unknown')}`",
        f"- Platform: `{meta.get('platform', 'unknown')}`",
        f"- Quick: `{meta.get('quick', 'unknown')}`",
        f"- With geometric: `{meta.get('with_geometric', 'unknown')}`",
        "",
        "## Status Counts",
        "",
        "| Status | Rows |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| `{status}` | {count} |")

    lines.extend(
        [
            "",
            "## Operation Counts",
            "",
            "| Operation | ok | skip | error |",
            "| --- | ---: | ---: | ---: |",
        ],
    )
    for op in sorted(op_counts):
        counts = op_counts[op]
        lines.append(f"| `{op}` | {counts['ok']} | {counts['skip']} | {counts['error']} |")

    lines.extend(
        [
            "",
            "## Slowest Rows",
            "",
            "| Operation | Layout | Shape | Dtype | Median ms |",
            "| --- | --- | --- | --- | ---: |",
        ],
    )
    for median, row in sorted(slowest, key=lambda item: item[0], reverse=True)[:10]:
        lines.append(
            f"| `{row.get('op', 'unknown')}` | `{row.get('layout', '?')}` | "
            f"`{_shape_text(row)}` | `{row.get('dtype', '?')}` | {median:.6g} |",
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    summary = _summarize(args.json_path)
    if args.output_md is not None:
        args.output_md.write_text(summary)
    else:
        sys.stdout.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
