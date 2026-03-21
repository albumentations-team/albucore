#!/usr/bin/env python3
"""Build a Markdown report comparing two ``benchmark_router_synthetic.py`` JSON outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _key(r: dict[str, object]) -> tuple[str, str, str, str]:
    shape = r["shape"]
    sh = ",".join(str(x) for x in shape) if isinstance(shape, list) else str(shape)
    return (str(r["op"]), str(r["layout"]), sh, str(r["dtype"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("new_json", type=Path)
    p.add_argument("old_json", type=Path)
    p.add_argument("out_md", type=Path)
    args = p.parse_args()

    new_d = json.loads(args.new_json.read_text(encoding="utf-8"))
    old_d = json.loads(args.old_json.read_text(encoding="utf-8"))

    new_m = {_key(r): r for r in new_d["rows"]}
    old_m = {_key(r): r for r in old_d["rows"]}
    keys = sorted(set(new_m) & set(old_m))

    table_lines: list[str] = []
    ratios: list[tuple[tuple[str, str, str, str], float]] = []
    by_op: dict[str, list[float]] = defaultdict(list)

    for k in keys:
        a, b = new_m[k], old_m[k]
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        nm = a["ms_median"]
        om = b["ms_median"]
        if nm is None or om is None or nm <= 0:
            continue
        ratio = om / nm
        ratios.append((k, ratio))
        by_op[k[0]].append(ratio)
        op, layout, sh, dt = k
        table_lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {nm:.4f} | {om:.4f} | {ratio:.2f}x |")

    regress = [(k, r) for k, r in ratios if r < 0.85]
    wins = [(k, r) for k, r in ratios if r > 1.15]
    regress.sort(key=lambda x: x[1])
    wins.sort(key=lambda x: -x[1])

    med_lines = []
    for op in sorted(by_op):
        xs = sorted(by_op[op])
        med = xs[len(xs) // 2]
        med_lines.append(f"- `{op}`: median **old/new** = **{med:.2f}x** ({len(xs)} cells)")

    lines = [
        "# Router synthetic benchmark — comparison",
        "",
        f"- **New:** `{new_d['meta'].get('distribution_version')}` ({args.new_json.name})",
        f"- **Old:** `{old_d['meta'].get('distribution_version')}` ({args.old_json.name})",
        "",
        "## Summary",
        "",
        f"- Comparable **ok/ok** cells: **{len(ratios)}**",
        f"- **New slower** (old/new &lt; 0.85): **{len(regress)}** cells",
        f"- **New faster** (old/new &gt; 1.15): **{len(wins)}** cells",
        "",
        "Ratio **old_ms / new_ms**: **&gt;1** ⇒ new build faster on that cell.",
        "",
        "### Median old/new by op",
        "",
        *med_lines,
        "",
        "### Largest regressions (new slower)",
        "",
        "| op | layout | shape | dtype | old/new |",
        "|----|--------|-------|-------|--------:|",
    ]
    for k, r in regress[:25]:
        op, layout, sh, dt = k
        lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {r:.2f}x |")
    if not regress:
        lines.append("| — | — | — | — | — |")

    lines += [
        "",
        "### Largest wins (new faster)",
        "",
        "| op | layout | shape | dtype | old/new |",
        "|----|--------|-------|-------|--------:|",
    ]
    for k, r in wins[:25]:
        op, layout, sh, dt = k
        lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {r:.2f}x |")
    if not wins:
        lines.append("| — | — | — | — | — |")

    lines += [
        "",
        "## Full table",
        "",
        "Only rows where **both** runs are `ok`.",
        "",
        "| op | layout | shape | dtype | new_ms | old_ms | old/new |",
        "|----|--------|-------|-------|-------:|-------:|--------:|",
        *table_lines,
        "",
    ]

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
