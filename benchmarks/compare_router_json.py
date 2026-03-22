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


def _fmt_ms_pm_std(row: dict[str, object]) -> str:
    """Median ms with ± sample-std when present (error bar from repeated timings)."""
    m = row.get("ms_median")
    if m is None:
        return "—"
    med = float(m)
    std = row.get("ms_std")
    if isinstance(std, (int, float)) and float(std) > 0:
        return f"{med:.4f} ± {float(std):.4f}"
    return f"{med:.4f}"


def _fmt_mad(row: dict[str, object]) -> str:
    mad = row.get("ms_mad")
    if mad is None:
        return "—"
    return f"{float(mad):.4f}"


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

    new_only_ok: list[tuple[tuple[str, str, str, str], dict[str, object]]] = []
    for k, row in new_m.items():
        if row.get("status") != "ok":
            continue
        old_row = old_m.get(k)
        if old_row is None or old_row.get("status") != "ok":
            new_only_ok.append((k, row))

    old_only_ok: list[tuple[tuple[str, str, str, str], dict[str, object]]] = []
    for k, row in old_m.items():
        if row.get("status") != "ok":
            continue
        new_row = new_m.get(k)
        if new_row is None or new_row.get("status") != "ok":
            old_only_ok.append((k, row))

    new_only_ok.sort(key=lambda x: x[0])
    old_only_ok.sort(key=lambda x: x[0])

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
        ratio = float(om) / float(nm)
        ratios.append((k, ratio))
        by_op[k[0]].append(ratio)
        op, layout, sh, dt = k
        table_lines.append(
            f"| {op} | {layout} | ({sh}) | {dt} | {_fmt_ms_pm_std(a)} | {_fmt_ms_pm_std(b)} | "
            f"{_fmt_mad(a)} | {_fmt_mad(b)} | {ratio:.2f}x |",
        )

    regress = [(k, r) for k, r in ratios if r < 0.85]
    wins = [(k, r) for k, r in ratios if r > 1.15]
    regress.sort(key=lambda x: x[1])
    wins.sort(key=lambda x: -x[1])

    med_lines = []
    for op in sorted(by_op):
        xs = sorted(by_op[op])
        med = xs[len(xs) // 2]
        med_lines.append(f"- `{op}`: median **old/new** = **{med:.2f}x** ({len(xs)} cells)")

    nr = new_d["meta"].get("repeats", "?")
    nw = new_d["meta"].get("warmup", "?")
    or_ = old_d["meta"].get("repeats", "?")
    ow = old_d["meta"].get("warmup", "?")

    nv = new_d["meta"].get("distribution_version", "?")
    ov = old_d["meta"].get("distribution_version", "?")
    nl = new_d["meta"].get("benchmark_label")
    ol = old_d["meta"].get("benchmark_label")
    nsk = new_d["meta"].get("skip_ops", [])
    osk = old_d["meta"].get("skip_ops", [])
    lines = [
        "# Router synthetic benchmark — comparison",
        "",
        f"- **New (current run):** `{args.new_json.name}` — `distribution_version` = `{nv}`",
        f"- **Old (baseline):** `{args.old_json.name}` — `distribution_version` = `{ov}`",
    ]
    if nl:
        lines.append(f"- **New label:** `{nl}`")
    if ol:
        lines.append(f"- **Old label:** `{ol}`")
    if nsk:
        lines.append(f"- **New `skip_ops`:** `{nsk}`")
    if osk:
        lines.append(f"- **Old `skip_ops`:** `{osk}`")
    lines += [
        "",
        "## Summary",
        "",
        f"- Comparable **ok/ok** cells: **{len(ratios)}**",
        f"- **New slower** (old/new &lt; 0.85): **{len(regress)}** cells",
        f"- **New faster** (old/new &gt; 1.15): **{len(wins)}** cells",
        "",
        "Ratio **old_ms / new_ms** (medians): **&gt;1** ⇒ new build faster on that cell.",
        "",
        f"- **New** run: **repeats={nr}**, **warmup={nw}**",
        f"- **Old** run: **repeats={or_}**, **warmup={ow}**",
        "",
        "**Error bars:** `ms_median ± ms_std` from repeated wall-time samples (see JSON). "
        "**MAD** = median absolute deviation from the median (robust).",
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
        "## New build only (`ok` in new, not `ok` in old or missing in old)",
        "",
        "Routers / shapes introduced or fixed since baseline; not part of **old/new** ratio table above.",
        "",
        "| op | layout | shape | dtype | new_ms | old status |",
        "|----|--------|-------|-------|--------:|------------|",
    ]
    for k, row in new_only_ok[:80]:
        op, layout, sh, dt = k
        old_row = old_m.get(k)
        ost = old_row.get("status", "missing") if old_row is not None else "missing"
        lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {_fmt_ms_pm_std(row)} | {ost} |")
    if len(new_only_ok) > 80:
        lines.append(f"| … | | | | *{len(new_only_ok) - 80} more rows* | |")

    lines += [
        "",
        "## Baseline only (`ok` in old, not `ok` in new or missing in new)",
        "",
        "| op | layout | shape | dtype | old_ms | new status |",
        "|----|--------|-------|-------|--------:|------------|",
    ]
    for k, row in old_only_ok[:40]:
        op, layout, sh, dt = k
        new_row = new_m.get(k)
        nst = new_row.get("status", "missing") if new_row is not None else "missing"
        lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {_fmt_ms_pm_std(row)} | {nst} |")
    if len(old_only_ok) > 40:
        lines.append(f"| … | | | | *{len(old_only_ok) - 40} more rows* | |")

    lines += [
        "",
        "## Full table",
        "",
        "Only rows where **both** runs are `ok`. Times are **median ± σ** (ms) when `ms_std` is present.",
        "",
        "| op | layout | shape | dtype | new_ms | old_ms | new_MAD | old_MAD | old/new |",
        "|----|--------|-------|-------|--------|--------|--------:|--------:|--------:|",
        *table_lines,
        "",
    ]

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
