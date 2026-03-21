#!/usr/bin/env python3
"""Build a Markdown table comparing two ``benchmark_router_synthetic.py`` JSON outputs."""

from __future__ import annotations

import argparse
import json
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

    new_m = { _key(r): r for r in new_d["rows"] }
    old_m = { _key(r): r for r in old_d["rows"] }
    keys = sorted(set(new_m) & set(old_m))

    lines = [
        "# Router synthetic benchmark — comparison",
        "",
        f"- **New:** `{new_d['meta'].get('distribution_version')}` ({args.new_json.name})",
        f"- **Old:** `{old_d['meta'].get('distribution_version')}` ({args.old_json.name})",
        "",
        "Ratio **old_ms / new_ms** (>1 ⇒ new build faster on that cell). Only rows where **both** runs are `ok`.",
        "",
        "| op | layout | shape | dtype | new_ms | old_ms | old/new |",
        "|----|--------|-------|-------|-------:|-------:|--------:|",
    ]

    for k in keys:
        a, b = new_m[k], old_m[k]
        if a["status"] != "ok" or b["status"] != "ok":
            continue
        nm = a["ms_median"]
        om = b["ms_median"]
        if nm is None or om is None or nm <= 0:
            continue
        ratio = om / nm
        op, layout, sh, dt = k
        lines.append(f"| {op} | {layout} | ({sh}) | {dt} | {nm:.4f} | {om:.4f} | {ratio:.2f}× |")

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
