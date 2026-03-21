---
name: albucore-benchmarks
description: Running albucore micro-benchmarks under benchmarks/, synthetic router timings, comparing PyPI releases with uv --no-project. Use when adding benchmarks, comparing performance across versions, or documenting benchmark workflow.
---

# Albucore benchmarks

## Layout

- **`benchmarks/`** — Python timing scripts (synthetic + NumKong tables). Run from **repo root**: `uv run python benchmarks/<script>.py`.
- **`benchmarks/timing.py`** — shared `median_ms` for scripts executed as `python benchmarks/foo.py` (puts `benchmarks/` on `sys.path[0]`).
- **`./benchmark.sh`** — dataset-driven runner; expects an external `benchmark` package (not always present in-tree). Prefer synthetic scripts for CI-style checks.
- **`benchmarks/benchmark_router_synthetic.py`** — times **public routers** on synthetic `uint8` / `float32` arrays: `HWC`, plus **`NHWC` for `mean` / `std` / `mean_std` only** (OpenCV paths are 3D).
- **`benchmarks/compare_router_json.py`** — Markdown table from two JSON outputs.

## Compare current tree vs PyPI release

```bash
# Editable / workspace albucore
uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-main.json

# Pinned wheel (no workspace project)
uv run --no-project --with albucore==0.0.40 --with opencv-python-headless \
  --with simsimd --with stringzilla --with numpy \
  python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-0.0.40.json

uv run python benchmarks/compare_router_json.py benchmarks/results/router-main.json \
  benchmarks/results/router-0.0.40.json benchmarks/results/REPORT_router_compare.md
```

Use **`--quick`** for smaller shape/channel grids while iterating.

## Docs

- NumKong tables & methodology: `docs/numkong-performance.md`
- Research notes (archived writeups): `docs/research/`
