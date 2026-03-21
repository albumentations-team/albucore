# Benchmark output

Committed artifacts (regenerate before releases or when routing changes):

- `router_synthetic_0.0.41.json` / `router_synthetic_0.0.40.json` — full grid from `benchmark_router_synthetic.py` (includes **1024×1024** HWC).
- `REPORT_router_compare.md` — `compare_router_json.py` **new** then **old** (ratio = old_ms / new_ms; **&gt;1** means new is faster).

**Current tree** (editable install):

```bash
uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router_synthetic_0.0.41.json
```

**Older wheel** (run from `/tmp` so the repo’s `albucore/` is not on `sys.path`; use a venv with that version + `opencv-python-headless`):

```bash
cd /tmp && /path/to/venv-with-0.0.40/bin/python /path/to/repo/benchmarks/benchmark_router_synthetic.py \
  --output-json /path/to/repo/benchmarks/results/router_synthetic_0.0.40.json
```

**Compare** (first arg = new, second = old):

```bash
uv run python benchmarks/compare_router_json.py \
  benchmarks/results/router_synthetic_0.0.41.json \
  benchmarks/results/router_synthetic_0.0.40.json \
  benchmarks/results/REPORT_router_compare.md
```
