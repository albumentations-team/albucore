# Benchmark output

Committed artifacts (regenerate before releases or when routing changes):

- `router_synthetic_current.json` — full grid from the **editable** tree (`benchmark_router_synthetic.py`).
- `router_synthetic_0.0.40.json` — same script, **albucore 0.0.40** in a separate venv (see below).
- `REPORT_router_compare.md` — `compare_router_json.py` **new** then **old** (ratio = old_median / new_median; **&gt;1** means new is faster).

Each JSON row (when `status: ok`) includes **`ms_median`**, **`ms_mean`**, **`ms_std`** (sample std of timed runs — use as error bar), **`ms_mad`**, **`timing_n`**. Meta records **`repeats`** and **`warmup`**.

**Current tree** (recommended: higher repeats for stable medians):

```bash
uv run python benchmarks/benchmark_router_synthetic.py \
  --repeats 21 --warmup 5 \
  --output-json benchmarks/results/router_synthetic_current.json
```

**Older wheel** (run from `/tmp` so the repo’s `albucore/` is not on `sys.path`):

```bash
cd /tmp && /path/to/venv-with-0.0.40/bin/python /path/to/repo/benchmarks/benchmark_router_synthetic.py \
  --repeats 21 --warmup 5 \
  --output-json /path/to/repo/benchmarks/results/router_synthetic_0.0.40.json
```

**Compare** (first arg = new, second = old):

```bash
uv run python benchmarks/compare_router_json.py \
  benchmarks/results/router_synthetic_current.json \
  benchmarks/results/router_synthetic_0.0.40.json \
  benchmarks/results/REPORT_router_compare.md
```
