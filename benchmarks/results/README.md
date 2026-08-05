# Benchmark results

This directory stores generated benchmark output used during PR review and
routing changes. The benchmark scripts are the source of truth; committed
Markdown and JSON files are evidence for a particular run, not permanent API
documentation.

## Generate current results

Run from the repository root:

```bash
uv run python benchmarks/benchmark_router_synthetic.py \
  --repeats 41 --warmup 12 \
  --output-json benchmarks/results/router-current.json

uv run python benchmarks/compare_router_json.py \
  benchmarks/results/router-current.json \
  benchmarks/results/router-previous.json \
  benchmarks/results/REPORT_router_current_vs_previous.md
```

For an advisory PR run, use fewer repetitions. For a release or routing
decision, keep the full matrix and record the hardware, dependency versions,
thread settings, shapes, dtypes, warmups, repetitions, and correctness checks
in the generated report.

## Deep-dive output

Backend-specific scripts may write Markdown under this directory. Keep a report
only when it supports a current routing decision, a reproducible regression
baseline, or an open benchmark question. If the decision changes, replace the
old report or remove it; do not accumulate version-to-version narratives here.

The maintained scripts and their workloads are listed in
[`benchmarks/README.md`](../README.md). Current routing summaries belong in
[`docs/numkong-performance.md`](../../docs/numkong-performance.md) and
[`docs/performance-optimization.md`](../../docs/performance-optimization.md).
