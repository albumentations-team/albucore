# Benchmark output (local)

JSON / comparison reports from [`benchmark_router_synthetic.py`](../benchmark_router_synthetic.py) are gitignored by default.

Example:

```bash
uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-main.json
uv run --no-project --with albucore==0.0.40 --with opencv-python-headless \\
  --with simsimd --with stringzilla --with numpy \\
  python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-0.0.40.json

uv run python benchmarks/compare_router_json.py benchmarks/results/router-main.json \\
  benchmarks/results/router-0.0.40.json benchmarks/results/REPORT_router_compare.md
```
