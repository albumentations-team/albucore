# Performance Policy

Albucore backend routing is benchmark-driven. Performance checks protect public atomic operations
from silent regressions.

## Benchmark Scope

Use `benchmarks/benchmark_router_synthetic.py` as the public-router regression guard. It times
routers exported from `albucore.functions.__all__` and can include public geometric routers with
`--with-geometric`.

Use targeted scripts for backend-specific routing evidence:

- `benchmarks/benchmark_numkong_vs_albucore_backends.py`
- `benchmarks/benchmark_lut_shared_routing.py`
- `benchmarks/benchmark_scale_vs_lut.py`
- `benchmarks/benchmark_stats.py`
- `benchmarks/benchmark_multiply_add_numkong.py`

Benchmark grids must include non-square H/W sizes so height-width swaps are visible.

## Modes

PR quick mode:

- advisory only;
- non-square HWC sizes;
- channels 1 and 3;
- `uint8` and `float32`;
- quick public-router registry.

Scheduled/release mode:

- compares current code to a published PyPI release baseline;
- includes larger non-square HWC sizes;
- includes channels 1, 3, and 9;
- includes stats batch cases;
- includes geometric routers when practical.

## Baselines

- Attach release baseline JSON to each validated release-candidate artifact bundle.
- Compare scheduled `main` runs against the selected PyPI release baseline.
- Compare release candidates against the previous published PyPI release.
- Keep raw noisy artifacts in workflow artifacts.
- Update committed baselines only intentionally, with a PR explanation.

## Thresholds

Start advisory on hosted runners.

PR warning:

- more than 15% slowdown on a release-blocking hot-path cell;
- more than 10% median slowdown across a router family;
- more than 25% slowdown on one noisy cell remains warning-only until repeated.

Release review:

- more than 5% slowdown on a release-blocking hot path requires explanation;
- more than 10% slowdown blocks release unless accepted by a maintainer;
- large memory regression blocks release unless intentional and documented.

Initial release-blocking hot paths:

- `normalize`, `normalize_per_image`
- `apply_uint8_lut`, `sz_lut`
- `add`, `add_array`, `add_vector`, `add_weighted`
- `multiply`, `multiply_by_array`, `multiply_by_vector`, `multiply_add`
- `mean`, `std`, `mean_std`, `reduce_sum`
- `to_float`, `from_float`
- `hflip`, `vflip`, `resize`, `remap`

Other public routers are reported first and promoted to blocking after baselines stabilize.

## Memory Smoke

Run `tools/run_memory_smoke.py` in advisory mode for selected hot paths:

- shared and per-channel LUT;
- `normalize` and `normalize_per_image`;
- array-valued arithmetic;
- `mean_std`;
- geometry with more than four channels.

The first implementation uses `tracemalloc` peak bytes. RSS or ASV memory checks can be added later
if these advisory checks are not enough.
