# Performance Policy

Albucore backend routing is benchmark-driven. Performance checks protect public atomic operations
from silent regressions.

## Benchmark Scope

Use `benchmarks/benchmark_router_synthetic.py` as the public-router regression guard. It times
operations exported from `albucore.__all__` and can include public geometric routers with
`--with-geometric`.

Use targeted scripts for backend-specific routing evidence:

- `benchmarks/benchmark_numkong_vs_albucore_backends.py`
- `benchmarks/benchmark_lut_backends.py`
- `benchmarks/benchmark_scale_vs_lut.py`
- `benchmarks/benchmark_stats.py`
- `benchmarks/benchmark_numkong_arithmetic.py`

Benchmark grids must include non-square H/W sizes so height-width swaps are visible.

## Modes

PR mode:

- advisory in CI;
- non-square HWC sizes;
- channels 1 and 3;
- `uint8`, `float16`, and `float32`;
- quick public-router registry.

Targeted local mode:

- use the full or targeted scripts relevant to the route being changed;
- include larger non-square HWC sizes when the change is shape-sensitive;
- include channels 1, 3, and 9 when OpenCV channel limits or high-channel paths are relevant;
- include stats batch cases for stats routing changes;
- include geometric routers for geometry routing changes.

## Same-dtype acceptance and dtype diagnostics

Compare baseline and candidate with the same input dtype, container, layout, controls, and allocation contract.
Correctness, runtime, and memory acceptance apply independently on Linux x86-64 and macOS ARM64.
A substantial code simplification may slow an individual public operation by at most 5% only when numerical accuracy
is preserved, memory does not increase, and the full AlbumentationsX route has no sustained slowdown.
The 5% allowance does not apply to the complete AX pipeline or to memory growth.
Fixed resident library-code pages are reported separately when a memory map identifies them. The no-growth gate
continues to cover inputs, outputs, workspace, backend heap, and retained arrays; it is not a general RSS allowance.

Float16 remains fully supported. Compare complete float16 and float32 routes on equivalent logical data as mandatory
optimization diagnostics, including casts, bridges, allocations, and output restoration. This cross-dtype comparison
is not an independent blocker for architecture migration. Missing fast kernels remain actionable upstream work;
they do not justify removing float16 support.

## Baselines

- PR benchmarks compare against the PR target branch.
- Keep raw noisy artifacts in workflow artifacts.
- Update committed baselines only intentionally, with a PR explanation.

## Thresholds

Start advisory on hosted runners.

PR warning:

- more than 15% slowdown on a hot-path cell;
- more than 10% median slowdown across a router family;
- more than 25% slowdown on one noisy cell remains warning-only until repeated.

Initial hot paths:

- `normalize`, `normalize_per_image`
- `apply_uint8_lut`
- `add` with scalar/channel/field controls, `add_weighted`
- `multiply` with scalar/channel/field controls, `multiply_add`
- `mean`, `std`, `mean_std`, `reduce_sum`
- `to_float`, `from_float`
- `hflip`, `vflip`, `resize`, `remap`

Other public routers are reported first and reviewed when baselines stabilize.

## Memory Smoke

Run `tools/run_memory_smoke.py` in advisory mode for selected hot paths:

- shared and per-channel LUT;
- `normalize` and `normalize_per_image`;
- array-valued arithmetic;
- `mean_std`;
- geometry with more than four channels.

The advisory smoke uses `tracemalloc` peak bytes. Architecture acceptance also requires live-buffer accounting,
fresh-process peak RSS, and steady RSS after repeated calls on both target platforms. Tracemalloc alone does not
account for every native allocation, and subtracting cumulative RSS high-watermarks does not give an exact kernel
allocation size.
