# NumKong route decisions

Albucore chooses NumPy, OpenCV, LUT, NumKong, and eager CPU Torch by measured
full-path performance. This page records the current NumKong routes and the
benchmark questions that can change them. It is a decision guide, not an archive
of one machine's timings.

## Current production routes

| Operation | NumKong route | Scope and fallback |
|---|---|---|
| `add_weighted` | `nk.blend` | uint8 and single-channel float32. Multi-channel float32 uses OpenCV for HWC/two contiguous inputs and NumKong for the measured strided higher-rank region. |
| `pairwise_distances_squared` | `nk.cdist(metric="sqeuclidean")` | `n1 * n2 < 1000`; larger point sets use the NumPy vectorized formula. |
| Global uint8 `mean`, `std`, `mean_std` | `nk.moments` | One contiguous reduction over all elements. |
| Per-channel `mean`, `std`, `mean_std` | `nk.moments` where measured | `mean` keeps its OpenCV/NumKong/NumPy split. `std` and `mean_std` use OpenCV for 3D float32 RGB/RGBA-like inputs and NumKong for uint8, single-channel, high-channel, and supported batch/volume cases. |
| `multiply_add` | `nk.scale` | Scalar float32 operands use NumKong in the measured region; large eligible arrays may use the Torch CPU route. Vector and full-array operands use NumPy. |
| `add_array` | `add_array_numkong` (`nk.blend`) | Same-shape, same-dtype uint8 arrays. Float32 uses NumPy; other uint8 layouts use OpenCV. |
| `from_float` (`float32` → `uint8`) | `nk.astype(..., out=...)` | The fallback uses one NumKong scale-and-round work buffer followed by NumKong's saturating, round-to-even cast; it beat the prior NumPy `rint`+clip route across the HWC/XHWC contiguous and strided matrix. |

The public routers preserve their documented dtype, layout, and aliasing
contracts. A dispatch branch is implementation routing; it is not a second
validation pass. Callers validate dtype, shape, layout, and control parameters
before entering Albucore.

## Routes that stay outside NumKong

| Operation | Current backend | Reason to keep the route |
|---|---|---|
| Global float32 `mean`, `std`, `mean_std` | NumPy | NumPy's float64 accumulators are faster or provide the required semantics on the supported matrix. |
| `min`, `max`, and paired global min/max | NumPy | The current NumKong candidate has no stable full-path win. Re-run the benchmark if NumKong or the target CPU changes. |
| `multiply_by_constant` | NumPy for float32, LUT for uint8 | The measured NumKong helper does not beat the production routes after allocation and saturation costs. |
| `add_constant` | NumPy for float32, OpenCV for uint8 | Allocating `nk.scale(1, beta)` has no stable win. |
| `multiply_by_array` | NumPy for float32, OpenCV plus saturation for uint8 | `nk.fma` does not win the complete operation. |
| `multiply_by_vector` and `add_vector` | LUT/OpenCV | A per-channel `scale` loop loses to the existing routes on the large float32 cells. |
| `exp`, `log`, `sqrt` | NumPy/OpenCV | NumKong exposes no matching elementwise kernels. A NumKong `minmax` guard makes `log` slower and needs extra special-value handling. |

## Reproduce a routing decision

Run the complete public path and compare it with every viable backend. Keep the
workload, thread counts, allocation mode, correctness tolerance, and caller
preconditions identical.

```bash
uv run python benchmarks/benchmark_add_weighted.py
uv run python benchmarks/benchmark_numkong_vs_albucore_backends.py
uv run python benchmarks/benchmark_multiply_add_numkong.py
uv run python benchmarks/benchmark_stats.py
uv run python benchmarks/benchmark_elementwise.py
uv run python benchmarks/benchmark_minmax_ravel.py
uv run python benchmarks/benchmark_numkong_astype.py
```

The [NumKong buffer-first cast report](../benchmarks/results/benchmark_numkong_astype.md) records the accepted
`from_float` fallback across HWC/XHWC contiguous and strided inputs.

For a new route, record the benchmark date, hardware, dependency versions,
shapes, channels, dtypes, strides, thread settings, warmups, repetitions,
correctness tolerance, and rejected candidates. Save large result tables under
`benchmarks/results/`; keep this page limited to the decision and the reason.

Use [`docs/performance-optimization.md`](performance-optimization.md) for the
full acceptance policy. In particular, accept a NumKong route only when the
complete connected region wins and the route does not add a material regression
near its boundary.

## Open questions

These are future benchmark tasks, not promises to change production routing:

- compare a fused float32 global `mean_std` with the current NumPy two-pass path;
- repeat min/max and multiply/add sweeps after NumKong or OpenCV upgrades;
- measure whether a NumKong route still wins after a caller-to-Albucore
  conversion or a required output allocation;
- add a route only when the saved full-path report supports a stable region.

## Related references

- [`docs/performance-optimization.md`](performance-optimization.md) — general
  routing and benchmark workflow;
- [`docs/torch-performance-optimization.md`](torch-performance-optimization.md)
  — eager CPU Torch candidates and bridge costs;
- [`benchmarks/README.md`](../benchmarks/README.md) — maintained benchmark
  scripts and result format;
- [`docs/public-api.md`](public-api.md) — public routers and internal shims.
