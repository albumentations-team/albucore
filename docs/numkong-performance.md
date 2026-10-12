# NumKong route decisions

Albucore chooses NumPy, OpenCV, LUT, NumKong, and eager CPU Torch by measured
full-path performance. This page records the current NumKong routes and the
benchmark questions that can change them. It is a decision guide, not an archive
of one machine's timings.

## Current production routes

| Operation | NumKong route | Scope and fallback |
|---|---|---|
| `add_weighted` | `nk.blend` for uint8 and single-channel float32; capability-gated NumKong half blend | Multi-channel float32 uses OpenCV for HWC/two contiguous inputs and NumKong for the measured strided higher-rank region. Contiguous float16 inputs with `neonhalf=true`, at least 20,480 elements, and weights `0.5, 0.5` use NumKong; the 21-sample HWC grid measured 0.52–0.55× float32 time for C=3/9. Other float16 controls retain the CV16F/float32 routes. A direct Torch half candidate was fast for normalized random data but failed one-ULP parity under extreme cancellation; a CV16F→CV32F OpenCV candidate was 2.61–3.30× slower. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503). |
| `pairwise_distances_squared` | `nk.cdist(metric="sqeuclidean")` | `n1 * n2 < 1000`; larger point sets use the NumPy vectorized formula. |
| Global uint8 `mean`, `std`, `mean_std` | `nk.moments` | One contiguous reduction over all elements. |
| Global float16 `mean` | OpenCV `CV_32F mean` after one widening pass | Contiguous arrays with at least 4,096 elements; smaller or non-contiguous inputs use NumPy float64. Direct `CV_16F mean` lost low-order contributions on skewed subnormal inputs and exceeded tolerance on a positive near-constant C=1 input without subnormals. |
| Global NumPy float16 `reduce_sum` | CPU Torch `sum(dtype=float64)` over a zero-copy NumPy view | Contiguous, writable inputs with 20,480+ elements and `keepdims=False`. A 21-cell, 101-sample HWC/NHWC grid matched NumPy float64 bitwise, ranged from 0.968–1.106× float32 time, and had no robust slow cell. Other layouts and `keepdims=True` use the existing accumulator route. See [`reduce_sum_global_torch64_public_full_101_20261011.json`](../_internal/float16_probe_20261010/reduce_sum_global_torch64_public_full_101_20261011.json). |
| Global float16 Tensor `mean` | CPU Torch `mean(dtype=float32)` | C>1 and at least 40,960 elements; returns float64. C=1 keeps the faster measured NumPy route. |
| Global float16 Tensor `mean_std` | OpenCV `CV_16F meanStdDev` | Contiguous channel-first storage with at least 20,480 elements is flattened as a zero-copy scalar view; near-constant and non-finite inputs widen to NumPy float64. |
| Global float16 Tensor `std` | NumPy float32 accumulator in range; Torch float64 fallback | Contiguous tensors with at least 40,960 elements use the float32 accumulator only when input values are within [-1, 1] and std exceeds 1e-3. Other ranges use a two-pass Torch float64 reduction; non-contiguous inputs retain NumPy float64. |
| Global float16 `std`, `mean_std` | OpenCV `CV_16F meanStdDev` where measured | Contiguous HWC/NHWC arrays with C=2–4 use one direct statistics call; C=1 uses measured size windows. Near-constant, non-finite, and unmeasured layouts use NumPy float64 reductions. |
| Per-channel `mean` | OpenCV `CV_32F mean` after widening for float16 HWC C≤4; zero-copy Torch `mean(dtype=float64)` for contiguous rank-4 float16 C=3; Torch global `sum(dtype=float64)` for contiguous rank-4 float16 C=1 | Direct CV16F mean can lose precision on subnormal and near-constant values. The rank-4 C=3 route applies from 20,480 elements; C=1 uses the scalar-sum bridge from 20,480 through 8,000,000 elements. On the current M4 Max stats grid, HWC C=1/3 widening remains 1.41–1.57× slower than float32; C9 is 1.21–1.27×. Direct NumPy float32 accumulation, CV32F reduction, and Torch alternatives did not win in the tested public regions. A temporary NumKong PR #400 moments route with the subnormal scan bypassed measured HWC C=1 mean at 0.93–1.02× float32, but C=3 strided means remained 1.15–1.17× and C=9 nine-call means were 4.85–5.00×, and existing `nk.sum(axis=(0,1))` was 4.82–4.93×; a C3 planar copy was 3.03–3.33×. These candidates met `rtol=1e-7` but only C1 passes the no-slower screen; do not enable before PR #400 is released. Grouped axis-aware `moments` is tracked in [NumKong #311](https://github.com/ashvardanian/NumKong/issues/311#issuecomment-6112109242). Data: [`mean_per_channel_c1_c3_numkong_pr400_public_101_20261011.json`](../_internal/float16_probe_20261010/mean_per_channel_c1_c3_numkong_pr400_public_101_20261011.json), [`mean_per_channel_c9_numkong_pr400_public_101_20261011.json`](../_internal/float16_probe_20261010/mean_per_channel_c9_numkong_pr400_public_101_20261011.json), and [`mean_per_channel_c1_c3_planar_numkong_pr400_public_101_20261011.json`](../_internal/float16_probe_20261010/mean_per_channel_c1_c3_planar_numkong_pr400_public_101_20261011.json). Float32 rank-4 C=2–4 continues to use NumKong `nk.sum`; see [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| Per-channel `std`, `mean_std` | Capability-gated `nk.moments` for contiguous float16 HWC C≤4; matching float32 NumKong/OpenCV route for inputs containing subnormals | The float16 moments kernel loses subnormal contributions on NEON-half, so Albucore scans half bit patterns before selecting it. Subnormal-containing inputs widen to float32 and mirror the float32 backend; near-constant inputs use NumPy float64. On the 101-sample M4 Max stats grid, 67/168 pairs remain robustly >1.10× float32, including HWC C=3 `mean_std_per_channel` at 1.41–1.59×. Accuracy and performance are tracked in [NumKong #399](https://github.com/ashvardanian/NumKong/issues/399); candidate F64 NEON accumulators are in the open [PR #400](https://github.com/ashvardanian/NumKong/pull/400), measured at 0.25–0.31× a vectorized F16→F32 moments bridge for contiguous and HWC C=3 strided kernels. Albucore keeps its fallback until the fix is merged and released. A temporary full-public-route probe with the PR #400 wheel bypassed only the subnormal scan and tested HWC C=1/2/3/4 `std` and `mean_std` at four sizes: all 32 cells were 0.27–0.86× float32 with no robust slow pair; maximum error versus NumPy float64 was 1.37e-10 for means and 1.67e-8 for std. A separate positive-range `[0.1,0.9]` check with signed subnormal sentinels passed the mean `rtol=1e-7` reference in all 16 HWC C=1–4 cells; maximum mean/std errors were 2.38e-8/6.66e-8, within the current `mean_std` test tolerances. Results are in [`stats_router_numkong_pr400_positive_accuracy_20261011.json`](../_internal/float16_probe_20261010/stats_router_numkong_pr400_positive_accuracy_20261011.json). The candidate is not enabled in the released-dependency route; see [NumKong #399 benchmark](https://github.com/ashvardanian/NumKong/issues/399), [positive-range accuracy follow-up](https://github.com/ashvardanian/NumKong/issues/399), and [`stats_router_numkong_pr400_moments_full_101_20261011.json`](../_internal/float16_probe_20261010/stats_router_numkong_pr400_moments_full_101_20261011.json). |
| Per-channel float16 `reduce_sum` | Torch float64 accumulation in measured HWC windows and contiguous rank-4 channel-last C=1 arrays up to 307,200 elements without `keepdims`, plus C=3 arrays up to 245,760 elements; OpenCV `CV_64F reduce` after widening for C=9 beyond the Torch window | Rank-4 C=3 is exact and 4.4–5.3× faster than the float32 peer; rank-4 C=1 is exact and 0.965–1.006× float32 time in the tested N=1–4, 128×160 and 240×320 windows. `cv2.reduce` rejects CV_16F; the current correct C=9 route remains 1.21–1.30× slower than the float32 peer, and tested direct Torch/NumKong/NumPy alternatives were slower. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| NumPy float16 `median_blur` | Ordered binary16 keys through OpenCV `CV_16U` | `ksize=5`, HWC C=1, at least 76,800 elements. NaNs and zero medians from mixed signed-zero inputs use the float32 route; every other region keeps its existing float32 or quantized path. |
| Float16 Tensor `reduce_sum` | CPU Torch `sum(dtype=float64)` | Contiguous inputs of 16,384+ elements; global, per-channel, and explicit-axis reductions return the existing float64 result without the NumPy layout bridge. The paired grid has no cell above 1.10× and shows >5% speedups in 11/21 global and 16/21 per-channel pairs. |
| `multiply_add` | `nk.scale` | Float32 scalars use NumKong in its measured region. Contiguous float16 NumPy inputs also use it when `neonhalf=true`, both scalar controls are exactly representable in float16, and the image has at least 20,480 elements. Other float16 controls use the measured Torch/NumPy path; vector and full-array controls do not select the half NumKong route. |
| NumPy HWC C=1 float16 `add_vector`, `multiply_by_vector` | `nk.scale` | Contiguous writable inputs with `neonhalf=true`, 20,480+ elements, one control, and a control exactly representable in float16. Addition uses `alpha=1`; multiplication uses a negative-zero bias to preserve signed zero. For 14 exact controls, all 65,536 input bit patterns matched the float32-rounded-half result on finite outputs; NaN/Inf classifications and infinity signs also matched. The 101-sample public grid measured float16/float32 ratios of 0.836/0.667/0.563 for add and 0.866/0.685/0.561 for multiply at 128×160/240×320/480×640; a separate 768×1024 check measured 0.535/0.517. At 128×160, 1,000 calls per sample measured 0.976×/0.991× and 301 samples of 100 calls measured 0.979×/1.029×; neither showed a robust >1.10× slowdown. See [`c1_vector_numkong_public_101_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_public_101_20261011.json), [`c1_vector_numkong_128x160_batched_301_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_128x160_batched_301_20261011.json), [`c1_vector_numkong_128x160_batched_1000_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_128x160_batched_1000_20261011.json), and [`c1_c3_vector_public_768_20261011.json`](../_internal/float16_probe_20261010/c1_c3_vector_public_768_20261011.json). Direct CV16F OpenCV C1 candidates were exact on the tested finite inputs but took 2.22–5.92× float32 public time; see [the C1 benchmark](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [the C3/C9 follow-up](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). Other channels, layouts, sizes, and controls retain measured Torch/NumPy paths. |
| NumPy HWC C=3 float16 `add_vector`, `multiply_by_vector` | OpenCV `CV_16F add`/`multiply` | Contiguous writable inputs, 61,440+ elements, and three exact half controls. Exhaustive binary16 outputs matched the float32-rounded-half reference, including special-value classifications. Full public ratios at 128×160/240×320/480×640 were 0.304/0.284/0.280 for add and 0.474/0.469/0.479 for multiply; 768×1024 measured 0.270/0.450. See [`c3_vector_cv16f_public_101_20261011.json`](../_internal/float16_probe_20261010/c3_vector_cv16f_public_101_20261011.json) and [`c1_c3_vector_public_768_20261011.json`](../_internal/float16_probe_20261010/c1_c3_vector_public_768_20261011.json); the results are also posted in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).
| NumPy HWC C=9 float16 `add_vector` | NumKong `nk.add` with broadcast half controls | Contiguous writable C9 inputs, `neonhalf=true`, half-exact float32 controls, 184,320+ elements; exact binary16 route, with public ratios described below. |
| NumPy HWC C=9 float16 `multiply_by_vector` | OpenCV `CV_16F multiply` with a zero-stride control array | Contiguous writable C9 images with 184,320+ elements and finite float32 controls. Exact half controls match bitwise; inexact controls are accepted only when every magnitude is ≤1 and each nonzero magnitude is at least the float16 normal minimum, where exhaustive binary16 inputs stay within 1 ULP. Other controls keep the NumPy float32 ufunc-to-half fallback. The runtime-enabled 101-sample public grid ranged from 0.842× to 0.922× float32 time across eight sizes from 128×160 through 768×1024. See [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json), [`c9_multiply_safe_inexact_controls_public_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_multiply_safe_inexact_controls_public_101_20261011_1715.json), and [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| `add_array` | `add_array_numkong` (`nk.blend`) | Same-shape, same-dtype uint8 arrays. Float32 uses NumPy; other uint8 layouts use OpenCV. |
| `from_float` (`float32` → `uint8`) | `nk.astype(..., out=...)` | The fallback uses one NumKong scale-and-round work buffer followed by NumKong's saturating, round-to-even cast; it beat the prior NumPy `rint`+clip route across the HWC/XHWC contiguous and strided matrix. |
| NumPy `from_float` (`float16` → `uint8`) | CPU Torch float32 work buffer | Widen once, scale/round/clip in place, then cast. This preserves exact float32 rounding; six large HWC full-path pairs were 1.29–1.31× slower than float32. An exhaustive 65,536-entry conversion LUT was exact but 1.39–2.46× slower than the current half route, so it was rejected. A fused CPU conversion is tracked in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503) and the rejected LUT screen in [#97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503). |

### C9 vector arithmetic: evidence and limits

- HWC C9 `add_vector` uses NumKong `nk.add` with a zero-stride control view for finite half-exact float32 controls. On the runtime-enabled M4 Max, the 101-sample public ratios at 128×160/160×200/240×320/320×448/400×640/480×640/640×800/768×1024 were 0.935/0.932/0.899/0.899/0.881/0.877/0.883/0.857; all tested outputs matched the float32-rounded-half result bitwise. Direct OpenCV CV16F addition was slower than NumKong at all four canonical sizes. Results: [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json) and [`c9_add_cv16f_vs_numkong_runtime_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_add_cv16f_vs_numkong_runtime_101_20261011_1715.json).
- HWC C9 `multiply_by_vector` uses OpenCV `CV_16F multiply` for exact half controls and inexact controls in `[-1, 1]` outside the half-subnormal range. Exhaustive binary16 tests stay within 1 ULP and preserve NaN/Inf classifications and signed zero. The 101-sample public ratios are 0.842–0.922× float32 over eight canonical sizes.
- For non-exact controls above one, the current route uses the NumPy float32 ufunc loop to half output; `1.1` measures 1.13–1.15× float32 at 400×640/480×640/768×1024. A mixed CV16F-output / float32-control OpenCV candidate matched all half bit patterns but took 1.497–1.563× float32. Results are in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [`c9_multiply_cv16f_mixed_controls_over1_public_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_cv16f_mixed_controls_over1_public_101_20261011.json).
- NumKong candidates on the signed-zero fix branch did not close this region: mixed float16×float32 promotion plus narrowing was 2.28–2.33× float32, and `out_dtype=float16` was 5.38–5.54×. Rounding controls to half was 0.90–0.92× but changed six overflow-boundary finite/Inf classifications across the binary16 input domain. Keep the current fallback; see [NumKong #395](https://github.com/ashvardanian/NumKong/issues/395#issuecomment-6096156733) and [#401](https://github.com/ashvardanian/NumKong/issues/401). Raw data: [`c9_multiply_numkong_mixed_controls_pr402_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_numkong_mixed_controls_pr402_101_20261011.json) and [`c9_multiply_numkong_inexact_over1_pr402_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_numkong_inexact_over1_pr402_101_20261011.json).
- The signed-zero defect in direct NumKong C9 multiplication is tracked in [#401](https://github.com/ashvardanian/NumKong/issues/401#issuecomment-6111217588) and proposed for correction in open [PR #402](https://github.com/ashvardanian/NumKong/pull/402). OpenCV’s safe-control results are in [#25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json).
- OpenCV rejects a nine-channel scalar tuple. Splitting into three CV16F C3 calls was exact on random tests but 5.2–5.74× slower than float32 public calls; measurements: [`c9_vector_cv16f_chunks_101_20261011.json`](../_internal/float16_probe_20261010/c9_vector_cv16f_chunks_101_20261011.json).
The public routers preserve their documented dtype, layout, and aliasing
contracts. A dispatch branch is implementation routing; it is not a second
validation pass. Callers validate dtype, shape, layout, and control parameters
before entering Albucore.

Albucore checks `nk.get_capabilities()["neonhalf"]` once at import and enables
only measured float16 NumKong regions when that capability is present. A
different ARM64 execution environment for NumKong 7.8.5 reported
`neonhalf=False`; there, direct `nk.scale` and `nk.blend` were 6.3–19.8× slower
than the full float32 round-trip across HWC 128×160×3, 240×320×3, and
480×640×3. Those results remain documented in the upstream
[performance issue](https://github.com/ashvardanian/NumKong/issues/395#issuecomment-6096156733)
and do not describe the current M4 Max runtime, where `neonhalf=true`.

A separate one-thread C=1 `nk.scale` comparison used 101 paired samples:

- Direct float16 was 16.4–24.7× slower than the float32 round-trip on HWC
  128×160, 240×320, and 480×640; both paths exactly matched the wide reference.
- The results are in [NumKong #395](https://github.com/ashvardanian/NumKong/issues/395#issuecomment-6096156733).
- The existing resolving [PR #398](https://github.com/ashvardanian/NumKong/pull/398) remains open.

With NumKong 7.8.5 and `neonhalf=true` on an Apple M4 Max, exact half controls
were measured through the full NumPy public route. For `nk.scale` controls
`alpha=0.875, beta=0.125`, direct half took 0.28–0.51× the float32 round-trip
across HWC C=1/3/9 and the 128×160, 480×640, and 768×1024 sizes; all 65,536
binary16 values matched exactly. For `nk.blend` weights `0.5, 0.5`, direct half
took 0.32–0.51× the bridge across the same grid; the tested random and
adversarial outputs stayed within one binary16 ULP. Albucore enables the
measured `add_weighted` and scalar `multiply_add` regions only when
`neonhalf=true`, inputs are contiguous, controls are exactly representable in
half, and the size threshold is met. The paired candidate matrix is recorded in
[`numkong_half_host_101_20261011.json`](../_internal/float16_probe_20261010/numkong_half_host_101_20261011.json).

NumKong PR [#398](https://github.com/ashvardanian/NumKong/pull/398) remains an
upstream improvement for widened NEON-half `scale`/`blend`/`fma` behavior across
more controls. It is separate from the capability-gated regions Albucore has
already measured. NumKong `moments(float16)` loses subnormal contributions; the
precision issue and public-route cost are tracked in
[NumKong #399](https://github.com/ashvardanian/NumKong/issues/399). The candidate
F64 NEON accumulator and its contiguous/strided bridge benchmark are in
[PR #400](https://github.com/ashvardanian/NumKong/pull/400). It has not been
merged or released, so Albucore keeps the current correctness fallback.
The current PR head also uses float64 accumulation in its unaligned-byte and
wide-stride scalar fallback; a 3-byte-stride regression checks direct NEON and
dispatched calls against a float64 reference before the existing float32 output
cast.
An odd-byte-stride `nk_bench` case with 1,048,576 values measured the direct
fallback at 0.910 ms and F16→F32 conversion plus NEON F32 moments at 1.120 ms
(0.812× bridge time). The float32 work buffer was preallocated; calls were
single-threaded, with CPU pinning unavailable. Raw samples are in
[`numkong_f16_oddstride_bench_20261011.json`](../_internal/float16_probe_20261010/numkong_f16_oddstride_bench_20261011.json).
The binary16 ULP issue in NumKong's generic test helper is tracked separately
in [#397](https://github.com/ashvardanian/NumKong/issues/397).

OpenCV 5.0.0.93 returns incorrect `CV_16F meanStdDev` results for constant
HWC inputs at C=3/4/9. Albucore limits this direct route to C≤4 and uses a
correct fallback for higher channel counts and for near-constant C≤4 data until
[OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769)
is fixed.

Global float16 `mean` also cannot use direct CV16F accumulation: a 128×128×1
array containing one `1.0` and otherwise the smallest half subnormal differed
from the float64 mean by 4.97e-9 (8.1e-5 relative). The current route widens once
and calls CV32F `mean`; a full 21-cell, 101-sample HWC/NHWC grid matched float64
with a maximum absolute error of 4.34e-19 and took 0.696–0.876× the float32
public time. The reproducer is in
[OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

For HWC float16 C=1 `mean_per_channel`, direct `cv2.mean` on CV_16F took about
0.51× the float32 public time in a 101-sample grid, but skewed data with
smallest-subnormal residuals differed from the float64 mean by 4.97e-9. This
exceeds the current per-channel tolerance, so the float32-widened route remains
selected. Torch float64 and direct NumPy float64 reductions were also slower
than the float32 public peer across 128×160–768×1024. Reproduction and full
timings are in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

Contiguous writable rank-4 channel-last float16 per-channel `mean` uses different
measured routes: C=3 calls zero-copy Torch `mean(dtype=float64)` from 20,480 elements; C=1 calls
the global Torch float64 sum from 20,480 through 8,000,000 elements. The
101-sample public grids matched NumPy float64 exactly. C=3 took 0.14–0.23× the
float32 public time; selected C=1 cells took 0.951–1.036× with no robust slow
cell. C=1 inputs above that size cap retain the existing NumPy float64 route.
See the [C=3 grid](../_internal/float16_probe_20261010/mean_per_channel_nhwc_c3_public_101_20261011.json)
and [C=1 grid](../_internal/float16_probe_20261010/mean_per_channel_nhwc_c1_public_sum_route_101_20261011.json).
The fallback above the C=1 cap remains slow: NHWC N=16 at 768×1024 measured
1.295–1.302× float32 time. A float32-accumulator Torch candidate was faster on
that shape but failed the strict subnormal-residual case; see the additional
[PyTorch CPU reduction evidence](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).

A second C=1 input containing only normal positive half values around 0.5 also
exceeded `rtol=1e-7`: the direct CV_16F result was 0.4999966621398926 versus
0.4999966025352478 from both CV_32F and float64. A guard that only looks for
subnormal values therefore cannot make the direct path safe; the reproducer is
in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

## NumPy float16 nearest resize

OpenCV rejects `CV_16F` in `resize`, but nearest-neighbor interpolation copies
source samples without arithmetic. For contiguous HWC float16 input, Albucore
passes a zero-copy `uint16` view through `cv2.resize`, then views the result as
float16. It uses an exact slice-and-copy for single-channel 2× downscales.
Row-strided inputs with dense pixels also use the `uint16` view; exact 2×
downscales with pixel strides copy the sampled values. Other pixel-strided
layouts keep the float32 path. These routes preserve sampled float16 bits and
avoid widening and restoring the full image.

A one-thread comparison used float32 peers widened from the same float16 values:

- The 24-case grid covered HWC sizes 128×160, 240×320, 480×640, and 768×1024;
  C=1/3/9; half-size and 1.5× outputs; five warmups; and 21 interleaved samples.
- The largest float16/float32 ratio was 1.019×, with no robust slow pair under
  the `median ± 2 MAD` screen; C=1/3 ranged from 0.679–1.019×.
- A focused 101-sample C=9 rerun measured 0.60–0.67×.
- A focused 101-sample strided-view grid covered row and column steps with
  exact 2× downscales. Its six C=1/3/9 float16/float32 ratios ranged from
  0.137–0.922×.

These routes cover `INTER_NEAREST` calls with explicit positive `dsize`.
Other interpolation modes and unmeasured pixel-strided regions retain the
float32 working path. The remaining CV_16F resize capability request and
linear-path measurements are in
[OpenCV #30216](https://github.com/opencv/opencv/issues/30216).

For NumPy HWC float16 linear downscales, a zero-copy channels-last Torch
interpolation candidate was also bitwise equal to the tested float32 reference
but slower than float32 public resize in every cell. Half Torch took 3.46–5.01×
for C=1, 2.88–3.04× for C=3, and 1.54–1.61× for C=9; float32 Torch took
2.22–7.10×. Results are in
[OpenCV #30216](https://github.com/opencv/opencv/issues/30216).

## NumPy float16 median blur

For HWC C=1, `ksize=5`, and at least 76,800 elements, Albucore maps each
binary16 value to an unsigned key with the same numeric order, filters the
`uint16` view with OpenCV, and maps the selected values back to float16. NaNs
retain the existing float32 route because their ordering is not numeric. If the
selected output contains zero and the input contains both signed-zero encodings,
the call also uses CV32F so OpenCV keeps its original zero tie selection. Other
channel counts and kernel sizes keep their existing route.

A full one-thread NumPy grid used four non-square sizes, C=1/3/9, `ksize=3/5/7`,
five warmups, and 101 interleaved samples. The selected C=1 `ksize=5` route
matched the float32 result bitwise across every nonzero non-NaN binary16 value,
including infinities and subnormals. Zero and NaN fallback tests preserve the
existing bit patterns and classification. Its float16/float32 median ratios were:

- 128×160: 1.049×; this size stays on the existing route.
- 240×320: 0.980×.
- 480×640: 0.992×.
- 768×1024: 0.761×.

The full benchmark also covers public CHW/NCHW Tensor calls. CHW ratios at
128×160/240×320/480×640/768×1024 were 1.044×/0.982×/0.948×/0.774×; NCHW N=4
ratios were 1.043×/0.980×/0.949×/0.941×. The smallest Tensor size remains
robustly slower; the larger cells have no robust slow pair.
For C=3/9, the ordered-key candidate took 2.54–2.74× the float32 time, so it
was rejected. The full NumPy timings are in
[`median_blur_ordered_u16_public_101_20261011.json`](../_internal/float16_probe_20261010/median_blur_ordered_u16_public_101_20261011.json);
the multi-channel candidate results are recorded in
[OpenCV #30217](https://github.com/opencv/opencv/issues/30217).
A direct `uint16` view for nonnegative C=1 `ksize=3` also matched bitwise but
took 2.80–4.91× the float32 public time; that candidate is rejected as well
([OpenCV #30217](https://github.com/opencv/opencv/issues/30217)).

Value distribution still controls the `ksize=5` result. At routed sizes
(240×320 and larger), the same one-thread, 101-sample public comparison gave
0.777–0.998× float32 time for positive-zero and sparse-zero inputs, and
0.841–1.069× for negative-zero and negative-random inputs. Mixed signed-zero
inputs use the exact CV32F fallback and took 1.302–1.374×; mixed-sign random
inputs use ordered keys and took 2.292–2.595×.
These cases remain open performance gates and are included by `--with-data-patterns`
in the benchmark JSON's `data_pattern_rows`; the matrix is also in
[OpenCV #30217](https://github.com/opencv/opencv/issues/30217).

## NumPy float16 nearest affine warp

`warp_affine(..., INTER_NEAREST)` copies source pixels and does not interpolate
their values. For contiguous HWC float16 images with the default zero constant
border, Albucore passes the half storage through an integer view and restores
the float16 view after OpenCV. It uses CV_16U for C=1/3 and a two-byte-per-pixel
CV_8U view for C=9 because the byte kernel did not preserve the same coordinate
mapping for the smaller channel layouts. The route falls back to float32 for
other borders, non-contiguous input, or other interpolation modes.

A one-thread full-public grid used four non-square input sizes, C=1/3/9, identity
and affine matrices, 10 warmups, and 101 interleaved samples. All 24 outputs
matched the float32 public result cast to float16 bitwise; no pair was robustly
slower and the maximum ratio was 1.031×. Code and measurements are in
[OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

## NumPy float16 nearest perspective warp

`warp_perspective(..., INTER_NEAREST)` copies selected pixels, so contiguous HWC
float16 inputs with the default zero constant border can pass through an integer
view. CV_16U is used for C=1/3 and a CV_8U byte view for C=9. Other channels,
borders, strides, `dst` outputs, non-float32 matrices, and interpolation modes
use the float32 working route.

A one-thread public grid used four non-square sizes, C=1/3/9, a non-identity
projective matrix, 5 warmups, and 101 interleaved samples. All 12 outputs
matched the float32 public result cast to float16 bitwise; the half route took
0.574–0.907× float32 time with no robust slow cell. The measurements and matrix
are in
[`warp_perspective_nearest_public_101_20261011.json`](../_internal/float16_probe_20261010/warp_perspective_nearest_public_101_20261011.json)
and [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

## NumPy float16 nearest remap

For contiguous HWC float16 input and contiguous float32 coordinate maps, the
default zero-border `INTER_NEAREST` path remaps an integer view of the half
storage. It uses CV_16U for C=1/3 and a CV_8U byte view for C=9. C=1 converts its
maps to fixed-point before remapping because direct float maps with a CV_16U
single-channel source can choose a different pixel at exact half-coordinate
ties. The route uses the converted map for C=1/3/9 so every integer view follows
the same float32 sampling coordinates. Map conversion is included in public
timing. Other channels, borders, strides, dst outputs, and interpolation modes
use the float32 working route.

A one-thread public grid used four non-square sizes, C=1/3/9, a non-identity
float32 map with tie coordinates, 5 warmups, and 101 interleaved samples. All 12
outputs matched the float32 public result cast to float16 bitwise; none was
robustly slower. The public float16 route took 0.543–0.960× float32 time. Raw
samples and thread/version metadata are in
[`remap_nearest_public_fixed_maps_101_20261011.json`](../_internal/float16_probe_20261010/remap_nearest_public_fixed_maps_101_20261011.json).
The missing native OpenCV `CV_16F remap` capability and these measurements are
reported in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).

## Routes that stay outside NumKong

| Operation | Current backend | Reason to keep the route |
|---|---|---|
| Global float32 `mean`, `std`, `mean_std` | NumPy | NumPy's float64 accumulators are faster or provide the required semantics on the supported matrix. |
| `min`, `max`, and paired global min/max | NumPy | The current NumKong candidate has no stable full-path win. Re-run the benchmark if NumKong or the target CPU changes. |
| `multiply_by_constant` | NumPy for float32, LUT for uint8 | The measured NumKong helper does not beat the production routes after allocation and saturation costs. |
| `add_constant` | NumPy for float32, OpenCV for uint8 | Allocating `nk.scale(1, beta)` has no stable win. |
| `multiply_by_array` | NumPy for float32, OpenCV plus saturation for uint8 | `nk.fma` does not win the complete operation. |
| Float16 `multiply_by_array` | Float32 working path | For full-shape float32 controls, direct OpenCV CV_16F multiply matched the tested reference but was 1.11–1.42× slower than the same CV_32F multiply-and-cast bridge. Casting controls to float16 before CV16F multiply was still 1.475× slower than the float32 public route at HWC 480×640×3, with up to 1 ULP output error. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and the [control-cast follow-up](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| Float32 `multiply_by_vector`, `add_vector` | NumPy/OpenCV | A NumKong per-channel scale loop loses on the measured large float32 cells. Float16 HWC C=1/C=3 routes are documented in the production-route table above. |
| `matmul` | NumPy `@` / OpenCV GEMM | `nk.dots_packed` with float32 packing was 4.46–11.22× slower than the public float32 route on the tested matrix grid; casting its output to float16 also failed exact wide-reference parity. No NumKong matrix route is selected. |
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
