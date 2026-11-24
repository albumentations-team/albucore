# NumKong route decisions

Albucore chooses NumPy, OpenCV, LUT, NumKong, and eager CPU Torch by measured
full-path performance. This page records the current NumKong routes and the
benchmark questions that can change them. It is a decision guide, not an archive
of one machine's timings.

The [7.8.7 integration review](numkong-7.8.7.md) records released fixes, removed
workarounds, and the precision/performance guards that remain necessary.

## Current production routes

| Operation | NumKong route | Scope and fallback |
|---|---|---|
| `add_weighted` | `nk.blend` for uint8 and single-channel float32; capability-gated NumKong half blend | Multi-channel float32 uses OpenCV for HWC/two contiguous inputs and NumKong for the measured strided higher-rank region. Contiguous float16 inputs with `neon=true`, at least 20,480 elements, and weights `0.5, 0.5` use NumKong; the 21-sample HWC grid measured 0.52–0.55× float32 time for C=3/9. Other float16 controls retain the CV16F/float32 routes. A direct Torch half candidate was fast for normalized random data but failed one-ULP parity under extreme cancellation; a CV16F→CV32F OpenCV candidate was 2.61–3.30× slower. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503). |
| `pairwise_distances_squared` | `nk.cdist(metric="sqeuclidean")` | `n1 * n2 < 1000` or `D >= 9`; contiguous float32 D=8 sets also use NumKong through 2,048 output pairs. M4 Max measured 1.24–2.00× and GCP e2-standard-4 measured 1.63–2.24× at 1,024–2,048 pairs. Larger D=8 sets and other layouts use the NumPy scratch route. The norm/dot identity is avoided because common offsets can overflow or erase small distances. See [the correctness and timing report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip). |
| Global uint8 `mean`, `std`, `mean_std` | `nk.moments` | One contiguous reduction over all elements. |
| Global float32 `std`, `mean_std` | OpenCV `CV_32F meanStdDev` | Contiguous inputs with at least two elements use one flattened OpenCV reduction on Linux x86-64 and macOS ARM64. Non-finite and low-relative-variance results (`std² <= 1e-4 * mean²`) fall back to the NumPy float64 path; an exact zero standard deviation uses `np.ptp` to recognize constant arrays without a second wide reduction. One-thread, 101-sample GCP timings measured 4.7–8.1× for `mean_std` and 3.7–8.0× for `std`; the M4 Max measured 2.4–3.1× and 2.3–2.8×. Random data, offsets through 1e8, and float32 subnormal cases matched NumPy float64 within `rtol=5e-10, atol=1e-12`; near-constant 1e6 and mixed-subnormal inputs take the fallback. Reproduce with [`benchmark_mean_std_global.py`](../benchmarks/benchmark_mean_std_global.py). |
| Tensor batch `remap3d`, float32 | Per-volume PyTorch grid-sample loop writes into one batch output | Linux x86-64 contiguous NCDHW inputs with N≥4, C>1, and at least 1,000,000 output elements use this route for nearest at either border and linear with nonzero constant fill. The 21-sample eligible-cell GCP screen measured 1.006–1.137× against the list-plus-stack reference (median 1.059×). Three fresh-process 8×3×32×256×256 nearest runs lowered median peak-extra RSS by 24,904 KiB (24.3 MiB) for a 192-MiB output batch. M4 Max float32 timing was neutral (0.977–1.015×) and uint8 was slower, so other dtypes and platforms keep the existing route. Reproduce with [`benchmark_remap3d_batch_preallocation.py`](../benchmarks/benchmark_remap3d_batch_preallocation.py) using `--eligible-only --rss --rss-repeats 3`. |
| `gaussian_blur3d` | OpenCV `sepFilter2D` for packed H/W planes, then Torch for depth | Linux x86-64 only, uint8/float32, all three Gaussian axes active, and packed `D*C ≤128`. NumPy requires contiguous DHWC with C=1/3/5/9. Tensor inputs use contiguous CDHW C1/3/5; channel-last-contiguous views use uint8 C1/3/5/9 and float32 C1/3/9. The float32 C5 channel-last view measured 0.92× and stays on Torch. At `16×128×160×C`, NumPy C1/3/5 and contiguous Tensor C1/3/5 measured 1.06–2.44× faster; thin C9 (`D=5`) measured 1.6–1.8×. The route writes OpenCV's XY result into its packed buffer. Fresh-process float32 C3 peak RSS above the input-ready baseline fell from 43,984 to 39,612 KiB. M4 Max C3 volume timing was about 20% slower, so macOS keeps the Torch path. FP16 differs by up to 0.0004883 and keeps the exact float32-work path. Reproduce with [`benchmark_gaussian_blur3d.py`](../benchmarks/benchmark_gaussian_blur3d.py); the `--rss` mode uses Linux `/proc` files and is unavailable on macOS. |
| Float32 `convolve` with separable blur kernel | OpenCV `sepFilter2D` | Linux x86-64 and macOS ARM64, contiguous HWC, finite float32 input, default border, and nonnegative normalized rank-1 7×7/15×15 kernels. The measured size gates are C3: 921,600 elements for k7 and 230,400 for k15; C9: 2,764,800 for either kernel. GCP e2-standard-4 paired timings measured 1.14–5.62× on selected cells: 240×320 C3 k15 was 4.36×; 480×640 C3/C9 k7 was 1.14–1.17× and k15 was 4.70–5.62×. The 240×320 C3 k7 cell was 0.99× and stays on `filter2D`. Nonfinite inputs, other kernels, dtypes, layouts, and borders use `filter2D`. Outputs matched the direct `filter2D` reference within `rtol=1e-5, atol=1e-5`. Reproduce with [`benchmark_convolve_separable.py`](../benchmarks/benchmark_convolve_separable.py). |
| Tensor batch `normalize_per_image`, float32 | Batched NumPy statistics over NCHW/NCDHW storage | Contiguous batches with N≥4 and absent or contiguous output use measured regions: `image` uses C=3/9 from 20,480 through 2,000,000 elements; `image_per_channel` uses NCHW C=1/3 and macOS ARM64 NCDHW C=3; `min_max` uses C=3; `min_max_per_channel` uses C=3/9. The last three modes start at 20,480 elements. Non-finite stats and low-relative-variance per-channel data fall back per item. Selected cells measured 1.07–4.30× on M4 Max and 1.13–4.95× on GCP. Linux NCDHW `image_per_channel`, NCHW C1 `image`, global min/max for C1/C9, and per-channel min/max for C1 were slower and retain the per-item path. Global `image` above 2,000,000 elements also stays on the per-item route on Linux. Reproduce with [`benchmark_normalize_per_image_tensor_batch.py`](../benchmarks/benchmark_normalize_per_image_tensor_batch.py). |
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
| `multiply_add` | `nk.scale` | Float32 scalars use NumKong in its measured region. Contiguous float16 NumPy inputs also use it when `neon=true`, both scalar controls are exactly representable in float16, and the image has at least 20,480 elements. Other float16 controls use the measured Torch/NumPy path; vector and full-array controls do not select the half NumKong route. |
| NumPy float16 `add(..., scalar)`, `multiply(..., scalar)` | `nk.scale` | Contiguous inputs and a scalar exactly representable in float16 (outputs bit-identical to the Torch route over all 65,536 half patterns). macOS `neon=true`: 20,480+ elements; `multiply` joins `add` there (0.63-0.93× of Torch from 20K to 300K elements, 0.96-0.98× above). Linux x86-64 with the NumKong `sapphire` (AVX-512 FP16) capability: 3,072-2,359,296 elements for `add` and 3,072-691,200 for `multiply`, measured on Sapphire Rapids at 0.04-0.09× of the previous route below 16,384 elements, 0.21-0.45× to 307K, and 0.72-0.90× (`add`) or 0.87× (`multiply`) above; larger sizes tie and keep Torch. Other x86 profiles keep the Torch/NumPy route. The C=1 single-control vector shares these regions. |
| NumPy HWC C=1 float16 `add(..., channel_vector)`, `multiply(..., channel_vector)` | `nk.scale` | Contiguous writable inputs with `neon=true`, 20,480+ elements, one control, and a control exactly representable in float16. Addition uses `alpha=1`; multiplication uses the signed-zero-preserving default bias in NumKong 7.8.7. For 14 exact controls, all 65,536 input bit patterns matched the float32-rounded-half result on finite outputs; NaN/Inf classifications and infinity signs also matched. The 101-sample public grid measured float16/float32 ratios of 0.836/0.667/0.563 for add and 0.866/0.685/0.561 for multiply at 128×160/240×320/480×640; a separate 768×1024 check measured 0.535/0.517. At 128×160, 1,000 calls per sample measured 0.976×/0.991× and 301 samples of 100 calls measured 0.979×/1.029×; neither showed a robust >1.10× slowdown. See [`c1_vector_numkong_public_101_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_public_101_20261011.json), [`c1_vector_numkong_128x160_batched_301_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_128x160_batched_301_20261011.json), [`c1_vector_numkong_128x160_batched_1000_20261011.json`](../_internal/float16_probe_20261010/c1_vector_numkong_128x160_batched_1000_20261011.json), and [`c1_c3_vector_public_768_20261011.json`](../_internal/float16_probe_20261010/c1_c3_vector_public_768_20261011.json). Direct CV16F OpenCV C1 candidates were exact on the tested finite inputs but took 2.22–5.92× float32 public time; see [the C1 benchmark](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [the C3/C9 follow-up](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). Other channels, layouts, sizes, and controls retain measured Torch/NumPy paths. |
| NumPy HWC C=3 float16 `add(..., channel_vector)`, `multiply(..., channel_vector)` | OpenCV `CV_16F add`/`multiply` | Contiguous writable inputs, 61,440+ elements, and three exact half controls. Exhaustive binary16 outputs matched the float32-rounded-half reference, including special-value classifications. Full public ratios at 128×160/240×320/480×640 were 0.304/0.284/0.280 for add and 0.474/0.469/0.479 for multiply; 768×1024 measured 0.270/0.450. See [`c3_vector_cv16f_public_101_20261011.json`](../_internal/float16_probe_20261010/c3_vector_cv16f_public_101_20261011.json) and [`c1_c3_vector_public_768_20261011.json`](../_internal/float16_probe_20261010/c1_c3_vector_public_768_20261011.json); the results are also posted in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).
| NumPy HWC C=9 float16 `add(..., channel_vector)` | NumKong `nk.add` with broadcast half controls | Contiguous writable C9 inputs, `neonhalf=true`, half-exact float32 controls, 184,320+ elements; exact binary16 route, with public ratios described below. |
| NumPy HWC C=9 float16 `multiply(..., channel_vector)` | OpenCV `CV_16F multiply` with a zero-stride control array | Contiguous writable C9 images with 184,320+ elements and finite float32 controls. Exact half controls match bitwise; inexact controls are accepted only when every magnitude is ≤1 and each nonzero magnitude is at least the float16 normal minimum, where exhaustive binary16 inputs stay within 1 ULP. Other controls keep the NumPy float32 ufunc-to-half fallback. The runtime-enabled 101-sample public grid ranged from 0.842× to 0.922× float32 time across eight sizes from 128×160 through 768×1024. See [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json), [`c9_multiply_safe_inexact_controls_public_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_multiply_safe_inexact_controls_public_101_20261011_1715.json), and [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| `add(..., field)` | `nk.blend` | Same-shape, same-dtype uint8 arrays. Float32 uses NumPy; other uint8 layouts use OpenCV. |
| `convert_dtype` (`float32` → `uint8`) | `nk.astype(..., out=...)` | The fallback uses one NumKong scale-and-round work buffer followed by NumKong's saturating, round-to-even cast; it beat the prior NumPy `rint`+clip route across the HWC/XHWC contiguous and strided matrix. |
| NumPy `convert_dtype` (`uint8` → `float32`, `max_value=1`) | NumPy raw cast / copy-to-output for contiguous sources | Unit scale preserves raw byte values; bypassing the LUT matches ARM throughput and cuts Linux median time to 0.406× the LUT reference. See [paired conversion measurements](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip). |
| NumPy `convert_dtype` (`uint8` → `float16`) | OpenCV float16 LUT | Divide the 256-entry table in float32, cast the table to half, and write half output directly. This avoids a full float32 image buffer. See the [ARM64 conversion report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip). |
| Tensor `convert_dtype` (`uint8` → `float16`) | Torch float32 workspace on Linux x86-64; native half conversion and division on macOS ARM64 | The Xeon half-normalization path was up to 2.884× the previous chain; a float32 workspace restored a 0.993 median and 1.032 maximum ratio. ARM64 keeps direct half normalization, using half workspace for strided destinations. See the [Linux correction report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip) and [ARM64 candidates](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip). |
| NumPy `convert_dtype` (`float16` → `uint8`, default scale) | Cached OpenCV `CV_16U` LUT | Contiguous inputs with contiguous or absent output use a lazily built 65,536-entry, 64-KiB table from 12,288 elements on macOS ARM64 and 8,192 on Linux x86-64. The fixed table is 64 KiB; at the route thresholds it exceeds the old FP32 workspace by 16 KiB on macOS and 32 KiB on Linux. Three fresh-process GCP samples at HWC 480×640×3 measured 2,944 KiB lower peak RSS and 608–616 KiB higher post-call RSS than the old route. At 8,192 elements, cold-process post-call RSS was 512–640 KiB higher. The resident difference stays below 1 MiB per process; the cached table is not reallocated per call. All 65,536 binary16 encodings match the old route bitwise, including read-only inputs. One-thread M4 Max timings were 1.53× at HWC 480×640×3 (also with `out`) and 1.53–2.13× across tested shapes; GCP Linux timings were 1.99–3.54×, including `out`. Strided input/output and custom scales keep the existing path. |
| Tensor `apply_uint8_lut` with a shared table | StringZilla `translate(src, out=...)` over zero-copy NumPy views | Contiguous CPU tensors use direct translation to the new result or supplied output, removing the pre-translation copy. M4 Max enables this from 61,440 elements; GCP x86-64 uses 230,400 elements in both modes. After 101 paired samples and ten warmups, selected cells measured 1.16–1.24× on M4 Max and 1.07–1.30× on Linux; a 301-sample boundary check at CHW 3×240×320 measured 1.09× allocating and 1.07× with `out`. Fallback medians stayed within 0.5% below to 6.2% above baseline. All output bytes matched exactly. NumPy byte indexing was slower across the tested grid (0.72–0.98× baseline). Non-contiguous inputs and non-target platforms keep the copy/strided path. Tensor writes through NumPy increment PyTorch version counters for supplied and exact-alias outputs; see [PyTorch `increment_version`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.graph.increment_version.html). Reproduce with [`benchmark_tensor_shared_lut.py`](../benchmarks/benchmark_tensor_shared_lut.py); add `--numpy-candidate` for the NumPy comparison. |
| NumPy `resize3d` linear, output depth one | NumPy selected depth slices plus existing H/W interpolation passes | For float16/float32 volumes with at least 4,096 input elements, compute depth interpolation from the two central slices, then resize H/W. On `63×128×160×3 → 1×96×160×3`, M4 Max measured 1.80× for float16 and 1.29× for float32; GCP Linux measured 6.93× and 1.36×. Linux's full screen measured 3.31–6.93× for float16 and 1.36–1.43× for float32 across depths 16–64. The M4 Max screen ranged from 1.12–1.59× across depths 3–64 and channels 1/3. If any half value is non-finite while NumPy invalid handling is enabled, the route widens the full volume to preserve signaling-NaN behavior. |
| NumPy `resize3d` nearest, all-axis downscale | NumPy broadcast gather | Contiguous C3 volumes with at least 900,000 input elements and a strict downscale on D/H/W use one index gather into the output. The M4 Max grid measured 1.34–3.89×; GCP Linux measured 2.45–16.20× across uint8/float16/float32 volumes from `16×128×160×3` through `48×240×320×3`. Results matched bitwise. Other channel counts, strides, interpolation modes, and non-downscales retain the OpenCV packing route. |
| Global NumPy `mean` for float32 | NumPy float64 reduction | The single-statistic `mean` route remains NumPy; the fused OpenCV path applies only when `std` is also requested. |
| Masked RGB `equalize_uint8(mode="cv")` | Packed OpenCV histograms followed by one per-channel LUT | Packed HWC C3 images with 76,800–2²⁴ pixels use `calcHist` on the original image, then map all channels in one LUT call. On M4 Max, public calls ran 1.25–1.43× faster at 240×320 and 480×640. GCP Linux measured 1.12–1.15× at those sizes for shared and row-crop masks; per-channel masks use this route from 307,200 pixels and measured 1.15× at 480×640. Results matched bitwise. |
| NumPy FP32 `box_blur` | OpenCV `blur(src, dst=output)` | Contiguous `ksize>1` calls write to one output allocation without copying the source. M4 Max enables this from 20,480 elements and measured 1.13–1.28× across HWC C=1/3/9. GCP Linux enables it for C>1 from 230,400 elements; measured speedups were 1.33× at 240×320×3, 1.45× at 480×640×3, and 1.81× at 768×1024×9. Linux C1 and smaller unmeasured regions keep the source-copy route. |
| NumPy uint8 `box_blur` | OpenCV `blur(src, dst=output)` | All `ksize>1` calls on both platforms, including row-padded crops; OpenCV copies other strided inputs itself. Outputs match the source-copy route bitwise. Paired one-thread timings over 128×160–768×1024 with C=1/3/9 and k=3/5/15 measured 1.00–1.18× on M4 Max and 1.05–1.42× for k≤5 on GCP Cascade Lake. Linux 768×1024 C1/C3 with k=9/15 stayed at 0.96–1.00× over two 101-sample reruns, within the 5% simplification allowance; k=31 and C9 measured 1.06–1.14×. Reproduce with [`benchmark_uint8_field_multiply_box_blur.py`](../benchmarks/benchmark_uint8_field_multiply_box_blur.py). |
| uint8 audit routes (2026-11-23) | OpenCV, LUT, NumKong and BLAS on existing storage | Public calls against the previous tree, one thread, 128×160–768×1024 with C=1/3/9 (medians of two alternating runs; min–median shown, M4 Max / GCP Cascade Lake). All outputs matched bitwise except the noted statistic. Scalar `add` uses one flat saturating `cv2.add`/`cv2.subtract` for every channel count (negative 9.8–19× / 4.9–16×, positive 0.96–1.1× / 0.99–1.1×). `exp(uint8)` uses a 256-entry float32 table (10.7–18× / 1.6–2.1×). Uint8 `matmul` uses exact float32 BLAS for K ≤ 258 and float64 above (10–36× / 9.8–15×). `copy_make_border` with C>4 per-channel values fills constant borders with NumPy and calls OpenCV once for other borders (7.6–29× / 6.7–19×). `median_blur` reads uint8 row crops directly (1.14–1.24× / 1.12–1.36× at k=3), filters C>4 planes separately for k≥7, and filters contiguous uint8 Tensors per plane (6.5–7.8× / 4.9–8.9× at k=3). Contiguous uint8 Tensors with scalar controls translate their storage through the shared table for `add`, `multiply`, `power` and `multiply_add` (5–20× / 4.9–12.5×), `normalize` uses one float LUT on storage (3.0–3.5× / 4.1–4.8×), and `add_weighted` blends storage with NumKong (8–9.3× / 18–24×). Uint8 per-channel `mean_std`/`std` with `keepdims=True` reuse the NumKong integer moments of `keepdims=False` (9–25× / 11–67×); std changes by at most 1e-11 and now equals the `keepdims=False` result. Rejected: float32 clip into a uint8 buffer (1.00×), Tensor uint8→float LUT conversion (0.42×), per-channel std through a squares LUT (0.22×), unmasked `equalize_uint8` through one per-channel LUT (0.89–1.30×), in-place 3D uint8 restoration and `resize3d` rounding (0.98×), Tensor uint8 `flip_volume` through int32 (1.00×), `unsharp_mask` with `CMP_LE` (1.02–1.05×, not durable), crop `convolve` without a copy (1.01×). Reproduce with [`benchmark_uint8_public_routes.py`](../benchmarks/benchmark_uint8_public_routes.py) run under each tree. |
| `erode` / `dilate` with a center-only kernel | Independent C-order copy | Kernels with exactly one nonzero value at OpenCV's default anchor return the source unchanged. The route preserves dtype and produces writable, independent storage. It uses 1×1 kernels for uint8/float16 at all sizes, and larger center-only kernels from 20,480 elements for float16 or 30,720 for uint8/float32. Float32 1×1 stays on OpenCV: the C9 480×640 GCP copy measured a 0.895× speedup. On selected cells, M4 Max measured 0.99–2.95× for uint8, 3.37–11.72× for float16, and 2.43–3.01× for float32. GCP measured 1.11–2.62×, 20.32–83.90×, and 1.14–2.18× respectively. Reproduce with [`benchmark_morphology_identity.py`](../benchmarks/benchmark_morphology_identity.py). |
| NumPy `sqrt(uint8)` | OpenCV `CV_32F` LUT | Contiguous HWC C=1/3 on M4 Max uses one 256-entry table from 20,480 elements; C9 uses it from 7,000,000 elements only without `out`. Linux x86-64 starts at 921,600 elements: C3 uses the LUT only without `out`, while C9 uses it with or without `out`. Linux measured 1.13× for C3 allocation and 1.23–1.69× for C9; C3 with `out` was neutral or slower and stays on NumPy. Results matched bitwise. Reproduce with `benchmark_elementwise.py`. |
| NumPy `pad3d` scalar fill | NumPy face writes | Contiguous C>1 inputs with at least 524,288 elements and output expansion no greater than 1.1× fill only the six non-overlapping faces before copying the center. One-cell padding measured 1.23–1.24× on M4 Max and 1.34–1.37× on GCP Linux at DHWC 32/64×128×160×3. Larger halos, C1, tuple fills, and strided inputs keep the existing route. Reproduce with `benchmark_pad3d.py --router-only`. |

### Float16 clipping

Contiguous NumPy float16 `clip(..., np.float16)` uses OpenCV integer min/max from 16,384 elements. A flattened single-channel view removes image-rank and channel-count limits without a copy. Positive finite/Inf encodings use unsigned minimum with the encoding of one; negative finite values use signed max/min. NaNs and negative zero retain NumPy's exact bit semantics through the original route. Strided outputs and inputs also keep that route.

The corrected 90-cell, one-thread comparison covers all four canonical HWC sizes with C=1/3/9, plus batches and volume batches, allocation, exact input alias and separate output, with positive and mixed-sign values. Sources reset before every mode and alias sample. Candidate/previous times are 0.022–0.135 on native Apple M4 Max and 0.018–0.112 on Linux x86-64. Exhaustive tests check all 65,536 half encodings plus the finite fast domain. The route creates only output storage and views; it does not widen the image or allocate a mask. Evidence: [ARM64 timings](../_internal/architecture/evidence/half-clip-full-native-macos-02.json), [Linux timings](../_internal/architecture/evidence/linux-binary-16-core/clip-full.json).

This Core result does not close the larger AX architecture acceptance. The half-brightness retained-heap investigation, uint8 C5 caller timing and remaining families are tracked in the [progress ledger](../_internal/architecture/progress.md).

### C9 vector arithmetic: evidence and limits

- HWC C9 `add(..., channel_vector)` uses NumKong `nk.add` with a zero-stride control view for finite half-exact float32 controls. On the runtime-enabled M4 Max, the 101-sample public ratios at 128×160/160×200/240×320/320×448/400×640/480×640/640×800/768×1024 were 0.935/0.932/0.899/0.899/0.881/0.877/0.883/0.857; all tested outputs matched the float32-rounded-half result bitwise. Direct OpenCV CV16F addition was slower than NumKong at all four canonical sizes. Results: [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json) and [`c9_add_cv16f_vs_numkong_runtime_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_add_cv16f_vs_numkong_runtime_101_20261011_1715.json).
- HWC C9 `multiply(..., channel_vector)` uses OpenCV `CV_16F multiply` for exact half controls and inexact controls in `[-1, 1]` outside the half-subnormal range. Exhaustive binary16 tests stay within 1 ULP and preserve NaN/Inf classifications and signed zero. The 101-sample public ratios are 0.842–0.922× float32 over eight canonical sizes.
- For non-exact controls above one, the current route uses the NumPy float32 ufunc loop to half output; `1.1` measures 1.13–1.15× float32 at 400×640/480×640/768×1024. A mixed CV16F-output / float32-control OpenCV candidate matched all half bit patterns but took 1.497–1.563× float32. Results are in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [`c9_multiply_cv16f_mixed_controls_over1_public_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_cv16f_mixed_controls_over1_public_101_20261011.json).
- NumKong candidates on the signed-zero fix branch did not close this region: mixed float16×float32 promotion plus narrowing was 2.28–2.33× float32, and `out_dtype=float16` was 5.38–5.54×. Rounding controls to half was 0.90–0.92× but changed six overflow-boundary finite/Inf classifications across the binary16 input domain. Keep the current fallback; see [NumKong #395](https://github.com/ashvardanian/NumKong/issues/395#issuecomment-6096156733) and [#401](https://github.com/ashvardanian/NumKong/issues/401). Raw data: [`c9_multiply_numkong_mixed_controls_pr402_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_numkong_mixed_controls_pr402_101_20261011.json) and [`c9_multiply_numkong_inexact_over1_pr402_101_20261011.json`](../_internal/float16_probe_20261010/c9_multiply_numkong_inexact_over1_pr402_101_20261011.json).
- NumKong 7.8.7 fixes the signed-zero defect reported in [#401](https://github.com/ashvardanian/NumKong/issues/401). The current OpenCV route remains faster in the [release screen](numkong-7.8.7.md). OpenCV’s safe-control results are in [#25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and [`c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json`](../_internal/float16_probe_20261010/c9_vector_final_public_runtime_neonhalf_101_20261011_1715.json).
- OpenCV 5.1 accepts compact `(1, 1, 9)` half coefficients. At 768×1024 on the M4 Max, public contiguous `multiply` dropped from 3.79 to 2.08 ms without `out` and with separate storage. It took 1.12× the previous time with an exact image alias. On Cascade Lake, the compact operand took 1.72–2.11× the expanded route across four sizes and three output modes. Keep the expanded operand on x86 and for aliasing. [OpenCV #30969](https://github.com/opencv/opencv/issues/30969) tracks the slower compact path on x86.

- OpenCV rejects a nine-channel scalar tuple. Splitting into three CV16F C3 calls was exact on random tests but 5.2–5.74× slower than float32 public calls; measurements: [`c9_vector_cv16f_chunks_101_20261011.json`](../_internal/float16_probe_20261010/c9_vector_cv16f_chunks_101_20261011.json).
The public routers preserve their documented dtype, layout, and aliasing
contracts. A dispatch branch is implementation routing; it is not a second
validation pass. Callers validate dtype, shape, layout, and control parameters
before entering Albucore.

Albucore gates float16 NumKong `scale` and `blend` on baseline `neon` from
7.8.7. Float16 sums and moments still require `neonhalf`; their dispatch was
not replaced by this release. Earlier 7.8.5 measurements with NEON disabled
are historical restricted-process results, not native-host acceptance data.

With NumKong 7.8.5 and `neonhalf=true` on an Apple M4 Max, exact half controls
were measured through the full NumPy public route. For `nk.scale` controls
`alpha=0.875, beta=0.125`, direct half took 0.28–0.51× the float32 round-trip
across HWC C=1/3/9 and the 128×160, 480×640, and 768×1024 sizes; all 65,536
binary16 values matched exactly. For `nk.blend` weights `0.5, 0.5`, direct half
took 0.32–0.51× the bridge across the same grid; the tested random and
adversarial outputs stayed within one binary16 ULP. That profile enabled the
measured `add_weighted` and scalar `multiply_add` regions only when
`neonhalf=true`, inputs are contiguous, controls are exactly representable in
half, and the size threshold is met. The paired candidate matrix is recorded in
[`numkong_half_host_101_20261011.json`](../_internal/float16_probe_20261010/numkong_half_host_101_20261011.json).

NumKong 7.8.7 supersedes [PR #398](https://github.com/ashvardanian/NumKong/pull/398)
and fixes [#395](https://github.com/ashvardanian/NumKong/issues/395) and
[#396](https://github.com/ashvardanian/NumKong/issues/396). Broader coefficient
regions remain excluded because they change current rounding and cancellation
results; see the [integration review](numkong-7.8.7.md).

NumKong `moments(float16)` loses subnormal contributions; the
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
NumKong 7.8.7 also corrects the generic binary16 ULP helper described in
[#397](https://github.com/ashvardanian/NumKong/issues/397).

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

## Exact FP16 and FP32 storage casts

`convert_dtype(float_image, np.float16/np.float32)` casts floating inputs without scaling. Uint8 inputs are divided
by `max_value` (default 255). Arithmetic, normalization, statistics, filtering, and geometry use the internal
`_cast_float_numpy` helper when a kernel needs float32 work or float16 restoration.

Contiguous NumPy inputs use NumKong from 16,384 elements when its `skylake` capability is enabled. NumKong writes
directly into `np.empty`, preserving independent writable storage and `OWNDATA`. Other CPU/layout/size regions
retain NumPy's order-K cast. Special values, narrowing overflow, and non-default underflow handling keep NumPy's
floating-error behavior; checking half finiteness through integer exponent bits avoids the slow x86 half ufunc.

On macOS ARM, contiguous Tensor casts from 61,440 elements use a NumPy view, one allocating cast, and zero-copy
Tensor wrapping. Same-dtype calls retain identity. Tensor casts retain Torch's floating-error policy.

One-thread, 51-sample Xeon measurements of a `(1,32,384,384)` crop reduced complete FP16→FP32 casts from
12.30 to 4.71 ms and FP32→FP16 casts from 23.89 to 4.29 ms. On M4 Max, NumPy is faster than NumKong even for
contiguous inputs; the selected Tensor route reduced the same casts from 0.445 to 0.224 ms and 0.460 to 0.264 ms.
See the [cast report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip) for versions, layouts, raw samples, complete
operation bridges, rejected candidates, and precision checks. NumKong [#404](https://github.com/ashvardanian/NumKong/issues/404)
and pending [#405](https://github.com/ashvardanian/NumKong/pull/405) track strided kernels; those routes remain on NumPy.

## Routes that stay outside NumKong

| Operation | Current backend | Reason to keep the route |
|---|---|---|
| Global float32 `mean` | NumPy float64 reduction | The single-statistic mean remains NumPy; `std` and `mean_std` use the measured OpenCV route above on contiguous Linux x86-64 and macOS ARM64 inputs. |
| `min`, `max`, and paired global min/max | NumPy | The current NumKong candidate has no stable full-path win. Re-run the benchmark if NumKong or the target CPU changes. |
| `multiply(..., scalar)` | NumPy for float32, LUT for uint8 | The measured NumKong helper does not beat the production routes after allocation and saturation costs. |
| `add(..., scalar)` | NumPy for float32, OpenCV for uint8 | Allocating `nk.scale(1, beta)` has no stable win. |
| `multiply(..., field)` | NumPy for float32, OpenCV for uint8 | `nk.fma` does not win the complete operation. Uint8 fields use one saturating `CV_8U` OpenCV multiply: integer products are exact, so the result matches the truncating float clip bitwise. Other uint8 field dtypes use mixed-depth `cv2.multiply(img, field, dtype=CV_32F)` without widening the image first, then clip. Paired one-thread timings over 128×160–768×1024 with C=1/3/9 measured 23–52× (M4 Max) and 7.4–18× (GCP Cascade Lake) for uint8 fields, and 1.12–1.31× (M4 Max) and a ~1.1× median (Linux, noisy cells 0.88–1.27× that flip sign between runs) for float32 fields. Reproduce with [`benchmark_uint8_field_multiply_box_blur.py`](../benchmarks/benchmark_uint8_field_multiply_box_blur.py). |
| Float16 `multiply(..., field)` | Float32 working path | For full-shape float32 controls, direct OpenCV CV_16F multiply matched the tested reference but was 1.11–1.42× slower than the same CV_32F multiply-and-cast bridge. Casting controls to float16 before CV16F multiply was still 1.475× slower than the float32 public route at HWC 480×640×3, with up to 1 ULP output error. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769) and the [control-cast follow-up](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769). |
| Float32 `multiply(..., channel_vector)`, `add(..., channel_vector)` | NumPy/OpenCV | A NumKong per-channel scale loop loses on the measured large float32 cells. Float16 HWC C=1/C=3 routes are documented in the production-route table above. |
| `matmul` | NumPy `@` / OpenCV GEMM | `nk.dots_packed` with float32 packing was 4.46–11.22× slower than the public float32 route on the tested matrix grid; casting its output to float16 also failed exact wide-reference parity. No NumKong matrix route is selected. |
| `exp`, `sqrt` | NumPy/OpenCV | NumKong exposes no matching elementwise kernels. |

## Reproduce a routing decision

Run the complete public path and compare it with every viable backend. Keep the
workload, thread counts, allocation mode, correctness tolerance, and caller
preconditions identical.

```bash
uv run python benchmarks/benchmark_add_weighted.py
uv run python benchmarks/benchmark_numkong_vs_albucore_backends.py
uv run python benchmarks/benchmark_numkong_arithmetic.py
uv run python benchmarks/benchmark_stats.py
uv run python benchmarks/benchmark_mean_std_global.py --repeats 101 --warmup 10 --output benchmarks/results/meanstd-global-cv2.json
uv run python benchmarks/benchmark_gaussian_blur3d.py --shape 16,128,160,3 --threads 1 --repeats 101 --warmup 10 --output benchmarks/results/gaussian-blur3d.md
uv run python benchmarks/benchmark_gaussian_blur3d.py --shape 16,128,160,3 --threads 1 --repeats 101 --warmup 10 --rss --rss-repeats 5 --output benchmarks/results/gaussian-blur3d-linux.md
uv run python benchmarks/benchmark_normalize_per_image_tensor_batch.py --repeats 51 --warmup 8 --output-json benchmarks/results/normalize-per-image-batch.json
uv run python benchmarks/benchmark_remap3d_batch_preallocation.py --threads 1 --eligible-only --repeats 21 --warmup 4 --rss --rss-repeats 3 --output-json benchmarks/results/remap3d-batch-out.json
uv run python benchmarks/benchmark_minmax_ravel.py
uv run python benchmarks/benchmark_numkong_astype.py
uv run python benchmarks/benchmark_float16_to_uint8.py --threads 1 --output-json benchmarks/results/float16-to-uint8.json
uv run python benchmarks/benchmark_resize3d.py --shape 63,128,160,3 --scenario unit --threads 1 --repeats 31 --warmup 5 --output benchmarks/results/resize3d-depth-one.md
uv run python benchmarks/benchmark_resize3d_nearest.py --threads 1 --output-json benchmarks/results/resize3d-nearest.json
uv run python benchmarks/benchmark_masked_equalize.py --threads 1 --output-json benchmarks/results/masked-equalize.json
uv run python benchmarks/benchmark_box_blur.py --threads 1 --output-json benchmarks/results/box-blur.json
uv run python benchmarks/benchmark_convolve_separable.py --repeats 51 --warmup 8 --output-json benchmarks/results/convolve-separable.json
uv run python benchmarks/benchmark_morphology_identity.py --repeats 51 --warmup 10 --output-json benchmarks/results/morphology-identity.json
uv run python benchmarks/benchmark_tensor_shared_lut.py --threads 1 --repeats 101 --warmup 10 --output-json benchmarks/results/tensor-shared-lut.json
uv run python benchmarks/benchmark_elementwise.py --threads 1 --repeats 31 --warmup 5 --output benchmarks/results/elementwise.md
uv run python benchmarks/benchmark_pad3d.py --router-only --rounds 11 --seconds 0.03 --output benchmarks/results/pad3d.json
```

The M4 Max screens used CPython 3.11.11, NumPy 2.4.6, OpenCV 5.1.0.95,
NumKong 7.8.7 (`neon=true`), StringZilla 5.3.2, and CPU PyTorch 2.15.0 with
one numerical-library thread. The Linux screen ran on a standard GCP
e2-standard-4 VM with AMD EPYC 9V74, CPython 3.11.18, NumPy 2.4.6,
OpenCV 5.1.0.95, NumKong 7.8.7 (`haswell=true`), StringZilla 5.3.2, CPU
PyTorch 2.15.0, and one numerical-library thread. Most GCP timing rows use 21
paired samples after four warmups; Tensor shared LUT uses 101 samples after ten
warmups. Routes were bitwise exact except D=8 pairwise distances, checked with
`rtol=4e-7, atol=2e-7`. The conversion screen checks every binary16 encoding.
The resize benchmark includes a float16 row for the `unit` scenario.

The [NumKong buffer-first cast report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip) records the accepted
`convert_dtype` fallback across HWC/XHWC contiguous and strided inputs.

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
