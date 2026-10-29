---
name: torch-performance-optimization
description: Optimize or review eager CPU-only Albucore PyTorch runtime paths with benchmark-backed decisions. Use when adding or changing Torch CPU kernels, Tensor/NumPy bridges, Torch backend routing, tensor layouts, allocations, threading, profiling, memory-format candidates, or Torch performance benchmarks.
---

# Torch Performance Optimization

Read [`docs/torch-performance-optimization.md`](../../../docs/torch-performance-optimization.md) completely before inspecting or editing a Torch path. Also read `../performance-optimization/SKILL.md`, its canonical performance guide, and `../albucore-benchmarks/SKILL.md` completely.

## Workflow

1. Establish a correctness baseline and benchmark the existing public path. State container, layout, dtype, shape, parameter, stride, thread, and allocation contracts.
2. Audit unnecessary Python work, full-volume passes, conversions, materializations, and output repairs before choosing a Torch operator.
3. Profile only to discover candidates. Benchmark every viable implementation end to end, including NumPy↔Torch bridges and layout conversion.
4. Compare NumPy, OpenCV, NumKong, StringZilla, and Torch where they share semantics. For float16, compare native half operations with float32 working paths plus output restoration. Treat fused operators, layout, `channels_last_3d`, and in-place reuse as hypotheses with their own correctness and performance matrices.
5. Select a route or threshold only from stable public-path evidence. Preserve rejected candidates in the benchmark report when they clarify a boundary.
6. Add correctness tests for each accepted route and update the benchmark evidence, public contract, and canonical guide when a reusable rule or limitation is discovered.

## Albucore Contract

- Same-dtype speed, memory, and correctness acceptance applies independently on Linux x86-64 and macOS ARM64.
  Use the canonical performance guide for the agreed simplification allowance. Float16/float32 comparisons are
  mandatory diagnostics, not an independent architecture-migration gate.
- AX owns reusable buffers. Input writes require explicit `out=img`; without `out`, views and identity returns may
  share storage but must not write it. Count necessary AX protective copies in full-path timing and memory.
- Torch is a required dependency. Assume it is already imported for benchmarks.
- Current public Tensor routes are eager CPU paths that do not record autograd inside the primitive and do not use `torch.compile`. Do not add device routes, graph-preserving primitive fallbacks, compilation, or their benchmark candidates.
- Tensor layouts are explicit and independent of NumPy layouts. Do not infer `NCHW` versus `CDHW` from shape sizes.
- `warp_affine3d` and `remap3d` support rank-4 `CDHW` volumes and rank-5 `NCDHW` batches; one affine matrix or remap grid is shared across the batch.
- For general 3D sampling, the measured CPU path folds `N>=4, C=1` batches into the channel axis for both routers. `warp_affine3d` uses rank-5 sampling for other channel counts. `remap3d` samples `N>=4, C>1` items independently while sharing grid and fill preparation because that beat a single native batch sampler in the full-call benchmark.
- XY-only linear affine matrices with unchanged depth share one plane grid. Single-channel inputs fold `N*D` into 2D channels; multi-channel batches merge `N` and `D` only when that is a view, otherwise sampling into a preallocated output per volume. Nearest interpolation retains the original 3D sampler.
- Caller-prevalidated 3D routers leave validation outside the hot path. Do not add it back while optimizing.
- A Tensor route must preserve the documented container, layout, border, interpolation, rounding, mutation, and aliasing behavior. Image dtypes are uint8, float16, and float32; the dtype contract must distinguish preserved input/output dtype from any explicit working conversion.
- For global float16 statistics, keep `mean`, `std`, and `mean_std` separate. Tensor `mean` uses float32 accumulation for C>1 at 40,960+ elements; contiguous Tensor global `std` at 40,960+ uses float32 accumulation only for finite inputs within [-1, 1] when std exceeds 1e-3, with two-pass Torch float64 fallback otherwise; non-contiguous inputs keep NumPy float64. Tensor `mean_std` uses flattened CV16F at 20,480+ elements with wide near-constant/non-finite fallback. PyTorch 2.14.1 CPU `std(float16)` returns half and rejects float32/float64 `out=` for global reductions; see [#97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- NumPy HWC float16 per-channel `mean` remains on CV32F/OpenCV until NumKong PR #400 is released. Direct PR #400 moments passed `rtol=1e-7` but measured C1 at 0.93–1.02× float32, C3 strided at 1.15–1.17×, and C9 at 4.85–5.00×; existing `nk.sum(axis=(0,1))` for C9 was 4.82–4.93×, and a C3 planar copy was 3.03–3.33×. Only the C1 candidate matched float32 timing in that historical screen. See [NumKong #311](https://github.com/ashvardanian/NumKong/issues/311#issuecomment-6112109242).
- NumPy HWC float16 per-channel `std`/`mean_std` still widens subnormal inputs and is 1.32–1.77× slower than float32 in the HWC C=1–4 measured region. A temporary PR #400 wheel with only that scan bypassed passed all 32 public cells at 0.27–0.86× float32 and within 1.67e-8 absolute error against NumPy float64. A positive `[0.1,0.9]` screen also passed mean `rtol=1e-7` in 16 cells and stayed within current `mean_std` test tolerances; do not enable it until the dependency PR is merged and released. See the [consolidated NumKong #399 results](https://github.com/ashvardanian/NumKong/issues/399).
- CPU float16 `matmul` keeps the current NumPy float32 working path. Direct Torch half GEMM was 7.2–123.6× slower than the float32 public peer on tested ARM64 matrices and differed from the wide result; a Torch float32-working candidate was exact but 1.42–1.97× slower than the float32 public path. See [PyTorch #146508](https://github.com/pytorch/pytorch/issues/146508#issuecomment-6106617861) and the [consolidated CPU float16 evidence](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor `resize(..., INTER_NEAREST)` uses native CPU half
  interpolation from 61,440 elements for NCHW and multi-channel CHW; CHW C=1
  keeps its existing route. The 20-cell, 101-sample grid matched the float32
  public output cast to half with no robust slow pair; test all binary16 patterns
  for the NCHW C=1 branch.
- For float16 Tensor `resize(..., INTER_LINEAR)` with C=1, the float32/OpenCV public bridge remains a performance gate: a 16-cell, 101-sample CHW/NCHW grid was 1.08–2.10× float32 and robustly slow in all cells. Direct Torch half was 1.19–5.21×, and a Torch float32 round-trip was 1.69–7.45×. Keep the current bridge until an upstream kernel wins the complete public path; track [OpenCV #30216](https://github.com/opencv/opencv/issues/30216) and [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Keep separable float16 CPU filtering in float32: direct half depthwise `conv3d` was 1.11–1.30× slower than the complete float32 round-trip and rounds after each pass. Recheck the numerical contract as well as full-path timing before enabling it; see the [consolidated CPU float16 evidence](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor scalar and channel-vector `add`/`multiply` use direct Torch half operations only from 16,384 elements and out-of-place. Scalars keep Torch's native scalar-to-half semantics; each channel-vector control must be exactly representable. Benchmark full public calls with layout conversion included; non-contiguous inputs, unsupported controls, and in-place calls retain the bridge.
- NumPy float16 per-channel `reduce_sum` has a measured zero-copy Torch float64 route for contiguous rank-4 channel-last C=3 arrays up to 245,760 elements. Full public calls are exact and 4.4–5.3× faster than the float32 peer for N=1–4 at 128×160; keep the route restricted to that measured region.
- NumPy rank-4 channel-last float16 per-channel `mean` uses zero-copy Torch float64 reduction for contiguous writable C=3 arrays from 20,480 elements, and the existing global Torch float64 sum for C=1 arrays from 20,480 through 8,000,000 elements. Public 101-sample grids matched NumPy float64 exactly; C=1 above the cap still has a robust slow cell. A faster float32 Torch accumulator failed the strict subnormal-residual test; see the [consolidated CPU float16 evidence](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- NumPy HWC float16 vector routes use NumKong `scale` for C1 exact controls, OpenCV CV16F for C3, and NumKong `add` with a zero-stride control view for C9 exact add controls. C9 multiply uses OpenCV CV16F for exact and safe inexact controls; other controls retain the NumPy float32 ufunc loop to half output and still include a 1.13–1.15× float32 slow region. Direct Torch half is slower than its float32 bridge ([PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503)). NumKong C9 multiply is excluded because of its signed-zero defect ([#401](https://github.com/ashvardanian/NumKong/issues/401)); its fix is under review in [PR #402](https://github.com/ashvardanian/NumKong/pull/402). Further C9 controls-above-one candidates did not meet the gate: mixed CV16F was 1.50–1.56× float32, NumKong mixed promotion 2.28–2.33×, and half-rounded controls changed overflow classifications ([OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769), [NumKong #395](https://github.com/ashvardanian/NumKong/issues/395#issuecomment-6096156733)).
- Contiguous float16 Tensor `normalize` with CHW/NCHW C=3/9, scalar or channel-vector controls, and 16,384+ elements uses per-plane CV16F `addWeighted` into a preallocated channel-first output; verify identity aliasing and at most one ULP from the wide reference. Other layouts use the zero-copy channel-last view and float32 working buffer.

- NumPy-backed two-step Torch arithmetic releases a no-longer-needed product before `Tensor.numpy()` creates its
  output view. Keeping the product alive across bridge initialization raised fresh-process Linux peak RSS by
  0.25–0.63MiB in the checked compound cases; early release restores the pre-migration peak without changing math.
- `add_weighted` uses concrete Tensor dispatch and explicit output storage. Preserve the whole contiguous half
  `0.5/0.5` NumKong region before per-item batch routing. Ordinary batch calls allocate one final channel-last
  result; supplied output views are filled item by item. Do not replace this with a result list plus `np.stack`.

## Required Handoff

Report the full-path baseline, selected route, removed work, allocation and copy changes, all viable backends and Torch-specific candidates considered, thread and benchmark matrix, correctness/memory evidence, regressions, rejected regions, and remaining follow-ups.

- Fixed-statistics `normalize` accepts result-dtype output. Positive-stride NumPy output can be wrapped for the
  existing Torch fused kernel; negative-stride output computes that same kernel before copying. Half CHW/NCHW
  plane kernels write contiguous Tensor output directly or copy their selected result to a strided destination.
  Non-half Tensor batches retain per-item numerical routing while filling one final output.

- Preserve batch output order: float32 NumPy and planar per-channel uint8 LUT kernels use K-order; shared uint8
  tables and whole-array OpenCV LUT use C-order. A single interleaved output can make planar ufunc writes much
  slower; compare every affected layout before replacing per-item allocation.

- Adaptive normalization fills one dense batch result while retaining per-item statistics and the first result's
  element order. A broadcasted empty_like template changes the batch-axis order; use an owned dense buffer view.
  Strided Tensor outputs may use native copy_ after the selected kernel. Floating aliases require statistics first.
- Preserve measured default allocation separately from explicit outputs. Float32 functional Torch allocation remains
  selected after a NumPy-owned default regressed; half may return its NumPy output directly. Do not assume every
  supplied Torch output is faster or allocation-free.

- Adaptive Half global image mean/std must not reuse its widened source for arithmetic: full AX batch calls
  regressed35–42%. Keep a distinct result and release widened input before final casting; other modes may reuse
  their newly owned contiguous work after statistics.
