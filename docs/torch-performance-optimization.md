# PyTorch Optimization Workflow

Use this guide when a change adds, reviews, profiles, or routes an Albucore Torch kernel. Its purpose is to find durable speedups without changing the public array contract or reporting a microbenchmark that disappears behind conversion, allocation, or dispatch cost.

PyTorch `>=2.14.1` is installed through the `torch` extra and is required in every environment that imports Albucore or exercises these routes.
This repository pins the development and CI profile to PyTorch's CPU index, so `uv sync --extra torch` does not
resolve Linux CUDA wheels.

The current Albucore scope is eager CPU execution without autograd through the primitive or `torch.compile`. NumPy inputs use Albucore channel-last layouts and Tensor inputs declare their own channel-first layout. AlbumentationsX validates CPU placement, layout, strides, autograd state, and parameter ranges before caller-prevalidated 3D primitives run. Do not repeat those checks in their hot paths.

`warp_affine3d` and `remap3d` accept a single rank-4 volume or rank-5 batch: NumPy `DHWC`/`NDHWC` and Tensor
`CDHW`/`NCDHW`. Each batch call reuses one matrix or grid across its items. Other 3D routers currently receive one
volume per call.

For general 3D sampling, the measured CPU route folds `N>=4, C=1` into the channel axis for both batch routers.
`warp_affine3d` uses the rank-5 sampler for other channel counts. For `remap3d`, `N>=4, C>1` uses per-volume sampling with grid and fill
preparation shared across items; a single native batch sampler was slower in the full-call comparison. See the
measurements in [`REPORT_sampling3d_batch.md`](../benchmarks/results/REPORT_sampling3d_batch.md).

An exact XY-only linear affine with unchanged depth shares one plane grid across all slices and batch items.
Single-channel inputs fold `N*D` into 2D channels. Multi-channel inputs use a 2D batch when merging `N` and `D`
requires no copy; other layouts sample each volume into one preallocated output while reusing the grid.
Nearest interpolation and every real Z change retain the original 3D sampler. See
[`REPORT_xy_volume_resampling.md`](../benchmarks/results/REPORT_xy_volume_resampling.md) for numerical and timing evidence.

## Start with the whole operation

Measure the path users execute:

1. Include container dispatch, `torch.from_numpy`, axis views, dtype casts, kernel construction, output layout repair, and `Tensor.numpy()` for a NumPy route.
2. For a Tensor route, accept an already-created CPU Tensor and state that its construction is excluded.
3. Use exactly one CPU thread per process for every candidate. Set `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`, and disable OpenCV internal parallelism with `cv2.setNumThreads(0)` before work starts; set inter-op threads only once per process. Apply the OpenMP/BLAS environment controls from the general performance guide before numerical imports and record effective settings. Use `num_threads=1` with `torch.utils.benchmark.Timer`. Benchmark additional thread counts only when the user explicitly requests thread scaling. [PyTorch thread controls](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html), [PyTorch benchmark docs](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
4. Warm up lazy initialization, collect repeated samples, and report median plus spread, hardware, library versions, shapes, dtype, strides, parameter values, and allocation mode.
5. Establish numerical, dtype, shape, range, layout, aliasing, and mutation parity before accepting a faster route.

Benchmark the canonical non-square image or DHWC volume grid, then add the dimension that controls the proposal: kernel radius, channel count, contiguity, or output size. A local kernel benchmark can explain a result. It cannot select a public route by itself.

## Remove Python and allocation work first

Audit every per-call conversion, allocation, reshape, and Python/Tensor boundary before changing an operator.

- `permute`, `movedim`, `unsqueeze`, and `squeeze` are usually metadata views. Do not add `.contiguous()` or `clone()` defensively; each can materialize the full volume. Benchmark any backend that requires materialization end to end.
- `torch.from_numpy` and CPU `Tensor.numpy()` can share storage. They require an eligible CPU buffer and do not repair unsupported negative strides, non-writable input, device placement, or autograd state. Copy exactly once only when the documented contract requires a repair.
- Return `np.asarray(result.numpy())` after the final layout view when the static checker needs an ndarray rather than `Any`. For an array already returned by `Tensor.numpy()`, `np.asarray` returns the same object without copying. `torch.from_numpy` goes in the opposite direction and cannot implement a Tensor-to-NumPy return.
- Build a scalar, per-channel Tensor, filter weight, affine matrix, or index map once per public call. Normalize a caller-supplied NumPy filter kernel to the documented working dtype before Tensor work; a default float64 NumPy kernel cannot be a Conv weight for a float32 volume. Cache it only when callers actually reuse an identical value and the cache has bounded lifetime, keying, device, dtype, and memory behavior.
- Keep decisions about sizes, modes, and parameter tuples in Python before the Tensor hot loop. Avoid `.item()`, `.tolist()`, or Python branching on Tensor values inside a repeated Tensor path. On accelerators these can synchronize execution; on CPU they still add wrapper work.
- Keep input, working, accumulator, and output dtypes explicit. A float16 route may compute in float32 and restore float16 when needed; include both conversions in the public-path benchmark. For uint8 output, preserve the single documented saturation/rounding step.
- Use `out=` or in-place Tensor operations only after proving ownership, aliasing, and backend semantics. In-place work is a candidate, not an automatic optimization.

## Preserve the Tensor contract

Every Tensor route states its logical layout independently of NumPy conventions. A 4D Tensor can mean `NCHW` or `CDHW`; never infer that choice from an axis size.

- Preserve the caller's container and declared layout.
- Do not silently move a Tensor, call `.detach()`, make it contiguous, or change a dtype to rescue an unsupported input unless the public contract explicitly documents that fallback. The caller-owned adapter or public contract decides those actions.
- "Same dtype as input" applies only to the listed supported dtypes. Callers reject other dtypes before dispatch. When a primitive deliberately has a documented fallback dtype, state its working and output dtype and test it, including any identity fast path.
- Run the primitive under `torch.no_grad()` or `torch.inference_mode()`; neither builds an autograd graph inside Albucore. Choose `inference_mode` only when the returned Tensor will not later be used in a graph. It removes more bookkeeping, but an inference Tensor cannot be saved by a downstream trainable module. Use `no_grad` when the public contract permits the result to feed later training. [PyTorch grad-mode guide](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- Keep the conversion bridge visible in the benchmark. NumPy input may choose a Torch kernel and Tensor input may choose NumPy/OpenCV only when the complete route wins and its numerical tolerance is documented.

## Choose Torch operators by measured regions

Compare vectorized NumPy, OpenCV, NumKong, StringZilla, and Torch where each can implement the same contract. Torch is mandatory as a candidate; it is not a default winner.

- Prefer fused Torch APIs when they remove a full pass and preserve semantics: `var_mean`, `std_mean`, `aminmax`, `addcmul`, and a single grouped convolution are examples to measure against separate calls.
- For separable spatial filters, compare three one-axis passes, a dense one-pass kernel only when its extra arithmetic may be offset by launch/setup cost, and any OpenCV packing route that can represent all channels. Count padding and weight materialization.
- On CPU, three direct float16 depthwise `conv3d` passes were 1.11–1.30× slower than widening once, computing in float32, and restoring half; they also rounded after each pass and differed from the one-final-rounding reference by up to one float16 step. Keep the float32 working route until an upstream kernel wins both full-path timing and the current precision contract. See [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- For reductions, preserve accumulator dtype and overflow semantics before comparing `torch.sum`, NumPy, and NumKong.
- For 2D `matmul`, preserve the current wide-accumulation result. On an ARM64 CPU,
  direct `torch.mm(float16)` was 7.2–123.6× slower than Albucore's float32 public
  path across 128×64×32, 240×320×160, and 480×640×320; it differed from the
  float32-accumulated half result by up to 0.03125. A Torch float32-working
  candidate matched the current result exactly but remained 1.42–1.97× slower
  than the float32 public route. Keep the NumPy working path until an upstream
  GEMM kernel meets both requirements; see the targeted
  [PyTorch matmul issue](https://github.com/pytorch/pytorch/issues/146508#issuecomment-6106617861)
  and the broader [CPU float16 issue](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor `reduce_sum` at 16,384+ elements now calls `torch.sum(dtype=float64)` directly for global,
  per-channel, and explicit axes. The paired grid has no float16/float32 pair above 1.10× for either global or
  per-channel sums; 11 of 21 global and 16 of 21 per-channel cases are at least 5% faster in float16.
- For NumPy float16 `reduce_sum(..., axis="per_channel")`, contiguous rank-4 channel-last C=3 arrays up to 245,760
  elements use the zero-copy Torch wrapper with a float64 accumulator. In the measured N=1–4, 128×160 window, the
  full public call is 4.4–5.3× faster than the float32 peer and exactly matches NumPy float64, including `keepdims`.
  Other rank-4 dtypes/channels retain their measured routes.
- For NumPy rank-4 channel-last float16 per-channel `mean`, contiguous writable C=3 arrays from 20,480 elements use
  zero-copy Torch `mean(dtype=float64)`; C=1 arrays use the existing global Torch `sum(dtype=float64)` route from
  20,480 through 8,000,000 elements. The 101-sample public grids match NumPy float64 exactly. C=3 takes 0.14–0.23×
  float32 time; the selected C=1 windows take 0.951–1.036× with no robust slow cell. Larger C=1 arrays retain the
  NumPy float64 route; N=16 at 768×1024 remains 1.29–1.30× slower than float32. A float32-accumulator candidate
  fails the strict subnormal-residual case, recorded in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
  See the [C=3 grid](../_internal/float16_probe_20261010/mean_per_channel_nhwc_c3_public_101_20261011.json)
  and [C=1 grid](../_internal/float16_probe_20261010/mean_per_channel_nhwc_c1_public_sum_route_101_20261011.json).
- For NumPy HWC float16 per-channel `mean`, a temporary PR #400 `nk.moments` candidate with the subnormal scan bypassed measured C=1 at 0.93–1.02× float32, but C=3 strided calls were 1.15–1.17× and C=9 nine-call routing was 4.85–5.00× and existing `nk.sum(axis=(0,1))` was 4.82–4.93×. A contiguous C3 planar copy was 3.03–3.33×. Keep the CV32F/OpenCV route until a released dependency route meets the no-slower gate; axis-aware moments are tracked in [NumKong #311](https://github.com/ashvardanian/NumKong/issues/311#issuecomment-6112109242).
- For NumPy HWC float16 per-channel `std` and `mean_std`, the current route scans for subnormals and widens to float32 when present; this remains 1.32–1.77× slower than float32 in a representative HWC C=1–4 grid. A temporary wheel from NumKong PR #400 passed 32 public cells at 0.27–0.86× float32 after bypassing only that scan, with maximum errors of 1.37e-10 for mean and 1.67e-8 for std against NumPy float64. Keep the current guard until the PR is merged and released; results are in [NumKong #399 benchmark](https://github.com/ashvardanian/NumKong/issues/399), the [positive-range accuracy follow-up](https://github.com/ashvardanian/NumKong/issues/399), and the [probe JSON](../_internal/float16_probe_20261010/stats_router_numkong_pr400_moments_full_101_20261011.json).
- For NumPy HWC float16 C=9 `mean_per_channel`, the current CV32F/OpenCV route remains 1.21–1.27× slower than float32. NumKong PR #400 `moments` called once per strided channel was 4.85–5.00× slower, despite meeting `rtol=1e-7`; keep the current route and track a grouped axis-aware reduction in [NumKong #311](https://github.com/ashvardanian/NumKong/issues/311#issuecomment-6112109242).
- For global float16 image statistics, benchmark each output separately. Current Tensor `mean` uses a float32 Torch
  accumulator (float64 output, C>1, at least 40,960 elements); contiguous Tensor `std` at 40,960+ elements uses a
  float32 NumPy accumulator only when values lie in [-1, 1] and std exceeds 1e-3, with float64 Torch fallback for other values; `mean_std` uses a
  flattened CV16F `meanStdDev` call (contiguous input, at least 20,480 elements) and widens near-constant/non-finite
  data. PyTorch 2.14.1 CPU `torch.std(float16)` returns half and rejects float32/float64 `out=` on global reduction;
  the output-precision gap is tracked in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor per-channel `mean` from 40,960 elements uses one CPU Torch reduction with a float64
  accumulator on the declared channel axis. This covers NCHW/NCDHW batches, CDHW volumes, and multi-channel CHW;
  single-channel CHW retains its faster measured OpenCV route. The full grid measured 20 of 21 pairs at least 5%
  faster than float32, with no robust slow pair. A separate 101-sample run measured 8–12× improvements for NCHW
  C=3/9 and CDHW C=3 large volumes, with maximum absolute error below 4e-10 against a float64 reference.
- Contiguous float16 Tensor scalar and channel-vector `add`/`multiply` use direct Torch half operations from 16,384
  elements. Scalars keep Torch's native scalar-to-half semantics; every channel-vector element must be exactly
  representable in float16. The route is out-of-place; non-contiguous inputs, other controls, and in-place calls keep
  the NumPy bridge. The paired full grid measured all 21 `add` and all 21 `multiply` cases at least 5% faster than
  their float32 peers. A separate 11-sample interleaved run with scalar `0.1` measured 0.26–0.47× for `add` and
  0.33–0.62× for `multiply` across CHW 128×160×3, CHW 480×640×3, and NCHW 4×3×240×320.
- NumPy HWC float16 vector routes are selected by channel count and measured region: C1 uses NumKong `scale` for
  exact controls from 20,480 elements; C3 uses OpenCV CV16F from 61,440 elements; C9 `add_vector` uses NumKong
  `add` with a zero-stride half control view for exact controls, while C9 `multiply_by_vector` uses the NumPy
  float32 ufunc loop into half output. The C9 multiply route is 1.12–1.17× float32 public time with one robust
  slow pair at 128×160; direct Torch half was 1.05–1.22× slower than the full float32 Torch bridge and is not
  selected. Track the PyTorch CPU candidate in [#97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503),
  the signed-zero mismatch in [NumKong #401](https://github.com/ashvardanian/NumKong/issues/401), and NumPy ufunc
  loop work in [#32970](https://github.com/numpy/numpy/issues/32970).
- Contiguous float16 Tensor `normalize` in CHW/NCHW with C=3/9, scalar or channel-vector controls, and 16,384+ elements uses per-plane CV16F
  `addWeighted` into a preallocated channel-first output. A zero second source preserves infinity semantics. The
  101-sample paired grid found no robust slow pair; median ratios ranged from 0.30× to 1.12× float32 time. Other
  Tensor layouts use the zero-copy channel-last view and float32 working buffer.
- For contiguous NumPy HWC float16 normalization, the eager CPU `torch.addcmul` half kernel matched a float32
  input/compute/output round-trip within 0.98–1.01× time across four 480×640 and 768×1024 C=3/9 cases, with bitwise
  identical outputs. The evidence is in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- For interpolation and sampling, differential-test coordinate conventions, border rules, `align_corners`, rounding, and uint8 restoration. A faster result with shifted samples is not a candidate.
- For contiguous float16 Tensor `resize(..., INTER_NEAREST)`, use native CPU half
  interpolation from 61,440 elements for NCHW and multi-channel CHW. CHW C=1 keeps
  the existing route because the native candidate was slower on a large upscale.
  A 20-cell, 101-sample grid matched the float32 public output cast to half with
  no robust slow cell; the maximum float16/float32 ratio was 0.900×. A binary16-
  domain test checks all 65,536 bit patterns on NCHW C=1.
- For float16 Tensor `resize(..., INTER_LINEAR)` with C=1, keep the current float32/OpenCV bridge until a faster backend is available. In a 16-cell 101-sample CHW/NCHW grid, the public half route was 1.08–2.10× the float32 route and passed the 2-MAD slow screen in all cells. Direct Torch half was slower in all 16 cells (1.19–5.21×); a Torch float32 round-trip was 1.69–7.45×. Direct-half upscales differed from the public float32 result rounded to half by up to 4 ULP. Track the OpenCV CV16F resize gap in [#30216](https://github.com/opencv/opencv/issues/30216) and the CPU half kernel limitation in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503); raw samples are in [`tensor_c1_linear_torch_candidate_public_101_20261011.json`](../_internal/float16_probe_20261010/tensor_c1_linear_torch_candidate_public_101_20261011.json).
- NumPy `float16 → uint8` keeps exact float32 scaling and rounding by widening into one float32 work buffer. The current full path is 1.29–1.31× slower than float32 on six large HWC cases; track a fused CPU route in [PyTorch #97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor `from_float(..., uint8)` may use half multiply/round/clip only when the scale is exactly
  representable in float16; all 65,536 binary16 patterns stay within one uint8 unit of the float32 route, and other
  scales retain float32 rounding.
- `resize3d` uses native CPU half trilinear interpolation outside the large all-axis upscale bridge. The bridge widens
  before the NumPy volume kernel and rounds back once. On normalized data, interpolation candidates are tested against
  the float32 reference with an absolute tolerance of `float16 eps / 2`.
- Keep Torch CPU, OpenMP, and MKL thread counts at one for backend comparisons. A thread-count sweep requires an explicit user request. `OMP_NUM_THREADS` controls OpenMP regions, and `MKL_NUM_THREADS` overrides it for MKL. Do not bake one machine's thread count into a router. [PyTorch threading variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)

## Escalate missing backend capabilities

When a viable optimization needs an operation that PyTorch, NumKong, OpenCV, or NumPy does not provide, open a Feature
Request in the relevant upstream project before adding a private substitute. Link that request from the Albucore issue,
benchmark note, or pull request so readers can see the dependency.

When we can implement the operation and the upstream contribution rules allow it, open a linked upstream pull request.
Until the operation is available, benchmark the portable alternatives and state the capability gap. See the general
[performance optimization workflow](performance-optimization.md) for the project-wide policy.

## Profile before adding a specialized route

Use a short profiler capture to find dominant operators and allocations before writing a threshold or alternative backend. `torch.profiler.profile` can record CPU/CUDA activity, shapes, and Tensor allocation/deallocation; shape and stack collection perturb timing, so profile first and benchmark separately. [PyTorch profiler docs](https://docs.pytorch.org/docs/stable/profiler.html)

Check the following before adding branches:

1. Does a bridge copy, dtype conversion, padding operation, or output repair dominate the operator?
2. Does a view become contiguous internally at the selected operator?
3. Can invariant parameters move outside a pass or a loop?
4. Does the candidate reduce full-volume reads/writes or merely replace one API call?
5. Does the result keep the current precision, borders, rounding, and ownership contract?

Keep a branch only when a sustained region wins on the public benchmark. Record both winning and rejected candidates with the shapes where they lose.

`torch.compile`, autograd through an Albucore primitive, GPU, MPS, CUDA graphs, mixed precision, and distributed execution are out of scope. Do not add them to a public route, a benchmark candidate, or a routing threshold. A future task that expands the contract must define device transfer, synchronization, compilation latency, dynamic-shape, graph, and correctness behavior before measuring them.

## Tests and review record

Add tests for every accepted route and its boundaries: uint8/float16/float32, declared layouts, `C=1/3/high`, unit and short axes, ordinary and strided inputs when supported, zero-work identity, border behavior, documented working/output dtype conversions, one final uint8 restoration, and container preservation. Differential tests use an independent reference or an existing documented backend.

Every Torch optimization handoff reports:

- full-path baseline and selected route;
- work, copies, allocations, and passes deleted or retained;
- Python wrapper, vectorization, fused-op, in-place, layout, memory-format, and backend candidates considered;
- thread settings, benchmark matrix, wins, regressions, and rejected regions;
- correctness and memory evidence;
- any follow-up that needs a profiler trace, platform matrix, or a wider API contract.
