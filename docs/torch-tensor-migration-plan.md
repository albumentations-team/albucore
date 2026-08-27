# Future CPU Tensor roadmap

This document lists the remaining work for accepting eager CPU `torch.Tensor`
data in Albucore and AlbumentationsX. It contains forward-looking decisions and
acceptance gates. Completed implementation details belong in the code,
`docs/public-api.md`, or a saved benchmark report.

## Target contract

The public container determines the output container:

| Input | Layout | Output |
|---|---|---|
| `np.ndarray` image | `HWC` | `np.ndarray` with the same layout |
| `np.ndarray` image batch | `NHWC` | `np.ndarray` with the same layout |
| `np.ndarray` volume | `DHWC` | `np.ndarray` with the same layout |
| `torch.Tensor` image | `CHW` | `torch.Tensor` with the same layout |
| `torch.Tensor` image batch | `NCHW` | `torch.Tensor` with the same layout |
| `torch.Tensor` volume | `CDHW` | `torch.Tensor` with the same layout |

AlbumentationsX supplies the target kind or explicit layout before calling a
low-level router. A four-dimensional Tensor is never classified from axis-size
heuristics. Callers also validate CPU placement, supported dtype, strides,
`requires_grad=False`, control data, and output semantics before entering
Albucore. The router performs backend dispatch and kernel work only.

The first supported path is eager CPU execution. A Tensor may use a NumPy or
OpenCV fallback through one adapter, and a NumPy input may use a Torch helper
only when the complete conversion-inclusive path wins its benchmark.

## Work remaining

### 1. Backend-neutral dispatch

- [ ] Define shared `NumpyImage` and `TorchImage` types plus container-preserving
  overloads for public routers.
- [ ] Define an internal layout descriptor with container, channel axis, spatial
  axes, and batch/depth context.
- [ ] Keep Tensor ↔ NumPy conversion in one adapter. Count every axis view,
  `.contiguous()` call, allocation, and output conversion.
- [ ] Keep backend-specific kernels out of package `__all__`; expose routers and
  shared types only.
- [ ] Store the current container and layout in Compose so adjacent helpers can
  share one backend segment.

Completion gate: an ambiguous four-dimensional Tensor is rejected at the
upstream boundary, and a valid Tensor round trip preserves container, layout,
dtype, values, and aliasing semantics documented by the router.

### 2. Tensor-aware wrappers

- [ ] Add container-aware implementations for `preserve_channel_dim`, `clipped`,
  `float32_io`, and `uint8_io`.
- [ ] Add separate channel-last and channel-first reshape tables to
  `batch_transform`.
- [ ] Keep `maybe_process_in_chunks` explicitly NumPy/OpenCV-specific.
- [ ] Test contiguous and strided inputs without adding defensive copies that
  the caller contract does not require.

Completion gate: wrapper tests cover `C=1`, `C=3`, and high-channel inputs for
every supported layout, and benchmark counters show no repeated conversion
between adjacent helpers on one backend.

### 3. Native Torch helpers

Port a helper only after its complete Tensor path beats the NumPy fallback or
removes a measured conversion. Candidate groups:

- [ ] conversion, arithmetic, elementwise math, and clipping;
- [ ] statistics and per-image normalization;
- [ ] flips, matrix operations, and squared distances;
- [ ] resize, remap, affine/perspective warp, and local-window operations;
- [ ] shared and per-channel LUTs, including index-buffer memory cost.

Each candidate needs differential tests for values, tolerance, range, shape,
layout, dtype, aliasing, and mutation. A faster isolated kernel is rejected if
its adapter or allocation makes the connected Compose segment slower.

### 4. AlbumentationsX integration

- [ ] Extend `ImageType`/`VolumeType` and dispatch targets for Tensor.
- [ ] Accept CPU Tensor at Compose input and reject `requires_grad=True` before
  the first Albucore call.
- [ ] Use target context to select `CHW`, `NCHW`, or `CDHW`.
- [ ] Keep one Tensor → NumPy transition before a NumPy fallback segment and one
  reverse transition after it; group adjacent helpers by backend.
- [ ] Preserve shared transform parameters across images, masks, boxes, and
  keypoints even when dense arrays change container.
- [ ] Verify Compose, replay, serialization, deterministic seeds, worker setup,
  and multiprocessing DataLoader behavior.

Completion gate: the complete Tensor Compose path is no slower than
`Tensor → existing NumPy Compose → Tensor`, and the complete NumPy path is no
slower than the current NumPy baseline.

## Benchmark gate

Record the full path, not only a primitive. Every accepted route must include:

- non-square `H/W` sizes and `C=1/3/high` cases;
- `uint8` and `float32` where the public contract permits them;
- contiguous and supported strided inputs;
- scalar, per-channel, and full-array parameters;
- fixed Torch, OpenCV, BLAS, and inter-op thread settings;
- warmups, repetitions, hardware, dependency versions, allocations, and peak
  memory;
- correctness tolerances and rejected candidates.

Use the acceptance rules in
[`docs/performance-optimization.md`](performance-optimization.md). Save the
report under `benchmarks/results/` and update the routing table or public API
doc only after the report supports a stable region.

## Deferred capabilities

Keep these outside the release gate until an end-to-end CPU Tensor path is
stable:

- `torch.compile` and `torch.func.vmap`;
- CUDA, MPS, GPU routing, and distributed execution;
- autograd through Albucore primitives;
- Tensor-native random-field and generator plumbing.

Any future expansion must define device transfer, synchronization, compilation
latency, dynamic-shape behavior, graph semantics, and correctness tests before
it receives a benchmark threshold.
