# PyTorch Optimization Workflow

Use this guide when a change adds, reviews, profiles, or routes an Albucore Torch kernel. Its purpose is to find durable speedups without changing the public array contract or reporting a microbenchmark that disappears behind conversion, allocation, or dispatch cost.

The current Albucore scope is eager CPU execution without autograd through the primitive or `torch.compile`. NumPy inputs use Albucore channel-last layouts and Tensor inputs declare their own channel-first layout. AlbumentationsX validates CPU placement, layout, strides, autograd state, and parameter ranges before caller-prevalidated 3D primitives run. Do not repeat those checks in their hot paths.

## Start with the whole operation

Measure the path users execute:

1. Include container dispatch, `torch.from_numpy`, axis views, dtype casts, kernel construction, output layout repair, and `Tensor.numpy()` for a NumPy route.
2. For a Tensor route, accept an already-created CPU Tensor and state that its construction is excluded.
3. Hold Torch, OpenCV, BLAS, and interop thread counts fixed. Set `torch.set_num_threads` before eager work; set `torch.set_num_interop_threads` once at process startup, before inter-op work. `torch.utils.benchmark.Timer` is useful for isolating one Torch core because it warms up and controls the Torch threadpool; Albucore's benchmark scripts remain the source of truth for public-route decisions. [PyTorch thread controls](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html), [PyTorch benchmark docs](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
4. Warm up lazy initialization, collect repeated samples, and report median plus spread, hardware, library versions, shapes, dtype, strides, parameter values, and allocation mode.
5. Establish numerical, dtype, shape, range, layout, aliasing, and mutation parity before accepting a faster route.

Benchmark the canonical non-square image or DHWC volume grid, then add the dimension that controls the proposal: kernel radius, channel count, contiguity, output size, or thread count. A local kernel benchmark can explain a result. It cannot select a public route by itself.

## Remove Python and allocation work first

Audit every per-call conversion, allocation, reshape, and Python/Tensor boundary before changing an operator.

- `permute`, `movedim`, `unsqueeze`, and `squeeze` are usually metadata views. Do not add `.contiguous()` or `clone()` defensively; each can materialize the full volume. Benchmark any backend that requires materialization end to end.
- `torch.from_numpy` and CPU `Tensor.numpy()` can share storage. They require an eligible CPU buffer and do not repair unsupported negative strides, non-writable input, device placement, or autograd state. Copy exactly once only when the documented contract requires a repair.
- Return `np.asarray(result.numpy())` after the final layout view when the static checker needs an ndarray rather than `Any`. For an array already returned by `Tensor.numpy()`, `np.asarray` returns the same object without copying. `torch.from_numpy` goes in the opposite direction and cannot implement a Tensor-to-NumPy return.
- Build a scalar, per-channel Tensor, filter weight, affine matrix, or index map once per public call. Normalize a caller-supplied NumPy filter kernel to the documented working dtype before Tensor work; a default float64 NumPy kernel cannot be a Conv weight for a float32 volume. Cache it only when callers actually reuse an identical value and the cache has bounded lifetime, keying, device, dtype, and memory behavior.
- Keep decisions about sizes, modes, and parameter tuples in Python before the Tensor hot loop. Avoid `.item()`, `.tolist()`, or Python branching on Tensor values inside a repeated Tensor path. On accelerators these can synchronize execution; on CPU they still add wrapper work.
- Prefer one float32 working conversion and one final uint8 saturation/rounding step. Do not cast or clip after every separable pass unless that is the documented numerical contract.
- Use `out=` or in-place Tensor operations only after proving ownership, aliasing, and backend semantics. In-place work is a candidate, not an automatic optimization.

## Preserve the Tensor contract

Every Tensor route states its logical layout independently of NumPy conventions. A 4D Tensor can mean `NCHW` or `CDHW`; never infer that choice from an axis size.

- Preserve the caller's container and declared layout.
- Do not silently move a Tensor, call `.detach()`, make it contiguous, or change a dtype to rescue an unsupported input unless the public contract explicitly documents that fallback. The caller-owned adapter or public contract decides those actions.
- "Same dtype as input" applies only to the listed supported dtypes. When a primitive deliberately has a fallback dtype, state its working and output dtype and test it, including any identity fast path. For example, `gaussian_blur3d` converts an unexpected float64 volume to float32 and returns float32.
- Run the primitive under `torch.no_grad()` or `torch.inference_mode()`; neither builds an autograd graph inside Albucore. Choose `inference_mode` only when the returned Tensor will not later be used in a graph. It removes more bookkeeping, but an inference Tensor cannot be saved by a downstream trainable module. Use `no_grad` when the public contract permits the result to feed later training. [PyTorch grad-mode guide](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- Keep the conversion bridge visible in the benchmark. NumPy input may choose a Torch kernel and Tensor input may choose NumPy/OpenCV only when the complete route wins and its numerical tolerance is documented.

## Choose Torch operators by measured regions

Compare vectorized NumPy, OpenCV, NumKong, StringZilla, and Torch where each can implement the same contract. Torch is mandatory as a candidate; it is not a default winner.

- Prefer fused Torch APIs when they remove a full pass and preserve semantics: `var_mean`, `std_mean`, `aminmax`, `addcmul`, and a single grouped convolution are examples to measure against separate calls.
- For separable spatial filters, compare three one-axis passes, a dense one-pass kernel only when its extra arithmetic may be offset by launch/setup cost, and any OpenCV packing route that can represent all channels. Count padding and weight materialization.
- For reductions, preserve accumulator dtype and overflow semantics before comparing `torch.sum`, NumPy, and NumKong.
- For interpolation and sampling, differential-test coordinate conventions, border rules, `align_corners`, rounding, and uint8 restoration. A faster result with shifted samples is not a candidate.
- Use Torch CPU thread counts as an explicit benchmark dimension. `OMP_NUM_THREADS` controls OpenMP regions, and `MKL_NUM_THREADS` overrides it for MKL. Do not bake one machine's thread count into a router. [PyTorch threading variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)

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

## Treat memory format as a measured CPU candidate

`channels_last_3d` is a stride layout for 5D `NCDHW` Tensors; it does not change the public logical axes. It can help supported convolution implementations, but a conversion can cost more than it saves for a one-off Albucore primitive. Compare a route that preserves the format through adjacent operators against the same complete route in contiguous format. [PyTorch memory-format docs](https://docs.pytorch.org/docs/stable/tensor_attributes.html)

`torch.compile`, autograd through an Albucore primitive, GPU, MPS, CUDA graphs, mixed precision, and distributed execution are out of scope. Do not add them to a public route, a benchmark candidate, or a routing threshold. A future task that expands the contract must define device transfer, synchronization, compilation latency, dynamic-shape, graph, and correctness behavior before measuring them.

## Tests and review record

Add tests for every accepted route and its boundaries: uint8/float32, declared layouts, `C=1/3/high`, unit and short axes, ordinary and strided inputs when supported, zero-work identity, border behavior, one final uint8 restoration, and container preservation. Differential tests use an independent reference or an existing documented backend.

Every Torch optimization handoff reports:

- full-path baseline and selected route;
- work, copies, allocations, and passes deleted or retained;
- Python wrapper, vectorization, fused-op, in-place, layout, memory-format, and backend candidates considered;
- thread settings, benchmark matrix, wins, regressions, and rejected regions;
- correctness and memory evidence;
- any follow-up that needs a profiler trace, platform matrix, or a wider API contract.
