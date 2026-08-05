# Torch CPU backend audit

## Decision

Albucore now uses Torch for four large CPU paths. Torch is a mandatory dependency and is imported with Albucore. Each path wraps a writable NumPy array with `torch.from_numpy`, computes on the CPU, and returns `Tensor.numpy()`. The input wrapper and final array share storage; no image-sized conversion buffer is added at either boundary.

The selected paths require at least 524,288 elements, writable storage, and no negative stride. The threshold excludes small and thread-pool-sensitive cells. A benchmark can disable the route with `TORCH_CPU_BACKEND_ENABLED` to measure its prior NumPy, OpenCV, LUT, or NumKong baseline; production keeps it enabled.

The caller's `torch.set_num_threads()` setting remains unchanged.

| Public router | Accepted Torch region | Fallback |
|---|---|---|
| `from_float` | float32 → uint8 | existing NumPy round/clip path |
| `multiply_add` | float32 scalar factor and scalar bias | NumKong `scale` |
| `normalize` | float32 with more than one channel | fused NumPy expression |
| `reduce_sum` | uint8 global; float32 global; float32 per-channel with C≤4 | existing NumPy, NumKong, and OpenCV routes |

## Reference benchmark

Run on 2026-08-02: macOS 26.4 arm64; Torch 2.13.0; NumPy 2.2.6; OpenCV 5.0.0; NumKong 7.7.0. Each cell timed the established public Albucore CPU backend against the complete NumPy → Torch → NumPy CPU path. The benchmark temporarily disables the production Torch route for its baseline. The test used 21 median samples after 5 warmups. It covered non-square HWC images with C=1/3/9, the canonical DHWC grid, and both one and twelve Torch/OpenCV CPU threads.

At one thread, the accepted regions produced these repeatable results:

| Candidate | HWC result | DHWC result |
|---|---:|---:|
| float32 → uint8 | 12/12 Torch wins; 2.50× mean | 11/11 wins; 2.62× mean |
| float32 scalar multiply-add | 12/12 wins; 1.68× mean | 11/11 wins; 1.82× mean |
| float32 normalize, C>1 | 8/8 wins; 1.71× mean | 7/7 wins; 1.55× mean |
| float32 global sum | 12/12 wins; 1.90× mean | 11/11 wins; 1.99× mean |
| float32 per-channel sum, C≤4 | 8/8 wins; 2.81× mean | 11/11 wins; 5.06× mean |

Twelve-thread results give larger wins for most large multi-channel images and volumes. Small inputs fluctuate because Torch thread-pool setup can dominate their runtime. The 524,288-element threshold keeps those cells on their established routes.

Use the maintained script to regenerate the full matrix:

```bash
uv sync --extra headless
uv run python benchmarks/benchmark_torch_cpu.py --full --threads 1
uv run python benchmarks/benchmark_torch_cpu.py --full --threads 12
uv run python benchmarks/benchmark_torch_cpu.py \
  --full --volume \
  --candidates from_float,multiply_add_scalar,normalize,reduce_sum_global_float32,reduce_sum_per_channel_float32,reduce_sum_global_uint8 \
  --threads 1
```

## Whole-code audit

| Area | Result | Reason |
|---|---|---|
| `to_float`, LUT, uint8 affine operations | Keep current paths | Torch treats uint8 indexing inputs as legacy boolean masks, so eager lookup needs an int32 or int64 full-size index buffer. int32 did not produce a durable CPU win in this matrix. Separately warmed `torch.compile` removes most of the execution cost for static shapes, but Albucore cannot charge compilation to one NumPy call. The open, actionable [PyTorch #61819](https://github.com/pytorch/pytorch/issues/61819) already requests native non-int64 indices, so no duplicate issue was created. |
| `add`, `multiply`, `add_weighted`, `power` | Keep current paths | Torch wins were sparse or changed with the CPU thread count. The public routes were faster on the one-thread reference grid. |
| `mean`, `std`, `mean_std` | Keep current paths | Torch reductions changed ranking with thread count. Existing paths also preserve their established float64 and NumKong-moments behavior. |
| `exp`, `log`, `sqrt` | Keep NumPy/OpenCV routing | Torch wins large cells with twelve threads; NumPy/OpenCV win the one-thread grid. A stable cross-thread threshold was not demonstrated. |
| `hflip`, `vflip` | Keep current routing | Torch flip performance depends strongly on HWC channel layout. Several C=3 cells regressed. |
| `matmul`, `pairwise_distances_squared` | Keep NumPy/NumKong | No durable Torch win across the size sweep. `torch.cdist` also computes a square root before squaring the result. |
| resize, border, remap, affine/perspective warp | Keep OpenCV | `interpolate` was slower in the valid resize cells. `grid_sample` does not reproduce OpenCV's interpolation, coordinate, and border semantics exactly. |
| median blur | Keep OpenCV | A Torch implementation needs `unfold`-style windows with much higher temporary memory. |
| decorators, shape helpers, RNG, validation | No Torch candidate | These paths have no full-array numerical kernel to accelerate. |

The audit intentionally excludes GPU execution and host-to-device transfers. Albucore's public inputs and outputs remain NumPy arrays.
