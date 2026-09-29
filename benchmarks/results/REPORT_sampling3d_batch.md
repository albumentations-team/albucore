# Batch performance: `warp_affine3d` and `remap3d`

The existing APIs now accept rank-5 volume batches. This report records the final CPU route comparison on PyTorch 2.14.0; no pre-upgrade timings are included.

## Environment and method

- macOS 27, arm64, Python 3.10.16; Albucore 0.2.18, NumPy 2.2.6, OpenCV 5.0.0, PyTorch 2.14.0.
- One Torch intra-op and inter-op thread; OpenCV parallelism disabled; OpenMP/BLAS set to one thread. MPS was outside this CPU-only run and was unavailable in the local process.
- Each benchmark used 2 warmups and 7 timed samples. Ratios below are candidate median divided by the matching public per-volume loop median; below 1.0 is faster. The p10–p90 range is across dtype, interpolation, border, container/layout, and, for remap, grid-container cases. It describes workload spread, not a confidence interval.
- Main shape: `DHW=(16,128,160)`, output `DHW=(8,96,120)`, channels `1/3/5/9`, batches `N=1/4/16`, uint8/float32, nearest/trilinear, constant-zero/per-channel-fill/replicate. NumPy used contiguous NDHWC; Tensor used contiguous and channel-last-strided NCDHW. Remap varied both NumPy and Tensor grids with one shared nonlinear grid.
- A compact `DHW=(16,32,40), C=1` shape checked the batch-size boundary. Every measured candidate was checked for exact equality with the stacked single-volume reference before timing.
- Peak RSS was 1262.7 MiB for the affine process and 1168.0 MiB for the final remap process. This is the maximum for each complete process, not a per-case memory measurement.
- Base revision: `6fe4af5`; benchmark and implementation changes were present in the working tree. Commands used:

  ```bash
  uv run python benchmarks/benchmark_warp_affine3d_batch.py --full \
    --shape 16,128,160,1 --shape 16,128,160,3 --shape 16,128,160,5 \
    --shape 16,128,160,9 --shape 16,32,40,1 \
    --threads 1 --warmup 2 --repeats 7

  uv run python benchmarks/benchmark_remap3d_batch.py --full \
    --shape 16,128,160,1 --shape 16,128,160,3 --shape 16,128,160,5 \
    --shape 16,128,160,9 --shape 16,32,40,1 \
    --threads 1 --warmup 2 --repeats 7
  ```

## Affine route

Each value is relative to the per-volume public loop plus stack. `Native dispatch` is the public rank-5 API. The bridge candidate converts the complete CPU Tensor route through NumPy and back.

| C | N | N×C folded | Native dispatch | Tensor→NumPy bridge |
|---:|---:|---:|---:|---:|
| 1 | 4 | 0.403× | **0.377×** (p10–p90 0.307–0.526) | 0.419× |
| 1 | 16 | 0.530× | **0.541×** (0.315–0.687) | 0.511× |
| 3 | 4 | 1.047× | **0.845×** (0.675–0.923) | 0.836× |
| 3 | 16 | 1.318× | **0.816×** (0.576–0.886) | 0.808× |
| 5 | 4 | 1.378× | **0.873×** (0.656–0.930) | 0.870× |
| 5 | 16 | 1.435× | **0.846×** (0.641–0.932) | 0.831× |
| 9 | 4 | 1.977× | **0.894×** (0.712–0.953) | 0.914× |
| 9 | 16 | 1.885× | **0.899×** (0.638–0.951) | 0.863× |

The public sampler folds `N>=4, C=1` into its channel axis; for other channel counts it keeps the native rank-5 call. Folding wider-channel volumes loses to the loop, while native dispatch is 10–18% faster by median. The bridge's roughly 1–4% median gains in some Tensor cases are small relative to the case spread and are not consistent across channels, so the runtime keeps direct dispatch.

For the compact `C=1` shape, public dispatch measured 0.337× at `N=4` and 0.258× at `N=16` relative to the same loop baseline.

## Remap route

`Native batch sampler` is the direct 5D `grid_sample` candidate. `Selected dispatch` is the final public path: channel folding for `C=1`, and per-volume sampling with shared grid/fill preparation for `C>1`.

| C | N | N×C folded | Native batch sampler | Selected dispatch | Tensor→NumPy bridge |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.518× | 0.482× | **0.504×** (p10–p90 0.334–0.626) | 0.518× |
| 1 | 16 | 0.672× | 0.671× | **0.631×** (0.214–0.757) | 0.677× |
| 3 | 4 | 1.056× | 1.017× | **1.006×** (0.955–1.059) | 0.999× |
| 3 | 16 | 1.582× | 1.092× | **1.002×** (0.990–1.020) | 1.001× |
| 5 | 4 | 1.236× | 1.074× | **1.011×** (0.987–1.052) | 1.002× |
| 5 | 16 | 1.887× | 1.150× | **1.005×** (0.960–1.062) | 1.003× |
| 9 | 4 | 1.815× | 1.214× | **1.014×** (0.951–1.087) | 1.003× |
| 9 | 16 | 2.825× | 1.184× | **1.003×** (0.945–1.069) | 0.986× |

For `C>1`, native batch sampling regressed against the loop by 1–21% by median. The selected shared-grid loop stays within 1.4% of the loop median across channels and batch sizes, avoiding repeated public dispatch, grid wrapping, and fill normalization. Tensor→NumPy→Tensor is effectively tied and varies by channel and batch size, so it is not selected. Folding is reserved for `C=1`.

For the compact `C=1` shape, public dispatch measured 0.585× at `N=4` and 0.439× at `N=16` relative to the per-volume loop.

## Decision

- Extend `warp_affine3d` and `remap3d` in place. Keep their existing rank-4 behavior and use rank 5 for batches: NumPy `NDHWC`, CPU Torch `NCDHW`.
- Reuse one affine matrix or remap grid for every volume. Keep the fill value shared across the batch.
- Keep the measured `N>=4, C=1` folding optimization for both APIs. Use native batch sampling for wider-channel affine. For wider-channel remap, sample volumes independently while reusing prepared controls.
- Preserve CPU-only eager Torch behavior. The batch API does not add MPS, autograd-through-the-primitive, or `torch.compile` support.

The benchmark scripts retain raw per-sample timings when run with `--output`; this report keeps the decision evidence and workload spread without duplicating the full machine-specific tables.
