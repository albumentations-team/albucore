# Batch performance: `warp_affine3d` and `remap3d`

The existing APIs now accept rank-5 volume batches. This report records the final CPU route comparison on PyTorch 2.14.0; no pre-upgrade timings are included.

## Environment and method

- macOS 27, arm64; Albucore 0.2.18, NumPy 2.2.6, OpenCV 5.0.0, PyTorch 2.14.0.
- One Torch intra-op and inter-op thread; OpenCV parallelism disabled; OpenMP/BLAS set to one thread. MPS was outside this CPU-only run and was unavailable in the local process.
- Each timing benchmark used 2 warmups and 7 timed samples. Ratios below are candidate median divided by the matching public per-volume loop median; below 1.0 is faster. The p10–p90 range is across dtype, interpolation, border, container/layout, and, for remap, grid-container cases. It describes workload spread, not a confidence interval.
- Main shape: `DHW=(16,128,160)`, output `DHW=(8,96,120)`, channels `1/3/5/9`, batches `N=1/4/16`, uint8/float32, nearest/trilinear, constant-zero/per-channel-fill/replicate. NumPy used contiguous NDHWC; Tensor used contiguous and channel-last-strided NCDHW. Remap varied both NumPy and Tensor grids with one shared nonlinear grid.
- A compact `DHW=(16,32,40), C=1` shape checked the batch-size boundary. Every timing candidate, including direct native-N sampling, was checked for exact equality with the stacked single-volume reference before timing.
- The direct native-N candidate bypasses channel-folding dispatch and calls the 5D sampler. Single-channel runs were repeated on the main and compact shapes for `N=4/16`; wide-channel affine direct-N results were repeated on the main shape. Wide-channel remap already used the direct sampler before this follow-up.
- Wide-channel remap timing rows retain the original full matrix: for `C>1`, that candidate already reached the native 5D sampler directly. The single-channel rows are replaced with the bypassed native-N rerun because the old candidate selected folding there.
- Timings and memory were measured from the current PR worktree on PyTorch 2.14.0. Isolated per-route peak RSS methodology and results are recorded below.
- Commands used for the review rerun:

  ```bash
  uv run python benchmarks/benchmark_warp_affine3d_batch.py --full \
    --shape 16,128,160,1 --shape 16,128,160,3 --shape 16,128,160,5 \
    --shape 16,128,160,9 --shape 16,32,40,1 --batch-sizes 4 16 \
    --threads 1 --warmup 2 --repeats 7

  uv run python benchmarks/benchmark_remap3d_batch.py --full \
    --shape 16,128,160,1 --shape 16,32,40,1 --batch-sizes 4 16 \
    --threads 1 --warmup 2 --repeats 7

  uv run python benchmarks/benchmark_sampling3d_batch_memory.py
  ```

## Affine route

Each value is relative to the per-volume public loop plus stack. `Direct native-N` bypasses runtime folding and calls the 5D sampler; `Public dispatch` includes matrix/grid preparation and the selected route. The bridge candidate converts the complete CPU Tensor route through NumPy and back.

| C | N | N×C folded | Direct native-N | Public dispatch | Tensor→NumPy bridge |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.390× | 0.694× (p10–p90 0.588–0.782) | **0.381×** (0.305–0.514) | 0.474× |
| 1 | 16 | 0.466× | 0.638× (0.531–0.749) | **0.459×** (0.287–0.679) | 0.492× |
| 3 | 4 | 0.952× | 0.837× (0.647–0.907) | **0.817×** (0.642–0.897) | 0.928× |
| 3 | 16 | 1.228× | 0.772× (0.607–0.888) | **0.783×** (0.588–0.869) | 0.834× |
| 5 | 4 | 1.315× | 0.867× (0.708–0.936) | **0.844×** (0.697–0.913) | 1.098× |
| 5 | 16 | 1.339× | 0.822× (0.660–0.906) | **0.827×** (0.650–0.895) | 0.989× |
| 9 | 4 | 1.909× | 0.888× (0.730–0.987) | **0.900×** (0.703–0.962) | 1.196× |
| 9 | 16 | 1.900× | 0.856× (0.642–0.939) | **0.867×** (0.666–0.938) | 1.072× |

For `C=1`, direct native-N takes 1.37–1.78× the folding median across the two batch sizes and loses the selected public route's speedup over the loop. For `C=3/5/9`, native-N dispatch is 10–23% faster than the loop by median; folding is slower for `C>=5` and inconsistent for `C=3`. The bridge gains vary across channels and have a wide workload spread, so the runtime keeps direct dispatch for wider channels.

For the compact `C=1` shape, folding/public dispatch measured `0.363×/0.361×` at `N=4` and `0.220×/0.221×` at `N=16`; direct native-N measured `0.523×` and `0.440×` respectively.

## Remap route

`Direct native-N` is the 5D `grid_sample` candidate with folding bypassed. `Selected dispatch` is the public path: channel folding for `C=1`, and per-volume sampling with shared grid/fill preparation for `C>1`.

| C | N | N×C folded | Direct native-N | Selected dispatch | Tensor→NumPy bridge |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.529× | 0.984× (p10–p90 0.897–1.053) | **0.503×** (0.358–0.634) | 0.527× |
| 1 | 16 | 0.562× | 1.031× (0.944–1.118) | **0.554×** (0.248–0.887) | 0.558× |
| 3 | 4 | 1.056× | 1.017× | **1.006×** (0.955–1.059) | 0.999× |
| 3 | 16 | 1.582× | 1.092× | **1.002×** (0.990–1.020) | 1.001× |
| 5 | 4 | 1.236× | 1.074× | **1.011×** (0.987–1.052) | 1.002× |
| 5 | 16 | 1.887× | 1.150× | **1.005×** (0.960–1.062) | 1.003× |
| 9 | 4 | 1.815× | 1.214× | **1.014×** (0.951–1.087) | 1.003× |
| 9 | 16 | 2.825× | 1.184× | **1.003×** (0.945–1.069) | 0.986× |

For `C=1`, direct native-N is 1.83–1.86× the folding median on the main shape and 1.54–2.39× on the compact shape. For `C>1`, direct native sampling regressed against the loop by 1–21% by median. The selected shared-grid loop stays within 1.4% of the loop median across channels and batch sizes, avoiding repeated public dispatch, grid wrapping, and fill normalization. Tensor→NumPy→Tensor is effectively tied and varies by channel and batch size, so it is not selected. Folding is reserved for `C=1`.

For the compact `C=1` shape, folding/public dispatch measured `0.523×/0.528×` at `N=4` and `0.327×/0.332×` at `N=16`; direct native-N measured `0.807×` and `0.783×` respectively.

## Route-specific peak memory

Each value is process high-water RSS from a fresh child process, measured separately for the listed route. The case uses a contiguous CPU Torch float32 batch with `N=16`, `C=1/3`, input `DHW=(16,128,160)`, output `DHW=(8,96,120)`, trilinear interpolation, and nonzero constant fill. RSS includes Python, imported libraries, input, output, and route temporaries; it is not an allocation delta. Each candidate used one CPU thread.

| Operation | C | Per-volume loop | N×C folded | Direct native-N | Public dispatch | Tensor→NumPy bridge |
|---|---:|---:|---:|---:|---:|---:|
| `warp_affine3d` | 1 | 254.4 MiB | 267.7 MiB | 267.2 MiB | 267.1 MiB | 266.8 MiB |
| `warp_affine3d` | 3 | 382.7 MiB | 358.4 MiB | 359.2 MiB | 357.8 MiB | 360.4 MiB |
| `remap3d` | 1 | 253.5 MiB | 265.4 MiB | 265.1 MiB | 265.5 MiB | 265.5 MiB |
| `remap3d` | 3 | 378.3 MiB | 356.6 MiB | 357.3 MiB | 378.4 MiB | 378.5 MiB |

The selected affine route adds about 13 MiB at `C=1` versus the loop and saves about 25 MiB at `C=3`. The selected wider-channel remap route has loop-like peak RSS because it accumulates per-volume outputs before stacking; direct native-N uses about 21 MiB less peak RSS but is slower. The single-channel folding route keeps its measured speedup with an approximately 12 MiB peak-RSS increase.

## Decision

- Extend `warp_affine3d` and `remap3d` in place. Keep their existing rank-4 behavior and use rank 5 for batches: NumPy `NDHWC`, CPU Torch `NCDHW`.
- Reuse one affine matrix or remap grid for every volume. Keep the fill value shared across the batch.
- Keep the measured `N>=4, C=1` folding optimization for both APIs. Use native batch sampling for wider-channel affine. For wider-channel remap, sample volumes independently while reusing prepared controls.
- Preserve CPU-only eager Torch behavior. The batch API does not add MPS, autograd-through-the-primitive, or `torch.compile` support.

The timing scripts retain raw per-sample timings when run with `--output`. The memory script launches each candidate in a fresh process and emits its route-specific high-water RSS table.
