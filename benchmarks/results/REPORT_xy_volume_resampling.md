# XY-only volume resampling

NumPy XY downscales with unchanged depth now avoid packing depth into OpenCV channels in the measured large-plane region. XY-only linear affine warps share one plane grid across slices and batch items. General 3D geometry, exact identity aliasing, nearest affine interpolation, and caller-owned validation retain their existing paths.

## Measurements

Measured on 2026-10-05: Apple M4 Max, macOS 27.0.1 arm64, NumPy 2.2.6, OpenCV 5.0.0, Torch 2.14.0. The baseline is source revision `c64a777eb2970dfcfbd7d6c63cbfacd5a06504f0` (manifest version 0.2.20); the candidate is this change. Both public implementations run in the same process. Imports and input construction are excluded. Dispatch, matrix preparation, NumPy/Torch views, allocation, sampling, and returned layout are included. OpenMP/BLAS environment variables and both Torch thread pools are set to one; OpenCV internal parallelism is disabled.

Each candidate receives ten warmups followed by nine rotating timing blocks. Per-candidate call counts target 50 ms per block, with a maximum of 3,000 calls. Values below are medians of block averages. All calls allocate independent output storage. Float32 inputs use seeded values in `[0, 1]`; affine fill is zero.

| Public NumPy workload | Dtype | Before ms | After ms | Speedup | Max difference |
|---|---|---:|---:|---:|---:|
| Resize `(30,256,256,1)` → `(30,32,32,1)` | uint8 | 1.303 | 0.083 | 15.6× | 1 |
| Resize `(30,256,256,1)` → `(30,32,32,1)` | float32 | 1.656 | 0.115 | 14.4× | 0 |
| XY affine `(30,256,256,1)`, same output shape | uint8 | 26.476 | 4.554 | 5.8× | 1 |
| XY affine `(30,256,256,1)`, same output shape | float32 | 26.148 | 4.242 | 6.2× | 1.79e-6 |

Affine uses a centred 13-degree rotation with subpixel XY translation. The paired campaign contains 136 cases: the canonical non-square DHWC sizes, the issue workload, depth 513, uint8/float32, C=1/3/5/9, direct Tensor views and contiguous CDHW inputs, and N=1/4/16 batches with NumPy, strided NCDHW and contiguous NCDHW inputs. NumPy affine speedups range from 2.9× to 6.7×, direct Tensor speedups from 2.9× to 6.1×, and batch speedups from 2.2× to 7.2×. No paired case regressed by more than 10%; unchanged small-plane resize cases stayed within 4% of the baseline.

A downstream Compose comparison used Python 3.12.7 and NumPy 2.5.1 with the same CPU/thread settings, replacing only the imported Albucore kernel between blocks. `Resize3D((30,32,32))` improved from 1.340 to 0.102 ms for uint8 and 1.992 to 0.121 ms for float32. A fixed 13-degree XY `Affine3D` improved from 25.965 to 5.667 ms for uint8 and 28.537 to 4.993 ms for float32. These retain the complete transform and Compose overhead.

## Routing and numerical limits

Resize uses the existing per-slice OpenCV helper when D is unchanged, both H and W shrink, D is greater than one, and one input plane contains at least 16,384 elements including channels. Tiny-plane loops were slower than packed calls, so they retain their previous routing. Fractional uint8 linear downscales also retain their previous routing: backend changes produced differences of 2–3 levels in additional checks. Integral uint8 downscales, nearest, and antialiased large-plane downscales are covered by the new route. Mixed scales, upscales and direct Tensor resize routing are unchanged.

Native batched Torch bilinear is retained as a resize benchmark candidate. A separate fixed-XY sweep over D=1–512 found modest float32 crossover regions, followed by ties or reversals at larger working sets. This change does not encode those noisy depth thresholds as runtime policy.

Affine selects the shared-plane route only for linear interpolation, unchanged output depth, an exact identity Z row, and zero XY/Z coupling. Single-channel arrays fold N*D into 2D channels. Other layouts merge N and D only when their strides permit a view; otherwise each volume is sampled into a preallocated output using the same grid. This avoids a full input repack for contiguous multi-channel Tensor batches.

Regression tests compare the affine route with the original native 3D sampler, including unit axes, per-channel fill, replicate borders, read-only/negative-stride NumPy inputs, contiguous/strided Tensors and depths 511/513. The normalized float32 cases use absolute tolerance 3e-5; uint8 permits one level. Nearest uses the original sampler and retains exact categorical half-voxel behaviour. Floating-point differences depend on value scale; the normalized-image tolerance is not an absolute bound for arbitrary-valued arrays.

## Memory and work removed

For `(30,256,256)`, the original affine grid occupies 22.5 MiB. The new grid stores one three-coordinate plane of 0.75 MiB and samples through its XY view. No full D×H×W grid is built. Grid preparation is shared across all batch items. Source and result remain separate; in-place resampling would overwrite values still needed by neighbouring samples.

A fresh-process peak RSS check used N=4, D=30, H=W=256, C=3, float32, a shared XY matrix and unchanged output size. It includes imports, input creation and one complete public call:

| Container | Before MiB | After MiB |
|---|---:|---:|
| NumPy NDHWC | 450.7 | 400.6 |
| Contiguous Tensor NCDHW | 451.6 | 423.1 |

These are single-process peak measurements, not isolated tensor allocation counts. NumPy/OpenCV and native Torch were compared; LUTs, integer grouped reductions, random generators, NumKong and StringZilla do not implement this spatial interpolation. Random input generation is outside the timed region. AlbumentationsX already delegates both operations to these primitives, so no downstream kernel duplication is introduced.

## Reproduce

```bash
uv run python benchmarks/benchmark_resize3d.py --quick
uv run python benchmarks/benchmark_resize3d.py --shape 30,256,256,1 --scenario xy_down
uv run python benchmarks/benchmark_warp_affine3d.py --quick
uv run python benchmarks/benchmark_warp_affine3d.py --shape 30,256,256,1 --scenario xy
uv run python benchmarks/benchmark_warp_affine3d_batch.py --quick --xy-only --threads 1
uv run python benchmarks/benchmark_warp_affine3d_batch.py --shape 16,32,40,9 --batch-sizes 1 4 16 --xy-only
```

Repeat `--shape D,H,W,C` at fixed H/W/C to isolate depth. The affine benchmarks retain the native 3D baseline; the resize benchmark includes packed, per-slice, native 3D Torch and native batched 2D Torch candidates. The depth and allocation evidence here applies to one CPU and one thread; broader platform thresholds require new measurements.
