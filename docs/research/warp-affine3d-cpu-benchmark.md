# CPU route decision for `warp_affine3d`

Run date: 2026-08-03. This note records the CPU route selected for one `warp_affine3d` volume. It does not describe a batch API: the router accepts NumPy `DHWC` or CPU Torch `CDHW` only.

## Decision

Production uses Torch `affine_grid` followed by 5D `grid_sample` with `align_corners=False`. NumPy input shares storage with Torch through `torch.from_numpy(...).permute(...)` when the array is writable and has no negative stride. Read-only and negative-stride arrays receive one explicit C-contiguous repair copy. CPU `grid_sample` does not accept 5D uint8 input, so uint8 uses one float32 sampling buffer and final saturating round-half-up conversion.

The matrix, grid, sampling, output allocation, dtype repair, and container conversion are part of the measured public path. No size threshold routes to a different production backend.

## Quick-matrix evidence

The 2026-08-03 quick run used macOS arm64, Torch 2.13.0, NumPy 2.2.6, OpenCV 5.0.0, one Torch/OpenCV CPU thread, 11 timed repetitions, and three warmups. It covered four non-batched shapes, uint8/float32, four output scenarios, and contiguous plus channel-last-strided Tensor inputs. That is 32 NumPy cells and 64 Tensor cells.

Two representative NumPy `DHWC` rows show that public validation has no material full-path cost. For `16×128×160×5` float32, the direct bridge and public router took 2.895 ms and 2.897 ms for `8×96×120`, then 26.688 ms and 26.062 ms for `32×192×240`.

The Tensor run compared two diagnostic alternatives. Manual broadcasted grid construction was faster in none of 64 cells and had a uint8 rounding difference at a sampling boundary. It fails both the stable-win and exact-parity gates. The coverage sampler won 4 of 16 nonzero-fill cells. On a contiguous `5×16×128×160` float32 Tensor with nonzero fill, native sampling took 3.864 ms and the coverage sampler took 5.404 ms. Production keeps the shifted-input fill adapter, which uses one data sampler instead of data plus coverage sampling.

An extended nine-shape sweep also completed: the NumPy path used three timed repetitions and the Tensor path completed one full correctness/timing pass. They cover 72 NumPy and 144 Tensor single-volume scenario cells. Every public candidate was bitwise equal to its native baseline. The Tensor one-pass timings are scope evidence, not a threshold-selection signal; a release-machine run with the standard repetitions remains required before introducing a size route.

## Rejected and deferred candidates

| Candidate | Result | Reason |
|---|---|---|
| Manual broadcasted grid | Rejected | It has no stable full-path win and changes boundary rounding for uint8. |
| Coverage sampler for nonzero fill | Rejected | A second grid-sample call loses on 12 of 16 nonzero-fill quick cells, including the large representative shape. |
| Tiled output-depth grid | Deferred | The quick matrix did not show a memory failure. Add it only after an isolated peak-RSS measurement triggers the memory route. |
| OpenCV and NumKong 3D warp | Unavailable | Evaluated local versions expose no public true 3D affine sampler. |
| Native extension | Deferred | It needs a profile that attributes a production bottleneck to dense-grid allocation or sampling after the current route passes the full matrix. |

## Reproduce

```bash
uv run python benchmarks/benchmark_warp_affine3d.py --quick --threads 1
uv run python benchmarks/benchmark_warp_affine3d_tensor.py --quick --threads 1
uv run python benchmarks/benchmark_warp_affine3d.py --full --threads 1 \
  --output benchmarks/results/warp-affine3d.md
uv run python benchmarks/benchmark_warp_affine3d_tensor.py --full --threads 1 \
  --output benchmarks/results/warp-affine3d-tensor.md
```

The full run must preserve public-path correctness before reporting timings. Repeat it on the release reference machine with its fixed thread setting before adding a routing threshold or a native backend.
