---
name: albucore-benchmarks
description: Running Albucore micro-benchmarks under benchmarks/, synthetic router timings, and comparing PyPI releases with uv --no-project. Use when adding benchmarks, comparing performance across versions, or documenting benchmark workflow.
---

# Albucore Benchmarks

Before designing a performance comparison, read `../performance-optimization/SKILL.md` and
`../../../docs/performance-optimization.md` completely. Extend the benchmark along the dimension that controls the
candidate, such as label density for `bincount`, table and channel layout for LUTs, or output size and dtype for random
generation.

## Layout

- `benchmarks/` - Python timing scripts. Run from repo root: `uv run python benchmarks/<script>.py`.
- `benchmarks/timing.py` - Shared `median_ms` helper for scripts executed as `python benchmarks/foo.py`.
- `./benchmark.sh` - Dataset-driven runner; expects an external `benchmark` package that is not always present in-tree. Prefer synthetic scripts for CI-style checks.
- `benchmarks/benchmark_router_synthetic.py` - Times public routers on synthetic `uint8` and `float32` arrays: HWC, plus NHWC for `mean`, `std`, and `mean_std` only.
- `benchmarks/compare_router_json.py` - Builds a Markdown table from two JSON outputs.
- `benchmarks/benchmark_resize3d_tensor.py` - Times direct Tensor, zero-copy Tensor→NumPy→Tensor, and public
  `resize3d` routes for contiguous and channel-last-strided CPU `CDHW` Tensors.
- `benchmarks/benchmark_warp_affine3d.py` - Times full single-volume NumPy `DHWC` affine paths, including the
  NumPy→Torch bridge and public router.
- `benchmarks/benchmark_warp_affine3d_tensor.py` - Times native Torch affine-grid, manual-grid and coverage-fill
  probes, and public single-volume `CDHW` routing.

## Canonical Shape Grid

Benchmark shape sweeps use channel-last Albucore conventions.

HWC images:

- `128x160` with 1, 3, 9 channels - small / warm-cache, non-square.
- `240x320` with 1, 3, 9 channels - mid-size crop, non-square.
- `480x640` with 1, 3, 9 channels - typical augmentation training crop, non-square.
- `768x1024` with 1, 3, 9 channels - high-res / full-image pass, non-square.

Use non-square H/W pairs so height-width swaps fail visibly. Avoid square-only benchmark grids.

DHWC volumes:

- `16x128x160x1`, `16x128x160x3` - thin slab, non-square in-plane.
- `32x128x160x1`, `32x128x160x3` - common nnU-Net patch depth.
- `64x128x160x3` - deeper slab.
- `96x128x160x1` - deep single-channel slab.
- `48x240x320x3` - large in-plane, multi-channel.

For `resize3d`, also include `C=5`, unit input/output spatial axes, and an explicit `D*C` value on both sides of the OpenCV encoded-channel boundary. Time its public NumPy route end-to-end, including channel packing, Torch conversions, and output repair. For Tensor input, sweep contiguous and channel-last-strided `CDHW`, direct interpolation, the zero-copy bridge, and the public router. Use `uv run python benchmarks/benchmark_resize3d.py --quick` and `uv run python benchmarks/benchmark_resize3d_tensor.py --quick` while iterating; retain the routing decision in `docs/research/resize3d-cpu-benchmark.md`.

For `warp_affine3d`, benchmark only one volume per call: NumPy `DHWC` or CPU Tensor `CDHW`. The full matrix uses
uint8/float32, `C=1/3/5/9`, canonical output sizes including a unit output axis, nearest/trilinear interpolation,
one 3×4 forward matrix per scenario, and zero/nonzero fill. NumPy timings use contiguous inputs; Tensor timings add
contiguous and channel-last-strided inputs. Test the equivalent homogeneous 4×4 representation as a contract, not a
timing route. Run `uv run python benchmarks/benchmark_warp_affine3d.py --quick --threads 1` and `uv run python
benchmarks/benchmark_warp_affine3d_tensor.py --quick --threads 1`. A manual grid, coverage sampler, tiled route, or
native extension remains a diagnostic candidate until it has exact correctness parity and a sustained full-path win.
`NDHWC` and `NCDHW` do not belong in this benchmark because the router has no batch contract.

NDHWC batch of volumes:

- `2x32x128x160x1`
- `2x32x128x160x3`
- `2x64x128x160x3`
- `4x16x128x160x3`

Channel choices: 1 for grayscale, 3 for RGB / 3-channel, and 9 for hyperspectral paths that exceed `MAX_OPENCV_WORKING_CHANNELS=4`.

## Compare Current Tree vs PyPI Release

```bash
uv run python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-main.json

uv run --no-project --with albucore==0.0.40 --with opencv-python-headless \
  --with simsimd --with stringzilla --with numpy \
  python benchmarks/benchmark_router_synthetic.py --output-json benchmarks/results/router-0.0.40.json

uv run python benchmarks/compare_router_json.py benchmarks/results/router-main.json \
  benchmarks/results/router-0.0.40.json benchmarks/results/REPORT_router_compare.md
```

Use `--quick` for smaller shape/channel grids while iterating.

## Docs

- NumKong tables and methodology: `docs/numkong-performance.md`
- Research notes: `docs/research/`
