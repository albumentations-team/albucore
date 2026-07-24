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
