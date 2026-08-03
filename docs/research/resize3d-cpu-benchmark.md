# CPU benchmark for `resize3d`

Run date: 2026-08-03. This note answers one question: which complete CPU path should public `resize3d` use for a NumPy `DHWC` volume? The timed path includes every permutation, NumPy/Torch wrapper, dtype conversion, interpolation call, and returned NumPy array.

## Decision

`resize3d` uses a small shape-aware router. A CPU `torch.Tensor` always uses native `F.interpolate`. A NumPy volume may use a zero-copy `DHWC → CDHW → DHWC` Torch route only for measured linear, non-antialiased regions. The input must be writable and have no negative stride.

The remaining NumPy regions use the fastest measured CPU path:

- all-axis float32 downscale and multi-channel downscale: Torch full route;
- uint8 `D → 1`: Torch full route; float32 `D → 1`: three float32 NumPy passes with one final uint8 round when applicable;
- moderate `D*C` with both H and W changing: joint H/W OpenCV packing plus a depth pass;
- an unchanged H or W axis: three OpenCV axis-packing passes;
- small one-channel all-axis downscale and upscales above OpenCV's encoded channel limit: per-slice OpenCV plus a depth pass.

Nearest-neighbour and NumPy `antialias=True` stay on OpenCV/NumPy paths. PyTorch does not support antialiased 5D trilinear interpolation; the public Tensor call reports that limitation.

## Method

The machine was macOS 26.4 arm64 with Torch 2.13.0, NumPy 2.2.6, and OpenCV 5.0.0. Each cell used 3 warmups and 11 timed repetitions. The script ran one process per representative shape to avoid retaining temporary buffers from the allocation-heavy NumPy baseline. It was run with one and twelve Torch/OpenCV CPU threads.

The candidate set was: pure NumPy three-pass reference, OpenCV axis packing, joint H/W packing plus depth, per-slice OpenCV plus depth, and complete NumPy-to-Torch-to-NumPy. Float32 candidates were compared with the half-pixel three-pass reference (`rtol=2e-4`, `atol=3e-5`). Uint8 candidates preserved shape, dtype, and range; OpenCV rounds between passes while Torch and the reference round once at the end.

## Representative one-thread results

The values below are median milliseconds for the public route. They show why no single backend is sufficient.

| DHWC input → output | dtype | Selected complete path | Public median (ms) |
|---|---|---|---:|
| `16×128×160×5 → 8×96×120×5` | uint8 | NumPy → Torch → NumPy | 1.029 |
| `16×128×160×5 → 8×96×120×5` | float32 | NumPy → Torch → NumPy | 0.853 |
| `16×128×160×5 → 32×192×240×5` | uint8 | joint H/W OpenCV + depth | 8.462 |
| `16×128×160×5 → 32×192×240×5` | float32 | joint H/W OpenCV + depth | 6.037 |
| `64×64×80×9 → 32×48×60×9` | uint8 | NumPy → Torch → NumPy | 1.626 |
| `64×64×80×9 → 128×96×120×9` | float32 | per-slice OpenCV + depth | 11.423 |
| `48×240×320×1 → 24×240×480×1` | uint8 | OpenCV axis packing | 2.721 |
| `48×240×320×1 → 1×180×320×1` | float32 | NumPy three-pass | 0.097 |

`64×64×80×9` has `D*C=576`, above the OpenCV packed-channel limit. That cell verifies the fallback without needing a synthetic enormous volume.

## Reproduce

```bash
uv run python benchmarks/benchmark_resize3d.py --quick --threads 1
uv run python benchmarks/benchmark_resize3d.py --shape 16,128,160,5 --threads 1
uv run python benchmarks/benchmark_resize3d.py --shape 64,64,80,9 --threads 12
uv run python benchmarks/benchmark_resize3d.py --shape 48,240,320,1 --threads 12
```

Use `--full` for the maintained canonical shape list. On memory-constrained machines, repeat `--shape` for each canonical cell; this keeps the NumPy three-pass reference from retaining unrelated volume buffers in the same process.

## Rejected universal routes

Pure NumPy creates three full float32 outputs. It is retained only for the measured float32 `D → 1` region. OpenCV joint packing cannot represent `D*C` above its encoded channel limit and loses when H or W is unchanged. Per-slice OpenCV adds Python calls and loses in most moderate-channel upscales. NumPy-to-Torch-to-NumPy is fast in several regions, but the conversion, uint8 float cast, and thread-pool cost make it slower in other regions. A custom C++ or Rust extension has no justified trigger from this matrix.
