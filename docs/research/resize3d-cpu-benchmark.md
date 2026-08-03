# CPU benchmark for `resize3d`

Run date: 2026-08-03. This note records which complete CPU path `resize3d` selects for NumPy `DHWC` volumes and Torch `CDHW` tensors. Timings include every adapter, conversion, kernel call, and output allocation used by the public router.

## Decision

`resize3d` uses a small shape-aware router. A NumPy volume may use a zero-copy `DHWC → CDHW → DHWC` Torch route only for measured linear, non-antialiased regions. The input must be writable and have no negative stride.

A Tensor input uses native `F.interpolate` for nearest, downscale, mixed, `D → 1`, and very small linear resize. For a linear resize that strictly enlarges depth, height, and width and produces at least 10,000 `C×D×H×W` elements, it uses zero-copy `CDHW → DHWC → CDHW` views and the selected NumPy/OpenCV route. This reduces the slow Tensor upscale region by 2.5–5.8× on the canonical measured cells. Float32 stays within `rtol=2e-4`, `atol=3e-5` of native interpolation; uint8 differs by at most one value on the selected large all-axis-upscale region. The caller has already supplied a CPU Tensor with `requires_grad=False`.

The remaining NumPy regions use the fastest measured CPU path:

- all-axis float32 downscale and multi-channel downscale: Torch full route;
- uint8 `D → 1`: Torch full route; float32 `D → 1`: three float32 NumPy passes with one final uint8 round when applicable;
- moderate `D*C` with both H and W changing: joint H/W OpenCV packing plus a depth pass;
- an unchanged H or W axis: three OpenCV axis-packing passes;
- small one-channel all-axis downscale and upscales above OpenCV's encoded channel limit: per-slice OpenCV plus a depth pass.

Nearest-neighbour and NumPy `antialias=True` stay on OpenCV/NumPy paths. PyTorch does not support antialiased 5D trilinear interpolation; the public Tensor call reports that limitation.

## Method

The machine was macOS 26.4 arm64 with Torch 2.13.0, NumPy 2.2.6, and OpenCV 5.0.0. Each cell used 3 warmups and 11 timed repetitions. The script ran one process per representative shape to avoid retaining temporary buffers from the allocation-heavy NumPy baseline. It was run with one and twelve Torch/OpenCV CPU threads.

The NumPy candidate set was: pure NumPy three-pass reference, OpenCV axis packing, joint H/W packing plus depth, per-slice OpenCV plus depth, and complete NumPy-to-Torch-to-NumPy. The Tensor candidate set was: native `F.interpolate`, zero-copy Tensor-to-NumPy bridge, and public `resize3d`. Float32 candidates were compared with the half-pixel three-pass reference (`rtol=2e-4`, `atol=3e-5`). Uint8 candidates preserved shape, dtype, and range; OpenCV rounds between passes while Torch and the reference round once at the end.

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

## Tensor-to-Tensor all-axis upscale, one thread

These medians include the zero-copy view adapters and output allocation. Torch import and input creation are outside the timed region.

| CDHW input → output | dtype | Native Torch | Selected bridge | Speedup |
|---|---|---:|---:|---:|
| `1×16×128×160 → 1×32×192×240` | uint8 | 3.885 ms | 0.674 ms | 5.76× |
| `1×16×128×160 → 1×32×192×240` | float32 | 3.652 ms | 1.379 ms | 2.65× |
| `5×16×128×160 → 5×32×192×240` | uint8 | 20.422 ms | 8.261 ms | 2.47× |
| `5×16×128×160 → 5×32×192×240` | float32 | 18.008 ms | 6.325 ms | 2.85× |

The bridge is deliberately limited to large all-axis upscale. At the 10,000-element threshold it beat float32 Torch by 1.11× with a 0.0004 ms median absolute deviation, and uint8 Torch by 1.35×; the threshold therefore retains a measured win while avoiding very small noisy cells. For `D → 1`, native float32 Tensor interpolation was 0.201 ms on `5×16×128×160`, while the bridge took 1.658 ms because the NumPy router correctly selected its three-pass reference path. For all-axis downscale and most mixed cases, the two paths tie or native Torch wins.

## Reproduce

```bash
uv run python benchmarks/benchmark_resize3d.py --quick --threads 1
uv run python benchmarks/benchmark_resize3d.py --shape 16,128,160,5 --threads 1
uv run python benchmarks/benchmark_resize3d_tensor.py --quick --threads 1
uv run python benchmarks/benchmark_resize3d.py --shape 64,64,80,9 --threads 12
uv run python benchmarks/benchmark_resize3d.py --shape 48,240,320,1 --threads 12
```

Use `--full` for the maintained canonical shape list. On memory-constrained machines, repeat `--shape` for each canonical cell; this keeps the NumPy three-pass reference from retaining unrelated volume buffers in the same process.

## Rejected universal routes

Pure NumPy creates three full float32 outputs. It is retained only for the measured float32 `D → 1` region. OpenCV joint packing cannot represent `D*C` above its encoded channel limit and loses when H or W is unchanged. Per-slice OpenCV adds Python calls and loses in most moderate-channel upscales. NumPy-to-Torch-to-NumPy is fast in several regions, but the conversion, uint8 float cast, and thread-pool cost make it slower in other regions. Tensor-to-NumPy bridging loses outside large all-axis upscale and has a wider uint8 difference there, so it is not a universal Tensor route. A custom C++ or Rust extension has no justified trigger from this matrix.
