# Public flip-view benchmark

Public `hflip` and `vflip` return NumPy slice views for every HWC channel count. The explicit `hflip_cv2` and `vflip_cv2` backends remain available when a materialized OpenCV result is required.

Run date: 2026-08-27. Platform: `macOS-26.4.1-arm64-arm-64bit`. Python: `3.10.16`. NumPy: `2.2.6`. OpenCV: `5.0.0` with 16 threads. NumKong: `7.8.0`. StringZilla: `5.1.2`. PyTorch: `2.13.0`. Warmup: 5 calls; timed repetitions: 21.

The public-router benchmark compared the current tree at commit `781a0f4` with an isolated published `albucore==0.2.15` environment using the same Python, NumPy, OpenCV, NumKong, StringZilla, and PyTorch versions. It covered non-square HWC inputs of `128x160`, `240x320`, `480x640`, and `768x1024`; `C=1/3/9`; and `uint8`/`float32` (48 cells overall, 24 per operation).

| Operation | Faster cells | Slower cells | Median old/new |
|---|---:|---:|---:|
| `hflip` | 24/24 | 0/24 | 223.53x |
| `vflip` | 24/24 | 0/24 | 201.95x |

These are direct public-router timings. A later backend that requires C-contiguous input performs its own copy; that full path is intentionally not claimed to be faster here. The contract avoids the copy for every downstream path that accepts native strides.

Reproduce with `benchmarks/benchmark_router_synthetic.py`, retaining only `hflip` and `vflip` via `--skip-ops`, then compare JSON outputs with `benchmarks/compare_router_json.py`. Run the release baseline outside the checkout so the local tree cannot shadow the published package.
