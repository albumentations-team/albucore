# CPU route decision for `gaussian_blur3d`

Run date: 2026-08-05. This note records the selected CPU implementation for one prevalidated NumPy `DHWC` or CPU Torch `CDHW` volume. It does not define target-level mask behavior.

## Decision

Production uses three grouped Torch `conv3d` passes, one for each of depth, height, and width. Each pass uses Torch reflect padding when the kernel radius is smaller than the source axis. A `BORDER_REFLECT_101` index map handles singleton axes and kernels wider than an axis. NumPy input crosses the zero-copy `DHWC → CDHW → DHWC` bridge when its storage permits it; read-only and negative-stride arrays receive the one required repair copy. The router filters in float32 and restores uint8 once after all three passes.

The call receives a prevalidated input and parameters from AlbumentationsX. It does not repeat dtype, rank, device, autograd, or range checks. Torch is imported before the timed call.

## Evidence

The full run used macOS arm64, Albucore 0.2.11 (commit `32de93a`), Torch 2.13.0, NumPy 2.2.6, OpenCV 5.0.0, one CPU thread, 11 timed repetitions, and three warmups. The matrix covered nine non-batched canonical `DHWC` shapes, uint8 and float32, and `C=1/3/5/9` where applicable. Every candidate preserved shape and dtype. Float32 candidates were checked against the selected route with `rtol=3e-5`, `atol=3e-5`; uint8 candidates differed by at most one value.

On `48×240×320×3`, the direct NumPy-to-Torch route took 31.711 ms for float32 and 32.204 ms for uint8. The all-NumPy three-pass route took 163.439 ms and 180.667 ms. On a `64×128×160×3` float32 Tensor, Torch reflect padding took 10.999 ms and universal index padding took 16.885 ms. The production path stays with Torch reflect padding and uses the index map only for axes that Torch cannot reflect-pad.

Packed OpenCV H/W plus a Torch depth pass was evaluated where `D*C` stayed within the installed OpenCV channel limit. It had no stable full-path win: on `16×128×160×1` float32 it took 0.879 ms versus 0.954 ms for the direct bridge, while on `32×128×160×3` it took 6.367 ms versus 5.380 ms. It is unavailable for cells whose packed channel count exceeds the OpenCV limit, so it does not become a size route.

The benchmark records wall time and candidate correctness. It does not yet isolate peak native RSS per candidate; add an OS-level isolated-process memory probe before introducing a memory-based route or a tiled implementation.

## Reproduce

```bash
uv run python benchmarks/benchmark_gaussian_blur3d.py --quick --threads 1
uv run python benchmarks/benchmark_gaussian_blur3d.py --full --threads 1 \
  --output benchmarks/results/gaussian-blur3d-cpu.md
```

The first command includes small and singleton axes. The second sweeps the canonical single-volume matrix. Both commands require Torch to be imported before timed calls; the script imports it during startup and excludes import cost from every timing cell.
