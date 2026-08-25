# remap3d Tensor-volume route decision

Decision: do not expose a public CPU Tensor `CDHW` overload in the first `remap3d` release.

The candidate was compared on the same Apple Silicon host, with one Torch/OpenCV thread, pre-created logically identical inputs, and 31 timed repetitions after seven warmups. It had to be no more than 1% slower than `Tensor → NumPy DHWC → remap3d → Tensor` in every cell. One repeatable loss rejects the overload because the issue prohibits size, dtype, channel, stride, or shape heuristics.

At `64×128×128×1`, `uint8`, contiguous `CDHW`, Tensor grid, nearest interpolation, and nonzero constant fill, the direct candidate median was 9.135 ms (MAD 0.261 ms); the bridge median was 8.629 ms (MAD 0.207 ms). The direct route was 5.9% slower. The same run found twelve more losses in the required matrix.

The raw samples, versions, thread setting, and process peak RSS are retained in [`benchmark_remap3d-route-gate-64x128x128.md`](benchmark_remap3d-route-gate-64x128x128.md). The accepted NumPy `DHWC` router still accepts NumPy or CPU Tensor float32 grids. Its full benchmark remains [`benchmark_remap3d.py`](../benchmark_remap3d.py).
