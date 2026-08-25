# remap3d direct Tensor sampler decision

Decision: expose public CPU Tensor `CDHW` support through the measured `Tensor → NumPy DHWC → remap3d → Tensor` bridge. Do not route calls to the direct CPU sampler in the first release.

The direct sampler is still benchmarked against the public Tensor bridge on the complete matrix: `32³`, `64×128×128`, and `128³`; `C=1/3`; `uint8/float32`; NumPy/Tensor grids; contiguous and channel-last-derived strided `CDHW`; nearest/trilinear interpolation; and zero/nonzero/replicate borders. The direct candidate has no heuristic route or public entry point.

At `128³×1`, `float32`, contiguous `CDHW`, NumPy grid, trilinear interpolation, and zero constant fill, the direct candidate median was 53.088 ms (MAD 1.824 ms); the public bridge median was 44.586 ms (MAD 2.865 ms). The direct route was 19.1% slower.

The complete raw samples, versions, thread setting, RSS, and allocation ledger are retained in [`benchmark_remap3d.md`](benchmark_remap3d.md).
