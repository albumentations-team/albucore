# remap3d direct Tensor sampler decision

Decision: expose public CPU Tensor `CDHW` support through the direct shared CPU Torch sampler. Keep the zero-copy `Tensor → NumPy DHWC → remap3d → Tensor` bridge only as a benchmark comparison baseline.

The complete matrix covers `32³`, `64×128×128`, and `128³`; `C=1/3`; `uint8/float32`; NumPy/Tensor grids; contiguous and channel-last-derived strided `CDHW`; nearest/trilinear interpolation; and zero/nonzero/replicate borders. Every Tensor comparison is paired: direct and bridge calls alternate order, and both include border-value normalization. There is no Tensor heuristic route.

The prior sequential report, which showed the bridge ahead, is discarded: it timed all bridge repetitions before direct repetitions and measured unequal setup. At `128³×1`, `float32`, contiguous `CDHW`, NumPy grid, trilinear interpolation, and zero constant fill, the paired report measures public direct Tensor sampling at 32.385 ms median (MAD 2.477 ms) and the bridge baseline at 33.878 ms (MAD 2.175 ms): direct is 4.4% faster for that cell.

The complete raw samples, versions, thread setting, RSS, and allocation ledger are retained in [`benchmark_remap3d.md`](benchmark_remap3d.md).
