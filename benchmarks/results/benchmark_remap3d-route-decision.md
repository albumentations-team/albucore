# remap3d direct Tensor sampler decision

Decision: expose public CPU Tensor `CDHW` support through the direct shared CPU Torch sampler. Keep the zero-copy `Tensor → NumPy DHWC → remap3d → Tensor` bridge only as a benchmark comparison baseline.

The complete matrix covers `32³`, `64×128×128`, and `128³`; `C=1/3`; `uint8/float32`; NumPy/Tensor grids; contiguous and channel-last-derived strided `CDHW`; nearest/trilinear interpolation; and zero/nonzero/replicate borders. Every Tensor comparison is paired: direct and bridge calls alternate order, and both include border-value normalization. There is no Tensor heuristic route.

The prior sequential report, which showed the bridge ahead, is discarded: it timed all bridge repetitions before direct repetitions and measured unequal setup. In the paired matrix, direct has the lower median in 174 of 288 Tensor cells (bridge: 110; ties: 4); the geometric aggregate puts bridge/direct at 1.0069. At `128³×1`, `float32`, contiguous `CDHW`, NumPy grid, trilinear interpolation, and zero constant fill, direct is 34.226 ms median (MAD 3.768 ms) and bridge is 34.349 ms (MAD 2.281 ms). That 0.4% cell difference is within observed noise, not a routing threshold. The direct public path remains selected because it preserves Tensor layout without an unnecessary container round-trip; the bridge has no public route.

The complete raw samples, versions, thread setting, RSS, and allocation ledger are retained in [`benchmark_remap3d.md`](benchmark_remap3d.md).
