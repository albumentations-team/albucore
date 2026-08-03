# Research notes

Supporting material for **backend choices** (extra tables, methodology, archived runs). It is **not** the primary user docs.

Runnable, maintained scripts live under **[`benchmarks/`](../../benchmarks/README.md)**.

| Note | Content |
|------|---------|
| [`minmax-ravel-benchmark.md`](minmax-ravel-benchmark.md) | Global min/max on raveled `(H,W,C)` — NumPy vs NumKong `minmax`. |
| [`sum-mean-std-ravel-benchmark.md`](sum-mean-std-ravel-benchmark.md) | Global sum / mean / std — NumPy vs NumKong on image-shaped arrays. |
| [`torch-cpu-backend-audit.md`](torch-cpu-backend-audit.md) | CPU Torch candidates, accepted NumPy→Torch→NumPy routes, benchmark matrix, and rejected paths. |
| [`resize3d-cpu-benchmark.md`](resize3d-cpu-benchmark.md) | Full CPU routes for `resize3d`: NumPy, OpenCV packing, and Torch CPU. |
| [`warp-affine3d-cpu-benchmark.md`](warp-affine3d-cpu-benchmark.md) | CPU route decision for single-volume `warp_affine3d`: native Torch grid sampling and rejected diagnostic candidates. |
