# XY-only nearest affine sampling

[Issue #179](https://github.com/albumentations-team/albucore/issues/179) is addressed by sharing the existing Torch plane sampler across unchanged-Z slices. The public API, native categorical ties, borders, dtype, identity aliasing, and input ownership are preserved.

Nearest sharing requires exact identity Z coefficients, zero XY/Z coupling, unchanged depth, at least 8,192 output voxels, and input depth at most 1,048,576. Larger batches also require `stride(0) == D * stride(2)` in NCDHW; single-item batches need no merge. Other nearest calls retain native 3D sampling. Linear routing is unchanged.

The size threshold avoids setup overhead. The depth bound preserves native float32 Z rounding: bypassing the native grid at `2**23+1` slices changed 524,288 labels. Tests retain that case and cover half-voxel ties, unit/deep axes, output-depth changes, tiny real Z changes, and batch-stride boundaries.

## Measurements

Captured on **2026-11-02**, after checking the Python 3.11 dependency profiles that day. Both hosts used NumPy 2.4.6, NumKong 7.8.7, StringZilla 5.3.2, and OpenCV headless 5.0.0.93. Apple M4 Max used Python 3.11.15/Torch 2.15.0; Linux c4-standard-2, Intel Xeon Platinum 8581C used Python 3.11.18/Torch 2.15.0+cpu. Torch intra/inter-op and numerical-library threads were one; OpenCV parallelism was disabled. Mac NumKong NEON was enabled.

Each host passed **1,062 exact comparisons** at `fbaf64d`: uint8/float16/float32, C=1/3/5/9, single volumes and N=1/4/16 batches, NumPy and contiguous/strided Tensors, fills, borders, and size boundaries. Calls include dispatch, matrix inversion, views, sampling, allocation, and dtype restoration. Three warmups and eleven alternating samples were followed by three 101-sample rechecks for robust >5% outliers; no sustained slowdown above 5% remained.

Complete float32 volume + uint8 mask calls use a centered 5-degree XY rotation, zero fill, linear volume sampling, and nearest mask sampling. Input creation is outside timing; returned allocations are included. Lower is better.

| Volume DHWC | Mac baseline ms | Mac candidate ms | Linux baseline ms | Linux candidate ms |
|---|---:|---:|---:|---:|
| 16x128x160x1 | 2.882 | 1.221 | 7.486 | 1.609 |
| 32x288x288x1 | 21.246 | 6.586 | 73.643 | 10.719 |
| 32x128x160x3 | 8.200 | 4.872 | 17.182 | 5.498 |

For the reported shape, the affine-grid allocation fell from 30.38 to 0.95 MiB. Mac peak RSS fell from 361.9–362.1 to 291.3–293.5 MiB; Linux peak/steady RSS fell from 435.6/405.4 to 383.5/357.1 MiB. Repeated depth-grid work is removed; image widening and output restoration remain.

The N=1 review follow-up at `8b3b7b9` reuses the already measured single-volume kernel. Its 48 matched Mac cases passed exact batch/native/single-peer comparisons; multi-channel batches improved 1.70–3.41x. No separate Linux N=1 timing claim is made; Linux CI covers its native-reference tests. The main single-volume pair paths above are unchanged.

[Mac timings](xy_nearest_macos.csv) and [Linux timings](xy_nearest_linux.csv) include medians, MADs, sample counts, and rechecks. Their [Mac metadata](xy_nearest_macos.json) and [Linux metadata](xy_nearest_linux.json) preserve complete package inventories, import origins, source hashes, RSS, and links to the immutable original raw captures. Timing summaries are rounded to one nanosecond. The benchmark now releases validation outputs before timing and writes CSV summaries plus JSON metadata; the captured measurements retain their original revisions and protocol.

OpenCV slices were rejected because the issue's half-voxel case differs in 22/198 labels. A C/D-fold candidate regressed a large single volume; per-volume batch loops and eager stride checks were also rejected. NumPy gathers would duplicate coordinate/tie handling. LUTs, RNGs, grouped reductions, NumKong, and StringZilla do not supply this spatial primitive; no new atom, dependency, cache, or output-buffer API is needed.

## Reproduce

Use a freshly resolved Python 3.11 environment. On Mac, run outside a sandbox that restricts native capability detection.

```bash
git show 2af7aaed196f357f8c2da73eb2cda0fb81ee79ca:albucore/affine3d.py > /tmp/affine3d_before.py
PYTHONPATH="$PWD" python benchmarks/benchmark_warp_affine3d_xy_nearest.py \
  --baseline /tmp/affine3d_before.py --output /tmp/xy_nearest.json
```

This writes `/tmp/xy_nearest.csv` and metadata JSON. For fresh-process memory, add `--rss baseline` or `--rss current`; each memory invocation writes only JSON.
