# XY nearest affine sampling

The fix for [Albucore #179](https://github.com/albumentations-team/albucore/issues/179) reuses the existing shared-plane Torch sampler for eligible nearest XY warps. The macOS ARM64 and Linux x86-64 acceptance matrices at `fbaf64d` each passed 1,062 exact comparisons. No sustained slowdown above 5% remained after extended timing rechecks.

## Behavior and routing

The categorical reference is the native 3D path from `2af7aaed196f357f8c2da73eb2cda0fb81ee79ca`. Nearest results require exact equality, including half-voxel ties, constant/replicate borders, and dtype restoration. OpenCV slices are unsuitable: the issue's `(2,9,11,1)` multiclass Y-half-voxel case still has 22/198 differing labels.

The shared plane requires exact identity Z coefficients, no XY/Z coupling, unchanged depth, at least 8,192 output voxels, and input depth at most 1,048,576. Single volumes and single-item batches are eligible; larger batches require `stride(0) == D * stride(2)` after conversion to NCDHW. Batches that need a per-volume loop retain native sampling, including single-channel batches with gaps between items. Linear routing is unchanged.

The minimum size avoids fixed setup overhead on tiny calls. Strides are read only after size and XY eligibility pass. The first Linux candidate's eager stride checks showed a 4.9–6.4% difference on one small native batch case. The final version measured baseline/candidate medians of 0.239/0.243 ms for that same case, about 1.8% apart. Both observations are retained in the Linux artifact.

At extreme depths, the native float32 normalized Z grid can select a neighboring slice even with identity Z: depth `2**23+1` exposed 524,288 differing labels when that grid was bypassed. The conservative depth limit preserves native behavior, with a complete-output regression test at that depth.

Public arguments, containers, layouts, identity aliasing, and ownership are unchanged. Read-only/negative-stride NumPy repair stays in the existing bridge. No caller-validation pass, cache, dependency, or new sampler was added.

## Environment and method

Measured on 2026-11-02 with Python 3.11 and freshly resolved compatible runtime dependencies:

| Host | Python | Torch | Other runtime dependencies |
|---|---|---|---|
| Apple M4 Max, macOS ARM64 | 3.11.15 | 2.15.0 | NumPy 2.4.6, NumKong 7.8.7, StringZilla 5.3.2, OpenCV headless 5.0.0.93 |
| GCP c4-standard-2, Intel Xeon Platinum 8581C, Linux x86-64 | 3.11.18 | 2.15.0+cpu | Same versions as Mac |

The final Linux run pinned the initial successful run's complete package inventory. NumKong NEON was enabled on Mac; Linux native capabilities are recorded. OpenMP/BLAS/Accelerate/NumExpr and Torch intra-op/inter-op thread counts were one. OpenCV internal parallelism was disabled.

The [Mac JSON](xy_nearest_macos.json.gz) and [Linux JSON](xy_nearest_linux.json.gz) retain every sample, package inventory, imported paths, source hashes, follow-ups, and RSS measurements. Mac evidence also includes AX source fingerprints and an allocation profile. The benchmark refuses an Albucore import outside its checkout. An initial installed-package run and two cloud setup/packaging failures are excluded from acceptance.

Calls include dispatch, matrix normalization/inversion, grid construction, container views, sampling, allocation, dtype restoration, and returned layout. Inputs and imports are outside timing. Candidate order alternates. Each cell uses three warmups and eleven samples; a >5% difference outside the combined 2-MAD ranges triggers three rechecks with ten warmups and 101 samples. One Mac and three Linux cells triggered rechecks; their long-run differences were below 1%.

Each 1,062-cell matrix covers canonical non-square volumes, the reported `32x288x288x1` case, C=1/3/5/9, single volumes and N=1/4/16 batches, NumPy and contiguous/strided CPU Tensor layouts, uint8/float16/float32, zero/nonzero constant fill, replicate borders, and both sides of the size threshold. Float16 and float32 share half-quantized logical values as a diagnostic; no new public dtype contract is introduced. Every cell checks exact output, container, shape, dtype, fresh output storage, and input preservation.

## Single-item batch review follow-up

A contiguous N=1 multi-channel Tensor batch now uses the same normalized sampler path as its corresponding single volume. No sampling kernel, working dtype, or full-array conversion changes. The existing two-platform matrices measure that single-volume kernel; they predate this batch-dispatch adjustment and retain their exact source hashes.

An additional 48-case Mac comparison checks the published `fbaf64d` batch path, the updated batch path, and the single-volume public peer, with five warmups and 31 samples. Shapes are `16x32x40` and `16x128x160`, C=1/3/5/9, uint8/float16/float32, and zero/nonzero fill. All three outputs match exactly. Multi-channel batches improve by 1.70–3.41x; one-channel controls have no sustained regression. For N=1, C=3, `16x128x160` float32, baseline/batch/single-peer medians are 2.254/0.660/0.664 ms.

The raw follow-up is stored in `single_item_batch_followup` in the Mac artifact, with its own source hash. It adds no separate Linux N=1 timing claim. Dedicated regression tests check the one-plane grid, native 3D parity, peer parity, borders, dtype, and ownership for these singleton batches; current-head CI validates the new route on Linux.

## Complete public-call results

Pair calls allocate a float32 linear volume and a uint8 nearest support mask using the same centered 5-degree forward matrix and zero fill.

| Volume DHWC | Mac baseline ms | Mac candidate ms | Linux baseline ms | Linux candidate ms |
|---|---:|---:|---:|---:|
| 16x128x160x1 | 2.882 | 1.221 | 7.486 | 1.609 |
| 32x288x288x1 | 21.246 | 6.586 | 73.643 | 10.719 |
| 32x128x160x3 | 8.200 | 4.872 | 17.182 | 5.498 |

The `32x288x288x1` uint8 mask alone fell from 17.591 to 2.643 ms on Mac and from 83.031 to 5.176 ms on Linux. The complete pair improved by 3.23x and 6.87x, respectively.

An AX integration diagnostic additionally included Compose, four XYZ keypoints, and background restoration. For the reported shape it measured 26.016/12.633 ms, with identical volumes, masks, and keypoints. This used a frozen snapshot of the dirty integration checkout based on `049b9f5e552f733c5ae2c05b8fa1f33ba8d6b91f`; all 117 Python source fingerprints and controls are retained. It predates the batch-only dispatch adjustment and exercises unchanged single-volume paths. It is not a measurement of that clean revision or a published AX release.

## Memory and implementation audit

For the reported shape, affine-grid `bmm` allocation shrank from 31,850,496 to 995,328 bytes (30.38 to 0.95 MiB). The four-coordinate base grid also shrank by depth. The profiler capture exercises unchanged single-volume code and records its source hash.

Fresh-process peak/steady RSS for five public pair calls with background restoration:

| Host | Baseline MiB | Candidate MiB |
|---|---:|---:|
| Mac, three processes per implementation | Peak 361.9–362.1; steady 361.9–362.1 | Peak/steady 291.3–293.5 |
| Linux, final candidate | Peak 435.6; steady 405.4 | Peak 383.5; steady 357.1 |

- Removed repeated depth-coordinate construction and 3D sampling of unchanged Z. Input widening and output restoration remain; eligible NumPy/Tensor bridges remain views.
- Reused the vectorized 2D Torch kernel and copy-free N/D folding. No per-pixel Python loop or full-volume index array was introduced.
- Rejected unconditional sharing on tiny arrays and per-volume batch loops. A C/D-fold candidate improved small contiguous batches but regressed a large single-volume case, so it was discarded.
- NumPy gathers need a separate coordinate/tie implementation; Torch already supplies the needed primitive. NumKong, StringZilla, LUTs, random generation, and grouped reductions such as `bincount` do not implement this spatial contract.
- The atom already belongs to Albucore, and AX uses its existing adapter. No duplicate atom or missing dependency primitive needs extraction.
- There is no explicit output-buffer API. Outputs remain separate from inputs; identity retains its alias contract.

## Validation and reproduction

The dedicated acceptance suites at `fbaf64d` passed **Mac 7,317/1 skipped and Linux 7,316/2 skipped**. The N=1 follow-up adds 24 regression cases; its full Mac suite passed **7,341/1 skipped**, with Linux coverage supplied by current-head CI. Linux emitted four existing invalid-multiply warnings in exhaustive half-conversion tests; that code is unchanged.

Coverage includes binary/multiclass half-voxel cases on both axes, unit/short/deep dimensions, size and leading-batch-stride boundaries, high channels, batches, nonzero per-channel fills, read-only/strided inputs, genuine Z changes on optimization-sized inputs, changed output depth, downstream trainable Tensor use, and 100 generated XY affine cases against native 3D coordinates. Final source passed the unslop/delete-first review and changed-file quality checks. Cloud jobs self-delete, with terminal results retained in GCS.

```bash
git show 2af7aaed196f357f8c2da73eb2cda0fb81ee79ca:albucore/affine3d.py > /tmp/affine3d_before.py
PYTHONPATH="$PWD" python benchmarks/benchmark_warp_affine3d_xy_nearest.py \
  --baseline /tmp/affine3d_before.py --output /tmp/xy_nearest.json
# Run each implementation in a fresh process for memory measurements.
PYTHONPATH="$PWD" python benchmarks/benchmark_warp_affine3d_xy_nearest.py \
  --baseline /tmp/affine3d_before.py --rss current --output /tmp/xy_nearest_rss.json
```

Use the Python 3.11 profiles above. On Mac, run outside a sandbox that restricts native capability detection. Substitute `--rss baseline` for the old route.
