# Pairwise distances: common-coordinate overflow and cancellation

Issue [#197](https://github.com/albumentations-team/albucore/issues/197) is reproducible at baseline
`fa126da86ea9b4a660c199549dd791a5f9584811`. The norm/dot identity squares absolute coordinates before
subtracting large terms. It produces NaNs for identical coordinates near `1e20` and loses a squared distance
of `0.03125` between coordinates near `10000`.

The replacement subtracts coordinates before squaring. NumKong handles fewer than 1,000 pairs and dimensions
`D >= 9`; other sets accumulate NumPy squared differences with one reusable `(N, M)` scratch matrix.
Removing the redundant NumKong non-negativity clamp also avoids one output pass. Neither route writes inputs.
The Tensor route retains its NumPy bridge and float32 output.

## Correctness

Regression tests require exactly `0`, `1`, and `0.03125` for the three reported examples, with separate
operands, 999/1,000/1,001 pairs, NumPy/Tensor containers, and contiguous/strided storage. Actual squared
distances outside float32 range remain infinity. Mixed points at a common offset cover dimensions
1/2/3/5/8/9/64 and contiguous, strided, reversed, and read-only NumPy storage. Oracles subtract coordinates
in float64 rather than repeating the norm/dot identity. Shape, dtype, non-negativity, and input preservation
are checked.

## Native macOS measurements

Apple M4 Max, macOS 27.0.1 ARM64, Python 3.11.15; NumPy 2.4.6, OpenCV 5.0.0.93, NumKong 7.8.7,
StringZilla 5.3.2, Torch 2.15.0. Dependencies were re-resolved on 2026-11-01. NumKong reported native NEON.
OpenCV parallelism was disabled; Torch intra-op/inter-op and numerical-library thread controls were one.
Each timing has five warmups and 101 samples, with no concurrent test workload.

The [216-case JSON](pairwise-distances-197-macos.json) includes dtype/container/stride variants, median
absolute deviations, package inventory, and imported source paths. The table shows allocating contiguous
NumPy float32 calls. Times are medians in microseconds; lower is better.

| N × M × D | Norm/dot baseline, µs | Fixed public route, µs |
|---|---:|---:|
| 10 × 10 × 2 | 1.71 | 1.25 |
| 32 × 32 × 2 | 6.42 | 3.42 |
| 1 × 999 × 2 | 6.42 | 5.88 |
| 1 × 1,000 × 2 | 10.71 | 2.87 |
| 1 × 1,001 × 2 | 10.75 | 2.92 |
| 100 × 100 × 2 | 18.13 | 13.63 |
| 1,000 × 100 × 2 | 86.25 | 100.17 |
| 4,096 × 5 × 2 | 76.42 | 42.33 |
| 20,480 × 16 × 2 | 361.92 | 368.37 |
| 307,200 × 16 × 2 | 5,683.71 | 5,835.17 |
| 100 × 100 × 3 | 18.21 | 18.87 |
| 100 × 100 × 5 | 17.42 | 35.04 |
| 100 × 100 × 8 | 17.83 | 59.71 |
| 100 × 100 × 9 | 18.42 | 61.75 |
| 100 × 100 × 64 | 24.67 | 161.96 |

This correctness fix has a performance cost: the 1,000 × 100 × 2 case takes 16% longer, and the 64-dimensional
case takes 6.57× the cancellation-prone baseline. The largest measured 2D call takes 2.7% longer. These
results do not establish a general speedup or Linux performance acceptance. Native Linux x86-64 timing
remains outstanding.

Fresh-process measurements for 307,200 × 16 × 2 reduced peak traced array allocation from 60,212,864 to
39,389,232 bytes, about 35%. Peak RSS, including imported libraries, fell from 301,465,600 to 277,102,592
bytes. Both runs retained one approximately 19.7 MB result. The NumPy route uses two distance-sized buffers
and avoids the `(N, M, D)` broadcast temporary and norm/dot intermediates.

## Alternatives and scope

- Direct NumKong for every matrix was slower on dense 2D inputs: a preliminary 307,200 × 16 × 2 screen
  measured about 24 ms versus 5.6 ms for coordinate accumulation. Higher-dimensional NumKong avoids a
  Python loop over every coordinate; at D=64 it took about 0.15 ms versus 0.44 ms for that loop.
- Full NumPy broadcasting and `einsum` allocate an `(N, M, D)` difference array and were slower in the
  dense 2D screen. Per-coordinate NumPy, OpenCV arithmetic, and flattened NumKong FMA were compared.
- Direct float32 Torch `cdist(..., compute_mode="donot_use_mm_for_euclid_dist").square()` changed some
  exactly representable mixed-point squared distances by one ULP because of its square-root round trip.
  Wider Torch arithmetic and fused per-coordinate Torch arithmetic were also screened; neither is added
  to this fix. Fused arithmetic would introduce another rounding policy and routing region.
- OpenCV's sorted nearest-neighbor `batchDistance` path required reordering, was slower, and did not
  provide the required infinity results. It is unsuitable as the full-matrix implementation here.
- LUTs, StringZilla byte translation, grouped integer-label reductions, and random generation do not apply
  to this distance calculation. No new public API, output alias, cache, or reusable atom is needed.
  AlbumentationsX already calls this atom from its TPS functions; no consumer changes are included.

Reproduce the public comparison with the latest Python 3.11-compatible environment:

```bash
python benchmarks/benchmark_pairwise_distances.py --output-json /tmp/pairwise-distances.json
```

On macOS ARM64, run outside the restricted sandbox; the script rejects a disabled NEON profile.
