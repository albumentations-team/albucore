# Dtype conversion route evidence — 2026-11-03

NumPy uint8→float16 conversion uses a float16 OpenCV LUT, avoiding the full float32
image buffer in the previous two-call chain. On macOS ARM64, Torch normalizes in half;
a strided half destination uses a contiguous half workspace before copying.
Linux x86-64 uses a float32 workspace after the [Linux follow-up](convert_dtype_linux_20261103.md).

## Original public-call comparison

The [432-case archive](convert_dtype_20261103.json.gz) preserves the original medians
and adds the run recipe, revisions, source hashes, layouts and strides. It compares
base `432f4853d5412c56e9e15f2fa01dc208b7b5e5a8` with the runtime implementation
committed in `7aafe236f72c3a48317d57b27efc76a58fb05ad8`.

Both sides received the same input and explicit output mode. This compares each
conversion with its previous implementation at the same source and target dtype.
It does not compare float16 identity calls with float32 casts.

| Source → target | Cases | Median new / previous runtime |
| --- | ---: | ---: |
| uint8 → float32 | 72 | 1.001 |
| float16 → uint8 | 72 | 1.000 |
| float32 → uint8 | 72 | 1.000 |
| float16 → float32 | 72 | 1.000 |
| float32 → float16 | 72 | 1.000 |
| uint8 → float16 | 72 | 0.818 |

Environment: Apple M4 Max, macOS 27.0.1 ARM64, Python 3.11.15, NumPy 2.4.6,
OpenCV 5.0.0.93, Torch 2.15.0, NumKong 7.8.7 and StringZilla 5.3.2. NumKong's
native NEON probe succeeded. Torch intra-op/inter-op threads were one; OpenCV
parallelism was disabled with `setNumThreads(0)`. BLAS/OpenMP environment variables
were set to one before imports. CPU affinity and power management were not controlled.
The archive includes the full package inventory and imported module paths.

The grid used 128×160, 240×320, 480×640 and 768×1024 images with C=1/3/9, NumPy
HWC and contiguous Torch CHW inputs. Each conversion used allocating, contiguous-out
and last-axis-step-2 out modes. With RNG seed 31, floating inputs were first rounded
to half and then cast to their source dtype. The default scale was 255. Input/output
shape, dtype, container and overlap were valid caller preconditions; no aliases were timed.

Each implementation ran once for an exact output comparison, followed by 61 pairs
with alternating baseline/current order. The first ten pairs warmed up; the remaining
51 calls per implementation were timed with `perf_counter_ns`. The original runner
retained per-cell medians and ratios, not raw timing samples. All 432 new outputs
matched a copied baseline output with `assert_array_equal`.

## Rejected candidates and the strided-output decision

The [54 candidate/public comparisons](convert_dtype_candidates_20261103.json.gz)
record a follow-up at the same dependency profile with raw samples, module paths,
capabilities, strides and thread settings. It used three representative HWC shapes:
128×160×1, 240×320×3 and 768×1024×9, with NumPy HWC and contiguous Torch CHW inputs.
Each candidate included conversion, normalization, allocation and any destination copy.
Calls alternated against the public router with ten warmups and 51 measured samples,
using RNG seed 73 and scale 255.

The table reports the median candidate / public runtime ratio across those shapes.

| Candidate | Allocate | Contiguous out | Strided out |
| --- | ---: | ---: | ---: |
| NumPy float32 public chain, then half cast | 1.206 | 1.214 | 1.006 |
| NumPy direct float16 LUT | 0.994 | 0.993 | 0.996 |
| Torch float32 public chain, then half cast | 1.272 | 1.280 | 1.246 |
| Torch float32 workspace, then half destination | 1.514 | 1.518 | 1.241 |
| Torch direct half destination division | 0.987 | 0.984 | 2.285 |
| Torch contiguous half workspace, then copy | 0.987 | 1.255 | 0.998 |

The direct NumPy LUT and the selected Torch candidates perform the same work as
the public implementation; their small ratios below one reflect dispatch overhead.
The float32 chains were rejected because they retain an extra full image conversion.
NumPy strided-output timings are effectively equal, while the half working image
uses two bytes per element instead of four.
For Torch, a half workspace loses for contiguous output but avoids expensive strided
division. This supports the existing destination-stride branch without a size threshold.

Every candidate matched the float32-divide-then-half reference exactly, preserved the
input and returned the supplied output object. Direct OpenCV CV_16F LUT and Torch half
normalization also matched that reference for all 256 uint8 values at scales 255,
128 and 17.3. These results concern uint8→float16; FP16↔FP32 casts and float→uint8
quantization are separate workloads.

The original timings above cover macOS ARM64. The [Linux follow-up](convert_dtype_linux_20261103.md)
adds x86-64 evidence and records the correction required for Tensor half normalization.
