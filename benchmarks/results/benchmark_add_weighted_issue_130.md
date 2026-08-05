# `add_weighted` float32 routing after removing clipping

## Decision

`add_weighted` returns the raw float32 weighted sum. Uint8 and single-channel float32 use NumKong. Multi-channel float32 uses OpenCV for HWC or when both rank-4 inputs are C-contiguous; a strided rank-4 input routes the pair to NumKong.

The issue #130 grid supports the channel split: NumKong won all three contiguous single-channel cells; OpenCV won four of the six contiguous multi-channel cells, while NumKong's two wins were within 1%. NumPy took about twice as long as the selected compiled backend on contiguous inputs. Every candidate was byte-exact against `img1 * 0.5 + img2 * 0.5` in this workload.

The rank-4 branch matters: NumKong wins whenever at least one input is strided. On fully contiguous rank-4 inputs, OpenCV and NumKong are close; the router keeps OpenCV for consistency with the HWC multi-channel route.

## Method

Run date: 2026-07-31. Environment: macOS 26.4.1, Apple M4 Max, Python 3.10.16, NumPy 2.2.6, OpenCV 5.0.0, and NumKong 7.7.0. OpenCV reported one thread after `cv2.setNumThreads(0)`.

Inputs were float32 arrays in `[0, 255]`; weights were `0.5, 0.5`. Each cell reports the median ± median absolute deviation in milliseconds from 101 repeats after 10 warmups. The harness shuffled candidate order on every repeat to reduce ordering bias and validated one candidate output at a time to avoid retaining four full result arrays.

## Issue #130 grid

| Shape | Public router | NumPy | OpenCV | NumKong | Fastest backend |
|---|---:|---:|---:|---:|---|
| 256×256×1 | 0.0093 ± 0.0004 | 0.0166 ± 0.0006 | 0.0095 ± 0.0004 | 0.0085 ± 0.0003 | NumKong |
| 256×256×3 | 0.0177 ± 0.0002 | 0.0380 ± 0.0002 | 0.0168 ± 0.0002 | 0.0166 ± 0.0002 | NumKong (<1%) |
| 256×256×5 | 0.0288 ± 0.0012 | 0.0647 ± 0.0016 | 0.0278 ± 0.0011 | 0.0277 ± 0.0015 | NumKong (<1%) |
| 512×512×1 | 0.0225 ± 0.0005 | 0.0503 ± 0.0012 | 0.0236 ± 0.0005 | 0.0217 ± 0.0004 | NumKong |
| 512×512×3 | 0.0708 ± 0.0044 | 0.1580 ± 0.0067 | 0.0705 ± 0.0048 | 0.0715 ± 0.0062 | OpenCV |
| 512×512×5 | 0.1333 ± 0.0096 | 0.2934 ± 0.0205 | 0.1322 ± 0.0117 | 0.1355 ± 0.0133 | OpenCV |
| 1024×1024×1 | 0.1095 ± 0.0154 | 0.2339 ± 0.0243 | 0.1065 ± 0.0150 | 0.1041 ± 0.0144 | NumKong |
| 1024×1024×3 | 0.3596 ± 0.0139 | 0.7706 ± 0.0283 | 0.3565 ± 0.0167 | 0.3761 ± 0.0185 | OpenCV |
| 1024×1024×5 | 0.5903 ± 0.0371 | 1.2849 ± 0.0610 | 0.5851 ± 0.0312 | 0.6037 ± 0.0310 | OpenCV |

### Strided HWC issue grid

| Shape | Public router | NumPy | OpenCV | NumKong | Fastest backend |
|---|---:|---:|---:|---:|---|
| 256×256×1 | 0.0432 ± 0.0007 | 0.0428 ± 0.0007 | 0.0437 ± 0.0009 | 0.0425 ± 0.0008 | NumKong |
| 256×256×3 | 0.4326 ± 0.0078 | 0.6128 ± 0.0109 | 0.4307 ± 0.0081 | 0.4307 ± 0.0068 | OpenCV / NumKong |
| 256×256×5 | 0.4835 ± 0.0095 | 0.7464 ± 0.0127 | 0.4820 ± 0.0081 | 0.4860 ± 0.0106 | OpenCV |
| 512×512×1 | 0.1670 ± 0.0016 | 0.1628 ± 0.0017 | 0.1681 ± 0.0027 | 0.1652 ± 0.0011 | NumPy (<2%) |
| 512×512×3 | 1.7381 ± 0.0308 | 2.4781 ± 0.0392 | 1.7222 ± 0.0267 | 1.7400 ± 0.0310 | OpenCV |
| 512×512×5 | 2.0259 ± 0.0870 | 3.1780 ± 0.1637 | 2.0006 ± 0.0806 | 2.0468 ± 0.1015 | OpenCV |
| 1024×1024×1 | 0.7267 ± 0.0257 | 0.7126 ± 0.0291 | 0.7280 ± 0.0405 | 0.7225 ± 0.0247 | NumPy (<2%) |
| 1024×1024×3 | 6.9580 ± 0.1197 | 10.0227 ± 0.1383 | 6.9495 ± 0.1032 | 6.9959 ± 0.0863 | OpenCV |
| 1024×1024×5 | 7.8807 ± 0.1630 | 12.2424 ± 0.2367 | 7.7921 ± 0.1297 | 7.8808 ± 0.1783 | OpenCV |

## Canonical non-square grid

| Shape | Public router | NumPy | OpenCV | NumKong | Fastest backend |
|---|---:|---:|---:|---:|---|
| 128×160×1 | 0.0038 ± 0.0001 | 0.0062 ± 0.0002 | 0.0045 ± 0.0001 | 0.0033 ± 0.0001 | NumKong |
| 128×160×3 | 0.0078 ± 0.0001 | 0.0165 ± 0.0002 | 0.0073 ± 0.0001 | 0.0080 ± 0.0001 | OpenCV |
| 128×160×9 | 0.0165 ± 0.0002 | 0.0357 ± 0.0003 | 0.0159 ± 0.0002 | 0.0157 ± 0.0003 | NumKong (<2%) |
| 240×320×1 | 0.0084 ± 0.0005 | 0.0167 ± 0.0004 | 0.0092 ± 0.0004 | 0.0078 ± 0.0004 | NumKong |
| 240×320×3 | 0.0200 ± 0.0002 | 0.0443 ± 0.0003 | 0.0196 ± 0.0002 | 0.0194 ± 0.0003 | NumKong (<2%) |
| 240×320×9 | 0.0718 ± 0.0096 | 0.1524 ± 0.0150 | 0.0715 ± 0.0114 | 0.0735 ± 0.0136 | OpenCV |
| 480×640×1 | 0.0256 ± 0.0004 | 0.0580 ± 0.0008 | 0.0270 ± 0.0004 | 0.0251 ± 0.0004 | NumKong |
| 480×640×3 | 0.0830 ± 0.0042 | 0.1857 ± 0.0082 | 0.0823 ± 0.0039 | 0.0834 ± 0.0050 | OpenCV |
| 480×640×9 | 0.2990 ± 0.0174 | 0.6462 ± 0.0302 | 0.2923 ± 0.0170 | 0.3118 ± 0.0181 | OpenCV |
| 768×1024×1 | 0.0848 ± 0.0026 | 0.1757 ± 0.0047 | 0.0830 ± 0.0035 | 0.0837 ± 0.0027 | OpenCV (<1%) |
| 768×1024×3 | 0.2254 ± 0.0137 | 0.5092 ± 0.0278 | 0.2226 ± 0.0163 | 0.2281 ± 0.0161 | OpenCV |
| 768×1024×9 | 0.8395 ± 0.0310 | 1.7781 ± 0.0848 | 0.8285 ± 0.0380 | 0.8592 ± 0.0428 | OpenCV |

## Higher-rank and asymmetric-layout grid

The full run covered C=1/3/5/9. The representative C=3 cells below show why layout must be part of routing. Rank 4 uses `4×128×160×C`.

| Rank | Input layouts | Public router | NumPy | OpenCV | NumKong | Fastest backend |
|---:|---|---:|---:|---:|---:|---|
| 4 | contiguous / contiguous | 0.0219 ± 0.0003 | 0.0473 ± 0.0005 | 0.0210 ± 0.0004 | 0.0208 ± 0.0005 | NumKong (<1%) |
| 4 | strided / strided | 0.5467 ± 0.0140 | 0.7810 ± 0.0220 | 1.3642 ± 0.0459 | 0.5388 ± 0.0103 | NumKong |
| 4 | contiguous / strided | 0.2824 ± 0.0073 | 0.4158 ± 0.0130 | 1.3356 ± 0.0434 | 0.2790 ± 0.0067 | NumKong |
| 4 | strided / contiguous | 0.2806 ± 0.0067 | 0.4132 ± 0.0128 | 1.3132 ± 0.0340 | 0.2778 ± 0.0064 | NumKong |

## Reproduce

```bash
uv run python benchmarks/benchmark_add_weighted.py \
  --grid issue --layouts contiguous --channels 1 3 5 \
  --repeats 101 --warmup 10

uv run python benchmarks/benchmark_add_weighted.py \
  --grid canonical --layouts contiguous --channels 1 3 9 \
  --repeats 101 --warmup 10

uv run python benchmarks/benchmark_add_weighted.py \
  --grid issue --layouts strided --channels 1 3 5 \
  --repeats 101 --warmup 10

uv run python benchmarks/benchmark_add_weighted.py \
  --grid rank --repeats 101 --warmup 10
```
