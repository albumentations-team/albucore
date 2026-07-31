# `add_weighted` float32 routing after removing clipping

## Decision

`add_weighted` returns the raw float32 weighted sum. The router uses NumKong for single-channel float32 inputs and OpenCV for float32 inputs with more than one channel. Uint8 keeps NumKong's saturating `blend` path.

The issue #130 grid supports the channel split: NumKong won all three single-channel cells; OpenCV won five of the six multi-channel cells, with the remaining result differing by less than 1%. NumPy took about twice as long as the selected compiled backend on contiguous inputs. Every candidate was byte-exact against `img1 * 0.5 + img2 * 0.5` in this workload.

The router does not add a strided-input branch. On the strided issue grid, the public route stayed within 3% of the fastest measured candidate. The differences between OpenCV and NumKong were smaller than the timing spread in several cells.

## Method

Run date: 2026-07-31. Environment: macOS 26.4.1, Apple M4 Max, Python 3.10.16, NumPy 2.2.6, OpenCV 5.0.0, and NumKong 7.7.0. OpenCV reported one thread after `cv2.setNumThreads(0)`.

Inputs were float32 HWC arrays in `[0, 255]`; weights were `0.5, 0.5`. Each cell reports the median ± median absolute deviation in milliseconds from 101 repeats after 10 warmups. The harness shuffled candidate order on every repeat to reduce ordering bias.

## Issue #130 grid

| Shape | Public router | NumPy | OpenCV | NumKong | Fastest backend |
|---|---:|---:|---:|---:|---|
| 256×256×1 | 0.0075 ± 0.0002 | 0.0148 ± 0.0001 | 0.0084 ± 0.0002 | 0.0069 ± 0.0002 | NumKong |
| 256×256×3 | 0.0180 ± 0.0002 | 0.0394 ± 0.0002 | 0.0175 ± 0.0002 | 0.0177 ± 0.0003 | OpenCV |
| 256×256×5 | 0.0293 ± 0.0011 | 0.0643 ± 0.0024 | 0.0287 ± 0.0012 | 0.0287 ± 0.0015 | OpenCV |
| 512×512×1 | 0.0236 ± 0.0017 | 0.0522 ± 0.0028 | 0.0248 ± 0.0015 | 0.0231 ± 0.0015 | NumKong |
| 512×512×3 | 0.0674 ± 0.0022 | 0.1509 ± 0.0053 | 0.0679 ± 0.0037 | 0.0677 ± 0.0036 | NumKong (<1%) |
| 512×512×5 | 0.1285 ± 0.0105 | 0.2885 ± 0.0185 | 0.1261 ± 0.0113 | 0.1375 ± 0.0147 | OpenCV |
| 1024×1024×1 | 0.1079 ± 0.0086 | 0.2348 ± 0.0154 | 0.1106 ± 0.0104 | 0.1072 ± 0.0100 | NumKong |
| 1024×1024×3 | 0.3652 ± 0.0193 | 0.7573 ± 0.0289 | 0.3573 ± 0.0199 | 0.3723 ± 0.0247 | OpenCV |
| 1024×1024×5 | 0.5550 ± 0.0431 | 1.2165 ± 0.0695 | 0.5470 ± 0.0413 | 0.5690 ± 0.0409 | OpenCV |

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
```
