
### Benchmark: `sz.translate` vs `cv2.LUT` (uint8 image, uint8 LUT)

_Median ms, repeats=41, warmup=12, seed=0; Darwin `arm64`, OpenCV 4.13.0, numpy 2.4.2.

#### Shared LUT `(256,)` — same table for every byte

| layout | shape | pixels | SZ 1× full buffer | SZ loop C× `sz_lut` | cv2 1× `LUT` | fastest (shared) |
|--------|-------|-------:|--------------------:|--------------------:|-------------:|------------------|
| HWC | 128×128×1 | 16384 | 0.0020 | 0.0027 | 0.0046 | SZ full |
| HWC | 128×128×3 | 49152 | 0.0050 | 0.0322 | 0.0128 | SZ full |
| HWC | 256×256×1 | 65536 | 0.0068 | 0.0083 | 0.0168 | SZ full |
| HWC | 256×256×3 | 196608 | 0.0175 | 0.1199 | 0.0495 | SZ full |
| HWC | 256×256×9 | 589824 | 0.0512 | 0.3615 | 0.1476 | SZ full |
| HWC | 512×512×1 | 262144 | 0.0236 | 0.0275 | 0.0331 | SZ full |
| HWC | 512×512×3 | 786432 | 0.0691 | 0.4747 | 0.0720 | SZ full |
| HWC | 512×512×9 | 2359296 | 0.3153 | 1.6098 | 0.2252 | cv2 |
| HWC | 1024×1024×1 | 1048576 | 0.0922 | 0.1447 | 0.0883 | cv2 |
| HWC | 1024×1024×3 | 3145728 | 0.4213 | 2.2855 | 0.2308 | cv2 |
| HWC | 1024×1024×9 | 9437184 | 0.8211 | 5.9310 | 0.3626 | cv2 |
| HWC | 96×96×9 | 82944 | 0.0091 | 0.0622 | 0.0213 | SZ full |
| DHWC | 8×64×64×3 | 98304 | 0.0101 | 0.0622 | 0.0285 | SZ full |
| DHWC | 6×32×32×9 | 55296 | 0.0058 | 0.0427 | 0.0146 | SZ full |
| NDHWC | 2×4×32×32×3 | 24576 | 0.0027 | 0.0178 | 0.0069 | SZ full |
| NDHWC | 2×3×24×24×9 | 31104 | 0.0032 | 0.0280 | 0.0086 | SZ full |

#### Per-channel LUT (different `256` table per channel)

| layout | shape | pixels | SZ loop C× `sz_lut` | cv2 (1× `(256,1,C)` on HWC else C× `LUT`) | fastest |
|--------|-------|-------:|--------------------:|----------------------------------------:|---------|
| HWC | 128×128×1 | 16384 | 0.0028 | 0.0054 | SZ loop |
| HWC | 128×128×3 | 49152 | 0.0329 | 0.0134 | cv2 |
| HWC | 256×256×1 | 65536 | 0.0287 | 0.0327 | SZ loop |
| HWC | 256×256×3 | 196608 | 0.1252 | 0.0515 | cv2 |
| HWC | 256×256×9 | 589824 | 0.3707 | 0.1529 | cv2 |
| HWC | 512×512×1 | 262144 | 0.0275 | 0.0401 | SZ loop |
| HWC | 512×512×3 | 786432 | 0.4807 | 0.0735 | cv2 |
| HWC | 512×512×9 | 2359296 | 1.6614 | 0.2662 | cv2 |
| HWC | 1024×1024×1 | 1048576 | 0.1560 | 0.1215 | cv2 |
| HWC | 1024×1024×3 | 3145728 | 2.2334 | 0.2472 | cv2 |
| HWC | 1024×1024×9 | 9437184 | 6.2912 | 0.6687 | cv2 |
| HWC | 96×96×9 | 82944 | 0.0624 | 0.0286 | cv2 |
| DHWC | 8×64×64×3 | 98304 | 0.0627 | 0.0798 | SZ loop |
| DHWC | 6×32×32×9 | 55296 | 0.0425 | 0.0490 | SZ loop |
| NDHWC | 2×4×32×32×3 | 24576 | 0.0178 | 0.0227 | SZ loop |
| NDHWC | 2×3×24×24×9 | 31104 | 0.0289 | 0.0345 | SZ loop |

**Notes:**
- **SZ full** is only valid when one LUT applies to every byte (scalar `apply_lut` path).
- **SZ loop** matches multi-channel `apply_lut` (`sz_lut` per channel).
- **cv2** shared `(256,)`: contiguous `HWC` / `DHWC` / `NDHWC` work here. Per-channel distinct: `(256,1,C)` is **only** valid for `ndim==3`; volumes/batches use **C×** `cv2.LUT(..., 1D)` on `img[..., c]` (OpenCV `lut.cpp` otherwise asserts).
- Regenerate on your CPU; routing should follow benchmarks, not assumptions.
