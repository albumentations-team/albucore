
### Benchmark: `sz.translate` vs `cv2.LUT` (uint8 image, uint8 LUT)

_Median ms, repeats=15, warmup=5, seed=0; Darwin `arm64`, OpenCV 5.0.0, numpy 2.2.6.

#### Shared LUT `(256,)` — same table for every byte

| layout | shape | pixels | SZ 1× full buffer | SZ loop C× `sz_lut` | cv2 1× `LUT` | fastest (shared) |
|--------|-------|-------:|--------------------:|--------------------:|-------------:|------------------|
| HWC | 128×128×1 | 16384 | 0.0020 | 0.0030 | 0.0049 | SZ full |
| HWC | 128×128×3 | 49152 | 0.0053 | 0.0337 | 0.0130 | SZ full |
| HWC | 256×256×1 | 65536 | 0.0071 | 0.0085 | 0.0178 | SZ full |
| HWC | 256×256×3 | 196608 | 0.0185 | 0.1227 | 0.0504 | SZ full |
| HWC | 256×256×9 | 589824 | 0.0507 | 0.3702 | 0.1540 | SZ full |
| HWC | 512×512×1 | 262144 | 0.0237 | 0.0276 | 0.0325 | SZ full |
| HWC | 512×512×3 | 786432 | 0.0683 | 0.4752 | 0.0667 | cv2 |
| HWC | 512×512×9 | 2359296 | 0.2020 | 1.4399 | 0.1715 | cv2 |
| HWC | 1024×1024×1 | 1048576 | 0.0890 | 0.1015 | 0.0911 | SZ full |
| HWC | 1024×1024×3 | 3145728 | 0.2717 | 1.9027 | 0.1513 | cv2 |
| HWC | 1024×1024×9 | 9437184 | 0.8309 | 5.8293 | 0.3514 | cv2 |
| HWC | 96×96×9 | 82944 | 0.0093 | 0.0625 | 0.0219 | SZ full |
| DHWC | 32×128×128×1 | 524288 | 0.0455 | 0.0526 | 0.1326 | SZ full |
| DHWC | 64×128×128×3 | 3145728 | 0.2763 | 1.9153 | 0.7915 | SZ full |
| DHWC | 128×128×128×1 | 2097152 | 0.1896 | 0.2002 | 0.5240 | SZ full |
| DHWC | 48×256×256×3 | 9437184 | 0.8167 | 5.7304 | 2.4360 | SZ full |
| DHWC | 96×160×160×3 | 7372800 | 0.6368 | 4.6008 | 1.9041 | SZ full |
| DHWC | 6×32×32×9 | 55296 | 0.0060 | 0.0456 | 0.0148 | SZ full |
| NDHWC | 2×32×128×128×3 | 3145728 | 0.2709 | 1.8676 | 0.8179 | SZ full |
| NDHWC | 2×64×128×128×3 | 6291456 | 0.5369 | 3.8311 | 1.6009 | SZ full |
| NDHWC | 1×128×128×128×1 | 2097152 | 0.1798 | 0.2002 | 0.5217 | SZ full |

#### Per-channel LUT (different `256` table per channel)

| layout | shape | pixels | SZ loop C× `sz_lut` | cv2 loop C× `LUT` | cv2 flat 1× `(256,1,C)` | fastest |
|--------|-------|-------:|--------------------:|------------------:|--------------------------:|---------|
| HWC | 128×128×1 | 16384 | 0.0030 | 0.0056 | 0.0049 | SZ loop |
| HWC | 128×128×3 | 49152 | 0.0339 | 0.0132 | 0.0131 | cv2 flat |
| HWC | 256×256×1 | 65536 | 0.0090 | 0.0193 | 0.0173 | SZ loop |
| HWC | 256×256×3 | 196608 | 0.1200 | 0.0500 | 0.0498 | cv2 flat |
| HWC | 256×256×9 | 589824 | 0.3705 | 0.1504 | 0.1587 | cv2 loop |
| HWC | 512×512×1 | 262144 | 0.0282 | 0.0372 | 0.0263 | cv2 flat |
| HWC | 512×512×3 | 786432 | 0.4658 | 0.0710 | 0.0664 | cv2 flat |
| HWC | 512×512×9 | 2359296 | 1.4304 | 0.2216 | 0.2235 | cv2 loop |
| HWC | 1024×1024×1 | 1048576 | 0.1070 | 0.1015 | 0.0910 | cv2 flat |
| HWC | 1024×1024×3 | 3145728 | 1.8679 | 0.1514 | 0.1510 | cv2 flat |
| HWC | 1024×1024×9 | 9437184 | 5.6543 | 0.3546 | 0.3376 | cv2 flat |
| HWC | 96×96×9 | 82944 | 0.0623 | 0.0220 | 0.0220 | cv2 loop |
| DHWC | 32×128×128×1 | 524288 | 0.0525 | 0.1403 | 0.0482 | cv2 flat |
| DHWC | 64×128×128×3 | 3145728 | 1.8818 | 2.4213 | 0.1587 | cv2 flat |
| DHWC | 128×128×128×1 | 2097152 | 0.2025 | 0.5542 | 0.1202 | cv2 flat |
| DHWC | 48×256×256×3 | 9437184 | 5.6579 | 7.3185 | 0.3016 | cv2 flat |
| DHWC | 96×160×160×3 | 7372800 | 4.4524 | 5.6776 | 0.2402 | cv2 flat |
| DHWC | 6×32×32×9 | 55296 | 0.0449 | 0.0505 | 0.0194 | cv2 flat |
| NDHWC | 2×32×128×128×3 | 3145728 | 1.8772 | 2.4338 | 0.1515 | cv2 flat |
| NDHWC | 2×64×128×128×3 | 6291456 | 3.7873 | 4.8565 | 0.2013 | cv2 flat |
| NDHWC | 1×128×128×128×1 | 2097152 | 0.2025 | 0.5523 | 0.1088 | cv2 flat |

**Notes:**
- **SZ full** is only valid when one LUT applies to every byte (scalar `apply_lut` path).
- **SZ loop** matches the non-contiguous multi-channel `apply_lut` fallback (`sz_lut` per channel).
- **cv2** shared `(256,)`: contiguous `HWC` / `DHWC` / `NDHWC` work here. Per-channel distinct: direct `(256,1,C)` is **only** valid for `ndim==3`, but contiguous volumes/batches can be reshaped to HWC and use the same one-shot OpenCV path.
- Regenerate on your CPU; routing should follow benchmarks, not assumptions.
