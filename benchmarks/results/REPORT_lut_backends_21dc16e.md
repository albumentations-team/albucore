
### Benchmark: `sz.translate` vs `cv2.LUT` (uint8 image, uint8 LUT)

_Median ms, repeats=31, warmup=7, seed=0; Darwin `arm64`, OpenCV 4.13.0, numpy 2.4.3.

#### Shared LUT `(256,)` — same table for every byte

| layout | shape | pixels | SZ 1× full buffer | SZ loop C× `sz_lut` | cv2 1× `LUT` | fastest (shared) |
|--------|-------|-------:|--------------------:|--------------------:|-------------:|------------------|
| HWC | 128×128×1 | 16384 | 0.0020 | 0.0035 | 0.0042 | SZ full |
| HWC | 128×128×3 | 49152 | 0.0047 | 0.0334 | 0.0132 | SZ full |
| HWC | 256×256×1 | 65536 | 0.0063 | 0.0087 | 0.0174 | SZ full |
| HWC | 256×256×3 | 196608 | 0.0180 | 0.1224 | 0.0507 | SZ full |
| HWC | 256×256×9 | 589824 | 0.0514 | 0.3670 | 0.1506 | SZ full |
| HWC | 512×512×1 | 262144 | 0.0210 | 0.0275 | 0.0318 | SZ full |
| HWC | 512×512×3 | 786432 | 0.0684 | 0.4738 | 0.0663 | cv2 |
| HWC | 512×512×9 | 2359296 | 0.3030 | 1.5747 | 0.2169 | cv2 |
| HWC | 1024×1024×1 | 1048576 | 0.0893 | 0.1230 | 0.0890 | cv2 |
| HWC | 1024×1024×3 | 3145728 | 0.3690 | 2.1178 | 0.1897 | cv2 |
| HWC | 1024×1024×9 | 9437184 | 0.8080 | 5.9545 | 0.3411 | cv2 |
| HWC | 96×96×9 | 82944 | 0.0089 | 0.0604 | 0.0215 | SZ full |
| DHWC | 32×128×128×1 | 524288 | 0.0452 | 0.0658 | 0.1327 | SZ full |
| DHWC | 64×128×128×3 | 3145728 | 0.3741 | 2.1045 | 0.9423 | SZ full |
| DHWC | 128×128×128×1 | 2097152 | 0.2578 | 0.3815 | 0.6501 | SZ full |
| DHWC | 48×256×256×3 | 9437184 | 0.8087 | 6.9410 | 2.4556 | SZ full |
| DHWC | 96×160×160×3 | 7372800 | 0.9176 | 5.7068 | 2.3497 | SZ full |
| DHWC | 6×32×32×9 | 55296 | 0.0062 | 0.0460 | 0.0158 | SZ full |
| NDHWC | 2×32×128×128×3 | 3145728 | 0.4044 | 2.2258 | 1.0065 | SZ full |
| NDHWC | 2×64×128×128×3 | 6291456 | 0.8193 | 4.8344 | 1.9913 | SZ full |
| NDHWC | 1×128×128×128×1 | 2097152 | 0.2652 | 0.4240 | 0.6650 | SZ full |

#### Per-channel LUT (different `256` table per channel)

| layout | shape | pixels | SZ loop C× `sz_lut` | cv2 loop C× `LUT` | cv2 flat 1× `(256,1,C)` | fastest |
|--------|-------|-------:|--------------------:|------------------:|--------------------------:|---------|
| HWC | 128×128×1 | 16384 | 0.0028 | 0.0054 | 0.0047 | SZ loop |
| HWC | 128×128×3 | 49152 | 0.0332 | 0.0136 | 0.0131 | cv2 flat |
| HWC | 256×256×1 | 65536 | 0.0139 | 0.0243 | 0.0170 | SZ loop |
| HWC | 256×256×3 | 196608 | 0.1700 | 0.0520 | 0.0505 | cv2 flat |
| HWC | 256×256×9 | 589824 | 0.3930 | 0.1690 | 0.1827 | cv2 loop |
| HWC | 512×512×1 | 262144 | 0.0614 | 0.0435 | 0.0335 | cv2 flat |
| HWC | 512×512×3 | 786432 | 0.5056 | 0.0692 | 0.0787 | cv2 loop |
| HWC | 512×512×9 | 2359296 | 1.6217 | 0.2791 | 0.2756 | cv2 flat |
| HWC | 1024×1024×1 | 1048576 | 0.1393 | 0.1458 | 0.0940 | cv2 flat |
| HWC | 1024×1024×3 | 3145728 | 2.2308 | 0.2120 | 0.2245 | cv2 loop |
| HWC | 1024×1024×9 | 9437184 | 6.1632 | 0.3824 | 0.3967 | cv2 loop |
| HWC | 96×96×9 | 82944 | 0.0602 | 0.0229 | 0.0235 | cv2 loop |
| DHWC | 32×128×128×1 | 524288 | 0.0879 | 0.1822 | 0.0661 | cv2 flat |
| DHWC | 64×128×128×3 | 3145728 | 2.2418 | 2.8030 | 0.2217 | cv2 flat |
| DHWC | 128×128×128×1 | 2097152 | 0.4309 | 0.8004 | 0.1610 | cv2 flat |
| DHWC | 48×256×256×3 | 9437184 | 7.1605 | 8.6764 | 0.3635 | cv2 flat |
| DHWC | 96×160×160×3 | 7372800 | 5.9308 | 7.3495 | 0.4510 | cv2 flat |
| DHWC | 6×32×32×9 | 55296 | 0.0451 | 0.0520 | 0.0165 | cv2 flat |
| NDHWC | 2×32×128×128×3 | 3145728 | 2.3846 | 2.7906 | 0.2162 | cv2 flat |
| NDHWC | 2×64×128×128×3 | 6291456 | 4.9196 | 6.1400 | 0.3178 | cv2 flat |
| NDHWC | 1×128×128×128×1 | 2097152 | 0.4034 | 0.8486 | 0.1390 | cv2 flat |

**Notes:**
- **SZ full** is only valid when one LUT applies to every byte (scalar `apply_lut` path).
- **SZ loop** matches the non-contiguous multi-channel `apply_lut` fallback (`sz_lut` per channel).
- **cv2** shared `(256,)`: contiguous `HWC` / `DHWC` / `NDHWC` work here. Per-channel distinct: direct `(256,1,C)` is **only** valid for `ndim==3`, but contiguous volumes/batches can be reshaped to HWC and use the same one-shot OpenCV path.
- Regenerate on your CPU; routing should follow benchmarks, not assumptions.
