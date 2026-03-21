# Router synthetic benchmark — comparison

- **New:** `0.0.41` (router_synthetic_0.0.41.json)
- **Old:** `0.0.40` (router_synthetic_0.0.40.json)

## Summary

- Comparable **ok/ok** cells: **410**
- **New slower** (old/new &lt; 0.85): **59** cells
- **New faster** (old/new &gt; 1.15): **65** cells

Ratio **old_ms / new_ms**: **&gt;1** ⇒ new build faster on that cell.

### Median old/new by op

- `add`: median **old/new** = **0.99x** (24 cells)
- `add_array`: median **old/new** = **2.86x** (24 cells)
- `add_constant`: median **old/new** = **1.01x** (24 cells)
- `add_vector`: median **old/new** = **0.99x** (24 cells)
- `add_weighted`: median **old/new** = **1.01x** (24 cells)
- `from_float`: median **old/new** = **1.01x** (12 cells)
- `hflip`: median **old/new** = **1.01x** (24 cells)
- `matmul`: median **old/new** = **0.96x** (1 cells)
- `median_blur`: median **old/new** = **1.01x** (12 cells)
- `multiply`: median **old/new** = **0.99x** (24 cells)
- `multiply_add`: median **old/new** = **1.03x** (24 cells)
- `multiply_by_array`: median **old/new** = **1.04x** (24 cells)
- `multiply_by_constant`: median **old/new** = **0.85x** (24 cells)
- `multiply_by_vector`: median **old/new** = **1.00x** (24 cells)
- `normalize`: median **old/new** = **1.01x** (24 cells)
- `normalize_per_image`: median **old/new** = **3.50x** (24 cells)
- `pairwise_distances_squared`: median **old/new** = **0.69x** (1 cells)
- `power`: median **old/new** = **1.01x** (24 cells)
- `sz_lut`: median **old/new** = **1.01x** (12 cells)
- `to_float`: median **old/new** = **1.02x** (12 cells)
- `vflip`: median **old/new** = **1.00x** (24 cells)

### Largest regressions (new slower)

| op | layout | shape | dtype | old/new |
|----|--------|-------|-------|--------:|
| multiply_by_constant | HWC | (512,512,1) | float32 | 0.42x |
| vflip | HWC | (128,128,9) | uint8 | 0.45x |
| multiply | HWC | (512,512,1) | float32 | 0.45x |
| multiply | HWC | (1024,1024,1) | float32 | 0.47x |
| normalize_per_image | HWC | (512,512,1) | float32 | 0.48x |
| vflip | HWC | (256,256,1) | float32 | 0.49x |
| normalize_per_image | HWC | (128,128,9) | float32 | 0.52x |
| normalize_per_image | HWC | (256,256,1) | float32 | 0.53x |
| vflip | HWC | (128,128,3) | uint8 | 0.54x |
| multiply_by_constant | HWC | (256,256,3) | float32 | 0.56x |
| normalize_per_image | HWC | (256,256,3) | float32 | 0.58x |
| from_float | HWC | (128,128,9) | float32 | 0.58x |
| normalize_per_image | HWC | (1024,1024,1) | float32 | 0.60x |
| multiply_by_constant | HWC | (256,256,9) | uint8 | 0.60x |
| multiply_by_constant | HWC | (1024,1024,1) | float32 | 0.60x |
| normalize_per_image | HWC | (512,512,3) | float32 | 0.66x |
| normalize | HWC | (1024,1024,3) | uint8 | 0.67x |
| normalize_per_image | HWC | (1024,1024,3) | float32 | 0.68x |
| normalize_per_image | HWC | (256,256,9) | float32 | 0.69x |
| multiply | HWC | (512,512,3) | float32 | 0.69x |
| pairwise_distances_squared | points | (24,16,3) | float32 | 0.69x |
| normalize_per_image | HWC | (1024,1024,9) | float32 | 0.71x |
| multiply | HWC | (128,128,3) | float32 | 0.72x |
| multiply_by_constant | HWC | (256,256,9) | float32 | 0.72x |
| multiply_by_constant | HWC | (1024,1024,9) | float32 | 0.73x |

### Largest wins (new faster)

| op | layout | shape | dtype | old/new |
|----|--------|-------|-------|--------:|
| normalize_per_image | HWC | (1024,1024,9) | uint8 | 10.25x |
| normalize_per_image | HWC | (512,512,1) | uint8 | 9.52x |
| normalize_per_image | HWC | (512,512,9) | uint8 | 8.47x |
| normalize_per_image | HWC | (512,512,3) | uint8 | 7.43x |
| add_array | HWC | (128,128,1) | uint8 | 7.15x |
| normalize_per_image | HWC | (256,256,3) | uint8 | 6.71x |
| normalize_per_image | HWC | (1024,1024,1) | uint8 | 6.12x |
| normalize_per_image | HWC | (128,128,9) | uint8 | 5.90x |
| normalize_per_image | HWC | (1024,1024,3) | uint8 | 5.56x |
| add_array | HWC | (128,128,3) | uint8 | 5.46x |
| add_array | HWC | (512,512,1) | uint8 | 5.04x |
| add_array | HWC | (256,256,1) | uint8 | 4.85x |
| normalize_per_image | HWC | (128,128,3) | uint8 | 4.25x |
| add_array | HWC | (1024,1024,1) | uint8 | 4.14x |
| normalize_per_image | HWC | (256,256,9) | uint8 | 3.77x |
| normalize_per_image | HWC | (256,256,1) | uint8 | 3.56x |
| add_array | HWC | (1024,1024,3) | uint8 | 3.52x |
| normalize_per_image | HWC | (128,128,1) | uint8 | 3.50x |
| add_array | HWC | (256,256,3) | uint8 | 3.38x |
| add_array | HWC | (128,128,9) | uint8 | 3.37x |
| add_array | HWC | (512,512,3) | uint8 | 3.29x |
| add_array | HWC | (1024,1024,9) | uint8 | 3.17x |
| add_array | HWC | (256,256,9) | uint8 | 2.92x |
| add_array | HWC | (512,512,9) | uint8 | 2.86x |
| to_float | HWC | (512,512,1) | uint8 | 2.11x |

## Full table

Only rows where **both** runs are `ok`.

| op | layout | shape | dtype | new_ms | old_ms | old/new |
|----|--------|-------|-------|-------:|-------:|--------:|
| add | HWC | (1024,1024,1) | float32 | 0.9568 | 0.9470 | 0.99x |
| add | HWC | (1024,1024,1) | uint8 | 0.0439 | 0.0430 | 0.98x |
| add | HWC | (1024,1024,3) | float32 | 1.2016 | 1.2101 | 1.01x |
| add | HWC | (1024,1024,3) | uint8 | 0.3040 | 0.2472 | 0.81x |
| add | HWC | (1024,1024,9) | float32 | 12.7787 | 13.7321 | 1.07x |
| add | HWC | (1024,1024,9) | uint8 | 0.3572 | 0.3395 | 0.95x |
| add | HWC | (128,128,1) | float32 | 0.0119 | 0.0118 | 0.99x |
| add | HWC | (128,128,1) | uint8 | 0.0054 | 0.0048 | 0.89x |
| add | HWC | (128,128,3) | float32 | 0.0227 | 0.0224 | 0.99x |
| add | HWC | (128,128,3) | uint8 | 0.0040 | 0.0035 | 0.88x |
| add | HWC | (128,128,9) | float32 | 0.0627 | 0.0642 | 1.02x |
| add | HWC | (128,128,9) | uint8 | 0.0079 | 0.0078 | 0.98x |
| add | HWC | (256,256,1) | float32 | 0.0307 | 0.0300 | 0.98x |
| add | HWC | (256,256,1) | uint8 | 0.0066 | 0.0066 | 1.00x |
| add | HWC | (256,256,3) | float32 | 0.0770 | 0.0774 | 1.00x |
| add | HWC | (256,256,3) | uint8 | 0.0102 | 0.0102 | 1.00x |
| add | HWC | (256,256,9) | float32 | 0.5748 | 0.5542 | 0.96x |
| add | HWC | (256,256,9) | uint8 | 0.0278 | 0.0222 | 0.80x |
| add | HWC | (512,512,1) | float32 | 0.1312 | 0.1233 | 0.94x |
| add | HWC | (512,512,1) | uint8 | 0.0145 | 0.0150 | 1.03x |
| add | HWC | (512,512,3) | float32 | 0.6323 | 0.6294 | 1.00x |
| add | HWC | (512,512,3) | uint8 | 0.0322 | 0.0307 | 0.95x |
| add | HWC | (512,512,9) | float32 | 0.9099 | 0.9574 | 1.05x |
| add | HWC | (512,512,9) | uint8 | 0.3105 | 0.2868 | 0.92x |
| add_array | HWC | (1024,1024,1) | float32 | 1.1684 | 0.9440 | 0.81x |
| add_array | HWC | (1024,1024,1) | uint8 | 0.0283 | 0.1172 | 4.14x |
| add_array | HWC | (1024,1024,3) | float32 | 1.1356 | 1.2227 | 1.08x |
| add_array | HWC | (1024,1024,3) | uint8 | 0.2550 | 0.8970 | 3.52x |
| add_array | HWC | (1024,1024,9) | float32 | 10.7769 | 9.0917 | 0.84x |
| add_array | HWC | (1024,1024,9) | uint8 | 0.2411 | 0.7644 | 3.17x |
| add_array | HWC | (128,128,1) | float32 | 0.0092 | 0.0120 | 1.31x |
| add_array | HWC | (128,128,1) | uint8 | 0.0017 | 0.0119 | 7.15x |
| add_array | HWC | (128,128,3) | float32 | 0.0210 | 0.0227 | 1.08x |
| add_array | HWC | (128,128,3) | uint8 | 0.0024 | 0.0130 | 5.46x |
| add_array | HWC | (128,128,9) | float32 | 0.0565 | 0.0581 | 1.03x |
| add_array | HWC | (128,128,9) | uint8 | 0.0054 | 0.0181 | 3.37x |
| add_array | HWC | (256,256,1) | float32 | 0.0270 | 0.0294 | 1.09x |
| add_array | HWC | (256,256,1) | uint8 | 0.0028 | 0.0133 | 4.85x |
| add_array | HWC | (256,256,3) | float32 | 0.0984 | 0.0769 | 0.78x |
| add_array | HWC | (256,256,3) | uint8 | 0.0065 | 0.0218 | 3.38x |
| add_array | HWC | (256,256,9) | float32 | 0.4166 | 0.5773 | 1.39x |
| add_array | HWC | (256,256,9) | uint8 | 0.0166 | 0.0485 | 2.92x |
| add_array | HWC | (512,512,1) | float32 | 0.1532 | 0.1144 | 0.75x |
| add_array | HWC | (512,512,1) | uint8 | 0.0085 | 0.0426 | 5.04x |
| add_array | HWC | (512,512,3) | float32 | 0.6071 | 0.8647 | 1.42x |
| add_array | HWC | (512,512,3) | uint8 | 0.0218 | 0.0717 | 3.29x |
| add_array | HWC | (512,512,9) | float32 | 0.8457 | 0.8310 | 0.98x |
| add_array | HWC | (512,512,9) | uint8 | 0.1887 | 0.5391 | 2.86x |
| add_constant | HWC | (1024,1024,1) | float32 | 1.0058 | 1.1112 | 1.10x |
| add_constant | HWC | (1024,1024,1) | uint8 | 0.0427 | 0.0406 | 0.95x |
| add_constant | HWC | (1024,1024,3) | float32 | 1.1978 | 1.2395 | 1.03x |
| add_constant | HWC | (1024,1024,3) | uint8 | 0.2767 | 0.2684 | 0.97x |
| add_constant | HWC | (1024,1024,9) | float32 | 11.6194 | 13.9795 | 1.20x |
| add_constant | HWC | (1024,1024,9) | uint8 | 0.3480 | 0.3364 | 0.97x |
| add_constant | HWC | (128,128,1) | float32 | 0.0113 | 0.0114 | 1.01x |
| add_constant | HWC | (128,128,1) | uint8 | 0.0042 | 0.0039 | 0.92x |
| add_constant | HWC | (128,128,3) | float32 | 0.0230 | 0.0223 | 0.97x |
| add_constant | HWC | (128,128,3) | uint8 | 0.0037 | 0.0038 | 1.05x |
| add_constant | HWC | (128,128,9) | float32 | 0.0619 | 0.0629 | 1.02x |
| add_constant | HWC | (128,128,9) | uint8 | 0.0076 | 0.0074 | 0.98x |
| add_constant | HWC | (256,256,1) | float32 | 0.0295 | 0.0295 | 1.00x |
| add_constant | HWC | (256,256,1) | uint8 | 0.0057 | 0.0062 | 1.09x |
| add_constant | HWC | (256,256,3) | float32 | 0.0803 | 0.0839 | 1.04x |
| add_constant | HWC | (256,256,3) | uint8 | 0.0097 | 0.0097 | 1.00x |
| add_constant | HWC | (256,256,9) | float32 | 0.5477 | 0.6250 | 1.14x |
| add_constant | HWC | (256,256,9) | uint8 | 0.0220 | 0.0219 | 0.99x |
| add_constant | HWC | (512,512,1) | float32 | 0.1292 | 0.1220 | 0.94x |
| add_constant | HWC | (512,512,1) | uint8 | 0.0141 | 0.0145 | 1.03x |
| add_constant | HWC | (512,512,3) | float32 | 0.6966 | 0.6267 | 0.90x |
| add_constant | HWC | (512,512,3) | uint8 | 0.0347 | 0.0317 | 0.91x |
| add_constant | HWC | (512,512,9) | float32 | 0.8865 | 0.9126 | 1.03x |
| add_constant | HWC | (512,512,9) | uint8 | 0.3326 | 0.3647 | 1.10x |
| add_vector | HWC | (1024,1024,1) | float32 | 1.6577 | 1.4719 | 0.89x |
| add_vector | HWC | (1024,1024,1) | uint8 | 0.1063 | 0.1056 | 0.99x |
| add_vector | HWC | (1024,1024,3) | float32 | 4.5751 | 4.5265 | 0.99x |
| add_vector | HWC | (1024,1024,3) | uint8 | 3.0171 | 2.6823 | 0.89x |
| add_vector | HWC | (1024,1024,9) | float32 | 17.7200 | 16.2506 | 0.92x |
| add_vector | HWC | (1024,1024,9) | uint8 | 6.5430 | 6.6087 | 1.01x |
| add_vector | HWC | (128,128,1) | float32 | 0.0139 | 0.0135 | 0.97x |
| add_vector | HWC | (128,128,1) | uint8 | 0.0073 | 0.0069 | 0.95x |
| add_vector | HWC | (128,128,3) | float32 | 0.0765 | 0.0724 | 0.95x |
| add_vector | HWC | (128,128,3) | uint8 | 0.0386 | 0.0394 | 1.02x |
| add_vector | HWC | (128,128,9) | float32 | 0.1292 | 0.1227 | 0.95x |
| add_vector | HWC | (128,128,9) | uint8 | 0.1093 | 0.1097 | 1.00x |
| add_vector | HWC | (256,256,1) | float32 | 0.0355 | 0.0317 | 0.89x |
| add_vector | HWC | (256,256,1) | uint8 | 0.0121 | 0.0124 | 1.02x |
| add_vector | HWC | (256,256,3) | float32 | 0.2830 | 0.2806 | 0.99x |
| add_vector | HWC | (256,256,3) | uint8 | 0.1253 | 0.1281 | 1.02x |
| add_vector | HWC | (256,256,9) | float32 | 0.8232 | 0.8684 | 1.05x |
| add_vector | HWC | (256,256,9) | uint8 | 0.3751 | 0.3659 | 0.98x |
| add_vector | HWC | (512,512,1) | float32 | 0.1380 | 0.1101 | 0.80x |
| add_vector | HWC | (512,512,1) | uint8 | 0.0415 | 0.0653 | 1.58x |
| add_vector | HWC | (512,512,3) | float32 | 1.7468 | 1.5906 | 0.91x |
| add_vector | HWC | (512,512,3) | uint8 | 0.4782 | 0.5335 | 1.12x |
| add_vector | HWC | (512,512,9) | float32 | 1.8983 | 1.9377 | 1.02x |
| add_vector | HWC | (512,512,9) | uint8 | 1.6159 | 2.1217 | 1.31x |
| add_weighted | HWC | (1024,1024,1) | float32 | 0.9817 | 0.7606 | 0.77x |
| add_weighted | HWC | (1024,1024,1) | uint8 | 0.0683 | 0.0667 | 0.98x |
| add_weighted | HWC | (1024,1024,3) | float32 | 1.2108 | 1.2246 | 1.01x |
| add_weighted | HWC | (1024,1024,3) | uint8 | 0.3960 | 0.3587 | 0.91x |
| add_weighted | HWC | (1024,1024,9) | float32 | 9.6773 | 10.0016 | 1.03x |
| add_weighted | HWC | (1024,1024,9) | uint8 | 0.6527 | 0.6089 | 0.93x |
| add_weighted | HWC | (128,128,1) | float32 | 0.0094 | 0.0093 | 0.99x |
| add_weighted | HWC | (128,128,1) | uint8 | 0.0022 | 0.0023 | 1.08x |
| add_weighted | HWC | (128,128,3) | float32 | 0.0224 | 0.0206 | 0.92x |
| add_weighted | HWC | (128,128,3) | uint8 | 0.0043 | 0.0046 | 1.07x |
| add_weighted | HWC | (128,128,9) | float32 | 0.0557 | 0.0558 | 1.00x |
| add_weighted | HWC | (128,128,9) | uint8 | 0.0107 | 0.0108 | 1.01x |
| add_weighted | HWC | (256,256,1) | float32 | 0.0265 | 0.0264 | 1.00x |
| add_weighted | HWC | (256,256,1) | uint8 | 0.0054 | 0.0056 | 1.04x |
| add_weighted | HWC | (256,256,3) | float32 | 0.0732 | 0.0852 | 1.16x |
| add_weighted | HWC | (256,256,3) | uint8 | 0.0138 | 0.0140 | 1.02x |
| add_weighted | HWC | (256,256,9) | float32 | 0.4566 | 0.4505 | 0.99x |
| add_weighted | HWC | (256,256,9) | uint8 | 0.0384 | 0.0381 | 0.99x |
| add_weighted | HWC | (512,512,1) | float32 | 0.0963 | 0.1422 | 1.48x |
| add_weighted | HWC | (512,512,1) | uint8 | 0.0180 | 0.0185 | 1.03x |
| add_weighted | HWC | (512,512,3) | float32 | 0.6711 | 0.6019 | 0.90x |
| add_weighted | HWC | (512,512,3) | uint8 | 0.0510 | 0.0506 | 0.99x |
| add_weighted | HWC | (512,512,9) | float32 | 0.8449 | 0.8860 | 1.05x |
| add_weighted | HWC | (512,512,9) | uint8 | 0.2421 | 0.2705 | 1.12x |
| from_float | HWC | (1024,1024,1) | float32 | 1.3873 | 1.4633 | 1.05x |
| from_float | HWC | (1024,1024,3) | float32 | 2.5172 | 2.5279 | 1.00x |
| from_float | HWC | (1024,1024,9) | float32 | 20.0694 | 19.2899 | 0.96x |
| from_float | HWC | (128,128,1) | float32 | 0.0147 | 0.0151 | 1.03x |
| from_float | HWC | (128,128,3) | float32 | 0.0365 | 0.0363 | 0.99x |
| from_float | HWC | (128,128,9) | float32 | 0.1922 | 0.1119 | 0.58x |
| from_float | HWC | (256,256,1) | float32 | 0.0487 | 0.0494 | 1.01x |
| from_float | HWC | (256,256,3) | float32 | 0.1956 | 0.2230 | 1.14x |
| from_float | HWC | (256,256,9) | float32 | 1.0198 | 1.5162 | 1.49x |
| from_float | HWC | (512,512,1) | float32 | 0.3219 | 0.3036 | 0.94x |
| from_float | HWC | (512,512,3) | float32 | 0.9903 | 1.4110 | 1.42x |
| from_float | HWC | (512,512,9) | float32 | 2.0707 | 1.9354 | 0.93x |
| hflip | HWC | (1024,1024,1) | float32 | 0.2678 | 0.2697 | 1.01x |
| hflip | HWC | (1024,1024,1) | uint8 | 0.0254 | 0.0222 | 0.87x |
| hflip | HWC | (1024,1024,3) | float32 | 0.4157 | 0.4093 | 0.98x |
| hflip | HWC | (1024,1024,3) | uint8 | 0.5370 | 0.4296 | 0.80x |
| hflip | HWC | (1024,1024,9) | float32 | 9.3037 | 10.4154 | 1.12x |
| hflip | HWC | (1024,1024,9) | uint8 | 1.7205 | 1.6813 | 0.98x |
| hflip | HWC | (128,128,1) | float32 | 0.0027 | 0.0028 | 1.03x |
| hflip | HWC | (128,128,1) | uint8 | 0.0016 | 0.0017 | 1.05x |
| hflip | HWC | (128,128,3) | float32 | 0.0066 | 0.0068 | 1.04x |
| hflip | HWC | (128,128,3) | uint8 | 0.0050 | 0.0050 | 1.00x |
| hflip | HWC | (128,128,9) | float32 | 0.1000 | 0.0985 | 0.99x |
| hflip | HWC | (128,128,9) | uint8 | 0.0264 | 0.0267 | 1.01x |
| hflip | HWC | (256,256,1) | float32 | 0.0067 | 0.0067 | 1.01x |
| hflip | HWC | (256,256,1) | uint8 | 0.0028 | 0.0027 | 0.97x |
| hflip | HWC | (256,256,3) | float32 | 0.0268 | 0.0261 | 0.97x |
| hflip | HWC | (256,256,3) | uint8 | 0.0197 | 0.0198 | 1.01x |
| hflip | HWC | (256,256,9) | float32 | 0.5053 | 0.5404 | 1.07x |
| hflip | HWC | (256,256,9) | uint8 | 0.1038 | 0.1028 | 0.99x |
| hflip | HWC | (512,512,1) | float32 | 0.0218 | 0.0216 | 0.99x |
| hflip | HWC | (512,512,1) | uint8 | 0.0066 | 0.0067 | 1.02x |
| hflip | HWC | (512,512,3) | float32 | 0.2493 | 0.2593 | 1.04x |
| hflip | HWC | (512,512,3) | uint8 | 0.0767 | 0.0888 | 1.16x |
| hflip | HWC | (512,512,9) | float32 | 1.8046 | 1.6753 | 0.93x |
| hflip | HWC | (512,512,9) | uint8 | 0.5148 | 0.4951 | 0.96x |
| matmul | 2D | (128,64,64,32) | float32 | 0.0021 | 0.0020 | 0.96x |
| median_blur | HWC | (1024,1024,1) | uint8 | 0.1615 | 0.1504 | 0.93x |
| median_blur | HWC | (1024,1024,3) | uint8 | 0.6492 | 0.5744 | 0.88x |
| median_blur | HWC | (1024,1024,9) | uint8 | 1.2632 | 1.2700 | 1.01x |
| median_blur | HWC | (128,128,1) | uint8 | 0.0073 | 0.0083 | 1.13x |
| median_blur | HWC | (128,128,3) | uint8 | 0.0128 | 0.0127 | 0.99x |
| median_blur | HWC | (128,128,9) | uint8 | 0.0308 | 0.0316 | 1.03x |
| median_blur | HWC | (256,256,1) | uint8 | 0.0163 | 0.0156 | 0.96x |
| median_blur | HWC | (256,256,3) | uint8 | 0.0349 | 0.0376 | 1.08x |
| median_blur | HWC | (256,256,9) | uint8 | 0.1121 | 0.0969 | 0.86x |
| median_blur | HWC | (512,512,1) | uint8 | 0.0437 | 0.0536 | 1.23x |
| median_blur | HWC | (512,512,3) | uint8 | 0.1186 | 0.1189 | 1.00x |
| median_blur | HWC | (512,512,9) | uint8 | 0.4682 | 0.4745 | 1.01x |
| multiply | HWC | (1024,1024,1) | float32 | 1.6715 | 0.7870 | 0.47x |
| multiply | HWC | (1024,1024,1) | uint8 | 0.0922 | 0.0914 | 0.99x |
| multiply | HWC | (1024,1024,3) | float32 | 1.2877 | 1.0830 | 0.84x |
| multiply | HWC | (1024,1024,3) | uint8 | 0.4252 | 0.4190 | 0.99x |
| multiply | HWC | (1024,1024,9) | float32 | 12.9899 | 9.8653 | 0.76x |
| multiply | HWC | (1024,1024,9) | uint8 | 0.8251 | 0.8460 | 1.03x |
| multiply | HWC | (128,128,1) | float32 | 0.0108 | 0.0085 | 0.79x |
| multiply | HWC | (128,128,1) | uint8 | 0.0058 | 0.0059 | 1.01x |
| multiply | HWC | (128,128,3) | float32 | 0.0260 | 0.0188 | 0.72x |
| multiply | HWC | (128,128,3) | uint8 | 0.0094 | 0.0095 | 1.01x |
| multiply | HWC | (128,128,9) | float32 | 0.0587 | 0.0535 | 0.91x |
| multiply | HWC | (128,128,9) | uint8 | 0.0176 | 0.0200 | 1.14x |
| multiply | HWC | (256,256,1) | float32 | 0.0288 | 0.0240 | 0.83x |
| multiply | HWC | (256,256,1) | uint8 | 0.0107 | 0.0109 | 1.02x |
| multiply | HWC | (256,256,3) | float32 | 0.0875 | 0.0732 | 0.84x |
| multiply | HWC | (256,256,3) | uint8 | 0.0213 | 0.0217 | 1.02x |
| multiply | HWC | (256,256,9) | float32 | 0.5519 | 0.4333 | 0.79x |
| multiply | HWC | (256,256,9) | uint8 | 0.0548 | 0.0582 | 1.06x |
| multiply | HWC | (512,512,1) | float32 | 0.1969 | 0.0888 | 0.45x |
| multiply | HWC | (512,512,1) | uint8 | 0.0273 | 0.0323 | 1.18x |
| multiply | HWC | (512,512,3) | float32 | 0.8834 | 0.6123 | 0.69x |
| multiply | HWC | (512,512,3) | uint8 | 0.0702 | 0.0735 | 1.05x |
| multiply | HWC | (512,512,9) | float32 | 0.9572 | 0.7967 | 0.83x |
| multiply | HWC | (512,512,9) | uint8 | 0.3088 | 0.3464 | 1.12x |
| multiply_add | HWC | (1024,1024,1) | float32 | 1.0369 | 1.3945 | 1.34x |
| multiply_add | HWC | (1024,1024,1) | uint8 | 0.0920 | 0.0918 | 1.00x |
| multiply_add | HWC | (1024,1024,3) | float32 | 1.2703 | 1.5453 | 1.22x |
| multiply_add | HWC | (1024,1024,3) | uint8 | 0.4301 | 0.4405 | 1.02x |
| multiply_add | HWC | (1024,1024,9) | float32 | 12.0293 | 17.0743 | 1.42x |
| multiply_add | HWC | (1024,1024,9) | uint8 | 0.8250 | 0.8139 | 0.99x |
| multiply_add | HWC | (128,128,1) | float32 | 0.0112 | 0.0152 | 1.37x |
| multiply_add | HWC | (128,128,1) | uint8 | 0.0059 | 0.0056 | 0.96x |
| multiply_add | HWC | (128,128,3) | float32 | 0.0235 | 0.0281 | 1.19x |
| multiply_add | HWC | (128,128,3) | uint8 | 0.0091 | 0.0092 | 1.01x |
| multiply_add | HWC | (128,128,9) | float32 | 0.0617 | 0.0787 | 1.27x |
| multiply_add | HWC | (128,128,9) | uint8 | 0.0177 | 0.0176 | 0.99x |
| multiply_add | HWC | (256,256,1) | float32 | 0.0322 | 0.0365 | 1.13x |
| multiply_add | HWC | (256,256,1) | uint8 | 0.0108 | 0.0110 | 1.02x |
| multiply_add | HWC | (256,256,3) | float32 | 0.1098 | 0.1128 | 1.03x |
| multiply_add | HWC | (256,256,3) | uint8 | 0.0213 | 0.0227 | 1.07x |
| multiply_add | HWC | (256,256,9) | float32 | 0.6437 | 0.9995 | 1.55x |
| multiply_add | HWC | (256,256,9) | uint8 | 0.0595 | 0.0546 | 0.92x |
| multiply_add | HWC | (512,512,1) | float32 | 0.1929 | 0.1441 | 0.75x |
| multiply_add | HWC | (512,512,1) | uint8 | 0.0284 | 0.0281 | 0.99x |
| multiply_add | HWC | (512,512,3) | float32 | 0.8355 | 1.0192 | 1.22x |
| multiply_add | HWC | (512,512,3) | uint8 | 0.0843 | 0.0708 | 0.84x |
| multiply_add | HWC | (512,512,9) | float32 | 0.9723 | 1.1805 | 1.21x |
| multiply_add | HWC | (512,512,9) | uint8 | 0.3333 | 0.3185 | 0.96x |
| multiply_by_array | HWC | (1024,1024,1) | float32 | 0.8161 | 0.9273 | 1.14x |
| multiply_by_array | HWC | (1024,1024,1) | uint8 | 1.7986 | 2.0614 | 1.15x |
| multiply_by_array | HWC | (1024,1024,3) | float32 | 1.1871 | 1.1510 | 0.97x |
| multiply_by_array | HWC | (1024,1024,3) | uint8 | 3.2221 | 3.2310 | 1.00x |
| multiply_by_array | HWC | (1024,1024,9) | float32 | 9.7523 | 9.2744 | 0.95x |
| multiply_by_array | HWC | (1024,1024,9) | uint8 | 18.2612 | 18.5525 | 1.02x |
| multiply_by_array | HWC | (128,128,1) | float32 | 0.0085 | 0.0117 | 1.39x |
| multiply_by_array | HWC | (128,128,1) | uint8 | 0.0238 | 0.0214 | 0.90x |
| multiply_by_array | HWC | (128,128,3) | float32 | 0.0202 | 0.0227 | 1.13x |
| multiply_by_array | HWC | (128,128,3) | uint8 | 0.0492 | 0.0509 | 1.03x |
| multiply_by_array | HWC | (128,128,9) | float32 | 0.0571 | 0.0593 | 1.04x |
| multiply_by_array | HWC | (128,128,9) | uint8 | 0.1375 | 0.1474 | 1.07x |
| multiply_by_array | HWC | (256,256,1) | float32 | 0.0306 | 0.0297 | 0.97x |
| multiply_by_array | HWC | (256,256,1) | uint8 | 0.0661 | 0.0673 | 1.02x |
| multiply_by_array | HWC | (256,256,3) | float32 | 0.0802 | 0.1023 | 1.28x |
| multiply_by_array | HWC | (256,256,3) | uint8 | 0.2591 | 0.2588 | 1.00x |
| multiply_by_array | HWC | (256,256,9) | float32 | 0.4181 | 0.4858 | 1.16x |
| multiply_by_array | HWC | (256,256,9) | uint8 | 0.8260 | 0.9358 | 1.13x |
| multiply_by_array | HWC | (512,512,1) | float32 | 0.1101 | 0.1248 | 1.13x |
| multiply_by_array | HWC | (512,512,1) | uint8 | 0.2761 | 0.3649 | 1.32x |
| multiply_by_array | HWC | (512,512,3) | float32 | 0.6246 | 0.6805 | 1.09x |
| multiply_by_array | HWC | (512,512,3) | uint8 | 1.1985 | 1.2084 | 1.01x |
| multiply_by_array | HWC | (512,512,9) | float32 | 0.9284 | 0.8990 | 0.97x |
| multiply_by_array | HWC | (512,512,9) | uint8 | 2.3048 | 2.2587 | 0.98x |
| multiply_by_constant | HWC | (1024,1024,1) | float32 | 1.2419 | 0.7512 | 0.60x |
| multiply_by_constant | HWC | (1024,1024,1) | uint8 | 0.0918 | 0.0988 | 1.08x |
| multiply_by_constant | HWC | (1024,1024,3) | float32 | 1.3307 | 1.0513 | 0.79x |
| multiply_by_constant | HWC | (1024,1024,3) | uint8 | 0.4217 | 0.5328 | 1.26x |
| multiply_by_constant | HWC | (1024,1024,9) | float32 | 13.1603 | 9.5621 | 0.73x |
| multiply_by_constant | HWC | (1024,1024,9) | uint8 | 0.8322 | 0.8089 | 0.97x |
| multiply_by_constant | HWC | (128,128,1) | float32 | 0.0108 | 0.0081 | 0.75x |
| multiply_by_constant | HWC | (128,128,1) | uint8 | 0.0070 | 0.0055 | 0.78x |
| multiply_by_constant | HWC | (128,128,3) | float32 | 0.0219 | 0.0187 | 0.85x |
| multiply_by_constant | HWC | (128,128,3) | uint8 | 0.0088 | 0.0091 | 1.04x |
| multiply_by_constant | HWC | (128,128,9) | float32 | 0.0581 | 0.0565 | 0.97x |
| multiply_by_constant | HWC | (128,128,9) | uint8 | 0.0173 | 0.0173 | 1.00x |
| multiply_by_constant | HWC | (256,256,1) | float32 | 0.0283 | 0.0236 | 0.83x |
| multiply_by_constant | HWC | (256,256,1) | uint8 | 0.0104 | 0.0104 | 1.00x |
| multiply_by_constant | HWC | (256,256,3) | float32 | 0.1132 | 0.0634 | 0.56x |
| multiply_by_constant | HWC | (256,256,3) | uint8 | 0.0217 | 0.0212 | 0.98x |
| multiply_by_constant | HWC | (256,256,9) | float32 | 0.5969 | 0.4316 | 0.72x |
| multiply_by_constant | HWC | (256,256,9) | uint8 | 0.0884 | 0.0533 | 0.60x |
| multiply_by_constant | HWC | (512,512,1) | float32 | 0.2500 | 0.1048 | 0.42x |
| multiply_by_constant | HWC | (512,512,1) | uint8 | 0.0273 | 0.0265 | 0.97x |
| multiply_by_constant | HWC | (512,512,3) | float32 | 0.8267 | 0.6347 | 0.77x |
| multiply_by_constant | HWC | (512,512,3) | uint8 | 0.0741 | 0.0694 | 0.94x |
| multiply_by_constant | HWC | (512,512,9) | float32 | 0.9787 | 0.7940 | 0.81x |
| multiply_by_constant | HWC | (512,512,9) | uint8 | 0.2717 | 0.4162 | 1.53x |
| multiply_by_vector | HWC | (1024,1024,1) | float32 | 1.0847 | 1.1921 | 1.10x |
| multiply_by_vector | HWC | (1024,1024,1) | uint8 | 0.1463 | 0.1113 | 0.76x |
| multiply_by_vector | HWC | (1024,1024,3) | float32 | 4.6499 | 4.4704 | 0.96x |
| multiply_by_vector | HWC | (1024,1024,3) | uint8 | 2.1472 | 2.4830 | 1.16x |
| multiply_by_vector | HWC | (1024,1024,9) | float32 | 13.9420 | 14.0733 | 1.01x |
| multiply_by_vector | HWC | (1024,1024,9) | uint8 | 6.4006 | 6.1059 | 0.95x |
| multiply_by_vector | HWC | (128,128,1) | float32 | 0.0150 | 0.0131 | 0.87x |
| multiply_by_vector | HWC | (128,128,1) | uint8 | 0.0070 | 0.0062 | 0.89x |
| multiply_by_vector | HWC | (128,128,3) | float32 | 0.0729 | 0.0725 | 1.00x |
| multiply_by_vector | HWC | (128,128,3) | uint8 | 0.0381 | 0.0381 | 1.00x |
| multiply_by_vector | HWC | (128,128,9) | float32 | 0.1098 | 0.1169 | 1.06x |
| multiply_by_vector | HWC | (128,128,9) | uint8 | 0.1078 | 0.1087 | 1.01x |
| multiply_by_vector | HWC | (256,256,1) | float32 | 0.0322 | 0.0333 | 1.03x |
| multiply_by_vector | HWC | (256,256,1) | uint8 | 0.0159 | 0.0118 | 0.74x |
| multiply_by_vector | HWC | (256,256,3) | float32 | 0.3006 | 0.2753 | 0.92x |
| multiply_by_vector | HWC | (256,256,3) | uint8 | 0.1463 | 0.1260 | 0.86x |
| multiply_by_vector | HWC | (256,256,9) | float32 | 0.7301 | 0.7669 | 1.05x |
| multiply_by_vector | HWC | (256,256,9) | uint8 | 0.4131 | 0.4209 | 1.02x |
| multiply_by_vector | HWC | (512,512,1) | float32 | 0.1457 | 0.1354 | 0.93x |
| multiply_by_vector | HWC | (512,512,1) | uint8 | 0.0312 | 0.0420 | 1.34x |
| multiply_by_vector | HWC | (512,512,3) | float32 | 1.5313 | 1.8545 | 1.21x |
| multiply_by_vector | HWC | (512,512,3) | uint8 | 0.5457 | 0.5127 | 0.94x |
| multiply_by_vector | HWC | (512,512,9) | float32 | 1.8931 | 1.9005 | 1.00x |
| multiply_by_vector | HWC | (512,512,9) | uint8 | 1.6480 | 1.6445 | 1.00x |
| normalize | HWC | (1024,1024,1) | float32 | 1.0267 | 0.8569 | 0.83x |
| normalize | HWC | (1024,1024,1) | uint8 | 0.2921 | 0.2623 | 0.90x |
| normalize | HWC | (1024,1024,3) | float32 | 0.7717 | 0.7500 | 0.97x |
| normalize | HWC | (1024,1024,3) | uint8 | 0.2768 | 0.1868 | 0.67x |
| normalize | HWC | (1024,1024,9) | float32 | 10.0318 | 10.7897 | 1.08x |
| normalize | HWC | (1024,1024,9) | uint8 | 1.4241 | 1.3430 | 0.94x |
| normalize | HWC | (128,128,1) | float32 | 0.0053 | 0.0061 | 1.16x |
| normalize | HWC | (128,128,1) | uint8 | 0.0085 | 0.0079 | 0.93x |
| normalize | HWC | (128,128,3) | float32 | 0.0139 | 0.0138 | 1.00x |
| normalize | HWC | (128,128,3) | uint8 | 0.0148 | 0.0151 | 1.02x |
| normalize | HWC | (128,128,9) | float32 | 0.0350 | 0.0357 | 1.02x |
| normalize | HWC | (128,128,9) | uint8 | 0.0415 | 0.0403 | 0.97x |
| normalize | HWC | (256,256,1) | float32 | 0.0193 | 0.0179 | 0.93x |
| normalize | HWC | (256,256,1) | uint8 | 0.0203 | 0.0202 | 1.00x |
| normalize | HWC | (256,256,3) | float32 | 0.0524 | 0.0584 | 1.11x |
| normalize | HWC | (256,256,3) | uint8 | 0.0517 | 0.0525 | 1.01x |
| normalize | HWC | (256,256,9) | float32 | 0.5044 | 0.4975 | 0.99x |
| normalize | HWC | (256,256,9) | uint8 | 0.2614 | 0.3102 | 1.19x |
| normalize | HWC | (512,512,1) | float32 | 0.1133 | 0.1305 | 1.15x |
| normalize | HWC | (512,512,1) | uint8 | 0.0448 | 0.0617 | 1.38x |
| normalize | HWC | (512,512,3) | float32 | 0.7203 | 0.8844 | 1.23x |
| normalize | HWC | (512,512,3) | uint8 | 0.1265 | 0.2310 | 1.83x |
| normalize | HWC | (512,512,9) | float32 | 0.5432 | 0.5357 | 0.99x |
| normalize | HWC | (512,512,9) | uint8 | 0.1713 | 0.2095 | 1.22x |
| normalize_per_image | HWC | (1024,1024,1) | float32 | 2.1653 | 1.3022 | 0.60x |
| normalize_per_image | HWC | (1024,1024,1) | uint8 | 0.4529 | 2.7720 | 6.12x |
| normalize_per_image | HWC | (1024,1024,3) | float32 | 4.2680 | 2.9116 | 0.68x |
| normalize_per_image | HWC | (1024,1024,3) | uint8 | 0.8190 | 4.5505 | 5.56x |
| normalize_per_image | HWC | (1024,1024,9) | float32 | 22.3417 | 15.7543 | 0.71x |
| normalize_per_image | HWC | (1024,1024,9) | uint8 | 2.1633 | 22.1845 | 10.25x |
| normalize_per_image | HWC | (128,128,1) | float32 | 0.0342 | 0.0270 | 0.79x |
| normalize_per_image | HWC | (128,128,1) | uint8 | 0.0111 | 0.0388 | 3.50x |
| normalize_per_image | HWC | (128,128,3) | float32 | 0.0731 | 0.0555 | 0.76x |
| normalize_per_image | HWC | (128,128,3) | uint8 | 0.0207 | 0.0882 | 4.25x |
| normalize_per_image | HWC | (128,128,9) | float32 | 0.2822 | 0.1473 | 0.52x |
| normalize_per_image | HWC | (128,128,9) | uint8 | 0.0539 | 0.3180 | 5.90x |
| normalize_per_image | HWC | (256,256,1) | float32 | 0.1296 | 0.0686 | 0.53x |
| normalize_per_image | HWC | (256,256,1) | uint8 | 0.0318 | 0.1132 | 3.56x |
| normalize_per_image | HWC | (256,256,3) | float32 | 0.3185 | 0.1844 | 0.58x |
| normalize_per_image | HWC | (256,256,3) | uint8 | 0.0665 | 0.4463 | 6.71x |
| normalize_per_image | HWC | (256,256,9) | float32 | 1.1785 | 0.8075 | 0.69x |
| normalize_per_image | HWC | (256,256,9) | uint8 | 0.3801 | 1.4332 | 3.77x |
| normalize_per_image | HWC | (512,512,1) | float32 | 0.5635 | 0.2698 | 0.48x |
| normalize_per_image | HWC | (512,512,1) | uint8 | 0.0823 | 0.7832 | 9.52x |
| normalize_per_image | HWC | (512,512,3) | float32 | 1.7007 | 1.1206 | 0.66x |
| normalize_per_image | HWC | (512,512,3) | uint8 | 0.2788 | 2.0725 | 7.43x |
| normalize_per_image | HWC | (512,512,9) | float32 | 3.0856 | 2.3230 | 0.75x |
| normalize_per_image | HWC | (512,512,9) | uint8 | 0.4115 | 3.4847 | 8.47x |
| pairwise_distances_squared | points | (24,16,3) | float32 | 0.0037 | 0.0025 | 0.69x |
| power | HWC | (1024,1024,1) | float32 | 1.8860 | 1.9772 | 1.05x |
| power | HWC | (1024,1024,1) | uint8 | 0.0990 | 0.1007 | 1.02x |
| power | HWC | (1024,1024,3) | float32 | 4.1045 | 4.1680 | 1.02x |
| power | HWC | (1024,1024,3) | uint8 | 2.1824 | 2.0607 | 0.94x |
| power | HWC | (1024,1024,9) | float32 | 19.2710 | 19.5467 | 1.01x |
| power | HWC | (1024,1024,9) | uint8 | 6.4871 | 6.2412 | 0.96x |
| power | HWC | (128,128,1) | float32 | 0.0256 | 0.0262 | 1.02x |
| power | HWC | (128,128,1) | uint8 | 0.0067 | 0.0074 | 1.10x |
| power | HWC | (128,128,3) | float32 | 0.0664 | 0.0685 | 1.03x |
| power | HWC | (128,128,3) | uint8 | 0.0407 | 0.0403 | 0.99x |
| power | HWC | (128,128,9) | float32 | 0.1970 | 0.1938 | 0.98x |
| power | HWC | (128,128,9) | uint8 | 0.1133 | 0.1142 | 1.01x |
| power | HWC | (256,256,1) | float32 | 0.0947 | 0.0917 | 0.97x |
| power | HWC | (256,256,1) | uint8 | 0.0121 | 0.0112 | 0.93x |
| power | HWC | (256,256,3) | float32 | 0.2804 | 0.2561 | 0.91x |
| power | HWC | (256,256,3) | uint8 | 0.1237 | 0.1265 | 1.02x |
| power | HWC | (256,256,9) | float32 | 1.1133 | 1.1070 | 0.99x |
| power | HWC | (256,256,9) | uint8 | 0.3845 | 0.4290 | 1.12x |
| power | HWC | (512,512,1) | float32 | 0.3645 | 0.3483 | 0.96x |
| power | HWC | (512,512,1) | uint8 | 0.0268 | 0.0275 | 1.02x |
| power | HWC | (512,512,3) | float32 | 1.4557 | 1.5202 | 1.04x |
| power | HWC | (512,512,3) | uint8 | 0.4827 | 0.5139 | 1.06x |
| power | HWC | (512,512,9) | float32 | 3.1808 | 3.0994 | 0.97x |
| power | HWC | (512,512,9) | uint8 | 1.7260 | 1.6308 | 0.94x |
| sz_lut | HWC | (1024,1024,1) | uint8 | 0.0748 | 0.0779 | 1.04x |
| sz_lut | HWC | (1024,1024,3) | uint8 | 0.2347 | 0.2376 | 1.01x |
| sz_lut | HWC | (1024,1024,9) | uint8 | 0.6955 | 0.6895 | 0.99x |
| sz_lut | HWC | (128,128,1) | uint8 | 0.0017 | 0.0018 | 1.10x |
| sz_lut | HWC | (128,128,3) | uint8 | 0.0040 | 0.0042 | 1.05x |
| sz_lut | HWC | (128,128,9) | uint8 | 0.0112 | 0.0111 | 0.99x |
| sz_lut | HWC | (256,256,1) | uint8 | 0.0053 | 0.0051 | 0.96x |
| sz_lut | HWC | (256,256,3) | uint8 | 0.0147 | 0.0146 | 0.99x |
| sz_lut | HWC | (256,256,9) | uint8 | 0.0480 | 0.0425 | 0.89x |
| sz_lut | HWC | (512,512,1) | uint8 | 0.0195 | 0.0192 | 0.99x |
| sz_lut | HWC | (512,512,3) | uint8 | 0.0564 | 0.0573 | 1.02x |
| sz_lut | HWC | (512,512,9) | uint8 | 0.1677 | 0.1732 | 1.03x |
| to_float | HWC | (1024,1024,1) | uint8 | 0.2352 | 0.3285 | 1.40x |
| to_float | HWC | (1024,1024,3) | uint8 | 0.2101 | 0.1833 | 0.87x |
| to_float | HWC | (1024,1024,9) | uint8 | 2.1413 | 1.5747 | 0.74x |
| to_float | HWC | (128,128,1) | uint8 | 0.0072 | 0.0074 | 1.03x |
| to_float | HWC | (128,128,3) | uint8 | 0.0143 | 0.0147 | 1.02x |
| to_float | HWC | (128,128,9) | uint8 | 0.0389 | 0.0390 | 1.00x |
| to_float | HWC | (256,256,1) | uint8 | 0.0205 | 0.0192 | 0.94x |
| to_float | HWC | (256,256,3) | uint8 | 0.0510 | 0.0512 | 1.01x |
| to_float | HWC | (256,256,9) | uint8 | 0.2561 | 0.2728 | 1.07x |
| to_float | HWC | (512,512,1) | uint8 | 0.0388 | 0.0816 | 2.11x |
| to_float | HWC | (512,512,3) | uint8 | 0.2695 | 0.2089 | 0.78x |
| to_float | HWC | (512,512,9) | uint8 | 0.1736 | 0.1997 | 1.15x |
| vflip | HWC | (1024,1024,1) | float32 | 0.3716 | 0.3670 | 0.99x |
| vflip | HWC | (1024,1024,1) | uint8 | 0.0590 | 0.0521 | 0.88x |
| vflip | HWC | (1024,1024,3) | float32 | 0.6090 | 0.6225 | 1.02x |
| vflip | HWC | (1024,1024,3) | uint8 | 0.2787 | 0.2300 | 0.83x |
| vflip | HWC | (1024,1024,9) | float32 | 5.1955 | 4.0853 | 0.79x |
| vflip | HWC | (1024,1024,9) | uint8 | 0.4483 | 0.4694 | 1.05x |
| vflip | HWC | (128,128,1) | float32 | 0.0040 | 0.0043 | 1.08x |
| vflip | HWC | (128,128,1) | uint8 | 0.0018 | 0.0018 | 1.00x |
| vflip | HWC | (128,128,3) | float32 | 0.0097 | 0.0101 | 1.04x |
| vflip | HWC | (128,128,3) | uint8 | 0.0025 | 0.0013 | 0.54x |
| vflip | HWC | (128,128,9) | float32 | 0.0283 | 0.0282 | 1.00x |
| vflip | HWC | (128,128,9) | uint8 | 0.0080 | 0.0036 | 0.45x |
| vflip | HWC | (256,256,1) | float32 | 0.0138 | 0.0067 | 0.49x |
| vflip | HWC | (256,256,1) | uint8 | 0.0044 | 0.0045 | 1.04x |
| vflip | HWC | (256,256,3) | float32 | 0.0383 | 0.0378 | 0.99x |
| vflip | HWC | (256,256,3) | uint8 | 0.0097 | 0.0097 | 1.00x |
| vflip | HWC | (256,256,9) | float32 | 0.2218 | 0.2065 | 0.93x |
| vflip | HWC | (256,256,9) | uint8 | 0.0315 | 0.0280 | 0.89x |
| vflip | HWC | (512,512,1) | float32 | 0.0505 | 0.0510 | 1.01x |
| vflip | HWC | (512,512,1) | uint8 | 0.0138 | 0.0137 | 1.00x |
| vflip | HWC | (512,512,3) | float32 | 0.2083 | 0.2592 | 1.24x |
| vflip | HWC | (512,512,3) | uint8 | 0.0377 | 0.0373 | 0.99x |
| vflip | HWC | (512,512,9) | float32 | 0.4496 | 0.4501 | 1.00x |
| vflip | HWC | (512,512,9) | uint8 | 0.1852 | 0.1686 | 0.91x |
