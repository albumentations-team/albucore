# Router synthetic benchmark — comparison

- **New:** `0.0.41` (router_synthetic_current.json)
- **Old:** `0.0.40` (router_synthetic_0.0.40.json)

## Summary

- Comparable **ok/ok** cells: **410**
- **New slower** (old/new &lt; 0.85): **70** cells
- **New faster** (old/new &gt; 1.15): **65** cells

Ratio **old_ms / new_ms**: **&gt;1** ⇒ new build faster on that cell.

### Median old/new by op

- `add`: median **old/new** = **0.98x** (24 cells)
- `add_array`: median **old/new** = **2.87x** (24 cells)
- `add_constant`: median **old/new** = **0.99x** (24 cells)
- `add_vector`: median **old/new** = **1.01x** (24 cells)
- `add_weighted`: median **old/new** = **0.96x** (24 cells)
- `from_float`: median **old/new** = **1.01x** (12 cells)
- `hflip`: median **old/new** = **0.98x** (24 cells)
- `matmul`: median **old/new** = **0.94x** (1 cells)
- `median_blur`: median **old/new** = **1.02x** (12 cells)
- `multiply`: median **old/new** = **1.00x** (24 cells)
- `multiply_add`: median **old/new** = **1.04x** (24 cells)
- `multiply_by_array`: median **old/new** = **1.02x** (24 cells)
- `multiply_by_constant`: median **old/new** = **1.00x** (24 cells)
- `multiply_by_vector`: median **old/new** = **1.00x** (24 cells)
- `normalize`: median **old/new** = **1.00x** (24 cells)
- `normalize_per_image`: median **old/new** = **2.63x** (24 cells)
- `pairwise_distances_squared`: median **old/new** = **0.69x** (1 cells)
- `power`: median **old/new** = **0.98x** (24 cells)
- `sz_lut`: median **old/new** = **1.00x** (12 cells)
- `to_float`: median **old/new** = **1.01x** (12 cells)
- `vflip`: median **old/new** = **0.96x** (24 cells)

### Largest regressions (new slower)

| op | layout | shape | dtype | old/new |
|----|--------|-------|-------|--------:|
| hflip | HWC | (512,512,1) | uint8 | 0.13x |
| add | HWC | (512,512,1) | float32 | 0.30x |
| multiply | HWC | (512,512,1) | float32 | 0.36x |
| add_vector | HWC | (1024,1024,1) | uint8 | 0.43x |
| normalize | HWC | (256,256,3) | float32 | 0.44x |
| vflip | HWC | (256,256,1) | float32 | 0.48x |
| vflip | HWC | (128,128,9) | uint8 | 0.49x |
| vflip | HWC | (1024,1024,9) | uint8 | 0.51x |
| add_vector | HWC | (512,512,1) | float32 | 0.53x |
| multiply_add | HWC | (512,512,1) | float32 | 0.54x |
| add_constant | HWC | (256,256,9) | float32 | 0.55x |
| from_float | HWC | (512,512,9) | float32 | 0.56x |
| add_array | HWC | (512,512,1) | float32 | 0.58x |
| multiply | HWC | (1024,1024,1) | float32 | 0.58x |
| multiply_by_array | HWC | (512,512,1) | uint8 | 0.58x |
| multiply_add | HWC | (256,256,3) | float32 | 0.60x |
| normalize | HWC | (512,512,1) | uint8 | 0.62x |
| add_weighted | HWC | (256,256,9) | float32 | 0.63x |
| multiply_by_constant | HWC | (512,512,9) | uint8 | 0.63x |
| vflip | HWC | (256,256,9) | float32 | 0.63x |
| median_blur | HWC | (1024,1024,9) | uint8 | 0.64x |
| vflip | HWC | (512,512,3) | float32 | 0.66x |
| vflip | HWC | (512,512,9) | uint8 | 0.66x |
| multiply_by_constant | HWC | (1024,1024,9) | uint8 | 0.66x |
| add_vector | HWC | (128,128,1) | uint8 | 0.68x |

### Largest wins (new faster)

| op | layout | shape | dtype | old/new |
|----|--------|-------|-------|--------:|
| normalize_per_image | HWC | (1024,1024,9) | uint8 | 9.11x |
| add_array | HWC | (128,128,1) | uint8 | 6.98x |
| normalize_per_image | HWC | (256,256,3) | uint8 | 6.87x |
| normalize_per_image | HWC | (512,512,9) | uint8 | 6.59x |
| normalize_per_image | HWC | (1024,1024,3) | uint8 | 6.31x |
| normalize_per_image | HWC | (128,128,9) | uint8 | 5.56x |
| add_array | HWC | (1024,1024,1) | uint8 | 5.38x |
| add_array | HWC | (512,512,1) | uint8 | 5.12x |
| normalize_per_image | HWC | (512,512,3) | uint8 | 5.11x |
| add_array | HWC | (256,256,1) | uint8 | 4.85x |
| normalize_per_image | HWC | (1024,1024,1) | uint8 | 4.77x |
| normalize_per_image | HWC | (256,256,1) | uint8 | 4.27x |
| normalize_per_image | HWC | (256,256,9) | uint8 | 4.22x |
| normalize_per_image | HWC | (128,128,3) | uint8 | 4.15x |
| add_array | HWC | (128,128,3) | uint8 | 4.09x |
| normalize_per_image | HWC | (512,512,1) | uint8 | 3.81x |
| add_array | HWC | (1024,1024,9) | uint8 | 3.66x |
| add_array | HWC | (1024,1024,3) | uint8 | 3.63x |
| add_array | HWC | (256,256,3) | uint8 | 3.38x |
| add_array | HWC | (512,512,3) | uint8 | 3.28x |
| add_array | HWC | (128,128,9) | uint8 | 3.02x |
| add_array | HWC | (256,256,9) | uint8 | 2.87x |
| add_array | HWC | (512,512,9) | uint8 | 2.87x |
| normalize_per_image | HWC | (128,128,1) | uint8 | 2.63x |
| to_float | HWC | (512,512,1) | uint8 | 2.13x |

## Full table

Only rows where **both** runs are `ok`.

| op | layout | shape | dtype | new_ms | old_ms | old/new |
|----|--------|-------|-------|-------:|-------:|--------:|
| add | HWC | (1024,1024,1) | float32 | 0.9961 | 0.9470 | 0.95x |
| add | HWC | (1024,1024,1) | uint8 | 0.0440 | 0.0430 | 0.98x |
| add | HWC | (1024,1024,3) | float32 | 1.2128 | 1.2101 | 1.00x |
| add | HWC | (1024,1024,3) | uint8 | 0.2520 | 0.2472 | 0.98x |
| add | HWC | (1024,1024,9) | float32 | 6.5875 | 13.7321 | 2.08x |
| add | HWC | (1024,1024,9) | uint8 | 0.3468 | 0.3395 | 0.98x |
| add | HWC | (128,128,1) | float32 | 0.0117 | 0.0118 | 1.01x |
| add | HWC | (128,128,1) | uint8 | 0.0050 | 0.0048 | 0.95x |
| add | HWC | (128,128,3) | float32 | 0.0219 | 0.0224 | 1.02x |
| add | HWC | (128,128,3) | uint8 | 0.0047 | 0.0035 | 0.75x |
| add | HWC | (128,128,9) | float32 | 0.0625 | 0.0642 | 1.03x |
| add | HWC | (128,128,9) | uint8 | 0.0076 | 0.0078 | 1.02x |
| add | HWC | (256,256,1) | float32 | 0.0364 | 0.0300 | 0.83x |
| add | HWC | (256,256,1) | uint8 | 0.0067 | 0.0066 | 0.98x |
| add | HWC | (256,256,3) | float32 | 0.0775 | 0.0774 | 1.00x |
| add | HWC | (256,256,3) | uint8 | 0.0100 | 0.0102 | 1.02x |
| add | HWC | (256,256,9) | float32 | 0.6402 | 0.5542 | 0.87x |
| add | HWC | (256,256,9) | uint8 | 0.0262 | 0.0222 | 0.85x |
| add | HWC | (512,512,1) | float32 | 0.4128 | 0.1233 | 0.30x |
| add | HWC | (512,512,1) | uint8 | 0.0149 | 0.0150 | 1.01x |
| add | HWC | (512,512,3) | float32 | 0.6915 | 0.6294 | 0.91x |
| add | HWC | (512,512,3) | uint8 | 0.0315 | 0.0307 | 0.97x |
| add | HWC | (512,512,9) | float32 | 0.9565 | 0.9574 | 1.00x |
| add | HWC | (512,512,9) | uint8 | 0.3520 | 0.2868 | 0.81x |
| add_array | HWC | (1024,1024,1) | float32 | 1.1480 | 0.9440 | 0.82x |
| add_array | HWC | (1024,1024,1) | uint8 | 0.0218 | 0.1172 | 5.38x |
| add_array | HWC | (1024,1024,3) | float32 | 1.1762 | 1.2227 | 1.04x |
| add_array | HWC | (1024,1024,3) | uint8 | 0.2470 | 0.8970 | 3.63x |
| add_array | HWC | (1024,1024,9) | float32 | 6.6788 | 9.0917 | 1.36x |
| add_array | HWC | (1024,1024,9) | uint8 | 0.2086 | 0.7644 | 3.66x |
| add_array | HWC | (128,128,1) | float32 | 0.0103 | 0.0120 | 1.17x |
| add_array | HWC | (128,128,1) | uint8 | 0.0017 | 0.0119 | 6.98x |
| add_array | HWC | (128,128,3) | float32 | 0.0208 | 0.0227 | 1.09x |
| add_array | HWC | (128,128,3) | uint8 | 0.0032 | 0.0130 | 4.09x |
| add_array | HWC | (128,128,9) | float32 | 0.0563 | 0.0581 | 1.03x |
| add_array | HWC | (128,128,9) | uint8 | 0.0060 | 0.0181 | 3.02x |
| add_array | HWC | (256,256,1) | float32 | 0.0276 | 0.0294 | 1.06x |
| add_array | HWC | (256,256,1) | uint8 | 0.0028 | 0.0133 | 4.85x |
| add_array | HWC | (256,256,3) | float32 | 0.0773 | 0.0769 | 0.99x |
| add_array | HWC | (256,256,3) | uint8 | 0.0065 | 0.0218 | 3.38x |
| add_array | HWC | (256,256,9) | float32 | 0.6990 | 0.5773 | 0.83x |
| add_array | HWC | (256,256,9) | uint8 | 0.0169 | 0.0485 | 2.87x |
| add_array | HWC | (512,512,1) | float32 | 0.1987 | 0.1144 | 0.58x |
| add_array | HWC | (512,512,1) | uint8 | 0.0083 | 0.0426 | 5.12x |
| add_array | HWC | (512,512,3) | float32 | 0.7163 | 0.8647 | 1.21x |
| add_array | HWC | (512,512,3) | uint8 | 0.0219 | 0.0717 | 3.28x |
| add_array | HWC | (512,512,9) | float32 | 0.8683 | 0.8310 | 0.96x |
| add_array | HWC | (512,512,9) | uint8 | 0.1879 | 0.5391 | 2.87x |
| add_constant | HWC | (1024,1024,1) | float32 | 1.2551 | 1.1112 | 0.89x |
| add_constant | HWC | (1024,1024,1) | uint8 | 0.0440 | 0.0406 | 0.92x |
| add_constant | HWC | (1024,1024,3) | float32 | 1.2734 | 1.2395 | 0.97x |
| add_constant | HWC | (1024,1024,3) | uint8 | 0.2378 | 0.2684 | 1.13x |
| add_constant | HWC | (1024,1024,9) | float32 | 13.2763 | 13.9795 | 1.05x |
| add_constant | HWC | (1024,1024,9) | uint8 | 0.3358 | 0.3364 | 1.00x |
| add_constant | HWC | (128,128,1) | float32 | 0.0113 | 0.0114 | 1.00x |
| add_constant | HWC | (128,128,1) | uint8 | 0.0053 | 0.0039 | 0.74x |
| add_constant | HWC | (128,128,3) | float32 | 0.0219 | 0.0223 | 1.02x |
| add_constant | HWC | (128,128,3) | uint8 | 0.0042 | 0.0038 | 0.92x |
| add_constant | HWC | (128,128,9) | float32 | 0.0610 | 0.0629 | 1.03x |
| add_constant | HWC | (128,128,9) | uint8 | 0.0088 | 0.0074 | 0.84x |
| add_constant | HWC | (256,256,1) | float32 | 0.0378 | 0.0295 | 0.78x |
| add_constant | HWC | (256,256,1) | uint8 | 0.0060 | 0.0062 | 1.03x |
| add_constant | HWC | (256,256,3) | float32 | 0.0827 | 0.0839 | 1.01x |
| add_constant | HWC | (256,256,3) | uint8 | 0.0096 | 0.0097 | 1.01x |
| add_constant | HWC | (256,256,9) | float32 | 1.1379 | 0.6250 | 0.55x |
| add_constant | HWC | (256,256,9) | uint8 | 0.0241 | 0.0219 | 0.91x |
| add_constant | HWC | (512,512,1) | float32 | 0.1360 | 0.1220 | 0.90x |
| add_constant | HWC | (512,512,1) | uint8 | 0.0141 | 0.0145 | 1.03x |
| add_constant | HWC | (512,512,3) | float32 | 0.7129 | 0.6267 | 0.88x |
| add_constant | HWC | (512,512,3) | uint8 | 0.0297 | 0.0317 | 1.07x |
| add_constant | HWC | (512,512,9) | float32 | 0.9497 | 0.9126 | 0.96x |
| add_constant | HWC | (512,512,9) | uint8 | 0.3690 | 0.3647 | 0.99x |
| add_vector | HWC | (1024,1024,1) | float32 | 1.1105 | 1.4719 | 1.33x |
| add_vector | HWC | (1024,1024,1) | uint8 | 0.2442 | 0.1056 | 0.43x |
| add_vector | HWC | (1024,1024,3) | float32 | 4.4808 | 4.5265 | 1.01x |
| add_vector | HWC | (1024,1024,3) | uint8 | 2.3702 | 2.6823 | 1.13x |
| add_vector | HWC | (1024,1024,9) | float32 | 15.9253 | 16.2506 | 1.02x |
| add_vector | HWC | (1024,1024,9) | uint8 | 6.3810 | 6.6087 | 1.04x |
| add_vector | HWC | (128,128,1) | float32 | 0.0135 | 0.0135 | 0.99x |
| add_vector | HWC | (128,128,1) | uint8 | 0.0102 | 0.0069 | 0.68x |
| add_vector | HWC | (128,128,3) | float32 | 0.0747 | 0.0724 | 0.97x |
| add_vector | HWC | (128,128,3) | uint8 | 0.0397 | 0.0394 | 0.99x |
| add_vector | HWC | (128,128,9) | float32 | 0.1219 | 0.1227 | 1.01x |
| add_vector | HWC | (128,128,9) | uint8 | 0.1072 | 0.1097 | 1.02x |
| add_vector | HWC | (256,256,1) | float32 | 0.0321 | 0.0317 | 0.99x |
| add_vector | HWC | (256,256,1) | uint8 | 0.0122 | 0.0124 | 1.01x |
| add_vector | HWC | (256,256,3) | float32 | 0.2719 | 0.2806 | 1.03x |
| add_vector | HWC | (256,256,3) | uint8 | 0.1252 | 0.1281 | 1.02x |
| add_vector | HWC | (256,256,9) | float32 | 1.2414 | 0.8684 | 0.70x |
| add_vector | HWC | (256,256,9) | uint8 | 0.3765 | 0.3659 | 0.97x |
| add_vector | HWC | (512,512,1) | float32 | 0.2073 | 0.1101 | 0.53x |
| add_vector | HWC | (512,512,1) | uint8 | 0.0310 | 0.0653 | 2.11x |
| add_vector | HWC | (512,512,3) | float32 | 1.8978 | 1.5906 | 0.84x |
| add_vector | HWC | (512,512,3) | uint8 | 0.4953 | 0.5335 | 1.08x |
| add_vector | HWC | (512,512,9) | float32 | 1.9405 | 1.9377 | 1.00x |
| add_vector | HWC | (512,512,9) | uint8 | 2.4379 | 2.1217 | 0.87x |
| add_weighted | HWC | (1024,1024,1) | float32 | 0.9825 | 0.7606 | 0.77x |
| add_weighted | HWC | (1024,1024,1) | uint8 | 0.0696 | 0.0667 | 0.96x |
| add_weighted | HWC | (1024,1024,3) | float32 | 1.4530 | 1.2246 | 0.84x |
| add_weighted | HWC | (1024,1024,3) | uint8 | 0.3549 | 0.3587 | 1.01x |
| add_weighted | HWC | (1024,1024,9) | float32 | 7.2816 | 10.0016 | 1.37x |
| add_weighted | HWC | (1024,1024,9) | uint8 | 0.6071 | 0.6089 | 1.00x |
| add_weighted | HWC | (128,128,1) | float32 | 0.0113 | 0.0093 | 0.83x |
| add_weighted | HWC | (128,128,1) | uint8 | 0.0023 | 0.0023 | 1.02x |
| add_weighted | HWC | (128,128,3) | float32 | 0.0240 | 0.0206 | 0.86x |
| add_weighted | HWC | (128,128,3) | uint8 | 0.0045 | 0.0046 | 1.01x |
| add_weighted | HWC | (128,128,9) | float32 | 0.0659 | 0.0558 | 0.85x |
| add_weighted | HWC | (128,128,9) | uint8 | 0.0106 | 0.0108 | 1.02x |
| add_weighted | HWC | (256,256,1) | float32 | 0.0328 | 0.0264 | 0.81x |
| add_weighted | HWC | (256,256,1) | uint8 | 0.0056 | 0.0056 | 1.01x |
| add_weighted | HWC | (256,256,3) | float32 | 0.0868 | 0.0852 | 0.98x |
| add_weighted | HWC | (256,256,3) | uint8 | 0.0137 | 0.0140 | 1.03x |
| add_weighted | HWC | (256,256,9) | float32 | 0.7188 | 0.4505 | 0.63x |
| add_weighted | HWC | (256,256,9) | uint8 | 0.0398 | 0.0381 | 0.96x |
| add_weighted | HWC | (512,512,1) | float32 | 0.2048 | 0.1422 | 0.69x |
| add_weighted | HWC | (512,512,1) | uint8 | 0.0182 | 0.0185 | 1.02x |
| add_weighted | HWC | (512,512,3) | float32 | 0.6702 | 0.6019 | 0.90x |
| add_weighted | HWC | (512,512,3) | uint8 | 0.0649 | 0.0506 | 0.78x |
| add_weighted | HWC | (512,512,9) | float32 | 1.0657 | 0.8860 | 0.83x |
| add_weighted | HWC | (512,512,9) | uint8 | 0.3376 | 0.2705 | 0.80x |
| from_float | HWC | (1024,1024,1) | float32 | 1.5166 | 1.4633 | 0.96x |
| from_float | HWC | (1024,1024,3) | float32 | 2.4526 | 2.5279 | 1.03x |
| from_float | HWC | (1024,1024,9) | float32 | 14.1888 | 19.2899 | 1.36x |
| from_float | HWC | (128,128,1) | float32 | 0.0190 | 0.0151 | 0.80x |
| from_float | HWC | (128,128,3) | float32 | 0.0360 | 0.0363 | 1.01x |
| from_float | HWC | (128,128,9) | float32 | 0.1117 | 0.1119 | 1.00x |
| from_float | HWC | (256,256,1) | float32 | 0.0542 | 0.0494 | 0.91x |
| from_float | HWC | (256,256,3) | float32 | 0.2020 | 0.2230 | 1.10x |
| from_float | HWC | (256,256,9) | float32 | 1.2244 | 1.5162 | 1.24x |
| from_float | HWC | (512,512,1) | float32 | 0.3875 | 0.3036 | 0.78x |
| from_float | HWC | (512,512,3) | float32 | 1.1523 | 1.4110 | 1.22x |
| from_float | HWC | (512,512,9) | float32 | 3.4664 | 1.9354 | 0.56x |
| hflip | HWC | (1024,1024,1) | float32 | 0.3047 | 0.2697 | 0.89x |
| hflip | HWC | (1024,1024,1) | uint8 | 0.0238 | 0.0222 | 0.93x |
| hflip | HWC | (1024,1024,3) | float32 | 0.4258 | 0.4093 | 0.96x |
| hflip | HWC | (1024,1024,3) | uint8 | 0.4905 | 0.4296 | 0.88x |
| hflip | HWC | (1024,1024,9) | float32 | 10.0534 | 10.4154 | 1.04x |
| hflip | HWC | (1024,1024,9) | uint8 | 2.2778 | 1.6813 | 0.74x |
| hflip | HWC | (128,128,1) | float32 | 0.0027 | 0.0028 | 1.01x |
| hflip | HWC | (128,128,1) | uint8 | 0.0021 | 0.0017 | 0.78x |
| hflip | HWC | (128,128,3) | float32 | 0.0065 | 0.0068 | 1.05x |
| hflip | HWC | (128,128,3) | uint8 | 0.0052 | 0.0050 | 0.96x |
| hflip | HWC | (128,128,9) | float32 | 0.0985 | 0.0985 | 1.00x |
| hflip | HWC | (128,128,9) | uint8 | 0.0276 | 0.0267 | 0.97x |
| hflip | HWC | (256,256,1) | float32 | 0.0085 | 0.0067 | 0.79x |
| hflip | HWC | (256,256,1) | uint8 | 0.0028 | 0.0027 | 0.99x |
| hflip | HWC | (256,256,3) | float32 | 0.0264 | 0.0261 | 0.99x |
| hflip | HWC | (256,256,3) | uint8 | 0.0201 | 0.0198 | 0.98x |
| hflip | HWC | (256,256,9) | float32 | 0.5128 | 0.5404 | 1.05x |
| hflip | HWC | (256,256,9) | uint8 | 0.1018 | 0.1028 | 1.01x |
| hflip | HWC | (512,512,1) | float32 | 0.0217 | 0.0216 | 1.00x |
| hflip | HWC | (512,512,1) | uint8 | 0.0516 | 0.0067 | 0.13x |
| hflip | HWC | (512,512,3) | float32 | 0.2318 | 0.2593 | 1.12x |
| hflip | HWC | (512,512,3) | uint8 | 0.0788 | 0.0888 | 1.13x |
| hflip | HWC | (512,512,9) | float32 | 1.7376 | 1.6753 | 0.96x |
| hflip | HWC | (512,512,9) | uint8 | 0.5452 | 0.4951 | 0.91x |
| matmul | 2D | (128,64,64,32) | float32 | 0.0021 | 0.0020 | 0.94x |
| median_blur | HWC | (1024,1024,1) | uint8 | 0.1048 | 0.1504 | 1.43x |
| median_blur | HWC | (1024,1024,3) | uint8 | 0.5218 | 0.5744 | 1.10x |
| median_blur | HWC | (1024,1024,9) | uint8 | 1.9975 | 1.2700 | 0.64x |
| median_blur | HWC | (128,128,1) | uint8 | 0.0069 | 0.0083 | 1.19x |
| median_blur | HWC | (128,128,3) | uint8 | 0.0125 | 0.0127 | 1.02x |
| median_blur | HWC | (128,128,9) | uint8 | 0.0309 | 0.0316 | 1.02x |
| median_blur | HWC | (256,256,1) | uint8 | 0.0156 | 0.0156 | 1.00x |
| median_blur | HWC | (256,256,3) | uint8 | 0.0350 | 0.0376 | 1.07x |
| median_blur | HWC | (256,256,9) | uint8 | 0.0970 | 0.0969 | 1.00x |
| median_blur | HWC | (512,512,1) | uint8 | 0.0455 | 0.0536 | 1.18x |
| median_blur | HWC | (512,512,3) | uint8 | 0.1310 | 0.1189 | 0.91x |
| median_blur | HWC | (512,512,9) | uint8 | 0.5281 | 0.4745 | 0.90x |
| multiply | HWC | (1024,1024,1) | float32 | 1.3633 | 0.7870 | 0.58x |
| multiply | HWC | (1024,1024,1) | uint8 | 0.0972 | 0.0914 | 0.94x |
| multiply | HWC | (1024,1024,3) | float32 | 1.0230 | 1.0830 | 1.06x |
| multiply | HWC | (1024,1024,3) | uint8 | 0.4126 | 0.4190 | 1.02x |
| multiply | HWC | (1024,1024,9) | float32 | 6.8408 | 9.8653 | 1.44x |
| multiply | HWC | (1024,1024,9) | uint8 | 0.8573 | 0.8460 | 0.99x |
| multiply | HWC | (128,128,1) | float32 | 0.0085 | 0.0085 | 1.00x |
| multiply | HWC | (128,128,1) | uint8 | 0.0057 | 0.0059 | 1.04x |
| multiply | HWC | (128,128,3) | float32 | 0.0191 | 0.0188 | 0.98x |
| multiply | HWC | (128,128,3) | uint8 | 0.0093 | 0.0095 | 1.02x |
| multiply | HWC | (128,128,9) | float32 | 0.0501 | 0.0535 | 1.07x |
| multiply | HWC | (128,128,9) | uint8 | 0.0172 | 0.0200 | 1.16x |
| multiply | HWC | (256,256,1) | float32 | 0.0345 | 0.0240 | 0.69x |
| multiply | HWC | (256,256,1) | uint8 | 0.0108 | 0.0109 | 1.00x |
| multiply | HWC | (256,256,3) | float32 | 0.0690 | 0.0732 | 1.06x |
| multiply | HWC | (256,256,3) | uint8 | 0.0212 | 0.0217 | 1.02x |
| multiply | HWC | (256,256,9) | float32 | 0.4332 | 0.4333 | 1.00x |
| multiply | HWC | (256,256,9) | uint8 | 0.0616 | 0.0582 | 0.94x |
| multiply | HWC | (512,512,1) | float32 | 0.2456 | 0.0888 | 0.36x |
| multiply | HWC | (512,512,1) | uint8 | 0.0269 | 0.0323 | 1.20x |
| multiply | HWC | (512,512,3) | float32 | 0.7697 | 0.6123 | 0.80x |
| multiply | HWC | (512,512,3) | uint8 | 0.0743 | 0.0735 | 0.99x |
| multiply | HWC | (512,512,9) | float32 | 0.7730 | 0.7967 | 1.03x |
| multiply | HWC | (512,512,9) | uint8 | 0.3974 | 0.3464 | 0.87x |
| multiply_add | HWC | (1024,1024,1) | float32 | 1.2025 | 1.3945 | 1.16x |
| multiply_add | HWC | (1024,1024,1) | uint8 | 0.1021 | 0.0918 | 0.90x |
| multiply_add | HWC | (1024,1024,3) | float32 | 1.2218 | 1.5453 | 1.26x |
| multiply_add | HWC | (1024,1024,3) | uint8 | 0.4235 | 0.4405 | 1.04x |
| multiply_add | HWC | (1024,1024,9) | float32 | 12.3075 | 17.0743 | 1.39x |
| multiply_add | HWC | (1024,1024,9) | uint8 | 0.8641 | 0.8139 | 0.94x |
| multiply_add | HWC | (128,128,1) | float32 | 0.0109 | 0.0152 | 1.40x |
| multiply_add | HWC | (128,128,1) | uint8 | 0.0054 | 0.0056 | 1.04x |
| multiply_add | HWC | (128,128,3) | float32 | 0.0226 | 0.0281 | 1.24x |
| multiply_add | HWC | (128,128,3) | uint8 | 0.0094 | 0.0092 | 0.98x |
| multiply_add | HWC | (128,128,9) | float32 | 0.0667 | 0.0787 | 1.18x |
| multiply_add | HWC | (128,128,9) | uint8 | 0.0172 | 0.0176 | 1.02x |
| multiply_add | HWC | (256,256,1) | float32 | 0.0444 | 0.0365 | 0.82x |
| multiply_add | HWC | (256,256,1) | uint8 | 0.0108 | 0.0110 | 1.02x |
| multiply_add | HWC | (256,256,3) | float32 | 0.1882 | 0.1128 | 0.60x |
| multiply_add | HWC | (256,256,3) | uint8 | 0.0213 | 0.0227 | 1.07x |
| multiply_add | HWC | (256,256,9) | float32 | 0.8823 | 0.9995 | 1.13x |
| multiply_add | HWC | (256,256,9) | uint8 | 0.0598 | 0.0546 | 0.91x |
| multiply_add | HWC | (512,512,1) | float32 | 0.2680 | 0.1441 | 0.54x |
| multiply_add | HWC | (512,512,1) | uint8 | 0.0270 | 0.0281 | 1.04x |
| multiply_add | HWC | (512,512,3) | float32 | 0.9902 | 1.0192 | 1.03x |
| multiply_add | HWC | (512,512,3) | uint8 | 0.0718 | 0.0708 | 0.99x |
| multiply_add | HWC | (512,512,9) | float32 | 0.9682 | 1.1805 | 1.22x |
| multiply_add | HWC | (512,512,9) | uint8 | 0.3200 | 0.3185 | 1.00x |
| multiply_by_array | HWC | (1024,1024,1) | float32 | 0.8415 | 0.9273 | 1.10x |
| multiply_by_array | HWC | (1024,1024,1) | uint8 | 2.2695 | 2.0614 | 0.91x |
| multiply_by_array | HWC | (1024,1024,3) | float32 | 1.1520 | 1.1510 | 1.00x |
| multiply_by_array | HWC | (1024,1024,3) | uint8 | 3.4063 | 3.2310 | 0.95x |
| multiply_by_array | HWC | (1024,1024,9) | float32 | 6.8496 | 9.2744 | 1.35x |
| multiply_by_array | HWC | (1024,1024,9) | uint8 | 11.9582 | 18.5525 | 1.55x |
| multiply_by_array | HWC | (128,128,1) | float32 | 0.0077 | 0.0117 | 1.53x |
| multiply_by_array | HWC | (128,128,1) | uint8 | 0.0217 | 0.0214 | 0.98x |
| multiply_by_array | HWC | (128,128,3) | float32 | 0.0199 | 0.0227 | 1.14x |
| multiply_by_array | HWC | (128,128,3) | uint8 | 0.0502 | 0.0509 | 1.01x |
| multiply_by_array | HWC | (128,128,9) | float32 | 0.0563 | 0.0593 | 1.05x |
| multiply_by_array | HWC | (128,128,9) | uint8 | 0.1347 | 0.1474 | 1.09x |
| multiply_by_array | HWC | (256,256,1) | float32 | 0.0283 | 0.0297 | 1.05x |
| multiply_by_array | HWC | (256,256,1) | uint8 | 0.0761 | 0.0673 | 0.88x |
| multiply_by_array | HWC | (256,256,3) | float32 | 0.0722 | 0.1023 | 1.42x |
| multiply_by_array | HWC | (256,256,3) | uint8 | 0.2754 | 0.2588 | 0.94x |
| multiply_by_array | HWC | (256,256,9) | float32 | 0.4880 | 0.4858 | 1.00x |
| multiply_by_array | HWC | (256,256,9) | uint8 | 0.9493 | 0.9358 | 0.99x |
| multiply_by_array | HWC | (512,512,1) | float32 | 0.1104 | 0.1248 | 1.13x |
| multiply_by_array | HWC | (512,512,1) | uint8 | 0.6300 | 0.3649 | 0.58x |
| multiply_by_array | HWC | (512,512,3) | float32 | 0.6374 | 0.6805 | 1.07x |
| multiply_by_array | HWC | (512,512,3) | uint8 | 1.3202 | 1.2084 | 0.92x |
| multiply_by_array | HWC | (512,512,9) | float32 | 0.8855 | 0.8990 | 1.02x |
| multiply_by_array | HWC | (512,512,9) | uint8 | 2.7308 | 2.2587 | 0.83x |
| multiply_by_constant | HWC | (1024,1024,1) | float32 | 0.8655 | 0.7512 | 0.87x |
| multiply_by_constant | HWC | (1024,1024,1) | uint8 | 0.0975 | 0.0988 | 1.01x |
| multiply_by_constant | HWC | (1024,1024,3) | float32 | 1.0570 | 1.0513 | 0.99x |
| multiply_by_constant | HWC | (1024,1024,3) | uint8 | 0.4224 | 0.5328 | 1.26x |
| multiply_by_constant | HWC | (1024,1024,9) | float32 | 6.1395 | 9.5621 | 1.56x |
| multiply_by_constant | HWC | (1024,1024,9) | uint8 | 1.2192 | 0.8089 | 0.66x |
| multiply_by_constant | HWC | (128,128,1) | float32 | 0.0081 | 0.0081 | 1.00x |
| multiply_by_constant | HWC | (128,128,1) | uint8 | 0.0059 | 0.0055 | 0.92x |
| multiply_by_constant | HWC | (128,128,3) | float32 | 0.0185 | 0.0187 | 1.01x |
| multiply_by_constant | HWC | (128,128,3) | uint8 | 0.0090 | 0.0091 | 1.01x |
| multiply_by_constant | HWC | (128,128,9) | float32 | 0.0559 | 0.0565 | 1.01x |
| multiply_by_constant | HWC | (128,128,9) | uint8 | 0.0168 | 0.0173 | 1.03x |
| multiply_by_constant | HWC | (256,256,1) | float32 | 0.0231 | 0.0236 | 1.02x |
| multiply_by_constant | HWC | (256,256,1) | uint8 | 0.0104 | 0.0104 | 1.00x |
| multiply_by_constant | HWC | (256,256,3) | float32 | 0.0745 | 0.0634 | 0.85x |
| multiply_by_constant | HWC | (256,256,3) | uint8 | 0.0211 | 0.0212 | 1.00x |
| multiply_by_constant | HWC | (256,256,9) | float32 | 0.5606 | 0.4316 | 0.77x |
| multiply_by_constant | HWC | (256,256,9) | uint8 | 0.0534 | 0.0533 | 1.00x |
| multiply_by_constant | HWC | (512,512,1) | float32 | 0.1097 | 0.1048 | 0.95x |
| multiply_by_constant | HWC | (512,512,1) | uint8 | 0.0280 | 0.0265 | 0.95x |
| multiply_by_constant | HWC | (512,512,3) | float32 | 0.8223 | 0.6347 | 0.77x |
| multiply_by_constant | HWC | (512,512,3) | uint8 | 0.0727 | 0.0694 | 0.95x |
| multiply_by_constant | HWC | (512,512,9) | float32 | 0.7814 | 0.7940 | 1.02x |
| multiply_by_constant | HWC | (512,512,9) | uint8 | 0.6633 | 0.4162 | 0.63x |
| multiply_by_vector | HWC | (1024,1024,1) | float32 | 1.1321 | 1.1921 | 1.05x |
| multiply_by_vector | HWC | (1024,1024,1) | uint8 | 0.1587 | 0.1113 | 0.70x |
| multiply_by_vector | HWC | (1024,1024,3) | float32 | 4.4605 | 4.4704 | 1.00x |
| multiply_by_vector | HWC | (1024,1024,3) | uint8 | 2.5636 | 2.4830 | 0.97x |
| multiply_by_vector | HWC | (1024,1024,9) | float32 | 10.5975 | 14.0733 | 1.33x |
| multiply_by_vector | HWC | (1024,1024,9) | uint8 | 6.9825 | 6.1059 | 0.87x |
| multiply_by_vector | HWC | (128,128,1) | float32 | 0.0137 | 0.0131 | 0.95x |
| multiply_by_vector | HWC | (128,128,1) | uint8 | 0.0065 | 0.0062 | 0.96x |
| multiply_by_vector | HWC | (128,128,3) | float32 | 0.0726 | 0.0725 | 1.00x |
| multiply_by_vector | HWC | (128,128,3) | uint8 | 0.0390 | 0.0381 | 0.98x |
| multiply_by_vector | HWC | (128,128,9) | float32 | 0.1160 | 0.1169 | 1.01x |
| multiply_by_vector | HWC | (128,128,9) | uint8 | 0.1098 | 0.1087 | 0.99x |
| multiply_by_vector | HWC | (256,256,1) | float32 | 0.0332 | 0.0333 | 1.01x |
| multiply_by_vector | HWC | (256,256,1) | uint8 | 0.0118 | 0.0118 | 1.00x |
| multiply_by_vector | HWC | (256,256,3) | float32 | 0.2760 | 0.2753 | 1.00x |
| multiply_by_vector | HWC | (256,256,3) | uint8 | 0.1363 | 0.1260 | 0.92x |
| multiply_by_vector | HWC | (256,256,9) | float32 | 0.6971 | 0.7669 | 1.10x |
| multiply_by_vector | HWC | (256,256,9) | uint8 | 0.3639 | 0.4209 | 1.16x |
| multiply_by_vector | HWC | (512,512,1) | float32 | 0.1099 | 0.1354 | 1.23x |
| multiply_by_vector | HWC | (512,512,1) | uint8 | 0.0330 | 0.0420 | 1.27x |
| multiply_by_vector | HWC | (512,512,3) | float32 | 1.8694 | 1.8545 | 0.99x |
| multiply_by_vector | HWC | (512,512,3) | uint8 | 0.5433 | 0.5127 | 0.94x |
| multiply_by_vector | HWC | (512,512,9) | float32 | 1.8551 | 1.9005 | 1.02x |
| multiply_by_vector | HWC | (512,512,9) | uint8 | 2.1131 | 1.6445 | 0.78x |
| normalize | HWC | (1024,1024,1) | float32 | 0.9527 | 0.8569 | 0.90x |
| normalize | HWC | (1024,1024,1) | uint8 | 0.3298 | 0.2623 | 0.80x |
| normalize | HWC | (1024,1024,3) | float32 | 0.7491 | 0.7500 | 1.00x |
| normalize | HWC | (1024,1024,3) | uint8 | 0.2645 | 0.1868 | 0.71x |
| normalize | HWC | (1024,1024,9) | float32 | 6.8295 | 10.7897 | 1.58x |
| normalize | HWC | (1024,1024,9) | uint8 | 1.8280 | 1.3430 | 0.73x |
| normalize | HWC | (128,128,1) | float32 | 0.0064 | 0.0061 | 0.95x |
| normalize | HWC | (128,128,1) | uint8 | 0.0079 | 0.0079 | 1.00x |
| normalize | HWC | (128,128,3) | float32 | 0.0138 | 0.0138 | 1.00x |
| normalize | HWC | (128,128,3) | uint8 | 0.0152 | 0.0151 | 1.00x |
| normalize | HWC | (128,128,9) | float32 | 0.0351 | 0.0357 | 1.02x |
| normalize | HWC | (128,128,9) | uint8 | 0.0399 | 0.0403 | 1.01x |
| normalize | HWC | (256,256,1) | float32 | 0.0175 | 0.0179 | 1.03x |
| normalize | HWC | (256,256,1) | uint8 | 0.0201 | 0.0202 | 1.01x |
| normalize | HWC | (256,256,3) | float32 | 0.1340 | 0.0584 | 0.44x |
| normalize | HWC | (256,256,3) | uint8 | 0.0535 | 0.0525 | 0.98x |
| normalize | HWC | (256,256,9) | float32 | 0.5701 | 0.4975 | 0.87x |
| normalize | HWC | (256,256,9) | uint8 | 0.2632 | 0.3102 | 1.18x |
| normalize | HWC | (512,512,1) | float32 | 0.1465 | 0.1305 | 0.89x |
| normalize | HWC | (512,512,1) | uint8 | 0.0989 | 0.0617 | 0.62x |
| normalize | HWC | (512,512,3) | float32 | 0.7555 | 0.8844 | 1.17x |
| normalize | HWC | (512,512,3) | uint8 | 0.2168 | 0.2310 | 1.07x |
| normalize | HWC | (512,512,9) | float32 | 0.5893 | 0.5357 | 0.91x |
| normalize | HWC | (512,512,9) | uint8 | 0.1850 | 0.2095 | 1.13x |
| normalize_per_image | HWC | (1024,1024,1) | float32 | 1.3733 | 1.3022 | 0.95x |
| normalize_per_image | HWC | (1024,1024,1) | uint8 | 0.5817 | 2.7720 | 4.77x |
| normalize_per_image | HWC | (1024,1024,3) | float32 | 2.7241 | 2.9116 | 1.07x |
| normalize_per_image | HWC | (1024,1024,3) | uint8 | 0.7217 | 4.5505 | 6.31x |
| normalize_per_image | HWC | (1024,1024,9) | float32 | 14.1273 | 15.7543 | 1.12x |
| normalize_per_image | HWC | (1024,1024,9) | uint8 | 2.4350 | 22.1845 | 9.11x |
| normalize_per_image | HWC | (128,128,1) | float32 | 0.0299 | 0.0270 | 0.90x |
| normalize_per_image | HWC | (128,128,1) | uint8 | 0.0147 | 0.0388 | 2.63x |
| normalize_per_image | HWC | (128,128,3) | float32 | 0.0561 | 0.0555 | 0.99x |
| normalize_per_image | HWC | (128,128,3) | uint8 | 0.0212 | 0.0882 | 4.15x |
| normalize_per_image | HWC | (128,128,9) | float32 | 0.1422 | 0.1473 | 1.04x |
| normalize_per_image | HWC | (128,128,9) | uint8 | 0.0572 | 0.3180 | 5.56x |
| normalize_per_image | HWC | (256,256,1) | float32 | 0.0675 | 0.0686 | 1.02x |
| normalize_per_image | HWC | (256,256,1) | uint8 | 0.0265 | 0.1132 | 4.27x |
| normalize_per_image | HWC | (256,256,3) | float32 | 0.1862 | 0.1844 | 0.99x |
| normalize_per_image | HWC | (256,256,3) | uint8 | 0.0650 | 0.4463 | 6.87x |
| normalize_per_image | HWC | (256,256,9) | float32 | 1.0177 | 0.8075 | 0.79x |
| normalize_per_image | HWC | (256,256,9) | uint8 | 0.3397 | 1.4332 | 4.22x |
| normalize_per_image | HWC | (512,512,1) | float32 | 0.2384 | 0.2698 | 1.13x |
| normalize_per_image | HWC | (512,512,1) | uint8 | 0.2056 | 0.7832 | 3.81x |
| normalize_per_image | HWC | (512,512,3) | float32 | 1.0596 | 1.1206 | 1.06x |
| normalize_per_image | HWC | (512,512,3) | uint8 | 0.4053 | 2.0725 | 5.11x |
| normalize_per_image | HWC | (512,512,9) | float32 | 2.1862 | 2.3230 | 1.06x |
| normalize_per_image | HWC | (512,512,9) | uint8 | 0.5291 | 3.4847 | 6.59x |
| pairwise_distances_squared | points | (24,16,3) | float32 | 0.0037 | 0.0025 | 0.69x |
| power | HWC | (1024,1024,1) | float32 | 1.9576 | 1.9772 | 1.01x |
| power | HWC | (1024,1024,1) | uint8 | 0.1008 | 0.1007 | 1.00x |
| power | HWC | (1024,1024,3) | float32 | 4.2417 | 4.1680 | 0.98x |
| power | HWC | (1024,1024,3) | uint8 | 2.3972 | 2.0607 | 0.86x |
| power | HWC | (1024,1024,9) | float32 | 15.5757 | 19.5467 | 1.25x |
| power | HWC | (1024,1024,9) | uint8 | 7.2520 | 6.2412 | 0.86x |
| power | HWC | (128,128,1) | float32 | 0.0334 | 0.0262 | 0.78x |
| power | HWC | (128,128,1) | uint8 | 0.0061 | 0.0074 | 1.21x |
| power | HWC | (128,128,3) | float32 | 0.0672 | 0.0685 | 1.02x |
| power | HWC | (128,128,3) | uint8 | 0.0524 | 0.0403 | 0.77x |
| power | HWC | (128,128,9) | float32 | 0.1931 | 0.1938 | 1.00x |
| power | HWC | (128,128,9) | uint8 | 0.1123 | 0.1142 | 1.02x |
| power | HWC | (256,256,1) | float32 | 0.0986 | 0.0917 | 0.93x |
| power | HWC | (256,256,1) | uint8 | 0.0115 | 0.0112 | 0.98x |
| power | HWC | (256,256,3) | float32 | 0.2739 | 0.2561 | 0.93x |
| power | HWC | (256,256,3) | uint8 | 0.1410 | 0.1265 | 0.90x |
| power | HWC | (256,256,9) | float32 | 1.5716 | 1.1070 | 0.70x |
| power | HWC | (256,256,9) | uint8 | 0.3918 | 0.4290 | 1.09x |
| power | HWC | (512,512,1) | float32 | 0.3766 | 0.3483 | 0.92x |
| power | HWC | (512,512,1) | uint8 | 0.0289 | 0.0275 | 0.95x |
| power | HWC | (512,512,3) | float32 | 1.5092 | 1.5202 | 1.01x |
| power | HWC | (512,512,3) | uint8 | 0.5037 | 0.5139 | 1.02x |
| power | HWC | (512,512,9) | float32 | 3.1817 | 3.0994 | 0.97x |
| power | HWC | (512,512,9) | uint8 | 2.2730 | 1.6308 | 0.72x |
| sz_lut | HWC | (1024,1024,1) | uint8 | 0.0777 | 0.0779 | 1.00x |
| sz_lut | HWC | (1024,1024,3) | uint8 | 0.2367 | 0.2376 | 1.00x |
| sz_lut | HWC | (1024,1024,9) | uint8 | 0.6901 | 0.6895 | 1.00x |
| sz_lut | HWC | (128,128,1) | uint8 | 0.0016 | 0.0018 | 1.16x |
| sz_lut | HWC | (128,128,3) | uint8 | 0.0041 | 0.0042 | 1.02x |
| sz_lut | HWC | (128,128,9) | uint8 | 0.0100 | 0.0111 | 1.11x |
| sz_lut | HWC | (256,256,1) | uint8 | 0.0052 | 0.0051 | 0.98x |
| sz_lut | HWC | (256,256,3) | uint8 | 0.0146 | 0.0146 | 1.00x |
| sz_lut | HWC | (256,256,9) | uint8 | 0.0435 | 0.0425 | 0.98x |
| sz_lut | HWC | (512,512,1) | uint8 | 0.0199 | 0.0192 | 0.97x |
| sz_lut | HWC | (512,512,3) | uint8 | 0.0580 | 0.0573 | 0.99x |
| sz_lut | HWC | (512,512,9) | uint8 | 0.1850 | 0.1732 | 0.94x |
| to_float | HWC | (1024,1024,1) | uint8 | 0.2997 | 0.3285 | 1.10x |
| to_float | HWC | (1024,1024,3) | uint8 | 0.2523 | 0.1833 | 0.73x |
| to_float | HWC | (1024,1024,9) | uint8 | 1.3143 | 1.5747 | 1.20x |
| to_float | HWC | (128,128,1) | uint8 | 0.0067 | 0.0074 | 1.09x |
| to_float | HWC | (128,128,3) | uint8 | 0.0130 | 0.0147 | 1.13x |
| to_float | HWC | (128,128,9) | uint8 | 0.0389 | 0.0390 | 1.00x |
| to_float | HWC | (256,256,1) | uint8 | 0.0195 | 0.0192 | 0.99x |
| to_float | HWC | (256,256,3) | uint8 | 0.0512 | 0.0512 | 1.00x |
| to_float | HWC | (256,256,9) | uint8 | 0.3017 | 0.2728 | 0.90x |
| to_float | HWC | (512,512,1) | uint8 | 0.0383 | 0.0816 | 2.13x |
| to_float | HWC | (512,512,3) | uint8 | 0.2210 | 0.2089 | 0.95x |
| to_float | HWC | (512,512,9) | uint8 | 0.1973 | 0.1997 | 1.01x |
| vflip | HWC | (1024,1024,1) | float32 | 0.3824 | 0.3670 | 0.96x |
| vflip | HWC | (1024,1024,1) | uint8 | 0.0531 | 0.0521 | 0.98x |
| vflip | HWC | (1024,1024,3) | float32 | 0.5699 | 0.6225 | 1.09x |
| vflip | HWC | (1024,1024,3) | uint8 | 0.2744 | 0.2300 | 0.84x |
| vflip | HWC | (1024,1024,9) | float32 | 4.6672 | 4.0853 | 0.88x |
| vflip | HWC | (1024,1024,9) | uint8 | 0.9268 | 0.4694 | 0.51x |
| vflip | HWC | (128,128,1) | float32 | 0.0051 | 0.0043 | 0.85x |
| vflip | HWC | (128,128,1) | uint8 | 0.0019 | 0.0018 | 0.96x |
| vflip | HWC | (128,128,3) | float32 | 0.0097 | 0.0101 | 1.04x |
| vflip | HWC | (128,128,3) | uint8 | 0.0020 | 0.0013 | 0.68x |
| vflip | HWC | (128,128,9) | float32 | 0.0281 | 0.0282 | 1.00x |
| vflip | HWC | (128,128,9) | uint8 | 0.0074 | 0.0036 | 0.49x |
| vflip | HWC | (256,256,1) | float32 | 0.0140 | 0.0067 | 0.48x |
| vflip | HWC | (256,256,1) | uint8 | 0.0043 | 0.0045 | 1.05x |
| vflip | HWC | (256,256,3) | float32 | 0.0393 | 0.0378 | 0.96x |
| vflip | HWC | (256,256,3) | uint8 | 0.0099 | 0.0097 | 0.97x |
| vflip | HWC | (256,256,9) | float32 | 0.3285 | 0.2065 | 0.63x |
| vflip | HWC | (256,256,9) | uint8 | 0.0280 | 0.0280 | 1.00x |
| vflip | HWC | (512,512,1) | float32 | 0.0517 | 0.0510 | 0.99x |
| vflip | HWC | (512,512,1) | uint8 | 0.0147 | 0.0137 | 0.93x |
| vflip | HWC | (512,512,3) | float32 | 0.3915 | 0.2592 | 0.66x |
| vflip | HWC | (512,512,3) | uint8 | 0.0388 | 0.0373 | 0.96x |
| vflip | HWC | (512,512,9) | float32 | 0.4490 | 0.4501 | 1.00x |
| vflip | HWC | (512,512,9) | uint8 | 0.2542 | 0.1686 | 0.66x |
