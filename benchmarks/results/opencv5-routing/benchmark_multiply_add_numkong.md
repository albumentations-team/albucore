# multiply / add: albucore vs NumKong `scale` / `fma` / `blend` helpers

Median ms; **prod** = current `@clipped` public API. **NK scale** = `α·x+β` on ravel.

## `multiply_by_constant` — scalar

| dtype | H×W | C | prod | NK scale (alloc) | NK scale (inplace) | fastest |
|-------|-----|---|-----:|-----------------:|-------------------:|---------|
| uint8 | 240×320 | 1 | 0.0108 | 0.0044 | 0.0076 | NK_alloc |
| float32 | 240×320 | 1 | 0.0287 | 0.0066 | 0.0280 | NK_alloc |
| uint8 | 240×320 | 3 | 0.0249 | 0.0108 | 0.0171 | NK_alloc |
| float32 | 240×320 | 3 | 0.0742 | 0.0157 | 0.0762 | NK_alloc |
| uint8 | 240×320 | 9 | 0.0580 | 0.0296 | 0.0440 | NK_alloc |
| float32 | 240×320 | 9 | 0.2306 | 0.0502 | 0.2335 | NK_alloc |
| uint8 | 480×640 | 1 | 0.0280 | 0.0137 | 0.0213 | NK_alloc |
| float32 | 480×640 | 1 | 0.1030 | 0.0223 | 0.1012 | NK_alloc |
| uint8 | 480×640 | 3 | 0.0738 | 0.0386 | 0.0567 | NK_alloc |
| float32 | 480×640 | 3 | 0.2871 | 0.0604 | 0.2971 | NK_alloc |
| uint8 | 480×640 | 9 | 0.2215 | 0.1168 | 0.1624 | NK_alloc |
| float32 | 480×640 | 9 | 0.8880 | 0.1807 | 0.9032 | NK_alloc |
| uint8 | 768×1024 | 1 | 0.0619 | 0.0331 | 0.0482 | NK_alloc |
| float32 | 768×1024 | 1 | 0.2530 | 0.0515 | 0.2540 | NK_alloc |
| uint8 | 768×1024 | 3 | 0.1658 | 0.1018 | 0.1419 | NK_alloc |
| float32 | 768×1024 | 3 | 0.7437 | 0.1617 | 0.7780 | NK_alloc |
| uint8 | 768×1024 | 9 | 0.3209 | 0.2931 | 0.4054 | NK_alloc |
| float32 | 768×1024 | 9 | 2.3344 | 0.4940 | 2.3305 | NK_alloc |

## `multiply_add` — scalar affine `factor * img + value`

| dtype | H×W | C | prod | NumPy | NK scale (alloc) | OpenCV `addWeighted` | fastest |
|-------|-----|---|-----:|------:|-----------------:|---------------------:|---------|
| float32 | 240×320 | 1 | 0.0363 | 0.0354 | 0.0283 | 0.0288 | NK_alloc |
| float32 | 240×320 | 3 | 0.0958 | 0.0907 | 0.0772 | 0.0815 | NK_alloc |
| float32 | 240×320 | 9 | 0.2890 | 0.2584 | 0.2212 | 0.2311 | NK_alloc |
| float32 | 480×640 | 1 | 0.1262 | 0.1233 | 0.1015 | 0.1079 | NK_alloc |
| float32 | 480×640 | 3 | 0.3637 | 0.3590 | 0.2975 | 0.3087 | NK_alloc |
| float32 | 480×640 | 9 | 0.9199 | 1.0903 | 0.9354 | 0.9152 | OpenCV_addWeighted |
| float32 | 768×1024 | 1 | 0.3112 | 0.3132 | 0.2547 | 0.2642 | NK_alloc |
| float32 | 768×1024 | 3 | 0.7765 | 0.9085 | 0.7687 | 0.8170 | NK_alloc |
| float32 | 768×1024 | 9 | 2.3485 | 2.8848 | 2.3484 | 2.4119 | NK_alloc |

## `add_constant` — scalar

| dtype | H×W | C | prod | NK scale (alloc) | NK scale (inplace) | fastest |
|-------|-----|---|-----:|-----------------:|-------------------:|---------|
| uint8 | 240×320 | 1 | 0.0080 | 0.0086 | 0.0074 | NK_inplace |
| float32 | 240×320 | 1 | 0.0288 | 0.0343 | 0.0275 | NK_inplace |
| uint8 | 240×320 | 3 | 0.0115 | 0.0172 | 0.0167 | prod |
| float32 | 240×320 | 3 | 0.0774 | 0.0783 | 0.0756 | NK_inplace |
| uint8 | 240×320 | 9 | 0.0270 | 0.0443 | 0.0440 | prod |
| float32 | 240×320 | 9 | 0.2304 | 0.2303 | 0.2237 | NK_inplace |
| uint8 | 480×640 | 1 | 0.0167 | 0.0220 | 0.0216 | prod |
| float32 | 480×640 | 1 | 0.1044 | 0.1049 | 0.1028 | NK_inplace |
| uint8 | 480×640 | 3 | 0.0345 | 0.0571 | 0.0559 | prod |
| float32 | 480×640 | 3 | 0.2951 | 0.2945 | 0.2933 | NK_inplace |
| uint8 | 480×640 | 9 | 0.0962 | 0.1610 | 0.1603 | prod |
| float32 | 480×640 | 9 | 0.8883 | 0.9364 | 0.9241 | prod |
| uint8 | 768×1024 | 1 | 0.0334 | 0.0487 | 0.0482 | prod |
| float32 | 768×1024 | 1 | 0.2547 | 0.2568 | 0.2543 | NK_inplace |
| uint8 | 768×1024 | 3 | 0.0928 | 0.1396 | 0.1392 | prod |
| float32 | 768×1024 | 3 | 0.7580 | 0.7890 | 0.7760 | prod |
| uint8 | 768×1024 | 9 | 0.2093 | 0.4131 | 0.4193 | prod |
| float32 | 768×1024 | 9 | 2.3626 | 2.3240 | 2.3792 | NK_alloc |

## `multiply_by_array` — elementwise `img * value`

| dtype | H×W | C | prod (OpenCV) | NK `fma` (a·b+0·z) | NumPy (`multiply_numpy` path) | fastest |
|-------|-----|---|--------------:|-------------------:|------------------------------:|---------|
| uint8 | 240×320 | 1 | 0.0770 | N/A | 0.0286 | numpy |
| float32 | 240×320 | 1 | 0.0313 | 0.0503 | 0.0086 | numpy |
| uint8 | 240×320 | 3 | 0.2139 | N/A | 0.0826 | numpy |
| float32 | 240×320 | 3 | 0.0813 | 0.1425 | 0.0180 | numpy |
| uint8 | 240×320 | 9 | 0.6288 | N/A | 0.2337 | numpy |
| float32 | 240×320 | 9 | 0.2551 | 0.4225 | 0.0724 | numpy |
| uint8 | 480×640 | 1 | 0.2863 | N/A | 0.1068 | numpy |
| float32 | 480×640 | 1 | 0.1123 | 0.1815 | 0.0292 | numpy |
| uint8 | 480×640 | 3 | 0.8184 | N/A | 0.3345 | numpy |
| float32 | 480×640 | 3 | 0.3220 | 0.5648 | 0.0962 | numpy |
| uint8 | 480×640 | 9 | 2.4830 | N/A | 0.9710 | numpy |
| float32 | 480×640 | 9 | 1.0014 | 1.7405 | 0.2911 | numpy |
| uint8 | 768×1024 | 1 | 0.7142 | N/A | 0.2918 | numpy |
| float32 | 768×1024 | 1 | 0.2838 | 0.4843 | 0.0862 | numpy |
| uint8 | 768×1024 | 3 | 2.1942 | N/A | 0.8606 | numpy |
| float32 | 768×1024 | 3 | 0.8954 | 1.4576 | 0.2356 | numpy |
| uint8 | 768×1024 | 9 | 6.6145 | N/A | 2.6365 | numpy |
| float32 | 768×1024 | 9 | 2.5978 | 4.3615 | 0.7563 | numpy |

## `add_array` — elementwise `img + value`

| dtype | H×W | C | prod (OpenCV) | NK `add_array_numkong` (`blend`) | fastest |
|-------|-----|---|--------------:|---------------------------------:|---------|
| uint8 | 240×320 | 1 | 0.0039 | 0.0034 | NK_blend |
| float32 | 240×320 | 1 | 0.0320 | 0.0099 | NK_blend |
| uint8 | 240×320 | 3 | 0.0082 | 0.0076 | NK_blend |
| float32 | 240×320 | 3 | 0.0910 | 0.0190 | NK_blend |
| uint8 | 240×320 | 9 | 0.0210 | 0.0197 | NK_blend |
| float32 | 240×320 | 9 | 0.2516 | 0.0738 | NK_blend |
| uint8 | 480×640 | 1 | 0.0098 | 0.0094 | NK_blend |
| float32 | 480×640 | 1 | 0.1068 | 0.0245 | NK_blend |
| uint8 | 480×640 | 3 | 0.0226 | 0.0212 | NK_blend |
| float32 | 480×640 | 3 | 0.3349 | 0.0978 | NK_blend |
| uint8 | 480×640 | 9 | 0.0708 | 0.0660 | NK_blend |
| float32 | 480×640 | 9 | 1.0176 | 0.2458 | NK_blend |
| uint8 | 768×1024 | 1 | 0.0183 | 0.0177 | NK_blend |
| float32 | 768×1024 | 1 | 0.2684 | 0.0599 | NK_blend |
| uint8 | 768×1024 | 3 | 0.0476 | 0.0470 | NK_blend |
| float32 | 768×1024 | 3 | 0.8286 | 0.2420 | NK_blend |
| uint8 | 768×1024 | 9 | 0.1560 | 0.1705 | prod |
| float32 | 768×1024 | 9 | 2.5924 | 0.7704 | NK_blend |

## `multiply_by_vector` / `add_vector` — per-channel constants (broadcast)

| dtype | op | H×W | C | prod | NK channel-wise `scale` | fastest |
|-------|----|-----|---|-----:|------------------------:|---------|
| uint8 | mul_vec | 240×320 | 1 | 0.0118 | 0.0102 | NK_loop |
| uint8 | add_vec | 240×320 | 1 | 0.0120 | 0.0106 | NK_loop |
| float32 | mul_vec | 240×320 | 1 | 0.0298 | 0.0345 | prod |
| float32 | add_vec | 240×320 | 1 | 0.0296 | 0.0344 | prod |
| uint8 | mul_vec | 240×320 | 3 | 0.0660 | 0.1395 | prod |
| uint8 | add_vec | 240×320 | 3 | 0.0722 | 0.1376 | prod |
| float32 | mul_vec | 240×320 | 3 | 0.2891 | 0.2068 | NK_loop |
| float32 | add_vec | 240×320 | 3 | 0.3013 | 0.2049 | NK_loop |
| uint8 | mul_vec | 240×320 | 9 | 0.2405 | 0.4161 | prod |
| uint8 | add_vec | 240×320 | 9 | 0.2334 | 0.4119 | prod |
| float32 | mul_vec | 240×320 | 9 | 0.4965 | 0.7367 | prod |
| float32 | add_vec | 240×320 | 9 | 0.4952 | 0.7393 | prod |
| uint8 | mul_vec | 480×640 | 1 | 0.0283 | 0.0260 | NK_loop |
| uint8 | add_vec | 480×640 | 1 | 0.0283 | 0.0263 | NK_loop |
| float32 | mul_vec | 480×640 | 1 | 0.1050 | 0.1201 | prod |
| float32 | add_vec | 480×640 | 1 | 0.1034 | 0.1200 | prod |
| uint8 | mul_vec | 480×640 | 3 | 0.0883 | 0.5519 | prod |
| uint8 | add_vec | 480×640 | 3 | 0.0896 | 0.5307 | prod |
| float32 | mul_vec | 480×640 | 3 | 1.1517 | 0.8043 | NK_loop |
| float32 | add_vec | 480×640 | 3 | 1.1635 | 0.7919 | NK_loop |
| uint8 | mul_vec | 480×640 | 9 | 0.2735 | 1.5882 | prod |
| uint8 | add_vec | 480×640 | 9 | 0.2840 | 1.5737 | prod |
| float32 | mul_vec | 480×640 | 9 | 2.0534 | 3.1211 | prod |
| float32 | add_vec | 480×640 | 9 | 2.1096 | 3.1707 | prod |
| uint8 | mul_vec | 768×1024 | 1 | 0.0633 | 0.0601 | NK_loop |
| uint8 | add_vec | 768×1024 | 1 | 0.0632 | 0.0600 | NK_loop |
| float32 | mul_vec | 768×1024 | 1 | 0.2596 | 0.3097 | prod |
| float32 | add_vec | 768×1024 | 1 | 0.2614 | 0.2990 | prod |
| uint8 | mul_vec | 768×1024 | 3 | 0.1503 | 1.3470 | prod |
| uint8 | add_vec | 768×1024 | 3 | 0.1422 | 1.3662 | prod |
| float32 | mul_vec | 768×1024 | 3 | 3.0927 | 2.0348 | NK_loop |
| float32 | add_vec | 768×1024 | 3 | 2.9960 | 2.0500 | NK_loop |
| uint8 | mul_vec | 768×1024 | 9 | 0.3518 | 4.1271 | prod |
| uint8 | add_vec | 768×1024 | 9 | 0.3900 | 4.1320 | prod |
| float32 | mul_vec | 768×1024 | 9 | 5.3723 | 8.7035 | prod |
| float32 | add_vec | 768×1024 | 9 | 5.4171 | 8.5744 | prod |

## Readout (this machine)

- **Scalar affine** (`multiply_by_constant`, `add_constant`): three paths timed — prod, NK scale (alloc), NK scale (inplace).
  `inplace` uses `out=buf` (raw numpy array, not Tensor wrapper) per NumKong #326 fix.
- **Elementwise multiply** full array: **NK fma** only timed for **float32** (uint8 prod promotes to f32).
- **Elementwise add** full array: **NK** is existing **`add_array_numkong`** (`blend`).
- **Per-channel** vector: **NK** is **C separate `scale` calls** — usually loses to one OpenCV/LUT pass.
