# Albucore backends: NumKong vs OpenCV / NumPy / LUT

Median ms; **fastest alt** = min(OpenCV, NumPy, [LUT for uint8]).

## `add_weighted` — uint8 (weights 0.5 / 0.5)

| H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 240×320 | 1 | 76800 | 0.0061 | 0.0138 | 0.0593 | 0.0620 | OpenCV (0.0138) | NK 2.26× faster than OpenCV |
| 240×320 | 3 | 230400 | 0.0159 | 0.0352 | 0.1705 | 0.1450 | OpenCV (0.0352) | NK 2.22× faster than OpenCV |
| 240×320 | 9 | 691200 | 0.0442 | 0.1036 | 0.5130 | 0.4327 | OpenCV (0.1036) | NK 2.34× faster than OpenCV |
| 480×640 | 1 | 307200 | 0.0206 | 0.0485 | 0.2271 | 0.1424 | OpenCV (0.0485) | NK 2.35× faster than OpenCV |
| 480×640 | 3 | 921600 | 0.0601 | 0.1403 | 0.7187 | 0.2953 | OpenCV (0.1403) | NK 2.34× faster than OpenCV |
| 480×640 | 9 | 2764800 | 0.1777 | 0.4485 | 2.0854 | 0.7261 | OpenCV (0.4485) | NK 2.52× faster than OpenCV |
| 768×1024 | 1 | 786432 | 0.0513 | 0.1221 | 0.5766 | 0.2798 | OpenCV (0.1221) | NK 2.38× faster than OpenCV |
| 768×1024 | 3 | 2359296 | 0.1517 | 0.3645 | 1.7113 | 0.5344 | OpenCV (0.3645) | NK 2.40× faster than OpenCV |
| 768×1024 | 9 | 7077888 | 0.4725 | 1.0846 | 5.3019 | 1.4287 | OpenCV (1.0846) | NK 2.30× faster than OpenCV |

## `add_weighted` — float32 (weights 0.5 / 0.5; no LUT)

| H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|------------:|---------------:|
| 240×320 | 1 | 76800 | 0.0073 | 0.0084 | 0.0164 | OpenCV (0.0084) | NK 1.15× faster than OpenCV |
| 240×320 | 3 | 230400 | 0.0251 | 0.0241 | 0.0518 | OpenCV (0.0241) | OpenCV 1.04× faster than NK |
| 240×320 | 9 | 691200 | 0.0724 | 0.0702 | 0.1504 | OpenCV (0.0702) | OpenCV 1.03× faster than NK |
| 480×640 | 1 | 307200 | 0.0328 | 0.0330 | 0.0673 | OpenCV (0.0330) | NK 1.00× faster than OpenCV |
| 480×640 | 3 | 921600 | 0.0961 | 0.0932 | 0.2013 | OpenCV (0.0932) | OpenCV 1.03× faster than NK |
| 480×640 | 9 | 2764800 | 0.2452 | 0.2664 | 0.5900 | OpenCV (0.2664) | NK 1.09× faster than OpenCV |
| 768×1024 | 1 | 786432 | 0.0880 | 0.0694 | 0.1535 | OpenCV (0.0694) | OpenCV 1.27× faster than NK |
| 768×1024 | 3 | 2359296 | 0.1953 | 0.1928 | 0.4930 | OpenCV (0.1928) | OpenCV 1.01× faster than NK |
| 768×1024 | 9 | 7077888 | 0.7555 | 0.7295 | 1.6018 | OpenCV (0.7295) | OpenCV 1.04× faster than NK |

## `add_weighted` — batch / video `(N,H,W,C)`, N=4, H×W=240×320

Same weights **0.5 / 0.5**. Pixels = N×H×W×C.

### uint8

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 4×240×320 | 1 | 307200 | 0.0204 | 0.0465 | 0.2215 | 0.1920 | OpenCV (0.0465) | NK 2.28× faster than OpenCV |
| 4×240×320 | 3 | 921600 | 0.0596 | 0.1395 | 0.7147 | 0.5704 | OpenCV (0.1395) | NK 2.34× faster than OpenCV |
| 4×240×320 | 9 | 2764800 | 0.1833 | 0.4160 | 2.0805 | 1.7848 | OpenCV (0.4160) | NK 2.27× faster than OpenCV |

### float32 (no LUT)

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|------------:|---------------:|
| 4×240×320 | 1 | 307200 | 0.0329 | 0.0318 | 0.0676 | OpenCV (0.0318) | OpenCV 1.03× faster than NK |
| 4×240×320 | 3 | 921600 | 0.1025 | 0.0800 | 0.1881 | OpenCV (0.0800) | OpenCV 1.28× faster than NK |
| 4×240×320 | 9 | 2764800 | 0.2245 | 0.3127 | 0.6222 | OpenCV (0.3127) | NK 1.39× faster than OpenCV |

## Global **mean only** (uint8) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel, `mean = s/n`. OpenCV (C=1): `meanStdDev`, read scalar mean.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 240×320 | 1 | 76800 | 0.0298 | 0.0038 | 0.0149 | NumKong |
| 240×320 | 3 | 230400 | 0.0850 | 0.0110 | N/A | NumKong |
| 240×320 | 9 | 691200 | 0.2555 | 0.0335 | N/A | NumKong |
| 480×640 | 1 | 307200 | 0.1179 | 0.0149 | 0.0581 | NumKong |
| 480×640 | 3 | 921600 | 0.3385 | 0.0440 | N/A | NumKong |
| 480×640 | 9 | 2764800 | 1.0170 | 0.1310 | N/A | NumKong |
| 768×1024 | 1 | 786432 | 0.2850 | 0.0367 | 0.1494 | NumKong |
| 768×1024 | 3 | 2359296 | 0.8728 | 0.1180 | N/A | NumKong |
| 768×1024 | 9 | 7077888 | 2.6690 | 0.3347 | N/A | NumKong |

## Global **std only** (uint8) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.std()) + eps`. NumKong: full `moments` → population std + eps. OpenCV (C=1): `meanStdDev`, read scalar std + eps.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 240×320 | 1 | 76800 | 0.0847 | 0.0044 | 0.0153 | NumKong |
| 240×320 | 3 | 230400 | 0.2362 | 0.0119 | N/A | NumKong |
| 240×320 | 9 | 691200 | 0.7235 | 0.0332 | N/A | NumKong |
| 480×640 | 1 | 307200 | 0.3152 | 0.0152 | 0.0588 | NumKong |
| 480×640 | 3 | 921600 | 0.9142 | 0.0495 | N/A | NumKong |
| 480×640 | 9 | 2764800 | 2.8462 | 0.1334 | N/A | NumKong |
| 768×1024 | 1 | 786432 | 0.7807 | 0.0443 | 0.1551 | NumKong |
| 768×1024 | 3 | 2359296 | 2.3455 | 0.1142 | N/A | NumKong |
| 768×1024 | 9 | 7077888 | 7.3294 | 0.3272 | N/A | NumKong |

## Global **mean only** (float32) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel. OpenCV (C=1): `meanStdDev` on float32 image.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 240×320 | 1 | 76800 | 0.0116 | 0.0339 | 0.0674 | NumPy |
| 240×320 | 3 | 230400 | 0.0278 | 0.1051 | N/A | NumPy |
| 240×320 | 9 | 691200 | 0.0774 | 0.3028 | N/A | NumPy |
| 480×640 | 1 | 307200 | 0.0356 | 0.1347 | 0.2666 | NumPy |
| 480×640 | 3 | 921600 | 0.1024 | 0.4035 | N/A | NumPy |
| 480×640 | 9 | 2764800 | 0.3002 | 1.2354 | N/A | NumPy |
| 768×1024 | 1 | 786432 | 0.0910 | 0.3500 | 0.6979 | NumPy |
| 768×1024 | 3 | 2359296 | 0.2764 | 1.0269 | N/A | NumPy |
| 768×1024 | 9 | 7077888 | 0.7839 | 3.1140 | N/A | NumPy |

## Global **std only** (float32) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.std()) + eps` (population `ddof=0`). NumKong: `moments` → population std + eps.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 240×320 | 1 | 76800 | 0.0350 | 0.0344 | 0.0681 | NumKong |
| 240×320 | 3 | 230400 | 0.1005 | 0.1078 | N/A | NumPy |
| 240×320 | 9 | 691200 | 0.2446 | 0.3070 | N/A | NumPy |
| 480×640 | 1 | 307200 | 0.1168 | 0.1369 | 0.2730 | NumPy |
| 480×640 | 3 | 921600 | 0.3358 | 0.4091 | N/A | NumPy |
| 480×640 | 9 | 2764800 | 1.0011 | 1.2709 | N/A | NumPy |
| 768×1024 | 1 | 786432 | 0.3034 | 0.3455 | 0.6997 | NumPy |
| 768×1024 | 3 | 2359296 | 0.8461 | 1.0394 | N/A | NumPy |
| 768×1024 | 9 | 7077888 | 2.7224 | 3.1834 | N/A | NumPy |

## Global statistics — batch / video `(N,H,W,C)`, N=4, H×W=240×320 (same mean/std semantics as image: reduce over **all** pixels)

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

### Batch — global **mean only** (uint8)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×240×320 | 1 | 307200 | 0.1150 | 0.0147 | 0.0590 | NumKong |
| 4×240×320 | 3 | 921600 | 0.3406 | 0.0430 | N/A | NumKong |
| 4×240×320 | 9 | 2764800 | 1.0023 | 0.1330 | N/A | NumKong |

### Batch — global **std only** (uint8)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×240×320 | 1 | 307200 | 0.3230 | 0.0155 | 0.0589 | NumKong |
| 4×240×320 | 3 | 921600 | 0.9341 | 0.0447 | N/A | NumKong |
| 4×240×320 | 9 | 2764800 | 2.7730 | 0.1313 | N/A | NumKong |

### Batch — global **mean only** (float32)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×240×320 | 1 | 307200 | 0.0375 | 0.1381 | 0.2826 | NumPy |
| 4×240×320 | 3 | 921600 | 0.1054 | 0.4241 | N/A | NumPy |
| 4×240×320 | 9 | 2764800 | 0.3182 | 1.2829 | N/A | NumPy |

### Batch — global **std only** (float32)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×240×320 | 1 | 307200 | 0.1195 | 0.1369 | 0.2753 | NumPy |
| 4×240×320 | 3 | 921600 | 0.3308 | 0.4172 | N/A | NumPy |
| 4×240×320 | 9 | 2764800 | 0.9875 | 1.1961 | N/A | NumPy |

## Per-channel mean + std — `(H,W,C)`, `(N,H,W,C)`, `(N,D,H,W,C)`

Reduce over **all axes except channel** (`shape[-1]`). **NP mean** / **NP std**: separate full reductions over those axes. **NP both**: `mean` then `std` in one timed block. **albucore**: `mean_std(img, "per_channel", eps=…)` (3D: OpenCV + NumPy routing in `stats`; higher rank → NumPy axis-reduce). **NK**: one NumKong `moments` per channel (no batched per-channel API in this bench).

| dtype | layout (indices …×C) | C | pixels | NP mean | NP std | NP both | albucore | NK (C×moments) |
|-------|------------------------|---|--------|--------:|-------:|--------:|---------:|---------------:|
| uint8 | 240×320×1 | 1 | 76800 | 0.0303 | 0.0815 | 0.1114 | 0.0055 | 0.0045 |
| float32 | 240×320×1 | 1 | 76800 | 0.0110 | 0.0349 | 0.0457 | 0.0359 | 0.0347 |
| uint8 | 240×320×3 | 3 | 230400 | 0.4878 | 1.2826 | 1.7248 | 0.0148 | 0.0139 |
| float32 | 240×320×3 | 3 | 230400 | 0.3786 | 1.0115 | 1.3640 | 0.0779 | 0.0993 |
| uint8 | 240×320×9 | 9 | 691200 | 0.6283 | 1.6882 | 2.4642 | 0.2069 | 0.2063 |
| float32 | 240×320×9 | 9 | 691200 | 0.3765 | 1.1234 | 1.5681 | 0.9989 | 1.0220 |
| uint8 | 480×640×1 | 1 | 307200 | 0.1165 | 0.3050 | 0.4223 | 0.0173 | 0.0158 |
| float32 | 480×640×1 | 1 | 307200 | 0.0368 | 0.1206 | 0.1600 | 0.1415 | 0.1373 |
| uint8 | 480×640×3 | 3 | 921600 | 2.0238 | 5.2999 | 7.2864 | 0.0479 | 0.0469 |
| float32 | 480×640×3 | 3 | 921600 | 1.5328 | 4.0811 | 5.5527 | 0.3067 | 0.3988 |
| uint8 | 480×640×9 | 9 | 2764800 | 2.6039 | 7.3909 | 9.8674 | 0.8413 | 0.8518 |
| float32 | 480×640×9 | 9 | 2764800 | 1.5397 | 4.8160 | 6.4687 | 4.1146 | 4.0342 |
| uint8 | 768×1024×1 | 1 | 786432 | 0.2836 | 0.7855 | 1.0636 | 0.0385 | 0.0375 |
| float32 | 768×1024×1 | 1 | 786432 | 0.0901 | 0.3059 | 0.3908 | 0.3601 | 0.3586 |
| uint8 | 768×1024×3 | 3 | 2359296 | 5.0441 | 13.4407 | 18.3781 | 0.1137 | 0.1127 |
| float32 | 768×1024×3 | 3 | 2359296 | 4.0292 | 10.8510 | 15.0420 | 0.8127 | 1.0547 |
| uint8 | 768×1024×9 | 9 | 7077888 | 6.6748 | 18.1957 | 25.5996 | 2.2381 | 2.1816 |
| float32 | 768×1024×9 | 9 | 7077888 | 4.0683 | 12.5615 | 16.8616 | 10.3552 | 10.3726 |
| uint8 | 4×240×320×1 | 1 | 307200 | 0.1118 | 0.2965 | 0.4326 | 0.0164 | 0.0154 |
| float32 | 4×240×320×1 | 1 | 307200 | 0.0355 | 0.1306 | 0.1620 | 0.1385 | 0.1389 |
| uint8 | 4×240×320×3 | 3 | 921600 | 1.9570 | 5.2953 | 7.4423 | 0.0474 | 0.0505 |
| float32 | 4×240×320×3 | 3 | 921600 | 1.5572 | 4.2670 | 5.8360 | 0.3988 | 0.4248 |
| uint8 | 4×240×320×9 | 9 | 2764800 | 2.6200 | 7.2625 | 10.0618 | 0.8503 | 0.8529 |
| float32 | 4×240×320×9 | 9 | 2764800 | 1.5473 | 4.8540 | 6.2456 | 3.9170 | 3.9411 |
| uint8 | 2×4×64×80×1 | 1 | 40960 | 0.0178 | 0.0476 | 0.0654 | 0.0039 | 0.0028 |
| float32 | 2×4×64×80×1 | 1 | 40960 | 0.0079 | 0.0233 | 0.0300 | 0.0207 | 0.0195 |
| uint8 | 2×4×64×80×3 | 3 | 122880 | 0.2739 | 0.6911 | 0.9386 | 0.0095 | 0.0085 |
| float32 | 2×4×64×80×3 | 3 | 122880 | 0.1983 | 0.5477 | 0.7627 | 0.0568 | 0.0557 |
| uint8 | 2×4×64×80×9 | 9 | 368640 | 0.3453 | 0.9400 | 1.2907 | 0.1210 | 0.1325 |
| float32 | 2×4×64×80×9 | 9 | 368640 | 0.2075 | 0.6290 | 0.8305 | 0.5368 | 0.5326 |
