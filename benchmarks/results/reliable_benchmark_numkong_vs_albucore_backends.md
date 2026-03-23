# Albucore backends: NumKong vs OpenCV / NumPy / LUT

Median ms; **fastest alt** = min(OpenCV, NumPy, [LUT for uint8]).

## `add_weighted` — uint8 (weights 0.5 / 0.5)

| H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 256×256 | 1 | 65536 | 0.0051 | 0.0129 | 0.0497 | 0.0478 | OpenCV (0.0129) | NK 2.54× faster than OpenCV |
| 256×256 | 3 | 196608 | 0.0134 | 0.0308 | 0.2152 | 0.1941 | OpenCV (0.0308) | NK 2.30× faster than OpenCV |
| 256×256 | 9 | 589824 | 0.0381 | 0.0920 | 0.9127 | 0.7295 | OpenCV (0.0920) | NK 2.41× faster than OpenCV |
| 512×512 | 1 | 262144 | 0.0181 | 0.0430 | 0.2819 | 0.1114 | OpenCV (0.0430) | NK 2.38× faster than OpenCV |
| 512×512 | 3 | 786432 | 0.0511 | 0.1199 | 1.3887 | 0.5565 | OpenCV (0.1199) | NK 2.35× faster than OpenCV |
| 512×512 | 9 | 2359296 | 0.2616 | 0.4858 | 1.7601 | 0.6260 | OpenCV (0.4858) | NK 1.86× faster than OpenCV |
| 1024×1024 | 1 | 1048576 | 0.0662 | 0.1636 | 1.8790 | 0.9500 | OpenCV (0.1636) | NK 2.47× faster than OpenCV |
| 1024×1024 | 3 | 3145728 | 0.3860 | 0.6530 | 2.3719 | 0.8299 | OpenCV (0.6530) | NK 1.69× faster than OpenCV |
| 1024×1024 | 9 | 9437184 | 0.6112 | 1.4348 | 10.5172 | 3.2475 | OpenCV (1.4348) | NK 2.35× faster than OpenCV |

## `add_weighted` — float32 (weights 0.5 / 0.5; no LUT)

| H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-----|---|--------|--------:|-------:|------:|------------:|---------------:|
| 256×256 | 1 | 65536 | 0.0079 | 0.0131 | 0.0158 | OpenCV (0.0131) | NK 1.66× faster than OpenCV |
| 256×256 | 3 | 196608 | 0.0213 | 0.0345 | 0.0752 | OpenCV (0.0345) | NK 1.62× faster than OpenCV |
| 256×256 | 9 | 589824 | 0.1887 | 0.2222 | 0.4068 | OpenCV (0.2222) | NK 1.18× faster than OpenCV |
| 512×512 | 1 | 262144 | 0.0294 | 0.0478 | 0.0602 | OpenCV (0.0478) | NK 1.63× faster than OpenCV |
| 512×512 | 3 | 786432 | 0.2280 | 0.2972 | 0.5073 | OpenCV (0.2972) | NK 1.30× faster than OpenCV |
| 512×512 | 9 | 2359296 | 0.2144 | 0.4121 | 0.4582 | OpenCV (0.4121) | NK 1.92× faster than OpenCV |
| 1024×1024 | 1 | 1048576 | 0.3777 | 0.4355 | 0.8234 | OpenCV (0.4355) | NK 1.15× faster than OpenCV |
| 1024×1024 | 3 | 3145728 | 0.3460 | 0.5666 | 0.7110 | OpenCV (0.5666) | NK 1.64× faster than OpenCV |
| 1024×1024 | 9 | 9437184 | 4.5787 | 1.7129 | 5.6640 | OpenCV (1.7129) | OpenCV 2.67× faster than NK |

## `add_weighted` — batch / video `(N,H,W,C)`, N=4, H×W=256×256

Same weights **0.5 / 0.5**. Pixels = N×H×W×C.

### uint8

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | LUT | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|----:|------------:|---------------:|
| 4×256×256 | 1 | 262144 | 0.0175 | 0.0400 | 0.2914 | 0.2528 | OpenCV (0.0400) | NK 2.28× faster than OpenCV |
| 4×256×256 | 3 | 786432 | 0.0514 | 0.1212 | 1.4059 | 1.0674 | OpenCV (0.1212) | NK 2.36× faster than OpenCV |
| 4×256×256 | 9 | 2359296 | 0.2772 | 0.4787 | 1.6935 | 1.4918 | OpenCV (0.4787) | NK 1.73× faster than OpenCV |

### float32 (no LUT)

| N×H×W | C | pixels | NumKong | OpenCV | NumPy | fastest alt | NK vs best alt |
|-------|---|--------|--------:|-------:|------:|------------:|---------------:|
| 4×256×256 | 1 | 262144 | 0.0295 | 0.0473 | 0.0590 | OpenCV (0.0473) | NK 1.60× faster than OpenCV |
| 4×256×256 | 3 | 786432 | 0.2795 | 0.2856 | 0.5828 | OpenCV (0.2856) | NK 1.02× faster than OpenCV |
| 4×256×256 | 9 | 2359296 | 0.2573 | 0.4269 | 0.5378 | OpenCV (0.4269) | NK 1.66× faster than OpenCV |

## Global **mean only** (uint8) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel, `mean = s/n`. OpenCV (C=1): `meanStdDev`, read scalar mean.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0268 | 0.0050 | 0.0130 | NumKong |
| 256×256 | 3 | 196608 | 0.0735 | 0.0127 | N/A | NumKong |
| 256×256 | 9 | 589824 | 0.2272 | 0.0365 | N/A | NumKong |
| 512×512 | 1 | 262144 | 0.1028 | 0.0164 | 0.0503 | NumKong |
| 512×512 | 3 | 786432 | 0.2990 | 0.0421 | N/A | NumKong |
| 512×512 | 9 | 2359296 | 0.9050 | 0.2631 | N/A | NumKong |
| 1024×1024 | 1 | 1048576 | 0.4101 | 0.0646 | 0.1988 | NumKong |
| 1024×1024 | 3 | 3145728 | 1.2232 | 0.3554 | N/A | NumKong |
| 1024×1024 | 9 | 9437184 | 3.6890 | 0.5784 | N/A | NumKong |

## Global **std only** (uint8) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.std()) + eps`. NumKong: full `moments` → population std + eps. OpenCV (C=1): `meanStdDev`, read scalar std + eps.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.1428 | 0.0053 | 0.0130 | NumKong |
| 256×256 | 3 | 196608 | 0.3323 | 0.0130 | N/A | NumKong |
| 256×256 | 9 | 589824 | 0.9071 | 0.0375 | N/A | NumKong |
| 512×512 | 1 | 262144 | 0.4228 | 0.0168 | 0.0512 | NumKong |
| 512×512 | 3 | 786432 | 1.3566 | 0.0497 | N/A | NumKong |
| 512×512 | 9 | 2359296 | 2.3552 | 0.2575 | N/A | NumKong |
| 1024×1024 | 1 | 1048576 | 1.6369 | 0.0659 | 0.1931 | NumKong |
| 1024×1024 | 3 | 3145728 | 5.8331 | 0.3475 | N/A | NumKong |
| 1024×1024 | 9 | 9437184 | 16.9031 | 0.5778 | N/A | NumKong |

## Global **mean only** (float32) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.mean())`. NumKong: `moments` on contiguous ravel. OpenCV (C=1): `meanStdDev` on float32 image.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0108 | 0.0326 | 0.0601 | NumPy |
| 256×256 | 3 | 196608 | 0.0258 | 0.0953 | N/A | NumPy |
| 256×256 | 9 | 589824 | 0.0819 | 0.3918 | N/A | NumPy |
| 512×512 | 1 | 262144 | 0.0372 | 0.1307 | 0.2376 | NumPy |
| 512×512 | 3 | 786432 | 0.0954 | 0.5349 | N/A | NumPy |
| 512×512 | 9 | 2359296 | 0.3091 | 1.1496 | N/A | NumPy |
| 1024×1024 | 1 | 1048576 | 0.1287 | 0.7049 | 0.9661 | NumPy |
| 1024×1024 | 3 | 3145728 | 0.3848 | 1.5861 | N/A | NumPy |
| 1024×1024 | 9 | 9437184 | 1.2768 | 4.6500 | N/A | NumPy |

## Global **std only** (float32) — NumPy vs NumKong vs OpenCV

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

NumPy: `float(img.std()) + eps` (population `ddof=0`). NumKong: `moments` → population std + eps.

| H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-----|---|--------|------:|--------:|-------:|--------|
| 256×256 | 1 | 65536 | 0.0301 | 0.0326 | 0.0532 | NumPy |
| 256×256 | 3 | 196608 | 0.0763 | 0.0982 | N/A | NumPy |
| 256×256 | 9 | 589824 | 0.3681 | 0.4322 | N/A | NumPy |
| 512×512 | 1 | 262144 | 0.1079 | 0.1336 | 0.2490 | NumPy |
| 512×512 | 3 | 786432 | 0.4815 | 0.5431 | N/A | NumPy |
| 512×512 | 9 | 2359296 | 0.9383 | 1.1905 | N/A | NumPy |
| 1024×1024 | 1 | 1048576 | 0.6264 | 0.7438 | 0.9696 | NumPy |
| 1024×1024 | 3 | 3145728 | 1.1332 | 1.5831 | N/A | NumPy |
| 1024×1024 | 9 | 9437184 | 7.0622 | 4.8827 | N/A | NumKong |

## Global statistics — batch / video `(N,H,W,C)`, N=4, H×W=256×256 (same mean/std semantics as image: reduce over **all** pixels)

OpenCV `cv2.meanStdDev` only matches **global** scalar semantics for **C=1**; for **C>1** it is per-channel. It always computes mean and std internally — the **same full-call time** is shown in both the mean-only and std-only tables for C=1.

### Batch — global **mean only** (uint8)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.0984 | 0.0164 | 0.0467 | NumKong |
| 4×256×256 | 3 | 786432 | 0.2974 | 0.0488 | N/A | NumKong |
| 4×256×256 | 9 | 2359296 | 0.9064 | 0.2587 | N/A | NumKong |

### Batch — global **std only** (uint8)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.4509 | 0.0166 | 0.0514 | NumKong |
| 4×256×256 | 3 | 786432 | 1.2742 | 0.0503 | N/A | NumKong |
| 4×256×256 | 9 | 2359296 | 2.3337 | 0.2533 | N/A | NumKong |

### Batch — global **mean only** (float32)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.0338 | 0.1308 | 0.2358 | NumPy |
| 4×256×256 | 3 | 786432 | 0.0985 | 0.5455 | N/A | NumPy |
| 4×256×256 | 9 | 2359296 | 0.3084 | 1.1830 | N/A | NumPy |

### Batch — global **std only** (float32)

| N×H×W | C | pixels | NumPy | NumKong | OpenCV | fastest |
|-------|---|--------|------:|--------:|-------:|--------|
| 4×256×256 | 1 | 262144 | 0.1004 | 0.1306 | 0.2437 | NumPy |
| 4×256×256 | 3 | 786432 | 0.4518 | 0.5301 | N/A | NumPy |
| 4×256×256 | 9 | 2359296 | 0.8889 | 1.1985 | N/A | NumPy |

## Per-channel mean + std — `(H,W,C)`, `(N,H,W,C)`, `(N,D,H,W,C)`

Reduce over **all axes except channel** (`shape[-1]`). **NP mean** / **NP std**: separate full reductions over those axes. **NP both**: `mean` then `std` in one timed block. **albucore**: `mean_std(img, "per_channel", eps=…)` (3D: OpenCV + NumPy routing in `stats`; higher rank → NumPy axis-reduce). **NK**: one NumKong `moments` per channel (no batched per-channel API in this bench).

| dtype | layout (indices …×C) | C | pixels | NP mean | NP std | NP both | albucore | NK (C×moments) |
|-------|------------------------|---|--------|--------:|-------:|--------:|---------:|---------------:|
| uint8 | 256×256×1 | 1 | 65536 | 0.0272 | 0.1068 | 0.1655 | 0.0137 | 0.0062 |
| float32 | 256×256×1 | 1 | 65536 | 0.0104 | 0.0317 | 0.0447 | 0.0595 | 0.0342 |
| uint8 | 256×256×3 | 3 | 196608 | 0.3872 | 1.2328 | 1.7147 | 0.0346 | 0.0899 |
| float32 | 256×256×3 | 3 | 196608 | 0.3381 | 0.8997 | 1.2559 | 0.0686 | 0.2232 |
| uint8 | 256×256×9 | 9 | 589824 | 0.5120 | 1.8383 | 2.3885 | 0.1207 | 0.3019 |
| float32 | 256×256×9 | 9 | 589824 | 0.3236 | 1.0910 | 1.4085 | 0.1872 | 0.5908 |
| uint8 | 512×512×1 | 1 | 262144 | 0.0952 | 0.4409 | 0.5017 | 0.0513 | 0.0175 |
| float32 | 512×512×1 | 1 | 262144 | 0.0326 | 0.1030 | 0.1385 | 0.2434 | 0.1333 |
| uint8 | 512×512×3 | 3 | 786432 | 1.5426 | 5.0511 | 6.5681 | 0.1282 | 0.2780 |
| float32 | 512×512×3 | 3 | 786432 | 1.3268 | 3.8793 | 5.4784 | 0.2685 | 0.7905 |
| uint8 | 512×512×9 | 9 | 2359296 | 2.0390 | 5.5715 | 7.5965 | 0.4820 | 0.9100 |
| float32 | 512×512×9 | 9 | 2359296 | 1.3038 | 4.0387 | 5.4022 | 0.7573 | 2.5033 |
| uint8 | 1024×1024×1 | 1 | 1048576 | 0.3930 | 1.6737 | 2.1673 | 0.2006 | 0.0626 |
| float32 | 1024×1024×1 | 1 | 1048576 | 0.1267 | 0.6102 | 0.7416 | 0.9774 | 0.7507 |
| uint8 | 1024×1024×3 | 3 | 3145728 | 6.1530 | 19.2115 | 25.6857 | 0.5246 | 1.1508 |
| float32 | 1024×1024×3 | 3 | 3145728 | 5.3747 | 14.4492 | 19.8197 | 1.0827 | 4.4048 |
| uint8 | 1024×1024×9 | 9 | 9437184 | 8.1170 | 28.9840 | 37.1242 | 1.9425 | 3.6718 |
| float32 | 1024×1024×9 | 9 | 9437184 | 5.2332 | 19.4995 | 24.8993 | 3.0330 | 15.5247 |
| uint8 | 4×256×256×1 | 1 | 262144 | 0.1005 | 0.3963 | 0.5388 | 0.5184 | 0.0173 |
| float32 | 4×256×256×1 | 1 | 262144 | 0.0342 | 0.1048 | 0.1383 | 0.3879 | 0.1329 |
| uint8 | 4×256×256×3 | 3 | 786432 | 1.5405 | 4.9173 | 6.3971 | 6.3189 | 0.3341 |
| float32 | 4×256×256×3 | 3 | 786432 | 1.3473 | 3.8648 | 5.1686 | 6.3119 | 0.7813 |
| uint8 | 4×256×256×9 | 9 | 2359296 | 2.0195 | 5.5805 | 7.6140 | 7.5529 | 0.9566 |
| float32 | 4×256×256×9 | 9 | 2359296 | 1.3158 | 3.9902 | 5.2762 | 6.4393 | 2.5516 |
| uint8 | 2×4×64×64×1 | 1 | 32768 | 0.0140 | 0.0911 | 0.1066 | 0.0600 | 0.0030 |
| float32 | 2×4×64×64×1 | 1 | 32768 | 0.0062 | 0.0182 | 0.0243 | 0.0701 | 0.0187 |
| uint8 | 2×4×64×64×3 | 3 | 98304 | 0.1918 | 0.5860 | 0.7557 | 0.7745 | 0.0345 |
| float32 | 2×4×64×64×3 | 3 | 98304 | 0.1681 | 0.4374 | 0.6176 | 0.7580 | 0.0820 |
| uint8 | 2×4×64×64×9 | 9 | 294912 | 0.2502 | 0.8955 | 1.1614 | 1.1949 | 0.1059 |
| float32 | 2×4×64×64×9 | 9 | 294912 | 0.1634 | 0.4930 | 0.6522 | 1.0323 | 0.4051 |
