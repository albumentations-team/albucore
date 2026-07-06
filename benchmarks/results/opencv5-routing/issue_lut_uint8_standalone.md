
## Section 1 — Shared LUT `(256,)` on `HWC`

Production routes large multi-channel to **cv2**, small/C=1 to **SZ**.
Both are shown for every shape; `*` marks the production path.

_Median ms, repeats=15, warmup=5; Darwin `arm64`, OpenCV 5.0.0, numpy 2.2.6, stringzilla 4.6.2._
| layout | shape | pixels | cv2 new | cv2→dst | SZ ravel+reuse | prod | best |
|--------|-------|-------:|--------:|--------:|---------------:|:----:|------|
| HWC | 256×256×1 | 65536 | 0.0177 | 0.0173 | 0.0068 | SZ | SZ |
| HWC | 256×256×3 | 196608 | 0.0516 | 0.0505 | 0.0177 | SZ | SZ |
| HWC | 256×256×9 | 589824 | 0.1532 | 0.1471 | 0.0520 | SZ | SZ |
| HWC | 512×512×1 | 262144 | 0.0251 | 0.0244 | 0.0237 | SZ | SZ |
| HWC | 512×512×3 | 786432 | 0.0610 | 0.0575 | 0.0690 | SZ | cv2 |
| HWC | 512×512×9 | 2359296 | 0.1626 | 0.1687 | 0.1993 | cv2 | cv2 |
| HWC | 640×640×3 | 1228800 | 0.0765 | 0.0750 | 0.1086 | cv2 | cv2 |
| HWC | 640×640×9 | 3686400 | 0.1758 | 0.1793 | 0.3164 | cv2 | cv2 |
| HWC | 1024×1024×1 | 1048576 | 0.1107 | 0.2215 | 0.0887 | SZ | SZ |
| HWC | 1024×1024×3 | 3145728 | 0.1391 | 0.1489 | 0.2632 | cv2 | cv2 |
| HWC | 1024×1024×9 | 9437184 | 0.3363 | 0.3413 | 0.8213 | cv2 | cv2 |

## Section 2 — Shared LUT `(256,)` on `DHWC` / `NDHWC`

Production **always** uses SZ flat ravel. `cv2` columns are for comparison only.

_Median ms, repeats=15, warmup=5; Darwin `arm64`, OpenCV 5.0.0, numpy 2.2.6, stringzilla 4.6.2._
| layout | shape | pixels | cv2 new | cv2→dst | SZ ravel+reuse [prod] | best |
|--------|-------|-------:|--------:|--------:|----------------------:|------|
| DHWC | 32×128×128×1 | 524288 | 0.1308 | 0.1320 | 0.0455 | SZ |
| DHWC | 32×128×128×3 | 1572864 | 0.4005 | 0.4056 | 0.1373 | SZ |
| DHWC | 64×128×128×1 | 1048576 | 0.2654 | 0.2654 | 0.0876 | SZ |
| DHWC | 64×128×128×3 | 3145728 | 0.7924 | 0.7979 | 0.2689 | SZ |
| DHWC | 64×128×128×9 | 9437184 | 2.4262 | 2.4138 | 0.8098 | SZ |
| DHWC | 128×128×128×1 | 2097152 | 0.5211 | 0.5225 | 0.1746 | SZ |
| DHWC | 48×256×256×3 | 9437184 | 2.4410 | 2.4669 | 0.8303 | SZ |
| DHWC | 96×160×160×3 | 7372800 | 1.9339 | 1.9198 | 0.6287 | SZ |
| DHWC | 64×256×256×9 | 37748736 | 9.8500 | 9.7607 | 3.3845 | SZ |
| NDHWC | 2×32×128×128×1 | 1048576 | 0.2717 | 0.2709 | 0.0886 | SZ |
| NDHWC | 2×32×128×128×3 | 3145728 | 0.8452 | 0.8041 | 0.2836 | SZ |
| NDHWC | 2×64×128×128×3 | 6291456 | 1.7115 | 1.6412 | 0.5443 | SZ |
| NDHWC | 2×64×128×128×9 | 18874368 | 4.9937 | 4.9645 | 1.7019 | SZ |
| DHWC | 30×640×640×3 | 36864000 | 9.6350 | 9.5440 | 3.3175 | SZ |
| DHWC | 8×1024×1024×3 | 25165824 | 6.7393 | 6.6310 | 2.2766 | SZ |
| DHWC | 16×256×256×9 | 9437184 | 2.5348 | 2.4759 | 0.8387 | SZ |

## Section 3 — Per-channel LUT `(C,256)` on `HWC`

Production uses **`cv2.LUT(img, (256,1,C))`** one-shot for C>1 contiguous HWC.

_Median ms, repeats=15, warmup=5; Darwin `arm64`, OpenCV 5.0.0, numpy 2.2.6, stringzilla 4.6.2._
| layout | shape | C | pixels | cv2 new [prod C>1] | cv2→dst | SZ loop reuse ch | best |
|--------|-------|---|-------:|-------------------:|--------:|-----------------:|------|
| HWC | 256×256×1 | 1 | 65536 | 0.0174 | 0.0171 | 0.0079 | SZ |
| HWC | 256×256×3 | 3 | 196608 | 0.0507 | 0.0505 | 0.1174 | cv2→dst |
| HWC | 256×256×9 | 9 | 589824 | 0.1506 | 0.2014 | 0.3542 | cv2 new |
| HWC | 512×512×1 | 1 | 262144 | 0.0349 | 0.0327 | 0.0270 | SZ |
| HWC | 512×512×3 | 3 | 786432 | 0.0665 | 0.0675 | 0.4488 | cv2 new |
| HWC | 512×512×9 | 9 | 2359296 | 0.2248 | 0.2157 | 1.4154 | cv2→dst |
| HWC | 640×640×3 | 3 | 1228800 | 0.0749 | 0.0761 | 0.7326 | cv2 new |
| HWC | 640×640×9 | 9 | 3686400 | 0.2261 | 0.2283 | 2.2396 | cv2 new |
| HWC | 1024×1024×1 | 1 | 1048576 | 0.0881 | 0.0824 | 0.1020 | cv2→dst |
| HWC | 1024×1024×3 | 3 | 3145728 | 0.1587 | 0.1428 | 1.8787 | cv2→dst |
| HWC | 1024×1024×9 | 9 | 9437184 | 0.3870 | 0.4015 | 5.6710 | cv2 new |

## Section 4 — Per-channel LUT `(C,256)` on `DHWC` / `NDHWC`

Production uses **SZ loop per channel** (cv2 `(256,1,C)` rejected for ndim>3).

_Median ms, repeats=15, warmup=5; Darwin `arm64`, OpenCV 5.0.0, numpy 2.2.6, stringzilla 4.6.2._
| layout | shape | C | pixels | SZ loop reuse ch [prod] | cv2 flat (C=1 only) | best |
|--------|-------|---|-------:|------------------------:|--------------------:|------|
| DHWC | 32×128×128×1 | 1 | 524288 | 0.0515 | 0.0412 | cv2 |
| DHWC | 32×128×128×3 | 3 | 1572864 | 0.9270 | n/a (C>1) | SZ [only option] |
| DHWC | 64×128×128×1 | 1 | 1048576 | 0.1023 | 0.0852 | cv2 |
| DHWC | 64×128×128×3 | 3 | 3145728 | 1.8919 | n/a (C>1) | SZ [only option] |
| DHWC | 64×128×128×9 | 9 | 9437184 | 5.6000 | n/a (C>1) | SZ [only option] |
| DHWC | 128×128×128×1 | 1 | 2097152 | 0.2057 | 0.1392 | cv2 |
| DHWC | 48×256×256×3 | 3 | 9437184 | 5.6575 | n/a (C>1) | SZ [only option] |
| DHWC | 96×160×160×3 | 3 | 7372800 | 4.5275 | n/a (C>1) | SZ [only option] |
| DHWC | 64×256×256×9 | 9 | 37748736 | 22.9650 | n/a (C>1) | SZ [only option] |
| NDHWC | 2×32×128×128×1 | 1 | 1048576 | 0.1032 | 0.0944 | cv2 |
| NDHWC | 2×32×128×128×3 | 3 | 3145728 | 1.8334 | n/a (C>1) | SZ [only option] |
| NDHWC | 2×64×128×128×3 | 3 | 6291456 | 3.7050 | n/a (C>1) | SZ [only option] |
| NDHWC | 2×64×128×128×9 | 9 | 18874368 | 11.2249 | n/a (C>1) | SZ [only option] |
| DHWC | 30×640×640×3 | 3 | 36864000 | 22.1338 | n/a (C>1) | SZ [only option] |
| DHWC | 8×1024×1024×3 | 3 | 25165824 | 15.1968 | n/a (C>1) | SZ [only option] |
| DHWC | 16×256×256×9 | 9 | 9437184 | 5.7375 | n/a (C>1) | SZ [only option] |

### Legend

- `[production]`: path taken by albucore `apply_uint8_lut` for this shape.
- `cv2 new`: `cv2.LUT(src, lut)` — allocates output each call.
- `cv2→dst`: `cv2.LUT(src, lut, dst)` — reuses a preallocated output buffer.
- `SZ ravel+reuse`: `copyto` into a preallocated flat buffer, then `translate(inplace=True)`.
- `SZ loop reuse ch`: per-channel loop; one reused `H×W` plane + `translate(inplace=True)`.
- `cv2 loop 2D slices`: cv2 per-channel via iterating 2D slices — not standard production.
