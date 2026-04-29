# LUT production path benchmark

Median ms, repeats=31, warmup=7. `prod` = current router. Lower is better.

## uint8 per-channel LUT

| layout | shape | contiguous | prod | sz loop | cv2 loop | cv2 flat/copy | fastest | prod/fastest |
|--------|-------|------------|-----:|--------:|---------:|--------------:|---------|-------------:|
| HWC | 128x128x1 | True | 0.0027 | 0.0030 | 0.0054 | 0.0048 | prod | 1.00x |
| HWC | 128x128x3 | True | 0.0136 | 0.0358 | 0.0372 | 0.0128 | cv2 flat | 1.06x |
| HWC | 128x128x9 | True | 0.0448 | 0.1048 | 0.1219 | 0.0375 | cv2 flat | 1.19x |
| HWC | 256x256x1 | True | 0.0064 | 0.0075 | 0.0190 | 0.0172 | prod | 1.00x |
| HWC | 256x256x3 | True | 0.0543 | 0.1197 | 0.1507 | 0.0501 | cv2 flat | 1.08x |
| HWC | 256x256x9 | True | 0.1958 | 0.3629 | 0.4630 | 0.1577 | cv2 flat | 1.24x |
| HWC | 512x512x1 | True | 0.0203 | 0.0243 | 0.0409 | 0.0330 | prod | 1.00x |
| HWC | 512x512x3 | True | 0.0667 | 0.4542 | 0.4945 | 0.0698 | prod | 1.00x |
| HWC | 512x512x9 | True | 0.2572 | 1.5023 | 1.6660 | 0.2651 | prod | 1.00x |
| HWC | 1024x1024x1 | True | 0.0761 | 0.0961 | 0.1354 | 0.0934 | prod | 1.00x |
| HWC | 1024x1024x3 | True | 0.2415 | 2.0402 | 2.0979 | 0.2049 | cv2 flat | 1.18x |
| HWC | 1024x1024x9 | True | 0.4451 | 5.8214 | 5.7525 | 0.3539 | cv2 flat | 1.26x |
| DHWC | 32x128x128x1 | True | 0.0393 | 0.0458 | 0.1400 | 0.0440 | prod | 1.00x |
| DHWC | 64x128x128x3 | True | 0.2165 | 2.1565 | 2.7086 | 0.2009 | cv2 flat | 1.08x |
| DHWC | 6x32x32x9 | True | 0.0212 | 0.0517 | 0.0526 | 0.0150 | cv2 flat | 1.42x |
| NDHWC | 2x32x128x128x3 | True | 0.1844 | 2.1103 | 2.6463 | 0.2070 | prod | 1.00x |
| NDHWC | 1x128x128x128x1 | True | 0.2290 | 0.3425 | 0.7165 | 0.1314 | cv2 flat | 1.74x |
| HWC-stridedW | 512x512x3 | False | 0.4625 | 0.4671 | 0.5437 | 0.7471 | prod | 1.00x |
| HWC-stridedW | 512x512x9 | False | 0.9785 | 1.5343 | 1.8765 | 0.9985 | prod | 1.00x |
| DHWC-stridedW | 16x128x128x3 | False | 0.4848 | 0.4618 | 0.6269 | 0.7510 | sz loop | 1.05x |
| NDHWC-stridedW | 2x8x128x128x3 | False | 0.4694 | 0.4711 | 0.6370 | 0.7474 | prod | 1.00x |

## float32 output per-channel LUT

| layout | shape | contiguous | prod | cv2 loop | cv2 flat/copy | fastest | prod/fastest |
|--------|-------|------------|-----:|---------:|--------------:|---------|-------------:|
| HWC | 128x128x1 | True | 0.0056 | 0.0053 | 0.0047 | cv2 flat | 1.18x |
| HWC | 128x128x3 | True | 0.0139 | 0.0523 | 0.0129 | cv2 flat | 1.08x |
| HWC | 128x128x9 | True | 0.0504 | 0.1472 | 0.0375 | cv2 flat | 1.35x |
| HWC | 256x256x1 | True | 0.0403 | 0.0398 | 0.0169 | cv2 flat | 2.38x |
| HWC | 256x256x3 | True | 0.0532 | 0.1677 | 0.0514 | cv2 flat | 1.03x |
| HWC | 256x256x9 | True | 0.2310 | 0.6336 | 0.2476 | prod | 1.00x |
| HWC | 512x512x1 | True | 0.0730 | 0.0717 | 0.0310 | cv2 flat | 2.35x |
| HWC | 512x512x3 | True | 0.1432 | 0.6933 | 0.1305 | cv2 flat | 1.10x |
| HWC | 512x512x9 | True | 0.2165 | 1.9590 | 0.2271 | prod | 1.00x |
| HWC | 1024x1024x1 | True | 0.4332 | 0.4296 | 0.1525 | cv2 flat | 2.84x |
| HWC | 1024x1024x3 | True | 0.1460 | 2.1557 | 0.1393 | cv2 flat | 1.05x |
| HWC | 1024x1024x9 | True | 0.3600 | 8.4712 | 0.3642 | prod | 1.00x |
| DHWC | 32x128x128x1 | True | 0.3459 | 0.3382 | 0.0880 | cv2 flat | 3.93x |
| DHWC | 64x128x128x3 | True | 0.1497 | 3.1146 | 0.1503 | prod | 1.00x |
| DHWC | 6x32x32x9 | True | 0.0192 | 0.0587 | 0.0156 | cv2 flat | 1.23x |
| NDHWC | 2x32x128x128x3 | True | 0.1477 | 3.1083 | 0.1522 | prod | 1.00x |
| NDHWC | 1x128x128x128x1 | True | 1.3371 | 1.3065 | 0.2669 | cv2 flat | 5.01x |
| HWC-stridedW | 512x512x3 | False | 0.7618 | 0.7586 | 0.8435 | cv2 loop | 1.00x |
| HWC-stridedW | 512x512x9 | False | 0.9479 | 1.9241 | 0.9394 | cv2 flat | 1.01x |
| DHWC-stridedW | 16x128x128x3 | False | 0.7321 | 0.7724 | 0.8063 | prod | 1.00x |
| NDHWC-stridedW | 2x8x128x128x3 | False | 0.7844 | 0.7344 | 0.7987 | cv2 loop | 1.07x |
