
### Benchmark: NumKong normalize patterns vs Albucore-style backends

_Median ms, repeats=15, warmup=5, seed=0; Darwin `arm64`, numkong 7.7.0, numpy 2.2.6._

#### 1) ImageNet fixed mean/std (uint8 input → float32 output)

NumKong: **per-channel** ``nk.scale`` on **float32** ravel (write-up uses uint8 tensor; that returns uint8 buffers — wrong for training; we time the float32 analogue). NumPy/OpenCV: **one** vectorized pass with broadcast (same as ``normalize`` math).

| H×W×C | NK loop (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |
|-------|-------------:|-----------:|------------:|---------|------------:|
| 128×128×3 | 0.0666 | 0.1354 | 0.1511 | NK | 1.00× |
| 256×256×3 | 0.2374 | 0.5270 | 0.5806 | NK | 1.00× |
| 512×512×3 | 0.9287 | 2.0499 | 2.2897 | NK | 1.00× |
| 1024×1024×3 | 3.8113 | 8.4100 | 9.0999 | NK | 1.00× |
| 512×512×9 | 3.3575 | 3.6697 | 4.1004 | NK | 1.00× |

#### 2) Global min–max normalize (float32 → [0,1])

NK: ``minmax`` on flat tensor + ``nk.scale``. NumPy: ``min``/``max`` + affine. OpenCV: ``cv2.normalize``.

| H×W×C | NK (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |
|-------|--------:|-----------:|------------:|---------|------------:|
| 128×128×3 | 0.0356 | 0.0291 | 0.0150 | OpenCV | 2.38× |
| 256×256×3 | 0.1317 | 0.0971 | 0.0461 | OpenCV | 2.85× |
| 512×512×3 | 0.5151 | 0.3828 | 0.1618 | OpenCV | 3.18× |
| 1024×1024×3 | 2.0607 | 1.5242 | 0.5881 | OpenCV | 3.50× |
| 512×512×9 | 1.5542 | 1.1063 | 0.4025 | OpenCV | 3.86× |

#### 3) Per-channel mean & std (stats only, float32 HWC)

NK write-up: ``Tensor(H*W, C)`` then ``sum(axis=0)`` and ``norm(axis=0)`` → mean/std. vs ``cv2.meanStdDev`` (C≤4 only in Albucore) vs NumPy ``mean``/``std`` over spatial axes.

| H×W×C | NK (ms) | cv2.meanStdDev (ms) | NumPy (ms) | fastest | NK vs best |
|-------|--------:|--------------------:|-----------:|---------|------------:|
| 128×128×3 | 0.0488 | 0.0184 | 0.3807 | cv2 | 2.65× |
| 256×256×3 | 0.1837 | 0.0685 | 1.5284 | cv2 | 2.68× |
| 512×512×3 | 0.7230 | 0.2647 | 6.0234 | cv2 | 2.73× |
| 1024×1024×3 | 2.8599 | 1.0889 | 24.0709 | cv2 | 2.63× |
| 512×512×9 | 6.7927 | — | 8.1412 | NK | 1.00× |

#### Takeaway (regenerate on your machine)

- **Fixed constants (same math as ``normalize(img, mean, denominator)`` / ImageNet α,β):** on **this** run, a **3-channel** loop of ``nk.scale`` on float32 ravels beat **one-shot** NumPy and OpenCV; with **C=9** (tiled ImageNet vectors) **NumPy** won. Routing **inside** ``normalize`` should stay **benchmark-driven per C**.
- **Global min–max:** **OpenCV** ``cv2.normalize`` dominated **NK minmax + nk.scale** and NumPy here — aligns with keeping **min_max** on the OpenCV path in ``normalize_per_image``.
- **Per-channel mean/std stats:** **cv2.meanStdDev** won for **C=3**; for **C=9**, **NumPy** beat the write-up’s **sum+norm** NK recipe. No change suggested to ``stats._mean_std_per_channel`` from this bench.
- **Write-up caveat:** ``nk.scale`` on **uint8** ravel returns a **uint8** buffer (not a float feature map); use **float32** ravels (or a documented NK dtype) for training-style normalize.
