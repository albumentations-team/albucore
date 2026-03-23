
### Benchmark: NumKong normalize patterns vs Albucore-style backends

_Median ms, repeats=41, warmup=12, seed=0; Darwin `arm64`, numkong 7.0.0, numpy 2.4.2._

#### 1) ImageNet fixed mean/std (uint8 input → float32 output)

NumKong: **per-channel** ``nk.scale`` on **float32** ravel (write-up uses uint8 tensor; that returns uint8 buffers — wrong for training; we time the float32 analogue). NumPy/OpenCV: **one** vectorized pass with broadcast (same as ``normalize`` math).

| H×W×C | NK loop (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |
|-------|-------------:|-----------:|------------:|---------|------------:|
| 128×128×3 | 0.0674 | 0.1347 | 0.1502 | NK | 1.00× |
| 256×256×3 | 0.3048 | 0.5582 | 0.6515 | NK | 1.00× |
| 512×512×3 | 1.4271 | 2.6259 | 2.8610 | NK | 1.00× |
| 1024×1024×3 | 5.4278 | 8.0382 | 8.7021 | NK | 1.00× |
| 512×512×9 | 3.8735 | 3.5745 | 3.8256 | NumPy | 1.08× |

#### 2) Global min–max normalize (float32 → [0,1])

NK: ``minmax`` on flat tensor + ``nk.scale``. NumPy: ``min``/``max`` + affine. OpenCV: ``cv2.normalize``.

| H×W×C | NK (ms) | NumPy (ms) | OpenCV (ms) | fastest | NK vs best |
|-------|--------:|-----------:|------------:|---------|------------:|
| 128×128×3 | 0.0343 | 0.0271 | 0.0140 | OpenCV | 2.45× |
| 256×256×3 | 0.1255 | 0.0999 | 0.0442 | OpenCV | 2.84× |
| 512×512×3 | 0.6789 | 0.5502 | 0.2621 | OpenCV | 2.59× |
| 1024×1024×3 | 1.9626 | 1.4683 | 0.4973 | OpenCV | 3.95× |
| 512×512×9 | 1.4712 | 1.0980 | 0.4106 | OpenCV | 3.58× |

#### 3) Per-channel mean & std (stats only, float32 HWC)

NK write-up: ``Tensor(H*W, C)`` then ``sum(axis=0)`` and ``norm(axis=0)`` → mean/std. vs ``cv2.meanStdDev`` (C≤4 only in Albucore) vs NumPy ``mean``/``std`` over spatial axes.

| H×W×C | NK (ms) | cv2.meanStdDev (ms) | NumPy (ms) | fastest | NK vs best |
|-------|--------:|--------------------:|-----------:|---------|------------:|
| 128×128×3 | 0.0425 | 0.0175 | 0.3408 | cv2 | 2.43× |
| 256×256×3 | 0.1775 | 0.0656 | 1.3279 | cv2 | 2.70× |
| 512×512×3 | 0.7862 | 0.2586 | 5.2712 | cv2 | 3.04× |
| 1024×1024×3 | 2.7836 | 1.0300 | 20.1620 | cv2 | 2.70× |
| 512×512×9 | 7.7751 | — | 6.1096 | NumPy | 1.27× |

#### Takeaway (regenerate on your machine)

- **Fixed constants (same math as ``normalize(img, mean, denominator)`` / ImageNet α,β):** on **this** run, a **3-channel** loop of ``nk.scale`` on float32 ravels beat **one-shot** NumPy and OpenCV; with **C=9** (tiled ImageNet vectors) **NumPy** won. Routing **inside** ``normalize`` should stay **benchmark-driven per C**.
- **Global min–max:** **OpenCV** ``cv2.normalize`` dominated **NK minmax + nk.scale** and NumPy here — aligns with keeping **min_max** on the OpenCV path in ``normalize_per_image``.
- **Per-channel mean/std stats:** **cv2.meanStdDev** won for **C=3**; for **C=9**, **NumPy** beat the write-up’s **sum+norm** NK recipe. No change suggested to ``stats._mean_std_per_channel`` from this bench.
- **Write-up caveat:** ``nk.scale`` on **uint8** ravel returns a **uint8** buffer (not a float feature map); use **float32** ravels (or a documented NK dtype) for training-style normalize.
