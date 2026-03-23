# Albucore: High-Performance Image Processing Functions

[![PyPI version](https://img.shields.io/pypi/v/albucore.svg)](https://pypi.org/project/albucore/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/albucore.svg)](https://pypi.org/project/albucore/)
[![CI](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml/badge.svg)](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=albumentations-team/albucore@github)](https://gitads.dev/v1/ad-track?source=albumentations-team/albucore@github)

Albucore is a library of optimized atomic functions designed for efficient image processing. These functions serve as the foundation for [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX), a popular image augmentation library.

## Overview

Image processing operations can be implemented in various ways, each with its own performance characteristics depending on the image type, size, and number of channels. Albucore aims to provide the fastest implementation for each operation by leveraging different backends such as NumPy, OpenCV, and custom optimized code.

**Supported dtypes:** `uint8` and `float32` only.

Key features:

- Optimized atomic image processing functions
- Automatic selection of the fastest implementation based on input image characteristics
- Seamless integration with AlbumentationsX
- Benchmark scripts available (see `./benchmark.sh <data_dir>`)

## Installation

**Requires Python 3.10+.** Basic installation (you manage OpenCV separately):

```bash
pip install albucore
```

**With OpenCV headless** (recommended for servers/CI):

```bash
pip install albucore[headless]
```

**With OpenCV GUI support** (for local development with cv2.imshow):

```bash
pip install albucore[gui]
```

**With OpenCV contrib modules:**

```bash
pip install albucore[contrib]              # GUI version
pip install albucore[contrib-headless]     # Headless version
```

**Note:** If you already have `opencv-python` or `opencv-contrib-python` installed, just use `pip install albucore` to avoid package conflicts. Albucore will detect and use your existing OpenCV installation.

## Usage

```python
import numpy as np
import albucore

# Create a sample RGB image
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Apply a function
result = albucore.multiply(image, 1.5)

# For grayscale images, ensure the channel dimension is present
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
gray_result = albucore.multiply(gray_image, 1.5)
```

Albucore automatically selects the most efficient implementation based on the input image type and characteristics.

## Shape Conventions

Albucore expects images to follow specific shape conventions, with the channel dimension always present:

- **Single image**: `(H, W, C)` - Height, Width, Channels
- **Grayscale image**: `(H, W, 1)` - Height, Width, 1 channel
- **Batch of images**: `(N, H, W, C)` - Number of images, Height, Width, Channels
- **3D volume**: `(D, H, W, C)` - Depth, Height, Width, Channels
- **Batch of volumes**: `(N, D, H, W, C)` - Number of volumes, Depth, Height, Width, Channels

### Important Notes:

1. **Channel dimension is always required**, even for grayscale images (use shape `(H, W, 1)`)
2. Single-channel images should have shape `(H, W, 1)` not `(H, W)`
3. **Batch vs volume:** `(N, H, W, C)` is **N separate images**; a single **3D volume** is `(D, H, W, C)` with **depth** `D`. Do not confuse `N` (batch) with `D` (slices).

### Examples:

```python
import numpy as np
import albucore

# Grayscale image - MUST have explicit channel dimension
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)

# RGB image
rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Batch of 10 grayscale images
batch_gray = np.random.randint(0, 256, (10, 100, 100, 1), dtype=np.uint8)

# 3D volume with 20 slices
volume = np.random.randint(0, 256, (20, 100, 100, 1), dtype=np.uint8)

# Batch of 5 RGB volumes, each with 20 slices
batch_volumes = np.random.randint(0, 256, (5, 20, 100, 100, 3), dtype=np.uint8)
```

## Functions

All symbols below are exported via `from albucore import *` (or `from albucore.functions import …`).
Every function accepts `uint8` and `float32` images; inputs must have an explicit channel dimension
(`(H, W, C)` — never bare `(H, W)`).

### Arithmetic

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `multiply` | `(img, value, inplace=False)` | `img * value`, clipped to dtype range | uint8 scalar/vector → LUT; uint8 array → OpenCV; float32 → NumPy broadcast |
| `add` | `(img, value, inplace=False)` | `img + value`, clipped to dtype range | uint8 scalar → OpenCV saturate; uint8 vector → LUT; uint8 array → NumKong/OpenCV; float32 → NumPy |
| `power` | `(img, exponent, inplace=False)` | `img ** exponent`, clipped to dtype range | uint8 → LUT; float32 scalar → `cv2.pow`; float32 array → NumPy |
| `add_weighted` | `(img1, weight1, img2, weight2)` | `img1*w1 + img2*w2`, clipped | NumKong SIMD `blend`; large float32 → `cv2.addWeighted` |
| `multiply_add` | `(img, factor, value, inplace=False)` | `img * factor + value`, clipped | uint8 → LUT (fused, one table); float32 → NumPy broadcast |

`value` / `factor` / `exponent` can be a scalar, a length-`C` 1-D array (per-channel), or a
full image-shaped array.

### Normalization

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `normalize` | `(img, mean, denominator)` | `(img - mean) * denominator → float32` | uint8 → LUT (256-entry float32 table per channel); float32 → NumPy fused. Caller-supplied constants (e.g. ImageNet stats). |
| `normalize_per_image` | `(img, normalization)` | Normalize using stats computed from `img` → float32 | uint8 → LUT (except `"min_max"` → `cv2.normalize`); float32 → OpenCV/NumPy. `normalization ∈ {"image", "image_per_channel", "min_max", "min_max_per_channel"}` |

`normalize` is for **fixed** per-channel constants (ImageNet-style).
`normalize_per_image` **estimates** stats from the image at call time.

### Statistics

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `mean` | `(arr, axis=None, *, keepdims=False, dtype=None)` | Population mean | uint8 global → NumKong `moments`; per-channel HWC ≤ 4ch → `cv2.mean`; else NumPy |
| `std` | `(arr, axis=None, *, keepdims=False, eps=1e-4, dtype=None)` | Population std + eps | uint8 global → NumKong `moments`; per-channel HWC ≤ 4ch → `cv2.meanStdDev`; else NumPy |
| `mean_std` | `(arr, axis=None, *, keepdims=False, eps=1e-4)` | Mean and std+eps jointly | Single NumKong `moments` pass for uint8 global (faster than separate `mean`+`std`) |
| `reduce_sum` | `(arr, axis=None, *, keepdims=False)` | Sum with wide accumulator | uint8 → NumKong (avoids uint8 overflow); float32 → `np.sum(dtype=float64)` |

`axis` accepts `None`/`"global"` (scalar), `"per_channel"` (shape `(C,)`), or any NumPy-style
`int`/`tuple[int, ...]`.

### LUT (lookup tables)

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `apply_uint8_lut` | `(img, lut, *, inplace=False)` | Apply uint8→uint8 LUT; `lut` shape `(256,)` or `(C, 256)` | Shared `(256,)`: StringZilla or `cv2.LUT` by size heuristic. Per-channel `(C, 256)`: single `cv2.LUT` with `(256,1,C)` table on contiguous HWC; else StringZilla per channel |
| `sz_lut` | `(img, lut, inplace=True)` | Apply shared `(256,)` uint8 LUT via StringZilla `translate` | Raw byte translation — channel-unaware, fastest for small images and single-channel |

### Geometric / spatial

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `hflip` | `(img)` | Mirror left-right | `cv2.flip(img, 1)`; chunked for >512 channels |
| `vflip` | `(img)` | Mirror top-bottom | `cv2.flip(img, 0)` for ≤4 channels; NumPy slice for >4 channels |
| `median_blur` | `(img, ksize)` | Median filter (odd ksize ≥ 3) | `cv2.medianBlur`; ksize ≥ 7 falls back to chunk processing for >4 channels. Accepts float32 via `@uint8_io` |
| `matmul` | `(a, b)` | Matrix multiply (`a @ b`) | NumPy `@` (BLAS-backed); replaces `cv2.gemm` which lacks uint8 support |
| `pairwise_distances_squared` | `(points1, points2)` | Squared Euclidean distance matrix `(N, M)` | Small (N*M < 1000) → NumKong `cdist`; large → NumPy vectorized `‖a‖²+‖b‖²−2(a·b)` |

### Type conversion

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `to_float` | `(img, max_value=None)` | Convert to float32 in [0, 1] | float32 → no-op; uint8 → `cv2.LUT` (256-entry float32 table); others → NumPy divide |
| `from_float` | `(img, target_dtype, max_value=None)` | Scale float32 → integer dtype (round + clip) | float32 → NumPy `rint(img * max_value)` then clip; non-float32 → generic NumPy path |

### Decorators (re-exported)

| Decorator | What it does |
|---|---|
| `float32_io` | Wrap a function: cast input to float32, cast output back to original dtype |
| `uint8_io` | Wrap a function: cast input to uint8, cast output back to original dtype |

See [docs/decorators.md](docs/decorators.md) for `@preserve_channel_dim`, `@contiguous`,
`@clipped`, and `@batch_transform` (used internally, not re-exported).

### Batch Processing

Many functions in Albucore support batch processing out of the box. The library automatically handles different input shapes:

- Single images: `(H, W, C)`
- Batches: `(N, H, W, C)`
- Volumes: `(D, H, W, C)`
- Batch of volumes: `(N, D, H, W, C)`

Functions will preserve the input shape structure, applying operations efficiently across all images/slices in the batch.

See [docs/decorators.md](docs/decorators.md) for internal decorator documentation (`@preserve_channel_dim`, `@contiguous`, `@clipped`, `@batch_transform`).

## Performance

Albucore uses a combination of techniques to achieve high performance:

1. **Multiple Implementations**: Each function may have several implementations using different backends (NumPy, OpenCV, custom code).
2. **Automatic Selection**: The library automatically chooses the fastest implementation based on the input image type, size, and number of channels.
3. **Optimized Algorithms**: Custom implementations are optimized for specific use cases, often outperforming general-purpose libraries.
4. **NumKong**: SIMD `blend` for `add_weighted`; `cdist` for small `pairwise_distances_squared`; wide-accumulator `moments` for **uint8** global mean/std in `stats`; `scale` for **float32** `multiply_by_constant` (see [docs/numkong-performance.md](docs/numkong-performance.md)).

Micro-benchmarks vs NumPy/OpenCV: see [benchmarks/README.md](benchmarks/README.md); quick run: `uv run python benchmarks/benchmark_numkong.py` (OpenCV rows are skipped if `cv2` is not installed).

See [docs/performance-optimization.md](docs/performance-optimization.md) for detailed performance guidelines and best practices.

## Documentation

- [CLAUDE.md](CLAUDE.md) - AI development guidelines for working with this codebase
- [docs/image-conventions.md](docs/image-conventions.md) - Image shape conventions and requirements
- [docs/decorators.md](docs/decorators.md) - Decorator usage and patterns
- [docs/performance-optimization.md](docs/performance-optimization.md) - Performance optimization guidelines
- [docs/numkong-performance.md](docs/numkong-performance.md) - NumKong vs OpenCV/NumPy/LUT baselines (benchmark tables; sum/mean/std)
- [docs/public-api.md](docs/public-api.md) - Star-exported routers vs `albucore.functions` shims
- [benchmarks/README.md](benchmarks/README.md) - Python micro-benchmarks (`uv run python benchmarks/…`)
- [docs/research/](docs/research/) - Research notes (extra benchmark writeups; see [`benchmarks/README.md`](benchmarks/README.md) for scripts)

## License

MIT

## Acknowledgements

Albucore is part of the [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX) project. We'd like to thank all contributors to [AlbumentationsX](https://albumentations.ai/people) and the broader computer vision community for their inspiration and support.

<!-- GitAds-Verify: 1LSAKH1Y2GKIISALRDIFCG2T9YYNR5WD -->
