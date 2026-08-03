# Albucore: High-Performance Image Processing Functions

[![PyPI version](https://img.shields.io/pypi/v/albucore.svg)](https://pypi.org/project/albucore/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/albucore.svg)](https://pypi.org/project/albucore/)
[![CI](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml/badge.svg)](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Sponsored by GitAds](https://gitads.dev/v1/ad-serve?source=albumentations-team/albucore@github)](https://gitads.dev/v1/ad-track?source=albumentations-team/albucore@github)

Albucore is a library of optimized atomic functions designed for efficient image processing. These functions serve as the foundation for [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX), an image augmentation library.

## Overview

Image processing operations can be implemented in several ways, with performance depending on dtype, size, layout, and channel count. Albucore routes each operation to a benchmark-selected NumPy, OpenCV, NumKong, or StringZilla implementation.

Most image-processing routers support `uint8` and `float32`. The elementwise `exp`, `log`, and `sqrt` routers are `float32`-only; conversion helpers also support additional integer dtypes.

Key features:

- Optimized atomic image processing functions
- Automatic selection of the fastest implementation based on input image characteristics
- Seamless integration with AlbumentationsX
- Reproducible micro-benchmarks and committed routing reports (see [benchmarks/README.md](benchmarks/README.md))

## Installation

**Requires Python 3.10+.** Basic installation (you manage OpenCV separately):

```bash
pip install albucore
```

`torch>=2.13.0` is a required dependency and installs with Albucore. AlbumentationsX validates `resize3d` Tensor
inputs as CPU and `requires_grad=False` before calling the Albucore primitive.

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

The tables below highlight commonly used public routers. They are exported via `from albucore import *` and can also be imported from `albucore.functions`. See [docs/public-api.md](docs/public-api.md) for the complete export list and the distinction between public routers and backend-specific compatibility shims.

Image routers use channel-last inputs with an explicit channel dimension (`(H, W, C)`, never bare `(H, W)`) and generally support `uint8` and `float32`. Exceptions are stated in the tables.

### Arithmetic

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `multiply` | `(img, value, inplace=False)` | Raw float32 `img * value`; uint8 saturates | uint8 scalar/vector → LUT; uint8 array → OpenCV; float32 → NumPy broadcast |
| `add` | `(img, value, inplace=False)` | Raw float32 `img + value`; uint8 saturates | uint8 scalar → OpenCV saturate; uint8 vector → LUT; uint8 array → NumKong/OpenCV; float32 → NumPy |
| `power` | `(img, exponent, inplace=False)` | Raw float32 `img ** exponent`; uint8 saturates | uint8 → LUT; float32 scalar → `cv2.pow`; float32 array → NumPy |
| `add_weighted` | `(img1, weight1, img2, weight2)` | Raw float32 `img1*w1 + img2*w2`; uint8 saturates | uint8 and float32 C=1 → NumKong; float32 C>1 → OpenCV for HWC/contiguous inputs, NumKong for strided batch/volume inputs |
| `multiply_add` | `(img, factor, value, inplace=False)` | Raw float32 `img * factor + value`; uint8 saturates | uint8 → LUT (fused, one table); scalar float32 → NumKong `scale`; vector/array float32 → NumPy broadcast |

`value` / `factor` / `exponent` can be a scalar, a length-`C` 1-D array (per-channel), or a
full image-shaped array.

These arithmetic routers do not impose an image-range convention on float32 data. The explicit
`@clipped` decorator remains available for callers, including AlbumentationsX operations whose own
contract requires clipping.

### Elementwise math

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `exp` | `(array, *, inplace=False)` | Elementwise exponential; float32 only | Small arrays → NumPy; large contiguous or strided arrays → OpenCV at benchmark-derived thresholds |
| `log` | `(array, *, inplace=False)` | NumPy-compatible natural logarithm; float32 only | NumPy for special values and small/unsupported layouts; guarded OpenCV path for eligible large arrays |
| `sqrt` | `(array, *, inplace=False)` | NumPy-compatible square root; float32 only | NumPy wins across the benchmark grid |

These functions accept float32 arrays of any rank and preserve the exact input shape. With `inplace=True`, an owned writable buffer may be reused; views and read-only arrays are never mutated. See the [elementwise benchmark report](benchmarks/results/benchmark_elementwise.md) for routing thresholds, environment, and NumKong results.

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
| `mean` | `(arr, axis=None, *, keepdims=False, dtype=None)` | Population mean | uint8 global → NumKong `sum`; per-channel routes among NumKong, OpenCV, and NumPy by rank/channel count |
| `std` | `(arr, axis=None, *, keepdims=False, eps=1e-4, dtype=None)` | Population std + eps | uint8 global → NumKong `moments`; per-channel routes among NumKong, OpenCV, and NumPy |
| `mean_std` | `(arr, axis=None, *, keepdims=False, eps=1e-4)` | Mean and std+eps jointly | Single NumKong `moments` pass for uint8 global; selected per-channel paths use NumKong or OpenCV |
| `reduce_sum` | `(arr, axis=None, *, keepdims=False)` | Sum with wide accumulator | uint8 and selected float32 per-channel layouts → NumKong; other float32 routes use a float64 NumPy accumulator |

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
| `hflip` | `(img)` | Mirror left-right | `cv2.flip(img, 1)`; chunked above OpenCV's 128-channel limit |
| `vflip` | `(img)` | Mirror top-bottom | `cv2.flip(img, 0)` for ≤4 channels; NumPy slice for >4 channels |
| `median_blur` | `(img, ksize)` | Median filter (odd ksize ≥ 3) | uint8 → direct/chunked `cv2.medianBlur`; float32 ksize 3/5 → native OpenCV; float32 ksize ≥ 7 → uint8 conversion fallback |
| `matmul` | `(a, b)` | Matrix multiply (`a @ b`) | NumPy `@` (BLAS-backed); replaces `cv2.gemm` which lacks uint8 support |
| `pairwise_distances_squared` | `(points1, points2)` | Squared Euclidean distance matrix `(N, M)` | Small (N*M < 1000) → NumKong `cdist`; large → NumPy vectorized `‖a‖²+‖b‖²−2(a·b)` |

The package also star-exports multi-channel wrappers for `copy_make_border`, `remap`, `resize`, `resize3d`, `warp_affine`, and `warp_perspective`; see [docs/public-api.md](docs/public-api.md) and their docstrings for complete signatures. `resize3d` expects prevalidated NumPy `DHWC` volumes and Torch `CDHW` tensors.

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

### Array layouts and batch processing

Arithmetic, normalization, statistics, conversion, and elementwise routers operate on channel-last arrays and preserve these layouts where applicable:

- Single images: `(H, W, C)`
- Batches: `(N, H, W, C)`
- Volumes: `(D, H, W, C)`
- Batch of volumes: `(N, D, H, W, C)`

Spatial routers document their own image-shape requirements. Transform authors can use `@batch_transform` to adapt an image operation to batches and volumes while restoring the original layout.

See [docs/decorators.md](docs/decorators.md) for internal decorator documentation (`@preserve_channel_dim`, `@contiguous`, `@clipped`, `@batch_transform`).

## Performance

Albucore uses a combination of techniques to achieve high performance:

1. **Multiple Implementations**: Each function may have several implementations using NumPy, OpenCV, NumKong, or StringZilla.
2. **Automatic Selection**: The library chooses a backend from dtype, size, memory layout, channel count, and semantic constraints.
3. **Measured Routing**: Backend choices and thresholds come from repeatable benchmarks rather than backend preference.
4. **NumKong**: SIMD `blend` for uint8 and single-channel float32 `add_weighted`, plus same-shaped uint8 `add_array`; `cdist` for small `pairwise_distances_squared`; wide-accumulator `moments` for selected statistics routes (see [docs/numkong-performance.md](docs/numkong-performance.md)).

Micro-benchmarks vs NumPy/OpenCV/NumKong: see [benchmarks/README.md](benchmarks/README.md). Run `uv run python benchmarks/benchmark_elementwise.py` for `exp`/`log`/`sqrt`, or `uv run python benchmarks/benchmark_numkong.py` for a smaller NumKong sweep.

See [docs/performance-optimization.md](docs/performance-optimization.md) for detailed performance guidelines and best practices.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Pull request process and CLA acceptance paths
- [AGENTS.md](AGENTS.md) - AI development guidelines for working with this codebase
- [docs/image-conventions.md](docs/image-conventions.md) - Image shape conventions and requirements
- [docs/decorators.md](docs/decorators.md) - Decorator usage and patterns
- [docs/performance-optimization.md](docs/performance-optimization.md) - Performance optimization guidelines
- [docs/numkong-performance.md](docs/numkong-performance.md) - NumKong vs OpenCV/NumPy/LUT baselines (benchmark tables; sum/mean/std)
- [docs/public-api.md](docs/public-api.md) - Star-exported routers vs `albucore.functions` shims
- [benchmarks/README.md](benchmarks/README.md) - Python micro-benchmarks (`uv run python benchmarks/…`)
- [docs/research/](docs/research/) - Research notes (extra benchmark writeups; see [`benchmarks/README.md`](benchmarks/README.md) for scripts)

## License

Albucore is publicly available under the [MIT License](LICENSE), including
contributions accepted under the
[Albucore Contributor License Agreement Version 1.0](CLA.md). The CLA does not
change the repository's public license. Historical contributions remain
available under MIT and become CLA-covered only through an applicable Version
1.0 Acceptance Record. See [CONTRIBUTING.md](CONTRIBUTING.md) for the individual
CLA Assistant and entity acceptance paths.

## Acknowledgements

Albucore provides core image-processing primitives for [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX). We'd like to thank all [AlbumentationsX contributors](https://albumentations.ai/people) and the broader computer vision community for their inspiration and support.

<!-- GitAds-Verify: 1LSAKH1Y2GKIISALRDIFCG2T9YYNR5WD -->
