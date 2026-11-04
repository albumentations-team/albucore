# Albucore: High-Performance Image Processing Functions

[![PyPI version](https://img.shields.io/pypi/v/albucore.svg)](https://pypi.org/project/albucore/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/albucore.svg)](https://pypi.org/project/albucore/)
[![CI](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml/badge.svg)](https://github.com/albumentations-team/albucore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Albucore is a library of optimized atomic functions designed for efficient image processing. These functions serve as the foundation for [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX), an image augmentation library.

## Citing AlbumentationsX

If you use Albucore as part of an AlbumentationsX pipeline, please cite
[AlbumentationsX: One Augmentation Pipeline for Images and Related Annotations](https://arxiv.org/abs/2608.11123).
Your citation makes the project's research impact visible to funders and helps sustain maintenance.

```bibtex
@article{iglovikov2026albumentationsx,
    title = {AlbumentationsX: One Augmentation Pipeline for Images and Related Annotations},
    author = {Iglovikov, Vladimir},
    journal = {arXiv preprint arXiv:2608.11123},
    year = {2026},
    doi = {10.48550/arXiv.2608.11123},
    url = {https://arxiv.org/abs/2608.11123}
}
```

## Overview

Image processing operations can be implemented in several ways, with performance depending on dtype, size, layout, and channel count. Albucore routes each operation to a benchmark-selected NumPy, OpenCV, NumKong, StringZilla, or eligible CPU PyTorch implementation.

Image routers accept `uint8`, `float16`, and `float32`; float64 image inputs are not supported. NumPy uses channel-last layouts, while eager CPU Torch calls use explicit channel-first layouts where channel semantics matter. The specialized uint8 LUT operation is uint8-only.

## Installation

**Requires Python 3.11+.** CI covers Python 3.11, 3.12, 3.13, and 3.14. Choose and install the PyTorch build for your CPU,
CUDA, or MPS environment first. Then install Albucore with an OpenCV extra. For a Linux CPU-only headless application:

```bash
pip install "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu
pip install "albucore[headless,torch]"
```

**CUDA or macOS (MPS):** Select and install the matching PyTorch build with the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then install Albucore:

```bash
pip install "albucore[headless,torch]"
```

`pip install albucore` installs only the base dependency set. Use it when a transitive consumer only resolves
Albucore, such as during a documentation build. The current public import graph requires both OpenCV and PyTorch.

The `torch` extra declares Albucore's PyTorch runtime requirement but cannot select a CPU, CUDA, or MPS wheel through
standard package metadata. Use PyTorch's platform-specific installation command before installing Albucore. The examples
use the `headless` OpenCV extra. When Albucore installs OpenCV, choose exactly one extra: `headless`, `gui`,
`contrib`, or `contrib-headless`. Use `albucore[torch]` when reusing an existing OpenCV installation.

AlbumentationsX passes prevalidated CPU, strided Torch tensors with `requires_grad=False` to
`resize3d` and `warp_affine3d`; the low-level routers do not repeat those checks or move/detach Tensor data.
`warp_affine3d` and `remap3d` also accept rank-5 volume batches with one shared affine matrix or pull grid.

**With OpenCV GUI support** (for local development with cv2.imshow):

```bash
pip install "albucore[gui,torch]"
```

**With OpenCV contrib modules:**

```bash
pip install "albucore[contrib,torch]"              # GUI version
pip install "albucore[contrib-headless,torch]"     # Headless version
```

**Note:** If you already have one of the supported OpenCV distributions installed, use `pip install "albucore[torch]"`
after installing the platform-specific PyTorch build. This does not add another OpenCV package; Albucore uses the
existing installation.

## Usage

```python
import numpy as np
import albucore

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
result = albucore.multiply(image, 1.5)

# Grayscale images keep an explicit channel dimension
gray_image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
gray_result = albucore.multiply(gray_image, 1.5)
```

## Shape Conventions

The channel dimension is always present, even for grayscale (`(H, W, 1)`, never `(H, W)`). NumPy layouts are
channel-last:

- **Single image**: `(H, W, C)`
- **Batch of images**: `(N, H, W, C)`
- **3D volume**: `(D, H, W, C)`
- **Volume batch**: `(N, D, H, W, C)`, accepted only by `warp_affine3d` and `remap3d`

CPU Torch uses `CHW`, `NCHW`, `CDHW`, or `NCDHW`; common operations that need a channel axis take an explicit
`layout=` keyword. Spatial routers such as `resize` and `resize3d` define their layout by the operation. See
[docs/image-conventions.md](docs/image-conventions.md) for details.

## Functions

The tables below highlight public routers exported directly by `albucore`. Their implementations live in the
operation modules listed in [docs/public-api.md](docs/public-api.md). Image routers support `uint8`, `float16`, and
`float32`, except where a table states a narrower dtype contract.

### Arithmetic

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `clip` | `(img, dtype=None, *, out=None)` | Saturate to the selected dtype range; dtype follows `out` or input when omitted | NumPy/Torch clipping; writes only to explicit `out` |
| `multiply` | `(img, value, *, out=None, layout=None)` | Floating results preserve input dtype; uint8 saturates | uint8 scalar/vector → LUT; float16 HWC C=1 uses NumKong `scale` with `neonhalf=true` for exact half controls from 20,480 elements; C=3 uses OpenCV CV16F for exact half controls from 61,440 elements; other float16/float32 paths are benchmark-selected |
| `add` | `(img, value, *, out=None, layout=None)` | Floating results preserve input dtype; uint8 saturates | uint8 uses OpenCV/LUT/NumKong; float16 HWC C=1 uses NumKong `scale` with `neonhalf=true` for exact half controls from 20,480 elements; C=3 uses OpenCV CV16F for exact half controls from 61,440 elements; other float16/float32 paths are benchmark-selected |
| `power` | `(img, exponent, *, out=None, layout=None)` | Floating results preserve input dtype; uint8 saturates | uint8 → LUT; float32 scalar → `cv2.pow`; float16 scalar widens for OpenCV then restores half |
| `add_weighted` | `(img1, weight1, img2, weight2, *, out=None, layout=None)` | Floating results preserve input dtype; uint8 saturates | NumKong is used for selected uint8/float32 regions; float16 uses a benchmark-selected CPU route |
| `multiply_add` | `(img, factor, value, *, out=None, layout=None)` | Floating results preserve input dtype; uint8 saturates | uint8 → LUT; float32 scalar → NumKong `scale`; float16 and other controls use measured NumPy/Torch paths |

`value` / `factor` / `exponent` can be a scalar, a length-`C` 1-D array (per-channel), or a
full image-shaped array.

These arithmetic routers do not impose an image-range convention on floating-point data. Without `out`, they do not
write input storage; a supplied destination is returned by identity and may alias only where the operation allows it.
Tensor calls that depend on channel semantics require `layout=`. All operations express dtype conversion, clipping,
layout, and output ownership in their own signatures or bodies.

### Elementwise math

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `exp` | `(array, *, out=None)` | Elementwise exponential; float16/float32 preserve dtype, uint8 returns float32 | NumPy, OpenCV, or CPU Torch by dtype and measured region |
| `log` | `(array, *, out=None)` | NumPy-compatible natural logarithm; float16/float32 preserve dtype, uint8 returns float32 | NumPy special values; guarded OpenCV or Torch paths for eligible regions |
| `sqrt` | `(array, *, out=None)` | NumPy-compatible square root; float16/float32 preserve dtype, uint8 returns float32 | NumPy or CPU Torch by dtype and measured region |

These functions preserve the input shape. Without `out`, they leave input data unchanged. Supply `out=destination`
to reuse a writable result buffer; floating inputs may use `out=array`, including supported views. Uint8 input
requires a float32 destination. AX owns any protective copy needed before writing shared storage. See the [elementwise benchmark report](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip) for routing thresholds, environment, and NumKong results.

### Normalization

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `normalize` | `(img, mean, denominator, *, out=None, layout=None)` | Float16 input returns float16; uint8/float32 return float32 | uint8 → LUT; float16/float32 use benchmark-selected CPU paths. Caller-supplied constants (e.g. ImageNet stats). |
| `normalize_per_image` | `(img, normalization, *, out=None, layout=None)` | Float16 input returns float16; uint8/float32 return float32 | uint8 → LUT; floating inputs use OpenCV/NumPy working paths. `normalization ∈ {"image", "image_per_channel", "min_max", "min_max_per_channel"}` |

`normalize` is for **fixed** per-channel constants (ImageNet-style).
`normalize_per_image` **estimates** stats from the image at call time.

### Statistics

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `mean` | `(arr, axis=None, *, keepdims=False, dtype=None, layout=None)` | Population mean, wide output | uint8 and floating inputs use NumKong, OpenCV, Torch, or NumPy by region |
| `std` | `(arr, axis=None, *, keepdims=False, eps=1e-4, dtype=None, layout=None)` | Population std + eps, wide output | NumKong moments, OpenCV, Torch, or NumPy by region |
| `mean_std` | `(arr, axis=None, *, keepdims=False, eps=1e-4, layout=None)` | Mean and std+eps jointly | Wide-accumulator routes selected by input dtype and layout |
| `reduce_sum` | `(arr, axis=None, *, keepdims=False, layout=None)` | Sum with wide accumulator | uint8 → uint64; floating inputs → float64 with Torch, NumKong, or NumPy by region |

`axis` accepts `None`/`"global"` (scalar), `"per_channel"` (shape `(C,)`), or any NumPy-style
`int`/`tuple[int, ...]`.

### LUT (lookup tables)

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `apply_uint8_lut` | `(img, lut, *, out=None, layout=None)` | Apply uint8→uint8 LUT; `lut` shape `(256,)` or `(256, 1, C)` | Shared `(256,)`: StringZilla byte translation. Per-channel `(256, 1, C)`: single `cv2.LUT` with `(256,1,C)` table on contiguous HWC; else StringZilla per channel |

### Geometric / spatial

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `hflip` | `(img)` | Mirror left-right | NumPy slice view for every channel count; `hflip_cv2` remains the explicit materializing backend |
| `vflip` | `(img)` | Mirror top-bottom | NumPy slice view for every channel count; `vflip_cv2` remains the explicit materializing backend |
| `median_blur` | `(img, ksize, *, layout=None)` | Median filter (odd ksize ≥ 3) | uint8 → direct/chunked OpenCV; float16/float32 use the documented OpenCV working paths |
| `pad3d` | `(volume, padding, value=0)` | Add constant borders to uint8/float16/float32 volumes or int16 masks | NumPy fill/copy or CPU Torch `F.pad`; preserves NumPy `DHWC` or Tensor `CDHW` |
| `gaussian_blur3d` | `(volume, sigma, kernel_size=0)` | Blur one volume along depth, height, and width | One NumPy `DHWC` or CPU Torch `CDHW` volume; float32 working passes where needed, then restore the output dtype |
| `separable_filter3d` | `(volume, kernels)` | Apply three D/H/W kernels to one volume | One NumPy `DHWC` or CPU Torch `CDHW` volume; float32 working passes where needed, then restore the output dtype |
| `warp_affine3d` | `(volume, matrix, size, interpolation, border_mode, border_value)` | Apply one forward 3D affine matrix to a volume or batch | NumPy `DHWC`/`NDHWC` or CPU Torch `CDHW`/`NCDHW`; one matrix is shared across the batch; native Torch `affine_grid` + `grid_sample` |
| `remap3d` | `(volume, sampling_grid, interpolation, border_mode, border_value)` | Sample a volume or batch through one shared normalized 3D pull grid | NumPy `DHWC`/`NDHWC` or CPU Torch `CDHW`/`NCDHW`; grid layout `(D_out, H_out, W_out, 3)` |
| `matmul` | `(a, b)` | Matrix multiply (`a @ b`) | NumPy `@` (BLAS-backed); float16 uses float32 accumulation and casts back |
| `pairwise_distances_squared` | `(points1, points2)` | Squared Euclidean distance matrix `(N, M)` | Small (N*M < 1000) → NumKong `cdist`; large → NumPy vectorized `‖a‖²+‖b‖²−2(a·b)` |

The package also star-exports `copy_make_border`, `gaussian_blur3d`, `pad3d`, `remap`, `remap3d`, `resize`, `resize3d`, `separable_filter3d`, `warp_affine`, `warp_affine3d`, and `warp_perspective`; see [docs/public-api.md](docs/public-api.md) and their docstrings for complete signatures. `gaussian_blur3d`, `pad3d`, `separable_filter3d`, and `resize3d` expect one NumPy `DHWC` volume or CPU Torch `CDHW` tensor per call. `warp_affine3d` and `remap3d` accept those single-volume layouts or NumPy `NDHWC` and CPU Torch `NCDHW` batches.

### Type conversion

| Function | Signature | What it does | How it works |
|---|---|---|---|
| `convert_dtype` | `(img, dtype, *, max_value=None, out=None)` | Convert among uint8, float16, and float32; uint8↔float scales by 255 by default, float↔float casts, same dtype returns by identity | Uint8→float uses a result-dtype LUT; floating casts and uint8 quantization retain measured NumPy/NumKong/Torch routes |

## Performance

1. **Measured routing**: Each function may have several NumPy, OpenCV, NumKong, StringZilla, or CPU PyTorch implementations. The router picks one from dtype, size, memory layout, channel count, and semantic constraints, using thresholds from repeatable benchmarks.
2. **NumKong**: SIMD `blend` for uint8 and single-channel float32 `add_weighted`, plus same-shaped uint8 `add(..., field)`; `cdist` for small `pairwise_distances_squared`; wide-accumulator `moments` for selected statistics routes (see [docs/numkong-performance.md](docs/numkong-performance.md)).
3. **CPU PyTorch**: Selected eager 3D volume operations use full-path Torch routes after benchmarks include the NumPy/Tensor bridge, thread settings, and output repair (see [docs/torch-performance-optimization.md](docs/torch-performance-optimization.md)).

Micro-benchmarks vs NumPy/OpenCV/NumKong: see [benchmarks/README.md](benchmarks/README.md). Run `uv run python benchmarks/benchmark_elementwise.py` for `exp`/`log`/`sqrt`, or `uv run python benchmarks/benchmark_numkong.py` for a smaller NumKong sweep.

See [docs/performance-optimization.md](docs/performance-optimization.md) for the general workflow and [docs/torch-performance-optimization.md](docs/torch-performance-optimization.md) for eager CPU Torch routes.

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Pull request process and CLA acceptance paths
- [AGENTS.md](AGENTS.md) - AI development guidelines for working with this codebase
- [docs/image-conventions.md](docs/image-conventions.md) - Image shape conventions and requirements
- [docs/performance-optimization.md](docs/performance-optimization.md) - Performance optimization guidelines
- [docs/torch-performance-optimization.md](docs/torch-performance-optimization.md) - Eager CPU PyTorch routing and benchmark guidance
- [docs/numkong-performance.md](docs/numkong-performance.md) - Current NumKong routes and benchmark decisions
- [docs/public-api.md](docs/public-api.md) - Public operations and their owning modules
- [benchmarks/README.md](benchmarks/README.md) - Python micro-benchmarks (`uv run python benchmarks/…`)

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
