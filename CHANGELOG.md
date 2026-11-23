# Changelog

## Unreleased

This release speeds up many uint8 NumPy and Tensor routes, fixes scalar `add` overflow for C>4, and removes
public helpers, operations and switches that AlbumentationsX does not use. Results stay bitwise identical except
uint8 per-channel `mean_std`/`std` with `keepdims=True`, which now matches the `keepdims=False` result.

### Performance

Speedups are single-thread public calls, macOS M4 Max / GCP Cascade Lake medians:

- `multiply(uint8, uint8_field)`: one saturating OpenCV byte multiply, 37× / 13×.
- `multiply(uint8, float32_field)`: no float32 widening of the image, 1.3× / 1.1×.
- `add(uint8, scalar)`: one flat saturating OpenCV call for every channel count; negative offsets 19× / 16×.
- `exp(uint8)`: 256-entry float32 table, 18× / 2.1×.
- `matmul(uint8, uint8)`: exact float BLAS, 36× / 15×.
- `copy_make_border` with C>4 per-channel values: 18–29× / 12–19×.
- `median_blur(uint8)`: direct row crops, per-plane C>4 for k≥7, per-plane uint8 Tensors (7.8× / 8.9× at k=3).
- uint8 Tensors with scalar controls: `add`, `multiply`, `power`, `multiply_add` 8–20× / 7.5–12.5×;
  `normalize` 3.5× / 4.8×; `add_weighted` 9.3× / 24×.
- uint8 per-channel `mean_std`/`std` with `keepdims=True`: 25× / 67×.
- `box_blur(uint8)`: no source copy, 1.12× / 1.11×.

### Fixed

- `add(uint8 image with C>4, value > 255)` raised `OverflowError`; it now saturates like other channel counts.

### Removed

- Operations: `log` and the public `separable_filter3d`; `gaussian_blur3d` keeps the same filtering.
- Helpers: `TWO`, `NPDTYPE_TO_OPENCV_DTYPE`, `get_opencv_dtype_from_numpy`, `get_image_data`,
  `ops_misc.hflip_cv2`/`vflip_cv2`/`hflip_numpy`/`vflip_numpy`, and `torch_backend.TORCH_CPU_BACKEND_ENABLED`.
- Root exports only (still importable from their modules): `convert_value`, `MAX_OPENCV_WORKING_CHANNELS` and
  `NUM_RGB_CHANNELS` from `albucore.utils`; `ImageInput`, `ValueType`, `SupportedDType`, `NormalizationType` and
  `TensorImageLayout` from `albucore._types`.
- Unsupported inputs: float64 volumes are no longer converted by the 3D filter; `matmul` and
  `pairwise_distances_squared` take operands of one container.

### Migration

Replace `albucore.log(x)` with `np.log(x)` or `torch.log(x)`. Replace `hflip_cv2(img)` with
`np.ascontiguousarray(hflip(img))`. Import moved names from their owning module, for example
`from albucore.utils import convert_value`.

## 0.3.4 — 2026-11-04

This release reduces temporary memory in float16-to-float32 conversion and speeds up raw uint8-to-float32
widening on the measured Linux x86-64 path. It also removes the public `ImageType` typing alias: applications
that import it must update their annotations before upgrading.

### Migration: replace `ImageType` imports

`ImageType` is removed from both `albucore` and `albucore._types`. Replace it with `NDArray` annotations
that describe the dtypes your code accepts. For example, replace this float32 image annotation:

```python
from albucore import ImageType

image: ImageType
```

with:

```python
import numpy as np
from numpy.typing import NDArray

image: NDArray[np.float32]
```

For code accepting all three supported NumPy dtypes, use
`NDArray[np.uint8] | NDArray[np.float16] | NDArray[np.float32]`. Array shape and layout follow each operation's
contract; these annotations specify storage dtype. This change preserves the concrete dtype and container
relationships of public operation results.

Python and dependency minimums are unchanged from 0.3.3. To update an existing installation with its chosen
OpenCV and PyTorch builds:

```bash
python -m pip install --upgrade albucore==0.3.4
```

### Lower memory use when widening float16 arrays

The finiteness check used by the selected NumKong x86 cast path now reduces integer views of the input instead
of creating an image-sized mask. In the measured matrix, its largest temporary fell from **13.5 MiB to less
than 2 KiB**. This saves working memory even when the caller reuses a float32 destination through `out=`.
An allocating conversion still needs storage for the float32 result. Conversion values, special-value
handling, and NumPy floating-point error behavior are preserved.

### Faster raw uint8-to-float32 conversion

For contiguous NumPy input, `convert_dtype(image, np.float32, max_value=1)` now casts directly, avoiding lookup-table
construction and application. This is useful when a downstream operation expects floating-point pixel values
on the original 0–255 scale:

```python
import numpy as np
from albucore import convert_dtype

pixels = np.array([[[0], [128], [255]]], dtype=np.uint8)
raw = convert_dtype(pixels, np.float32, max_value=1)
assert raw.tolist() == [[[0.0], [128.0], [255.0]]]
```

The default `max_value=255` produces normalized floating-point intensities. Allocating calls preserve the source;
calls with `out=destination` return that destination. Other input layouts and scales retain their existing routes.

Representative public-call timings for **480 × 640 × 3** images on a **2.80 GHz Intel Xeon with AVX-512**,
using Python 3.11 and one native thread:

| Conversion | 0.3.3 | 0.3.4 |
| --- | ---: | ---: |
| float16 → float32, allocate result | 0.478 ms | 0.379 ms |
| float16 → float32, reuse `out` | 0.483 ms | 0.372 ms |
| uint8 → float32, `max_value=1`, allocate result | 0.452 ms | 0.185 ms |

The half-cast rows use mixed positive and negative finite values. The complete matrix covers four image sizes
from 128 × 160 to 768 × 1024, C=1/3/9, and allocating and explicit-output calls. Outputs match bitwise.
Apple M4 Max public-call timings were essentially unchanged; no measured public cell on either platform slowed
by more than 5%.
See the [complete benchmark and memory results](https://github.com/albumentations-team/albucore/releases/download/0.3.4/albucore-benchmark-data-20261104.zip)
for all cells, dependency versions, and reproduction commands.

[Implementation and tests: PR #211](https://github.com/albumentations-team/albucore/pull/211) ·
[Full changes since 0.3.3](https://github.com/albumentations-team/albucore/compare/0.3.3...0.3.4)

## Earlier 0.3.x changes

The following changes were already available in 0.3.3.

- Replace implicit mutation flags with explicit `out=` buffers across array operations. Tensor calls require `layout=`;
  calls without `out` leave input storage unchanged, though identity and view returns are allowed.
- Add float16 image support across NumPy and eager CPU Torch routers, preserving documented output dtype and
  wide-statistics policies. Specialized uint8 LUT APIs remain uint8-only.
- Write shared NumPy uint8 LUT results directly into separate contiguous outputs with StringZilla.
- Raise the NumKong and StringZilla minimums to 7.8.7 and 5.3.2, respectively; keep the Torch minimum at 2.14.1.
- Fix large uint8 statistics and adaptive normalization through NumKong 7.8.7's moments overflow correction.
- Require Python 3.11 or newer. CI covers Python 3.11, 3.12, 3.13, and 3.14.
