---
name: albucore-conventions
description: Albucore image processing conventions - NumPy/Torch layouts, dtypes (uint8/float16/float32), benchmark-driven backend routing (OpenCV, NumPy, Torch CPU, LUT, NumKong), tests, and lockfile discipline. Use when implementing or modifying albucore modules, writing tests, or reviewing image-processing code.
---

# Albucore Conventions

For runtime implementation or review, also read `../performance-optimization/SKILL.md` and its canonical reference
completely before acting.

## When to Apply

- Implementing or changing code under `albucore/`.
- Adding or modifying tests.
- Reviewing image-processing code.
- Answering questions about shapes, dtypes, backend choice, or dependency lock consistency.

## Critical Rules

### 1. Image Shapes - Always Explicit Channel Dimension

```python
# Correct
grayscale = (H, W, 1)
rgb = (H, W, 3)
batch = (N, H, W, C)
volume = (D, H, W, C)
volume_batch = (N, D, H, W, C)

# Wrong - never use implicit channels
gray = (H, W)
gray_vol = (D, H, W)
```

Dimension indexing is always:

```python
num_channels = image.shape[-1]
width = image.shape[-2]
height = image.shape[-3]
```

NumPy images use `HWC`, `NHWC`, `DHWC`, or `NDHWC`. An API that accepts a `torch.Tensor` defines its layout
independently; never infer a Tensor layout from its rank. Common image operations use explicit `CHW`, `NCHW`,
`CDHW`, or `NCDHW` layout where channel semantics matter. `resize3d` declares NumPy `DHWC` and Torch `CDHW`.
`warp_affine3d` and `remap3d` also accept NumPy `NDHWC` and Torch `NCDHW` batches. Their shared matrix or grid
applies to every item. Other 3D routers accept one volume. AlbumentationsX checks CPU, layout, strides, eager
(`requires_grad=False`) execution, and control data before the call.

### 1a. Precondition Boundary - Do Not Revalidate in Albucore

Albucore routers receive prevalidated inputs from the caller. The upstream transform layer owns validation of:

- container type, rank, layout, and explicit channel dimension;
- dtype, device, contiguity, and autograd state;
- spatial sizes, control-data shapes, interpolation, border, and fill contracts.

Do not add duplicate runtime checks for these preconditions inside Albucore routers. Keep only the dispatch needed to
select the NumPy or Torch implementation and the normalization/conversion required by the selected kernel. A caller
contract violation is outside the low-level router contract; tests and benchmarks must use valid inputs. Document the
precondition boundary in public-router docstrings.

### 2. Supported Dtypes

Image routers support `uint8`, `float16`, and `float32`; image inputs do not include `float64`. Statistics may return
wide `float64` or integer results. `apply_uint8_lut` and `sz_lut` remain `uint8`-only. Callers reject unsupported
dtypes before dispatch; runtime kernels may rely on each router's documented dtype contract.

### 3. Backend Routing - Benchmark-Driven Only

- Do not assume LUT is fastest for uint8; benchmark.
- NumKong is used where benchmarks win (`blend`, `moments`, `scale`, `cdist`, etc.); see `docs/numkong-performance.md`.
- The runtime minimum is `numkong>=7.8.5`. Albucore reads `nk.get_capabilities()["neonhalf"]` once and uses NumKong float16 elementwise kernels only in measured regions: contiguous `add_weighted` with weights `0.5, 0.5`, and scalar `multiply_add` with half-exact controls, both above 20,480 elements. Other controls keep their measured NumPy/Torch/OpenCV routes. PR [#398](https://github.com/ashvardanian/NumKong/pull/398) remains an upstream widening improvement for broader controls; Issues [#395](https://github.com/ashvardanian/NumKong/issues/395), [#396](https://github.com/ashvardanian/NumKong/issues/396), and [#397](https://github.com/ashvardanian/NumKong/issues/397) track the related performance and precision work.
- NumPy HWC float16 `add_vector` and `multiply_by_vector` use `nk.scale` for contiguous writable C=1 inputs from 20,480 elements when the one-element control is exactly representable in float16. Multiplication uses a negative-zero bias to preserve signed-zero semantics. The selected region matched the float32 reference rounded to half across exhaustive binary16 tests and had no robust slow pair in the measured grid; other layouts, channel counts, sizes, and controls keep their current measured routes.
- NumPy HWC float16 `add_vector` and `multiply_by_vector` use direct OpenCV CV16F scalar operations for contiguous writable C=3 inputs from 61,440 elements when all three controls are exactly representable in float16. Exhaustive binary16 outputs match the float32-rounded-half reference; the full public 101-sample grid has no robust slow pair. OpenCV C1 scalar candidates were rejected for speed; see `docs/numkong-performance.md` and [#25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).
- NumKong float16 `moments` can lose subnormal contributions even with `neonhalf=true`. Albucore currently scans for subnormals, widens those inputs, and mirrors the float32 backend; near-constant data keeps a float64 fallback. This still leaves robust slow cells and is tracked in [NumKong #399](https://github.com/ashvardanian/NumKong/issues/399). PR [#400](https://github.com/ashvardanian/NumKong/pull/400) adds F64 NEON accumulation and an unaligned-byte fallback; its 3-byte, 1M-value case measured 0.812× the F16→F32 bridge. A temporary Albucore public-route probe using that wheel and bypassing only the subnormal scan measured all 32 HWC C=1/2/3/4 `std`/`mean_std` cells at 0.27–0.86× float32, with maximum float64-reference errors of 1.37e-10 for mean and 1.67e-8 for std. A separate positive-range `[0.1,0.9]` screen passed mean `rtol=1e-7` in 16 cells and stayed within the current `mean_std` test tolerances. Keep the current guard until PR #400 is merged and released; see [the consolidated benchmark](https://github.com/ashvardanian/NumKong/issues/399).
- HWC float16 per-channel `mean` remains on CV32F/OpenCV until NumKong PR #400 is released. A temporary direct-moments candidate with the subnormal scan bypassed measured C1 at 0.93–1.02× float32, C3 strided at 1.15–1.17×, and C9 at 4.85–5.00×; C9 `nk.sum(axis=(0,1))` was 4.82–4.93× and a C3 planar copy was 3.03–3.33×. All candidates met `rtol=1e-7`, but only C1 meets the speed gate. Track grouped axis-aware `moments` in [NumKong #311](https://github.com/ashvardanian/NumKong/issues/311#issuecomment-6112109242).
- For NumPy float16 C=9 `reduce_sum`, keep the measured Torch window and widened CV_64F reduction route explicit; the current C=9 public bridge is 1.21–1.30× float32 time, while tested direct Torch/NumKong/NumPy routes were slower. OpenCV rejects CV_16F input in `reduce`; see [#25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).
- Contiguous float16 Tensor `reduce_sum` at 16,384+ elements uses direct Torch `sum(dtype=float64)` for global, per-channel, and explicit axes. It preserves the wide output and avoids channel-last bridge costs.
- Contiguous NumPy float16 global `reduce_sum` at 20,480+ elements with `keepdims=False` uses Torch `sum(dtype=float64)` through a zero-copy view; the full 101-sample HWC/NHWC grid matched NumPy float64 bitwise and had no robust slow cell (0.968–1.106× float32 time).
- NumPy float16 per-channel `reduce_sum` uses zero-copy Torch float64 reducers for contiguous rank-4 channel-last C=1 up to 307,200 elements without `keepdims` and C=3 up to 245,760 elements; the tested N=1–4, 128×160/240×320 public calls are exact. C=3 is 4.4–5.3× faster and C=1 is 0.965–1.006× the float32 peer. Keep other rank-4 channels on their measured paths.
- NumPy rank-4 channel-last float16 per-channel `mean` uses zero-copy Torch float64 reductions: C=3 from 20,480 elements and C=1 from 20,480 through 8,000,000 elements. Both preserve the NumPy float64 result; C=1 above the cap retains its measured fallback.
- Contiguous NumPy float16 `warp_affine(..., INTER_NEAREST)` with the default zero border uses a bit-preserving integer view: CV_16U for C=1/3 and CV_8U bytes for C=9. Other border, stride, and interpolation regions keep the float32 route; see `docs/numkong-performance.md`.
- Contiguous NumPy float16 `warp_perspective(..., INTER_NEAREST)` with a float32 matrix and the default zero border uses the shared bit-preserving integer view: CV_16U for C=1/3 and CV_8U bytes for C=9. Other matrices, borders, strides, dst outputs, and interpolation regions keep the float32 route.
- Contiguous NumPy float16 `remap(..., INTER_NEAREST)` with contiguous float32 maps and the default zero border converts maps to fixed-point, then uses CV_16U for C=1/3 and CV_8U bytes for C=9. Direct float maps with a CV_16U single-channel source can change tie rounding, so the fixed-point map is used for every selected channel count. Other channels, borders, strides, dst outputs, and interpolation modes use the float32 route.
- NumPy float16 `median_blur` uses ordered CV_16U keys for HWC C=1, `ksize=5`, and at least 76,800 elements; NaNs and zero medians with mixed signed-zero inputs use the float32 route, and other channel/kernel regions keep existing routes. The bridge exposes the same path to CHW/NCHW Tensor calls. See the measured matrix and rejected multi-channel candidate in `docs/numkong-performance.md` and OpenCV [#30217](https://github.com/opencv/opencv/issues/30217).
- For NumPy HWC float16 C=1, `add_vector` and `multiply_by_vector` use the scalar Torch path only when the one-element control is exactly representable in float16; nonrepresentable controls retain the float32-working route.
- Global float16 Tensor `mean` uses a float32 Torch accumulator and returns float64 for C>1 at 40,960 elements or larger. Global Tensor `std` at 40,960+ elements uses a float32 NumPy accumulator only for contiguous input within [-1, 1] with std>1e-3, and a two-pass Torch float64 fallback outside that region; non-contiguous inputs retain the wide NumPy route. Global Tensor `mean_std` may use zero-copy scalar-view CV16F `meanStdDev` at 20,480 elements or larger, with wide fallback for near-constant/non-finite inputs. PyTorch CPU half `std` cannot select an output dtype; see [#97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Contiguous float16 Tensor `mean(..., axis="per_channel")` at 40,960+ elements uses a Torch float64 accumulator over every axis except the declared channel axis for batch and volume layouts and multi-channel CHW. Single-channel CHW keeps its faster OpenCV route. Preserve `keepdims` layout and compare to a float64 reference.
- Global contiguous NumPy float16 `mean` at 4,096+ elements widens once to float32 before OpenCV `mean`; direct CV16F mean can lose low-order subnormal contributions. The complete 21-cell HWC/NHWC matrix matched the float64 reference and ran at 0.696–0.876× float32 time. Small or non-contiguous inputs keep the NumPy float64 reduction. See [OpenCV #25019](https://github.com/opencv/opencv/issues/25019#issuecomment-6112519769).
- For contiguous float16 Tensor `from_float(..., uint8)`, half multiply/round/clip is measured only for scales exactly representable in float16; exhaustive binary16 inputs showed at most one uint8-unit difference from float32. Unrepresentable scales use float32 rounding.
- StringZilla is a candidate for uint8 translation paths; compare it with the public LUT router and OpenCV.
- OpenCV has a 4-channel limit for many ops. Use `MAX_OPENCV_WORKING_CHANNELS`, then fall back to NumPy or chunking for more channels.
- Route from benchmark evidence in `benchmarks/` and `docs/numkong-performance.md`, not convention.

### 4. Torch CPU Is a Mandatory Backend

- `torch>=2.14.1` is a required runtime dependency, not an optional import. Use direct imports rather than
  `sys.modules` checks, class-name heuristics, or lazy imports.
- Training callers are expected to have imported Torch already. Do not optimize public CPU routing around deferred
  import cost.
- A public Tensor path must state its layout. For caller-prevalidated routers such as `resize3d`, `warp_affine3d`, and
  `remap3d`, AlbumentationsX owns CPU, strided-layout, and `requires_grad=False` validation. These routers do not
  silently detach or move data.
- For float16 Tensor `resize(..., INTER_LINEAR)` with C=1, the current float32/OpenCV bridge remains slower than float32 (1.08–2.10× across a 16-cell CHW/NCHW grid). Direct Torch half and Torch float32-round-trip candidates were slower in every cell; do not add either route without a full-path win. Track the capability gap in OpenCV [#30216](https://github.com/opencv/opencv/issues/30216) and PyTorch [#97068](https://github.com/pytorch/pytorch/issues/97068#issuecomment-6111732503).
- Benchmark NumPy-to-Torch routes end-to-end: wrapper creation, permutations, dtype casts, kernel execution, and
  returned NumPy layout all belong inside the timed region.
- For `resize3d`, benchmark direct Tensor and zero-copy Tensor→NumPy→Tensor routes separately. A linear all-axis
  upscale may select the bridge for speed; preserve its documented float32 tolerance and uint8 delta bound.
- For `warp_affine3d`, benchmark rank-4 single volumes and rank-5 batches separately. Include matrix conversion,
  shared affine-grid creation, sampling, fill correction, uint8 conversion, NumPy/Torch views, and output materialization.
  Compare per-volume calls, `N × C` folding, native batch sampling, and Tensor/NumPy bridges on the public path.

### 5. OpenCV LUT - Source vs Table Dtype

- `cv2.LUT` source image: for uint8 LUT paths, pass a uint8 `(H, W, C)` image.
- `cv2.LUT` lookup table: for float outputs, the table must be float32, not float64. OpenCV stats often promote to float64; cast the small LUT to float32 so the output does not widen.

### 6. Floating-Point Working Dtypes

Choose input, working, accumulator, and output dtypes per operation. Float16 outputs preserve float16 input where
documented; float32 working buffers are allowed when needed for precision or backend support. Include conversion and
output restoration in the public-path benchmark. Do not widen full arrays by default when a faster measured kernel can
write the final half output directly.
- On OpenCV 5.0.0.93, direct `CV_16F mean`/`meanStdDev` can lose precision on small, skewed, subnormal, and constant
  HWC inputs. NumPy HWC per-channel float16 C≤4 widens to CV_32F; high-channel routes use widened OpenCV or NumKong.
  Global CV16F routes remain only in measured regions with wide near-constant/non-finite fallbacks. Track accuracy and
  performance work in [OpenCV #25019](https://github.com/opencv/opencv/issues/25019).

### 7. Utilities and Decorators

- Utilities: `get_num_channels`, `convert_value`, `clip`, etc. in `albucore.utils`.
- Decorators: `@preserve_channel_dim`, `@clipped`, etc. in `albucore.decorators`.

### 8. Tests

- Test uint8, float16, and float32 for each supported image router; keep uint8 LUT APIs uint8-only.
- Test NumPy channel-last and declared CPU Tensor layouts, plus image batches and volumes where supported.
- Cover 1-channel and >4-channel edge cases.

### 9. Dependency Lock Consistency

- When changing dependencies in `pyproject.toml`, update `uv.lock` in the same PR.
- Validate with `uv lock --check`.
- Release flow uses `uv export --frozen`; stale `uv.lock` can break release artifact generation.

## Quick Reference

| Convention | Rule |
|------------|------|
| Grayscale shape | `(H, W, 1)` never `(H, W)` |
| Image dtypes | uint8, float16, float32; LUT-only APIs use uint8 |
| Backend | Choose by benchmark |
| LUT | uint8 image in; float32 LUT table when output is float32 |
| Working dtype | Select per operation; benchmark conversions and accumulators |
| OpenCV limit | 4 channels unless chunked or using NumPy |
| Torch Tensor route | Explicit layout; AlbumentationsX prevalidates CPU/strides/autograd before `warp_affine3d` |
| Benchmarks | `benchmarks/` and `docs/numkong-performance.md` |
| Lockfile | Keep `uv.lock` in sync with `pyproject.toml` |
