# Exact FP16 and FP32 cast routing

Large C-contiguous NumPy FP16↔FP32 casts use NumKong on the measured AVX-512 region from 16,384 elements. Other CPU/layout/size regions retain NumPy. Contiguous macOS ARM Tensor casts use NumPy storage conversion from 61,440 elements, followed by zero-copy Tensor wrapping. Same-dtype calls retain identity.

Measured 2026-10-12 with one native thread, five warmups, and 51 balanced repetitions per candidate. Allocation, dispatch, output adaptation, and compatibility guards are included in public-call timings. Inputs are half-quantized before either cast direction; input Tensor preparation is excluded. The Xeon run shared its CPU with ongoing training.

| Environment | NumPy | NumKong | Torch | OpenCV |
|---|---:|---:|---:|---:|
| M4 Max, macOS 27.0.1, CPython 3.11.11 | 2.4.6 | 7.8.5 | 2.14.1 | 5.0.0 |
| Xeon model 85, GCP a2-highgpu-1g, CPython 3.13.16 | 2.5.2 | 7.8.5 | 2.14.1+cu130 | 5.0.0 |

## Complete crop casts

Shape `(1,32,384,384)`, C-contiguous; medians in milliseconds.

| CPU / container | Cast | Previous route | Selected public route | Speedup |
|---|---|---:|---:|---:|
| Xeon / NumPy | FP16→FP32 | 12.300 | 4.709 | 2.61× |
| Xeon / NumPy | FP32→FP16 | 23.893 | 4.290 | 5.57× |
| M4 / Tensor | FP16→FP32 | 0.445 | 0.224 | 1.99× |
| M4 / Tensor | FP32→FP16 | 0.460 | 0.264 | 1.74× |

The full cast matrix covers the 16,383/16,384 and 61,439/61,440 boundaries, canonical HWC C=1/3/9 sizes, volume shapes, and the MRI crop. The selected Xeon region contains 38 cells: public/NumPy median ratios are 0.091–0.383. The selected M4 Tensor region contains 32 cells: public/native-Torch ratios are 0.461–0.791.

## Complete image bridges

The baseline explicitly widens with NumPy, invokes the same float32 operation, and casts back with NumPy. Every compared half output is bitwise identical. `power_opencv(value=1.3)` is the explicit backend used by scalar power consumers.

| Xeon operation, 768×1024×9 | NumPy bridge ms | Routed bridge ms | Speedup |
|---|---:|---:|---:|
| resize_linear | 32.881 | 15.533 | 2.12× |
| median3 | 76.266 | 33.284 | 2.29× |
| normalize_image | 82.283 | 44.068 | 1.87× |
| exp | 73.869 | 30.912 | 2.39× |
| power_opencv | 84.471 | 36.123 | 2.34× |

M4 keeps NumPy for these bridges; all measured public/baseline median ratios remain close to one. Faster storage conversion does not establish a native-half kernel win.

## Semantics and rejected candidates

- NumKong writes into a fresh `np.empty` output. This retains writable independent storage and `OWNDATA`, which an adapted NumKong Tensor does not provide.
- Half finiteness uses integer exponent bits; `np.isfinite(float16)` erased most of the Xeon widening gain. The temporary mask is released before output allocation. Narrowing uses scalar min/max guards; no new full-size float32 work buffer is allocated.
- NaN/infinity, narrowing overflow, and non-default underflow policy retain NumPy casts. Tensor conversion retains Torch error behavior. No scaling, clipping, accumulator, or intermediate-rounding policy changes.
- Guarded Torch casts into owned NumPy output did not establish a sustained win over guarded NumKong in the Xeon matrix under the median/2-MAD screen. They remain benchmark candidates.
- Released NumKong 7.8.5 loses to NumPy on M4 even for contiguous input. Strided NumKong is excluded; NumPy preserves order K while NumKong returns C output, so packing must be included in a future comparison. NumKong [#404](https://github.com/ashvardanian/NumKong/issues/404) and [PR #405](https://github.com/ashvardanian/NumKong/pull/405) track that work.

Validation: 3,647 local tests pass, including a separate full run forcing the NumKong cast selector. The initial native AVX-512 run passed 33 regression tests and skipped one macOS-specific NaN-payload test. Tests cover finite binary16 encodings, FP32 rounding boundaries, signed zero, special values, `np.errstate`, shape, dtype, source mutation, ownership, readonly/strided inputs, Tensor storage, and fast-route selection. Strict typing and 45 router contracts pass.

## Reproduce

```bash
uv run --extra headless --extra torch python benchmarks/benchmark_float_cast.py \
  --threads 1 --repeats 51 --with-routers \
  --output-json benchmarks/results/float_cast_current.json
```

Raw samples, effective thread settings, capabilities, versions, and measured source hashes: [M4 casts](float_cast_m4.json), [Xeon casts](float_cast_xeon.json), [M4 bridges](float_bridge_m4.json), [Xeon bridges](float_bridge_xeon.json).
