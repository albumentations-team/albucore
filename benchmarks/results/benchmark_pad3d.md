# Constant 3D padding: measured routing

`pad3d` selects complete NumPy-to-NumPy and Tensor-to-Tensor paths by measured dtype, size, and storage layout. For [issue #159](https://github.com/albumentations-team/albucore/issues/159), the laptop measurements select both NumPy allocation/fill/copy and Torch constant padding in different regions. `torch.full` plus insertion does not establish a separate winning region over `F.pad`.

Every candidate and final route matched the NumPy reference exactly in shape, dtype, and values; inputs remained unchanged. No final case was more than 15% slower than the baseline. These are Apple M4 Max results; Linux x86 routing performance was not measured.

| Main input matrix | Cases | Median baseline / public | Faster by >5% | Within 5% |
|---|---:|---:|---:|---:|
| NumPy, C contiguous | 36 | 1.47x | 34 | 2 |
| NumPy, positive row strides | 36 | 1.55x | 36 | 0 |
| Tensor, contiguous CDHW | 36 | 4.99x | 36 | 0 |
| Tensor, interleaved channels | 36 | 1.57x | 34 | 2 |

The Tensor baseline reproduces the AlbumentationsX fallback: CDHW Tensor → DHWC NumPy view → `np.pad` → CDHW Tensor view. The bridge shares input storage. A different output storage layout can make the padding copy expensive even when the wrappers themselves do not copy.

Fortran-contiguous NumPy controls preserve F order through `np.full(order="F")`. At `(32,128,160,C)`, C=1/3, the final route was 2.61–2.66x faster than `np.pad` for uint8 and 1.56x for float32.

## Selected routes

All sizes below count input elements, including channels. NumPy-to-Torch wrapping requires a writable, non-Fortran array with nonnegative strides divisible by the element size. Other arrays stay in NumPy.

| Input and scalar-fill region | Complete route |
|---|---|
| NumPy uint8/float32, C=1, at least 524,288 elements | DHWC NumPy → Torch view → `F.pad` with zero channel padding → NumPy view |
| NumPy uint8, non-contiguous, at least 1,048,576 elements | Same Torch bridge |
| NumPy float32, non-contiguous, 524,288–4,194,304 elements | Same Torch bridge |
| Other NumPy, including int16 masks and Fortran arrays | `np.full` in C or F order, followed by one interior assignment |
| Tensor with C>1 and channel stride 1 | DHWC NumPy view → `np.full` plus insertion → CDHW Tensor view |
| Other Tensor | Native CDHW `F.pad` under `torch.no_grad()` |

All-zero padding returns the original object before dispatch. Tuple fills retain `np.pad` before/after semantics, including corner precedence; Tensor tuples use the NumPy bridge. Integer fill normalization operates on a scalar, preserving uint8 wrapping and NumPy truncation without converting the volume. The caller adds/removes singleton channels for DHW data.

## Measurement and alternatives

Environment: Apple M4 Max, 128 GiB RAM, macOS-26.4.1-arm64-arm-64bit, Python 3.10.16, NumPy 2.2.6, Torch 2.13.0. Measured September 6, 2026. Torch intra-op/inter-op threads were both 1. OMP, OpenBLAS, MKL, Accelerate, and NumExpr limits were set to 1 before imports. OpenCV reported 16 threads after its thread-setting call; these padding paths do not use OpenCV.

The candidate sweep has 215 cases; final verification adds four Fortran controls for 219. The main 144 cases combine canonical non-square volumes and the issue grid, C=1/3/5/9, uint8/float32, both containers, and contiguous/strided storage. Controls cover int16, asymmetric padding, tuple fills, identity, unit axes, negative/read-only inputs, sliced Tensors, and 40 non-square cases around routing thresholds. JSON records every shape, stride, fill, padding, calibrated iteration count, and raw timing sample.

Each candidate receives three warmup calls and three calibration calls, then five rounds in alternating forward/reverse order. Iterations are calibrated per candidate to about 30 ms per round, with at least three calls. Timings include dispatch, scalar casts, wrappers, allocation, padding, and output conversion. Input generation, correctness checks, sampling, Compose, and downstream training are excluded.

Compared candidates: `np.pad`; NumPy full allocation and insertion; NumPy border-only writes; complete `F.pad` and `torch.full` bridges; native Tensor padding; a 5D Tensor view; DHWC Tensor padding; NumPy full allocation in CDHW; and Torch border-only writes. The full-allocation candidates write the interior twice. Border-only writes save that pass but lost across most measured regions; 5D views did not fix interleaved-channel copy costs. Native CDHW padding and DHWC padding avoid opposite layout penalties, which motivates storage-based routing. Isolated microsecond wins in tiny cases did not justify another backend and size threshold.

Padding performs vectorized fill and copy; label reductions, `bincount`, LUTs, and random generation do not apply. OpenCV, NumKong, and StringZilla provide no directly applicable single-call spatial-padding candidate for this contract. Growing the output prevents safe in-place mutation of caller-owned storage. AlbumentationsX call sites remain `Pad3D`, `PadIfNeeded3D`, and its crop-and-pad helper; this change contains the Albucore kernel only.

For float32 `(64,96,128,3)` with padding `(4,4,8,8,12,12)`, the output occupies 14,708,736 bytes. NumPy allocation tracking measured a 14,710,688-byte peak during the call, only 1,952 bytes above the output. A native Tensor profiler capture recorded one `aten::empty` allocation of 14,708,736 bytes, one fill, and one copy. Eligible bridges add views, and every allocating route retains the input dtype. These allocation observations do not measure whole-process peak RSS.

Validation covers fill casting, tuple corners, stride compatibility, routing boundaries, input ownership, output lifetime, and backward through a subsequent trainable convolution. The full test suite, pre-commit hooks, mypy, router contract manifest, and CI matrix check passed for this change.

## Reproduce

From the checkout with its NumPy/OpenCV/Torch dependencies installed:

```bash
python benchmarks/benchmark_pad3d.py --output benchmarks/results/benchmark_pad3d_candidates.json
python benchmarks/benchmark_pad3d.py --router-only --output benchmarks/results/benchmark_pad3d.json
```

Use `--quick` while iterating or `--container numpy` / `--container tensor` to isolate a container. The current script includes all 219 cases in a full run. Retained evidence: [candidate timings](benchmark_pad3d_candidates.json) and [final public-route timings](benchmark_pad3d.json).
