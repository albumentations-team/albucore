# NumKong 7.8.7 integration

Albucore requires NumKong 7.8.7 to fix the uint8 moments overflow reported in
[Albucore #196](https://github.com/albumentations-team/albucore/issues/196).
The [release](https://github.com/ashvardanian/NumKong/releases/tag/v7.8.7)
contains the following changes from 7.8.6.

| Released change | Effect on Albucore |
|---|---|
| Bound SIMD moments accumulators before widening; fixes NumKong #412 | Corrects large uint8 `mean_std`, `std`, and adaptive normalization. No local replacement kernel is needed. |
| Widen float16 elementwise arithmetic; resolves #395/#396 and supersedes PR #398 | `scale` and `blend` now need baseline NEON. Their Albucore gates use `neon`; float16 sums and moments retain `neonhalf`. |
| Preserve signed zero in multiplication and default `scale`; fixes #401 | Remove the explicit negative-zero bias from single-channel multiplication. Retain C9 OpenCV routing for performance. |
| Avoid reading NaN/Inf output storage as a multiplication addend | Albucore has no direct `nk.multiply` production call or output-zeroing workaround to delete. |
| Fuse elementwise weights | Keep the existing coefficient regions: general coefficients change current rounding and cancellation results. |
| Check x86 OS register-state support; fixes #415 | Improves NumKong dispatch. Albucore has no local CPUID/XCR0 override. |
| Preserve callee-saved registers around Scalable Matrix Extension size queries; fixes #416 | Fixes upstream packed operations. Albucore's existing matrix routes were selected by performance and remain unchanged. |
| Correct binary16/bfloat16 ULP measurement | Addresses the test-helper defect described in #397; no Albucore runtime workaround exists. |
| Change upstream release automation | No Albucore runtime change. |

The dependency license is still Apache-2.0. The 7.8.6 and 7.8.7 wheel license
files have identical SHA-256 hashes.

## Guards that remain necessary

- [Float16 moments precision, #399/PR #400](https://github.com/ashvardanian/NumKong/pull/400)
  is still unreleased. Keep the subnormal scan, widening, and near-constant fallback.
- [Sandboxed Apple NEON detection, #407/PR #408](https://github.com/ashvardanian/NumKong/pull/408)
  is still unreleased. Native benchmark acceptance still requires an unrestricted
  process with verified NEON.
- C9 multiplication now preserves signed zero, but the direct NumKong candidate
  took 1.08–1.12 times the current OpenCV public-call time on the four canonical
  image sizes. Keep the faster route.
- Scalar `multiply_add` with non-half-exact controls retains its current rounding.
  Replacing it with NumKong changed 10,112 finite binary16 results for controls
  `0.4, 0.6`, and 9,025 for `1.127, 0.07` in the exhaustive encoding probe.
- General-weight NumKong blends also change cancellation results. For half inputs
  `10272, -6848` and weights `0.4, 0.6`, the candidate returned approximately
  `0.00012970`, while the existing OpenCV route returned `-0.00023174`.
  The current `0.5, 0.5` region stays unchanged.

These probes used NumKong 7.8.7, Python 3.11.15, NumPy 2.4.6, OpenCV 5.0.0.93,
PyTorch 2.15.0, and StringZilla 5.3.2 on Apple M4 Max, with one native thread.
The C9 screen used 31 samples of five calls after five warmups. It compared
complete public OpenCV calls with a direct NumKong candidate including control
conversion, broadcasting, allocation, and the returned NumPy view; it is a
rejection screen, not evidence for a new production route.

The final before/after screen compared baseline commit `c5fd143` with this
cleanup using the same 7.8.7 environment: 40 public calls across four image
sizes, float16/float32, C=1/3/9 blends, and C1 allocating/separate-output
multiplication. Each cell used five warmups and 51 samples of five calls.
The median updated/baseline time ratio was 1.003, with no slowdown above 10%
after allowing two median absolute deviations on each measurement. Output
checksums matched. These ARM-only gates do not change x86 routing.

Validation: 6,658 tests passed and one skipped. A separate native process with
NEON-half disabled passed 1,042 affected tests using baseline NEON. Full
pre-commit, 37 public-router contracts, and the CI matrix check passed.
