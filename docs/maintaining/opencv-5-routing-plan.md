# OpenCV 5 Routing Refresh Plan

Status: implemented on 2026-07-06

Target: move Albucore to OpenCV 5, refresh the dependency lock with current NumPy,
StringZilla, NumKong, and related runtime packages, then revalidate every
performance route.

Scope: Albucore only. Downstream OpenCV 4 compatibility is not a constraint for
this work.

## Implementation Summary

Implemented changes:

- OpenCV optional dependencies now require `>=5.0.0.93`.
- `uv.lock` was refreshed to OpenCV 5.0.0.93, NumKong 7.7.0, StringZilla 4.6.2,
  and the latest resolver-compatible NumPy set.
- Added `benchmarks/probes/opencv_behavior_probe.py` and saved raw OpenCV 5
  behavior evidence in `benchmarks/results/opencv5-routing/opencv5_behavior_probe.json`.
- Updated `hflip_cv2` / `vflip_cv2` chunking to OpenCV 5's measured
  `cv2.flip` limit of 128 channels.
- Recalibrated shared HWC uint8 LUT routing for the OpenCV 5 + StringZilla 4.6.2
  stack.
- Routed `std(..., "per_channel")` and `mean_std(..., "per_channel")` through
  NumKong per-channel `moments` for the shapes where the refreshed benchmarks
  show it wins.

Retained workarounds:

- `cv2.LUT` still drops the explicit channel dimension for `(H, W, 1)`.
- High-channel cubic/Lanczos warp and remap paths still require chunking.
- High-channel `INTER_AREA` downscale still requires resize chunking.
- High-channel constant borders still need Albucore handling.
- Large-kernel high-channel `medianBlur` still needs safeguards.

Benchmark artifacts:

- OpenCV 4 baseline:
  `benchmarks/results/opencv5-routing/opencv4-current/router_opencv4_current_pre_upgrade.json`
- OpenCV 5 router run:
  `benchmarks/results/opencv5-routing/router_opencv5_latest.json`
- Router comparison report:
  `benchmarks/results/opencv5-routing/REPORT_router_opencv5_latest_vs_opencv4_current.md`
- Targeted backend reports:
  `benchmarks/results/opencv5-routing/benchmark_*.md`

## Goals

- Require OpenCV 5 instead of OpenCV 4 in the project metadata and lock file.
- Refresh benchmark dependencies to the latest resolver-compatible versions.
- Run the full router benchmark suite and targeted backend benchmarks.
- Decide whether each current OpenCV, NumPy, LUT, StringZilla, and NumKong route
  is still correct.
- Delete compatibility workarounds only when OpenCV 5 behavior and benchmark
  evidence prove they are obsolete.
- Record the benchmark evidence and routing decisions in committed artifacts.

## Source Facts To Recheck At Execution Time

- `opencv-python-headless 5.0.0.93` is the first OpenCV 5 Python wheel release
  to target this upgrade.
- OpenCV Python packages share the same `cv2` namespace. Benchmark one package
  variant per environment, with `opencv-python-headless` as the canonical
  benchmark variant.
- OpenCV 5 changed several image-processing kernels. In particular,
  `warpAffine`, `warpPerspective`, `remap`, and nearest-neighbor resize have
  changed accuracy or implementation details, so both correctness and speed need
  fresh measurement.
- OpenCV 5 added core dtype support such as unsigned 32-bit and 64-bit integer
  types, but Albucore remains a `uint8` and `float32` library unless a separate
  design expands that contract.

References:

- PyPI package: https://pypi.org/project/opencv-python-headless/
- OpenCV Python releases: https://github.com/opencv/opencv-python/releases
- OpenCV 5 overview: https://github.com/opencv/opencv/wiki/OpenCV-5
- OpenCV 4 to 5 migration notes: https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration

## Current Repo Touchpoints

Dependency metadata:

- `pyproject.toml` currently carries OpenCV optional dependencies.
- `uv.lock` must be updated in the same change as dependency metadata.
- Runtime performance packages currently include `numpy`, `numkong`, and
  `stringzilla`; the refresh must let the resolver select current compatible
  versions and record the exact resolved versions in benchmark metadata.

OpenCV-facing modules:

- `albucore/geometric.py`
- `albucore/lut.py`
- `albucore/ops_misc.py`
- `albucore/arithmetic.py`
- `albucore/convert.py`
- `albucore/normalize.py`
- `albucore/stats.py`
- `albucore/utils.py`

Benchmark documentation and policy:

- `benchmarks/README.md`
- `benchmarks/results/README.md`
- `docs/maintaining/performance-policy.md`
- `docs/numkong-performance.md`
- `docs/performance-optimization.md`

## Non-Negotiable Rules

- Keep explicit channel dimensions. Grayscale images are `(H, W, 1)`, never
  `(H, W)`, at Albucore boundaries.
- Supported public dtypes remain `uint8` and `float32`.
- Routing is benchmark-driven. Do not keep or change a route based on convention.
- Compare routes on realistic HWC, DHWC, and NDHWC shapes where the operation
  supports them.
- Benchmark non-square images. Do not rely only on square test cases.
- Keep raw OpenCV probes separate from Albucore wrapper benchmarks so it is clear
  whether a result is caused by OpenCV itself or by wrapper overhead.

## Phase 0: Baseline Current Main

Before changing dependencies, capture a clean OpenCV 4 baseline from the current
lock file.

1. Confirm the working tree state.

   ```bash
   git status --short
   ```

2. Record installed versions.

   ```bash
   uv run python - <<'PY'
   import cv2
   import numpy as np
   import numkong as nk
   import stringzilla as sz

   print("cv2", cv2.__version__)
   print("numpy", np.__version__)
   print("numkong", getattr(nk, "__version__", "unknown"))
   print("stringzilla", getattr(sz, "__version__", "unknown"))
   PY
   ```

3. Run the correctness suite on the current lock.

   ```bash
   uv run pytest
   uv run ruff check .
   uv run mypy .
   ```

4. Run the full public router baseline.

   ```bash
   uv run python benchmarks/benchmark_router_synthetic.py \
     --with-geometric \
     --repeats 41 \
     --warmup 12 \
     --output-json benchmarks/results/router_opencv4_current_pre_upgrade.json
   ```

5. Run the targeted benchmark set listed in "Full Benchmark Matrix" below and
   store outputs under `benchmarks/results/opencv5-routing/opencv4-current/`.

Deliverables:

- Current dependency version note.
- `router_opencv4_current_pre_upgrade.json`.
- Targeted benchmark JSON or markdown outputs for the current route set.

## Phase 1: Dependency Upgrade

Change the project metadata to require OpenCV 5.

Expected metadata edits:

- `opencv-python>=5.0.0.93`
- `opencv-python-headless>=5.0.0.93`
- `opencv-contrib-python>=5.0.0.93`
- `opencv-contrib-python-headless>=5.0.0.93`

Refresh the lock with the latest resolver-compatible packages:

```bash
uv lock \
  --upgrade-package opencv-python \
  --upgrade-package opencv-python-headless \
  --upgrade-package opencv-contrib-python \
  --upgrade-package opencv-contrib-python-headless \
  --upgrade-package numpy \
  --upgrade-package numkong \
  --upgrade-package stringzilla
```

Validate lock consistency:

```bash
uv lock --check
```

Record exact resolved versions:

```bash
uv run python - <<'PY'
import cv2
import numpy as np
import numkong as nk
import stringzilla as sz

print("cv2", cv2.__version__)
print("numpy", np.__version__)
print("numkong", getattr(nk, "__version__", "unknown"))
print("stringzilla", getattr(sz, "__version__", "unknown"))
PY
```

Do not benchmark with multiple OpenCV wheel variants installed in the same
environment. Use `opencv-python-headless` as the canonical performance
environment unless a specific package variant needs an import smoke test.

## Phase 2: OpenCV 5 Behavior Probes

Run small raw OpenCV probes before changing routing. These answer whether old
workarounds can even be candidates for deletion.

Probe matrix:

- `cv2.LUT` on `(H, W, 1)` input. Confirm whether it still drops the channel
  dimension.
- `cv2.flip` on many-channel arrays, especially 5, 9, 513, and 1024 channels.
- `cv2.warpAffine`, `cv2.warpPerspective`, and `cv2.remap` with 1, 3, 4, 5, and
  9 channels for `INTER_NEAREST`, `INTER_LINEAR`, `INTER_LINEAR_EXACT`,
  `INTER_CUBIC`, and `INTER_LANCZOS4`.
- `cv2.resize` with 1, 3, 4, 5, and 9 channels for `INTER_AREA`, upsample and
  downsample cases.
- `cv2.copyMakeBorder` with scalar and per-channel border values for 1, 3, 4, 5,
  and 9 channels.
- `cv2.medianBlur` for kernel sizes 3, 5, 7, and 9 on 1, 3, 4, 5, and 9
  channels.
- `cv2.add`, `cv2.multiply`, `cv2.subtract`, `cv2.divide`, and `cv2.addWeighted`
  with scalar, per-channel vector, and image-shaped values.
- `cv2.mean`, `cv2.meanStdDev`, `cv2.reduce`, `cv2.minMaxLoc`, and
  `cv2.normalize` on contiguous and non-contiguous arrays.

Save the probe script under `benchmarks/probes/` if the result drives a code
deletion. Commit the probe or convert it into a regression test when the behavior
matters for future routing.

Known candidates from preliminary checking:

- `cv2.LUT` still needs explicit shape preservation for `(H, W, 1)`.
- High-channel `cv2.flip` still needs a fallback or chunking.
- Cubic and Lanczos geometric transforms still need high-channel chunking.
- `cv2.medianBlur` still needs careful high-channel and large-kernel handling.
- `cv2.resize` with high-channel `INTER_AREA` may be simpler than the current
  code assumes and needs full verification.

## Phase 3: Correctness Suite On OpenCV 5

Run the full suite after the lock update and before route changes.

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

Then run focused OpenCV-facing tests if the full suite fails or if output changes
need isolation:

```bash
uv run pytest \
  tests/test_lut.py \
  tests/test_normalize.py \
  tests/test_normalize_per_image.py \
  tests/test_multiply_add.py \
  tests/test_geometric.py \
  tests/test_to_from_float.py \
  tests/test_median_blur.py
```

Special correctness checks:

- Review golden values for `warpAffine`, `warpPerspective`, `remap`, and nearest
  resize. OpenCV 5 may legitimately produce different pixels.
- Confirm Albucore wrappers still return explicit `(H, W, 1)` for single-channel
  results.
- Confirm unsupported dtypes still fail or are rejected as before.
- Confirm non-contiguous inputs are still handled by decorators or explicit
  conversion.

## Phase 4: Full Benchmark Matrix

Run each benchmark twice: once on the OpenCV 4 baseline from Phase 0 and once on
the OpenCV 5 plus latest dependency stack. Store raw results separately and
generate a comparison report.

Public router benchmark:

```bash
uv run python benchmarks/benchmark_router_synthetic.py \
  --with-geometric \
  --repeats 41 \
  --warmup 12 \
  --output-json benchmarks/results/opencv5-routing/router_opencv5_latest.json
```

Compare against the baseline:

```bash
uv run python benchmarks/compare_router_json.py \
  benchmarks/results/router_opencv4_current_pre_upgrade.json \
  benchmarks/results/opencv5-routing/router_opencv5_latest.json
```

Targeted backend benchmarks:

- `benchmarks/benchmark_lut_shared_routing.py`
- `benchmarks/benchmark_sz_lut_vs_cv2_lut.py`
- `benchmarks/benchmark_scale_vs_lut.py`
- `benchmarks/benchmark_add_constant_uint8_channels.py`
- `benchmarks/benchmark_grayscale_paths.py`
- `benchmarks/benchmark_multiply_add_numkong.py`
- `benchmarks/benchmark_numkong_vs_albucore_backends.py`
- `benchmarks/benchmark_numkong.py`
- `benchmarks/benchmark_stats.py`
- `benchmarks/benchmark_reduce_sum.py`
- `benchmarks/benchmark_sum_mean_std_ravel.py`
- `benchmarks/benchmark_minmax_ravel.py`
- `benchmarks/benchmark_normalize_numkong_patterns.py`

Geometric benchmark expansion:

The public router benchmark includes geometric operations, but the OpenCV 5
kernel changes justify a dedicated geometric benchmark if current coverage does
not separate interpolation and border modes enough.

Required dimensions:

- HWC: `128x160`, `240x320`, `480x640`, `768x1024`
- Channels: `1`, `3`, `4`, `5`, `9`
- Dtypes: `uint8`, `float32`
- Layouts where supported: HWC, DHWC, NDHWC
- Contiguity: contiguous and representative non-contiguous inputs

Required operation variants:

- `resize`: upsample and downsample with `INTER_NEAREST`, `INTER_LINEAR`,
  `INTER_LINEAR_EXACT`, `INTER_CUBIC`, `INTER_LANCZOS4`, and `INTER_AREA`
- `warp_affine`: identity, translation, rotation, scale, and shear
- `warp_perspective`: identity and non-trivial perspective transform
- `remap`: identity map, fractional map, and out-of-bounds map
- `copy_make_border`: constant scalar, constant per-channel, replicate, reflect,
  and wrap modes

Benchmark metadata to capture:

- CPU model and OS.
- Python version.
- Exact versions of OpenCV, NumPy, NumKong, StringZilla.
- Git commit SHA.
- Command line and repeat counts.
- Median, standard deviation, robust spread, and sample count.

## Phase 5: Routing Audit By Module

Use this checklist to review current code paths after benchmark data exists.

### `albucore/lut.py`

- Recalibrate `opencv_shared_uint8_lut_faster_hwc` with OpenCV 5 and latest
  StringZilla.
- Compare `cv2.LUT`, `sz.lookup`, NumPy advanced indexing, and any shared-LUT
  helpers for HWC, DHWC, and NDHWC.
- Keep channel-dimension restoration if `cv2.LUT` still returns `(H, W)` for
  single-channel HWC input.
- Check whether per-channel LUT splitting thresholds changed.
- Confirm LUT table dtype choices still avoid float64 widening.

### `albucore/geometric.py`

- Recheck every `maybe_process_in_chunks` path against raw OpenCV 5 behavior.
- Decide whether `resize` needs high-channel chunking for `INTER_AREA`
  downsample cases.
- Keep cubic and Lanczos chunking if OpenCV 5 still rejects high-channel inputs.
- Benchmark per-channel border value handling. Remove manual splitting only if
  OpenCV 5 supports it correctly and faster.
- Update tolerance or golden tests only after confirming pixel changes match
  OpenCV 5 migration behavior.

### `albucore/ops_misc.py`

- Benchmark `hflip` and `vflip` against OpenCV 5, NumPy slicing, and chunked
  fallbacks for normal and very high channel counts.
- Revalidate `median_blur` routes for dtype, channel count, and kernel size.
- Recheck any distance or miscellaneous helpers that route to NumKong.

### `albucore/arithmetic.py`

- Rebenchmark add, multiply, subtract, power, multiply-add, and weighted-add
  paths for scalar, per-channel, and array values.
- Compare OpenCV 5, NumPy, LUT, and NumKong routes separately for `uint8` and
  `float32`.
- Pay special attention to 1-channel and `>4` channel images because OpenCV
  scalar semantics and channel limits often differ there.

### `albucore/convert.py`

- Rebenchmark `to_float`, `from_float`, and LUT conversion paths.
- Confirm OpenCV 5 does not introduce hidden float64 promotion in conversion
  paths.
- Keep output dtype and clipping behavior identical to Albucore's public
  contract.

### `albucore/normalize.py`

- Rebenchmark fixed normalize and per-image normalize paths.
- Compare NumPy, OpenCV, LUT, and NumKong-backed helpers where applicable.
- Verify `cv2.normalize` behavior for min-max cases and single-channel HWC
  outputs.

### `albucore/stats.py`

- Rebenchmark `mean`, `std`, `mean_std`, `sum`, `min`, `max`, and normalization
  support statistics.
- Compare NumKong, OpenCV reductions, NumPy reductions, and ravel-based helpers.
- Separate global statistics from per-channel statistics.

### `albucore/utils.py`

- Recheck `MAX_OPENCV_WORKING_CHANNELS` assumptions against OpenCV 5 behavior.
- Keep `uint8` and `float32` public dtype validation even if OpenCV 5 exposes
  more native dtypes.
- Update comments that describe OpenCV limits only after probes confirm the new
  behavior.

## Phase 6: Code Simplification Rules

Only simplify code when all three conditions are true:

1. Raw OpenCV 5 probes show the workaround is no longer needed.
2. Albucore correctness tests cover the affected shape, dtype, and channel case.
3. Benchmarks show the simpler path is neutral or faster for the relevant route.

Likely simplification candidates:

- Remove stale `resize` chunking for high-channel `INTER_AREA` if OpenCV 5 handles
  it correctly and faster.
- Simplify comments and route guards that exist only for OpenCV 4 limitations
  that are gone in OpenCV 5.
- Collapse duplicated scalar or per-channel OpenCV paths if OpenCV 5 now supports
  the same behavior directly and faster.

Likely non-candidates unless probes contradict this:

- Single-channel shape preservation around `cv2.LUT`.
- High-channel flip fallback.
- Cubic and Lanczos chunking for high-channel warp and remap.
- Large-kernel or high-channel median blur safeguards.

## Phase 7: Decision Thresholds

Apply the existing performance policy:

- More than 15 percent slowdown on a hot path requires investigation.
- More than 10 percent median slowdown across a router family requires
  explanation or rerouting.
- Release-blocking hot paths include normalize, per-image normalize, LUT,
  arithmetic, statistics, `to_float`, `from_float`, flips, resize, and remap.

Routing decision format:

```text
Operation:
Current route:
Candidate route:
Shapes/dtypes tested:
Median speedup or slowdown:
Correctness impact:
Decision:
Follow-up:
```

Every route change should point to a benchmark artifact. Every retained route
that looks surprising should also point to evidence, especially when LUT is not
the fastest `uint8` path.

## Phase 8: Documentation Updates

Update docs only after benchmark decisions are final.

Required documentation checks:

- `docs/performance-optimization.md`: routing guidance and OpenCV 5 notes.
- `docs/numkong-performance.md`: benchmark tables if NumKong routes change.
- `benchmarks/results/README.md`: new OpenCV 5 result artifacts.
- `benchmarks/README.md`: any new geometric or probe benchmark.
- Inline comments near removed or retained OpenCV workarounds.

## Final Acceptance Criteria

- `pyproject.toml` requires OpenCV 5 optional packages.
- `uv.lock` is refreshed and `uv lock --check` passes.
- Exact resolved versions are recorded in benchmark output or a report.
- Full test suite passes on OpenCV 5.
- Full router benchmark and targeted backend benchmarks are run.
- A benchmark comparison report exists for OpenCV 4 current versus OpenCV 5
  latest stack.
- Each routing change has benchmark evidence.
- Each deleted workaround has a raw OpenCV 5 behavior probe or a regression test.
- Public dtype and shape contracts remain unchanged.

## Suggested Execution Order

1. Capture OpenCV 4 baseline tests and benchmarks.
2. Update metadata and lock to OpenCV 5 plus latest performance dependencies.
3. Run OpenCV 5 behavior probes.
4. Run the full correctness suite.
5. Run full benchmarks and generate comparisons.
6. Audit routes module by module.
7. Make code simplifications and routing changes.
8. Rerun focused tests and affected benchmarks.
9. Rerun full tests and router benchmark.
10. Update docs and benchmark result index.

## Risk Register

- Benchmark noise can hide small routing differences. Use enough repeats and
  rerun close calls.
- OpenCV 5 interpolation changes can make pixel-exact OpenCV 4 expectations
  obsolete. Treat those as correctness migrations, not automatic regressions.
- Installing more than one OpenCV wheel variant can produce confusing `cv2`
  imports. Keep benchmark environments clean.
- Latest NumPy, StringZilla, or NumKong may change the winner independently of
  OpenCV 5. Attribute decisions to the whole resolved stack, not OpenCV alone.
- New OpenCV dtype support should not leak into Albucore's public contract
  without a separate API decision.
