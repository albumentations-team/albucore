# Albucore micro-benchmarks

Scripts here time **NumKong vs OpenCV / NumPy / LUT** and compare **`albucore.stats`** to plain NumPy. They are **not** shipped with the wheel; run from a **git checkout** (or any tree where `albucore` is importable, e.g. editable install).

## Setup

```bash
uv sync --extra headless   # OpenCV + NumKong + numpy; required for most tables
```

Run each script from the **repository root** so imports resolve:

```bash
uv run python benchmarks/<script>.py
```

Shared helper: [`timing.py`](timing.py) — **`bench_wall_ms`** (median, mean, sample **std**, **MAD**, `n`) and **`median_ms`** (median only, backward compatible). Router JSON stores **`ms_median ± ms_std`**-style fields for error bars.

### Copies in timed loops (intentional)

Many microbenchmarks use **`img.copy()`** / **`np.ascontiguousarray(base.copy())`** so each iteration starts from an untouched buffer and matches APIs that expect contiguous inputs. That adds allocator traffic versus a single in-place buffer, but keeps comparisons fair and avoids cross-iteration mutation. The **router** harness times public functions the same way typical callers see them (often a fresh output array).

NumKong exposes **`out=`** on some APIs, but **`nk.zeros` + `out=`** can cost an extra full buffer zero-fill vs using the library’s returned `Tensor` + **`np.frombuffer`**; benchmark before switching production paths.

## Scripts

| Script | Purpose |
|--------|---------|
| [`benchmark_numkong_vs_albucore_backends.py`](benchmark_numkong_vs_albucore_backends.py) | Large Markdown tables: `add_weighted`, global mean/std, per-channel stats — matches methodology in [`docs/numkong-performance.md`](../docs/numkong-performance.md). |
| [`benchmark_multiply_add_numkong.py`](benchmark_multiply_add_numkong.py) | Scalar/array multiply & add: production APIs vs `nk.scale` / `blend` / `fma`. |
| [`benchmark_numkong.py`](benchmark_numkong.py) | Smaller sweeps: `cdist`, blend, 1D `scale`/`fma`, misc. |
| [`benchmark_stats.py`](benchmark_stats.py) | Quick smoke: `albucore.stats.mean_std` vs NumPy reference on a few shapes. |
| [`benchmark_router_synthetic.py`](benchmark_router_synthetic.py) | Every name in **`albucore.functions.__all__`** (except decorator factories). Defaults **`--repeats 21`**, **`--warmup 5`**; JSON includes spread (`ms_std`, `ms_mad`). **`sz_lut`** bench uses **`inplace=False`** so the image is not mutated across iterations. **`--skip-ops`** omits routers (no rows). **`--benchmark-label`** stored in JSON meta. |
| [`compare_router_json.py`](compare_router_json.py) | Markdown report: ratios from medians; full table shows **median ± σ** and MAD columns when present. Sections for **new-only** / **baseline-only** `ok` cells. |
| [`run_router_compare_0_0_41.sh`](run_router_compare_0_0_41.sh) | **`git worktree`** at tag **`0.0.41`** + its **`uv sync`** (simsimd era), router bench with **`--skip-ops`** stats+LUT; then current tree full bench; writes JSON + **`results/REPORT_router_0.0.41_vs_current.md`**. Env: **`REPEATS`**, **`WARMUP`**, **`ALBUCORE_041_WORKTREE`**. |
| [`benchmark_minmax_ravel.py`](benchmark_minmax_ravel.py) | Prints Markdown tables: `Tensor.minmax()` vs NumPy min+max on raveled `(H,W,C)`. |
| [`benchmark_normalize_numkong_patterns.py`](benchmark_normalize_numkong_patterns.py) | NumKong “how to normalize” patterns: per-channel `nk.scale` (ImageNet α/β), `minmax`+`scale`, vs OpenCV/NumPy; 2D `sum`/`norm` per-channel stats vs `cv2.meanStdDev` / NumPy. |
| [`benchmark_sum_mean_std_ravel.py`](benchmark_sum_mean_std_ravel.py) | Prints Markdown tables: NumPy vs NumKong sum/mean/std on `(H,W,C)`. |
| [`benchmark_add_constant_uint8_channels.py`](benchmark_add_constant_uint8_channels.py) | uint8 scalar add: OpenCV vs LUT vs NumKong vs NumPy vs `add_constant` wrapper (C=5..9, several spatial sizes). |
| [`benchmark_grayscale_paths.py`](benchmark_grayscale_paths.py) | Grayscale / routing sanity: uint8 per-channel multiply LUT vs OpenCV; float→uint8 NumPy vs cv2 (and cv2 (H,W,1) quirk). |
| [`benchmark_sz_lut_vs_cv2_lut.py`](benchmark_sz_lut_vs_cv2_lut.py) | `StringZilla` `translate` / `sz_lut` vs `cv2.LUT` on uint8: shared `(256,)` and per-channel `(256,1,C)` LUTs; shapes `HWC`, `DHWC`, `NDHWC`. |
| [`benchmark_lut_shared_routing.py`](benchmark_lut_shared_routing.py) | Grid sweep: when OpenCV beats StringZilla for **shared** HWC LUT vs `opencv_shared_uint8_lut_faster_hwc` (used by `apply_uint8_lut`). |

### What compares what (sanity / routing)

| Script | Wrapper (`functions` API) | NumKong | OpenCV | LUT | NumPy |
|--------|---------------------------|---------|--------|-----|-------|
| [`benchmark_router_synthetic.py`](benchmark_router_synthetic.py) | **Yes** — times whatever each export routes to | No (unless the export calls NK internally) | No (unless routed) | No (unless routed) | No (unless routed) |
| [`benchmark_numkong_vs_albucore_backends.py`](benchmark_numkong_vs_albucore_backends.py) | No | **Yes** | **Yes** | **Yes** (uint8 where applicable) | **Yes** |
| [`benchmark_multiply_add_numkong.py`](benchmark_multiply_add_numkong.py) | Partial (`multiply` / `add_array` paths) | **Yes** | **Yes** | **Yes** (uint8) | Via prod APIs |
| [`benchmark_add_constant_uint8_channels.py`](benchmark_add_constant_uint8_channels.py) | **`add_constant`** | **`add_constant_numkong`** | **`add_opencv`** | **`add_lut`** | Saturated int16 reference |
| [`benchmark_numkong.py`](benchmark_numkong.py) | No | **Yes** | If installed | — | **Yes** |

The **router** JSON is the regression guard vs an older wheel; it does **not** sweep alternate backends for the same op. Use the **multi-backend** scripts when checking “are we missing a faster library path?”.

## Research notes

Extra write-ups and archived tables: **[`docs/research/`](../docs/research/)** (not regenerated by the scripts above).

## Dataset-driven benchmarks

End-to-end throughput over real image folders (not in `benchmarks/`):

```bash
./benchmark.sh <data_dir> [options]
```

See [`benchmark.sh`](../benchmark.sh) for flags.
