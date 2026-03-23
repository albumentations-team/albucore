# Reliable-repeat benchmark suite (2026-03-21)

## Protocol

| Script / artifact | repeats | warmup | Output |
|-------------------|--------:|-------:|--------|
| `benchmark_router_synthetic.py` | 41 | 12 | `router_synthetic_reliable.json`, `router_synthetic_reliable_run2.json` |
| `compare_router_json.py` | — | — | `REPORT_reliable_run2_vs_run1.md`, `REPORT_reliable_vs_current.md` |
| `benchmark_lut_shared_routing.py` | 41 | 12 | `bench_lut_shared_routing_reliable.md` |
| `benchmark_sz_lut_vs_cv2_lut.py` | 41 | 12 | `bench_sz_lut_vs_cv2_reliable.md` |
| `benchmark_numkong_vs_albucore_backends.py` | 41 | 12 | `reliable_benchmark_numkong_vs_albucore_backends.md` |
| `benchmark_numkong.py` | 41 | 12 | `reliable_benchmark_numkong.md` |
| `benchmark_minmax_ravel.py` | 41 | 12 | `reliable_benchmark_minmax_ravel.md` |
| `benchmark_normalize_numkong_patterns.py` | 41 | 12 | `reliable_benchmark_normalize_numkong_patterns.md` |
| `benchmark_sum_mean_std_ravel.py` | 41 | 12 | `reliable_benchmark_sum_mean_std_ravel.md` |
| `benchmark_add_constant_uint8_channels.py` | 41 | 12 | `reliable_benchmark_add_constant_uint8_channels.md` |
| `benchmark_grayscale_paths.py` | 41 | 12 | `reliable_benchmark_grayscale_paths.md` |
| `benchmark_stats.py` | (fixed 7/2 in script) | — | `reliable_benchmark_stats.md` |

**Skipped:** `benchmark_multiply_add_numkong.py` — `ImportError: cannot import name 'multiply_by_constant_numkong' from 'albucore.functions'` (API drift).

## Regression check: `@preserve_channel_dim` on OpenCV shared LUT (`_cv2_lut_uint8`)

**Same code, same machine, same protocol** — two consecutive router synthetic runs:

- **`apply_uint8_lut`:** median **old/new = 0.99×** over 12 cells (`REPORT_reliable_run2_vs_run1.md`).
- **`sz_lut`:** median **0.97×** over 12 cells.

Interpretation: medians for the LUT routers sit at ~1× noise. Individual cells still move (e.g. ±15–20% or occasional outliers like **0.42×** on huge uint8 tensors) **without any code change** — that is wall-clock + thermal variance, not the decorator.

The wrapper is a **no-op on the OpenCV path that actually runs today** (`C ≥ 2` in `opencv_shared_uint8_lut_faster_hwc`): `preserve_channel_dim` only expands when `result.ndim == 2` and `shape[-1] == 1`, which does not happen for multi-channel `cv2.LUT` output.

## Do **not** use `REPORT_reliable_vs_current.md` as a decorator A/B

`router_synthetic_current.json` was collected with **repeats=21, warmup=5** and does **not** include `apply_uint8_lut` rows (older export set / schema). Comparing it to the new JSON mixes **different methodology** with **different op coverage**, which produces misleading “16× wins” on unrelated ops.

Use **`REPORT_reliable_run2_vs_run1.md`** for “how loud is noise at 41/12?”.

## LUT routing calibration (shared HWC)

See `bench_lut_shared_routing_reliable.md`: **5** heuristic mismatches (`!!`) on the 7×12×5-seed grid — same class of tuning noise as before; re-tune `opencv_shared_uint8_lut_faster_hwc` only if you want fewer crosses.

## Regenerating

From repo root:

```bash
cd benchmarks
R=41 W=12
uv run python benchmark_router_synthetic.py --output-json results/router_synthetic_reliable.json --repeats $R --warmup $W
# optional second run for variance:
uv run python benchmark_router_synthetic.py --output-json results/router_synthetic_reliable_run2.json --repeats $R --warmup $W
uv run python compare_router_json.py results/router_synthetic_reliable_run2.json results/router_synthetic_reliable.json results/REPORT_reliable_run2_vs_run1.md
```
