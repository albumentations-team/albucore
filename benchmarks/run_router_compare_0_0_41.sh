#!/usr/bin/env bash
# Router synthetic: tag 0.0.41 (separate worktree + venv) vs current checkout (repo venv).
#
# 0.0.41 tree uses simsimd (not numkong); the worktree has its own uv.lock from that tag.
# Current tree uses numkong; `uv run` from repo root uses the main .venv.
#
# Usage (from repo root):
#   ./benchmarks/run_router_compare_0_0_41.sh
# Env overrides:
#   REPEATS=41 WARMUP=12 ALBUCORE_041_WORKTREE=/path/to/wt

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WT="${ALBUCORE_041_WORKTREE:-$ROOT/../albucore-wt-041}"
REPEATS="${REPEATS:-41}"
WARMUP="${WARMUP:-12}"
OUT_OLD="$ROOT/benchmarks/results/router_synthetic_0.0.41_skip_stats_lut.json"
OUT_NEW="$ROOT/benchmarks/results/router_synthetic_CURRENT_main.json"
OUT_MD="$ROOT/benchmarks/results/REPORT_router_0.0.41_vs_current.md"

# Ops not benchmarked on the 0.0.41 profile (no stable public routers / match current feature set).
SKIP_041="mean,std,mean_std,apply_uint8_lut,sz_lut"

if [[ ! -d "$WT/.git" ]]; then
  echo "Adding worktree at $WT (tag 0.0.41)..."
  git -C "$ROOT" worktree add "$WT" 0.0.41
fi

echo "Syncing venv in worktree (simsimd + OpenCV)..."
(cd "$WT" && uv sync --extra headless)

PY_041="$WT/.venv/bin/python"
if [[ ! -x "$PY_041" ]]; then
  echo "Expected python at $PY_041" >&2
  exit 1
fi

echo "=== Benchmark albucore @ 0.0.41 (skip $SKIP_041), repeats=$REPEATS warmup=$WARMUP ==="
(cd "$ROOT/benchmarks" && "$PY_041" benchmark_router_synthetic.py \
  --output-json "$OUT_OLD" \
  --repeats "$REPEATS" \
  --warmup "$WARMUP" \
  --skip-ops "$SKIP_041" \
  --benchmark-label "tag 0.0.41 worktree; --skip-ops stats+LUT")

echo "=== Benchmark current checkout (full __all__) ==="
(cd "$ROOT/benchmarks" && uv run python benchmark_router_synthetic.py \
  --output-json "$OUT_NEW" \
  --repeats "$REPEATS" \
  --warmup "$WARMUP" \
  --benchmark-label "main working tree; full routers")

echo "=== Compare (new=current, old=0.0.41 profile) ==="
(cd "$ROOT/benchmarks" && uv run python compare_router_json.py "$OUT_NEW" "$OUT_OLD" "$OUT_MD")

echo "Wrote $OUT_OLD $OUT_NEW $OUT_MD"
