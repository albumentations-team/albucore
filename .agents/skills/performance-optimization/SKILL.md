---
name: performance-optimization
description: Systematic performance audit for Albucore runtime code before and after a change. Use when implementing, reviewing, profiling, or optimizing atomic image operations, backend routing, reductions, label maps, LUTs, random generation, dtype conversions, allocation-heavy paths, batch or volume kernels, or explicit output aliasing ("оптимизируй", "почему медленно", "ускорь").
---

# Performance Optimization

Read `../../../docs/performance-optimization.md` completely before inspecting or changing runtime code. That document
is the canonical workflow shared by Albucore and AlbumentationsX.

When the sibling AlbumentationsX checkout is available, verify the shared copy with
`cmp -s docs/performance-optimization.md ../AlbumentationsX/.agents/skills/performance-optimization/references/performance-optimization.md`.
A mismatch blocks completion.

Benchmark Core public operations only, including Core-owned dispatch, conversions, allocations, and memory use, on the
acceptance profile in `AGENTS.md`. AX functional, transform, and Compose performance checks need an explicit user
request.

## Workflow

1. Establish correctness and public-router performance baselines before editing.
2. Run every stage of the canonical optimization pass. Do not stop after the first plausible improvement.
3. Treat delete-first, vectorization, LUT, random generation, `bincount`, backend selection, routing thresholds, and
   explicit output aliasing as benchmark questions. Internal mutation is limited to owned working buffers.
4. Compare the complete public path, including dispatch, dtype conversion, contiguity, allocation, clipping, and
   shape repair.
5. Load the `albucore-benchmarks` skill and read it completely before measuring. Extend the matrix along the
   dimension that controls the candidate.
6. Add or preserve correctness tests for every accepted route and its boundary cases.
7. Update benchmark evidence and route documentation when public routing changes.

## Albucore boundary

Accept reusable atomic array operations with stable image semantics. Keep augmentation policy, stochastic parameter
sampling, target dispatch, and annotation rules in AlbumentationsX.

Before adding an atom:

- search Albucore for an existing router or lower-level operation;
- search AlbumentationsX for duplicate local implementations;
- define dtype, shape, channel, contiguity, aliasing, and error contracts;
- benchmark all viable backends before exposing a route.

Invalid direct calls are outside the caller-validated router contract in `AGENTS.md` and must not add hot-path
branches.

## NaN contract

Only vector fields expect NaN. AlbumentationsX calls only `remap`, `remap3d` and the `tensor_bridge` helpers on fields,
so only these must keep it exactly (`tests/test_nan_field_contract.py`). Every other operation (arithmetic, quantization,
statistics, normalization, LUT, flip, rotate, crop, pad, resize) assumes finite images, masks and volumes: delete NaN guards instead of adding them, and benchmark those
routes on finite data. See `docs/performance-optimization.md`.

## Speed over bit-exactness

Speed is the top routing priority. Accept rounding-level differences (ulp-scale backend gaps, fused versus two-rounding
arithmetic, float32 versus float64 accumulation) when a route is faster; document the tolerance in the docstring and the
public-API row, and test with that tolerance, not bitwise. Gross errors stay unacceptable: wrong saturation or overflow
(`1e10 * pixel` gives 255, not 0), a wrong `std` from near-constant cancellation, wrong shape or dtype, a crash on valid input.

## Required handoff

Report:

- work deleted or avoided;
- full-array passes, copies, conversions, and allocations removed;
- vectorization and grouped-reduction candidates evaluated;
- LUT, random-generator, and backend candidates compared;
- routing thresholds and safe `out=` alias decisions;
- AlbumentationsX duplicates replaced or follow-up extraction opportunities;
- correctness evidence, benchmark matrix, regressions, and rejected candidates.
