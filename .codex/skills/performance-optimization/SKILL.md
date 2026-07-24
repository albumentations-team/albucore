---
name: performance-optimization
description: Systematic performance audit for Albucore runtime code. Use whenever implementing, reviewing, profiling, or optimizing atomic image operations, backend routing, reductions, label maps, LUTs, random generation, dtype conversions, allocation-heavy paths, batch or volume kernels, or in-place behavior.
---

# Performance Optimization

Read `../../../docs/performance-optimization.md` completely before inspecting or changing runtime code. That document
is the canonical shared workflow for Albucore and AlbumentationsX.

When the sibling AlbumentationsX checkout is available, verify the fallback copy with
`cmp -s docs/performance-optimization.md ../AlbumentationsX/.codex/skills/performance-optimization/references/performance-optimization.md`.
A mismatch blocks completion.

## Workflow

1. Establish correctness and public-router performance baselines before editing.
2. Run every stage of the canonical optimization pass. Do not stop after finding the first plausible improvement.
3. Treat delete-first, vectorization, LUT, random generation, `bincount`, backend selection, routing thresholds, and
   in-place mutation as benchmark questions.
4. Compare the complete public path, including dispatch, dtype conversion, contiguity, allocation, clipping, and shape
   repair.
5. Read `../albucore-benchmarks/SKILL.md` completely before measuring. Extend the matrix along the dimension that
   controls the candidate.
6. Add or preserve correctness tests for every accepted route and its boundary cases.
7. Update benchmark evidence and route documentation when public routing changes.

## Albucore Boundary

Accept reusable atomic array operations with stable image semantics. Keep augmentation policy, stochastic parameter
sampling, target dispatch, and annotation rules in AlbumentationsX.

Before adding an atom:

- search Albucore for an existing router or lower-level operation;
- search AlbumentationsX for duplicate local implementations;
- define dtype, shape, channel, contiguity, aliasing, and error contracts;
- benchmark all viable backends before exposing a route.

## Required Handoff

Report:

- work deleted or avoided;
- full-array passes, copies, conversions, and allocations removed;
- vectorization and grouped-reduction candidates evaluated;
- LUT, random-generator, and backend candidates compared;
- routing thresholds and safe in-place decisions;
- AlbumentationsX duplicates replaced or follow-up extraction opportunities;
- correctness evidence, benchmark matrix, regressions, and rejected candidates.
