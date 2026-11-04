---
name: performance-optimization
description: Systematic performance audit for Albucore runtime code. Use whenever implementing, reviewing, profiling, or optimizing atomic image operations, backend routing, reductions, label maps, LUTs, random generation, dtype conversions, allocation-heavy paths, batch or volume kernels, or explicit output aliasing.
---

# Performance Optimization

Benchmark Albucore changes through Core public operations only, including their Core-owned dispatch, conversions,
allocations and memory use. AX functional/transform/Compose performance checks require an explicit user request and
are not an automatic gate for Core changes. Follow the benchmark scope in the canonical guide below.

Read `../../../docs/performance-optimization.md` completely before inspecting or changing runtime code. That document
is the canonical shared workflow for Albucore and AlbumentationsX.

For final correctness, speed, and memory acceptance, use Python 3.11 with the latest stable dependencies available
for Python 3.11 and the target platform. The current recorded profile uses PyTorch 2.15.0, including the CPU wheel on
Linux. Check resolved versions and imported source paths; a newer Python or stale `uv.lock` is not a substitute.

When the sibling AlbumentationsX checkout is available, verify the fallback copy with
`cmp -s docs/performance-optimization.md ../AlbumentationsX/.codex/skills/performance-optimization/references/performance-optimization.md`.
A mismatch blocks completion.

## Workflow

1. Establish correctness and public-router performance baselines before editing.
2. Run every stage of the canonical optimization pass. Do not stop after finding the first plausible improvement.
3. Treat delete-first, vectorization, LUT, random generation, `bincount`, backend selection, routing thresholds, and
   explicit output aliasing as benchmark questions. Internal mutation is limited to owned working buffers.
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

Albucore routers are called after the upstream validation boundary. The caller owns container type, rank/layout,
dtype, device, contiguity, autograd, and operation-control validation. Do not duplicate those checks in the runtime
router; retain only backend dispatch and kernel-required normalization. Invalid direct calls are outside the low-level
contract and must not add hot-path branches.

## Required Handoff

Report:

- work deleted or avoided;
- full-array passes, copies, conversions, and allocations removed;
- vectorization and grouped-reduction candidates evaluated;
- LUT, random-generator, and backend candidates compared;
- routing thresholds and safe `out=` alias decisions;
- AlbumentationsX duplicates replaced or follow-up extraction opportunities;
- correctness evidence, benchmark matrix, regressions, and rejected candidates.

## Generated benchmark artifacts

Keep generated benchmark reports, CSV/JSON/GZ data, profiler output and run logs outside Git. Local outputs may
use ignored results directories. Never force-add them, including when publishing a PR or preserving evidence.
Keep reproducible benchmark code and concise current route decisions in the existing documentation. Preserve raw
measurements as external artifacts and link them from PR descriptions or the archive index in benchmarks/README.md.
