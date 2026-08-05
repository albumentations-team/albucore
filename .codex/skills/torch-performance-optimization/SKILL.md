---
name: torch-performance-optimization
description: Optimize or review eager CPU-only Albucore PyTorch runtime paths with benchmark-backed decisions. Use when adding or changing Torch CPU kernels, Tensor/NumPy bridges, Torch backend routing, tensor layouts, allocations, threading, profiling, memory-format candidates, or Torch performance benchmarks.
---

# Torch Performance Optimization

Read [`docs/torch-performance-optimization.md`](../../../docs/torch-performance-optimization.md) completely before inspecting or editing a Torch path. Also read `../performance-optimization/SKILL.md`, its canonical performance guide, and `../albucore-benchmarks/SKILL.md` completely.

## Workflow

1. Establish a correctness baseline and benchmark the existing public path. State container, layout, dtype, shape, parameter, stride, thread, and allocation contracts.
2. Audit unnecessary Python work, full-volume passes, conversions, materializations, and output repairs before choosing a Torch operator.
3. Profile only to discover candidates. Benchmark every viable implementation end to end, including NumPy↔Torch bridges and layout conversion.
4. Compare NumPy, OpenCV, NumKong, StringZilla, and Torch where they share semantics. Treat fused operators, layout, `channels_last_3d`, and in-place reuse as hypotheses with their own correctness and performance matrices.
5. Select a route or threshold only from stable public-path evidence. Preserve rejected candidates in the benchmark report when they clarify a boundary.
6. Add correctness tests for each accepted route and update the benchmark evidence, public contract, and canonical guide when a reusable rule or limitation is discovered.

## Albucore Contract

- Torch is a required dependency. Assume it is already imported for benchmarks.
- Current public Tensor routes are eager CPU paths that do not record autograd inside the primitive and do not use `torch.compile`. Do not add device routes, graph-preserving primitive fallbacks, compilation, or their benchmark candidates.
- Tensor layouts are explicit and independent of NumPy layouts. Do not infer `NCHW` versus `CDHW` from shape sizes.
- Caller-prevalidated 3D routers leave validation outside the hot path. Do not add it back while optimizing.
- A Tensor route must preserve the documented container, layout, dtype, border, interpolation, rounding, mutation, and aliasing behavior.

## Required Handoff

Report the full-path baseline, selected route, removed work, allocation and copy changes, all viable backends and Torch-specific candidates considered, thread and benchmark matrix, correctness/memory evidence, regressions, rejected regions, and remaining follow-ups.
