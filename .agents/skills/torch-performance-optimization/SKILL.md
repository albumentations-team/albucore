---
name: torch-performance-optimization
description: Benchmark-backed optimization and review of eager CPU-only Albucore PyTorch paths. Use when adding or changing Torch CPU kernels, Tensor/NumPy bridges, Torch routing or thresholds, Tensor memory format or channels_last_3d performance, allocations, threading, or Torch benchmarks ("ускорь torch", "tensor bridge", "torch маршрут").
---

# Torch Performance Optimization

Read [`docs/torch-performance-optimization.md`](../../../docs/torch-performance-optimization.md) completely before
inspecting or editing a Torch path. Also load the `performance-optimization` and `albucore-benchmarks` skills and read
them, with the canonical performance guide, completely.

Use the acceptance profile in `AGENTS.md` (Python 3.11, newest stable CPU PyTorch: `+cpu` wheel on Linux x86-64).
Record the exact package inventory and confirm the imported Albucore path before running tests or benchmarks.
Albucore Tensor performance checks cover Core public calls and Core-owned bridges only. AX performance validation
requires an explicit user request.

## Workflow

1. Establish a correctness baseline and benchmark the existing public path. State container, layout, dtype, shape,
   parameter, stride, thread, and allocation contracts.
2. Audit unnecessary Python work, full-volume passes, conversions, materializations, and output repairs before
   choosing a Torch operator.
3. Profile only to discover candidates. Benchmark every viable implementation end to end, including NumPy↔Torch
   bridges and layout conversion.
4. Compare NumPy, OpenCV, NumKong, StringZilla, and Torch where they share semantics. For float16, compare native half
   operations with float32 working paths plus output restoration. Treat fused operators, layout, `channels_last_3d`,
   and explicit output aliasing as hypotheses with their own correctness and performance matrices.
5. Select a route or threshold only from stable public-path evidence. Keep rejected candidates in the benchmark
   report when they clarify a boundary.
6. Add correctness tests for each accepted route. Record the route, its measured region, and links to the external
   measurements in the "Choose Torch operators by measured regions" section of
   `docs/torch-performance-optimization.md`, not in this skill.

## Albucore contract

- Same-dtype speed, memory, and correctness acceptance applies independently on Linux x86-64 and macOS ARM64.
  Float16/float32 comparisons are mandatory diagnostics; same-dtype measurements decide acceptance.
- Public Tensor routes are eager CPU paths: no autograd recording inside the primitive, no `torch.compile`, no
  device routes, and no graph-preserving fallbacks or benchmark candidates for them.
- Tensor layouts are explicit and independent of NumPy layouts. Do not infer `NCHW` versus `CDHW` from shape sizes.
- `warp_affine3d` and `remap3d` accept rank-4 `CDHW` volumes and rank-5 `NCDHW` batches; one matrix or grid is shared
  across the batch. Caller-prevalidated 3D routers keep validation out of the hot path; do not add it back.
- A Tensor route preserves the documented container, layout, border, interpolation, rounding, mutation, and aliasing
  behavior, and distinguishes preserved input/output dtype from any explicit working conversion.
- AX owns reusable buffers. Input writes require explicit `out=img`; measure the Core call with the supplied buffers.

## Allocation rules learned from measurements

- Batch routes fill one final output (or the supplied output view item by item). Do not collect per-item results
  and `np.stack` them.
- Release intermediates that are no longer needed before `Tensor.numpy()` creates an output view; keeping them alive
  raises peak RSS.
- Measure default allocation separately from explicit outputs. A supplied output is not automatically faster or
  allocation-free.
- Compare every affected memory order (C-order, K-order, planar, interleaved) before replacing per-item allocation
  with one dense batch buffer.
- Compute all statistics before writing an aliased output.

## Required handoff

Report the full-path baseline, selected route, removed work, allocation and copy changes, all viable backends and
Torch-specific candidates considered, thread and benchmark matrix, correctness and memory evidence, regressions,
rejected regions, and remaining follow-ups.
