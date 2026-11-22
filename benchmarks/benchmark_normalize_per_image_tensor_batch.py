# ruff: noqa: E402, INP001
"""Compare per-image float32 Tensor batch loops with batched NumPy statistics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path, write_json_atomic
from albucore.normalize import normalize_per_image
from albucore.stats import DEFAULT_EPS
from albucore.tensor_bridge import numpy_to_tensor_layout, tensor_to_numpy_channel_last
from timing import bench_paired_wall_ms

SHAPES = (
    ("NCHW", (4, 3, 128, 160)),
    ("NCHW", (4, 1, 128, 160)),
    ("NCHW", (4, 9, 128, 160)),
    ("NCHW", (16, 3, 128, 160)),
    ("NCHW", (16, 9, 128, 160)),
    ("NCHW", (4, 3, 480, 640)),
    ("NCDHW", (4, 3, 16, 64, 80)),
    ("NCDHW", (4, 3, 32, 128, 160)),
    ("NCDHW", (4, 1, 32, 128, 160)),
    ("NCDHW", (4, 9, 16, 64, 80)),
)
MODES = ("image", "image_per_channel", "min_max", "min_max_per_channel")


def _batch_numpy_route(img: torch.Tensor, mode: str, out: torch.Tensor | None) -> torch.Tensor:
    source = img.numpy()
    if mode == "image":
        axes = tuple(range(1, source.ndim))
        center = source.mean(axis=axes, keepdims=True)
        divisor = source.std(axis=axes, keepdims=True) + DEFAULT_EPS
        lower, upper = -20, 20
    elif mode == "image_per_channel":
        axes = tuple(range(2, source.ndim))
        center = source.mean(axis=axes, keepdims=True)
        divisor = source.std(axis=axes, keepdims=True) + DEFAULT_EPS
        lower, upper = -20, 20
    elif mode == "min_max":
        axes = tuple(range(1, source.ndim))
        center = source.min(axis=axes, keepdims=True)
        divisor = source.max(axis=axes, keepdims=True) - center
        lower, upper = 0, 1
    else:
        per_channel = source.shape[1] > 1
        axes = tuple(range(2, source.ndim)) if per_channel else tuple(range(1, source.ndim))
        center = source.min(axis=axes, keepdims=True)
        divisor = source.max(axis=axes, keepdims=True) - center
        if per_channel:
            divisor = divisor + DEFAULT_EPS
        lower, upper = 0, 1
    result = np.empty_like(source) if out is None else out.numpy()
    np.subtract(source, center, out=result)
    if mode.startswith("min_max"):
        np.divide(result, divisor, out=result, where=divisor != 0)
    else:
        np.divide(result, divisor, out=result)
    np.clip(result, lower, upper, out=result)
    return torch.from_numpy(result) if out is None else out


def _per_item_route(img: torch.Tensor, mode: str, layout: str, out: torch.Tensor | None) -> torch.Tensor:
    item_layout = "CHW" if layout == "NCHW" else "CDHW"
    if out is not None:
        for index, item in enumerate(img):
            normalize_per_image(item, mode, out=out[index], layout=item_layout)
        return out

    image = tensor_to_numpy_channel_last(img, layout)
    first = normalize_per_image(image[0], mode)
    storage = np.empty(image.shape[0] * first.size, dtype=first.dtype)
    result = np.ndarray(image.shape, dtype=first.dtype, buffer=storage, strides=(first.nbytes, *first.strides))
    np.copyto(result[0], first)
    for index in range(1, image.shape[0]):
        normalize_per_image(image[index], mode, out=result[index])
    return numpy_to_tensor_layout(result, layout)


def _measure_case(
    tensor: torch.Tensor,
    layout: str,
    mode: str,
    *,
    with_output: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    baseline_out = torch.empty_like(tensor) if with_output else None
    candidate_out = torch.empty_like(tensor) if with_output else None

    def baseline() -> torch.Tensor:
        return _per_item_route(tensor, mode, layout, baseline_out)

    def candidate() -> torch.Tensor:
        return _batch_numpy_route(tensor, mode, candidate_out)

    expected = baseline()
    actual = candidate()
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5)
    if with_output and actual is not candidate_out:
        raise AssertionError("normalization candidate did not return the supplied output")
    baseline_timing, candidate_timing = bench_paired_wall_ms(
        baseline,
        candidate,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    baseline_ms, candidate_ms = baseline_timing.median, candidate_timing.median
    return {
        "layout": layout,
        "mode": mode,
        "shape": list(tensor.shape),
        "elements": tensor.numel(),
        "output": "supplied" if with_output else "allocated",
        "baseline_ms": baseline_ms,
        "baseline_samples_ms": baseline_timing.raw,
        "candidate_ms": candidate_ms,
        "candidate_samples_ms": candidate_timing.raw,
        "speedup": baseline_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)

    rng = np.random.default_rng(251121)
    rows = []
    for layout, shape in SHAPES:
        tensor = torch.from_numpy(rng.random(shape, dtype=np.float32))
        for mode in MODES:
            rows.extend(
                _measure_case(
                    tensor,
                    layout,
                    mode,
                    with_output=with_output,
                    args=args,
                )
                for with_output in (False, True)
            )

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "threads": 1,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "paired_order": "baseline starts first; call order alternates every sample",
        "rows": rows,
    }
    write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
