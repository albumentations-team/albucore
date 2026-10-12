"""Compare allocating FP16/FP32 casts through the public NumPy and CPU Tensor routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from functools import partial
from pathlib import Path

from benchmark_threads import THREAD_COUNT, configure_libraries

import cv2
import numkong as nk
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from albucore import exp, from_float, median_blur, normalize_per_image, resize
from albucore._float_cast import _numkong_cast_preserves_numpy_errors
from albucore.functions import power_opencv


def measure(calls, repeats):
    samples = {name: [] for name in calls}
    for call in calls.values():
        for _ in range(5):
            call()
    names = list(calls)
    for index in range(repeats):
        order = names[index % len(names) :] + names[: index % len(names)]
        if index % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter_ns()
            result = calls[name]()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
            del result
    return {
        name: {
            "median_ms": float(np.median(values)),
            "mad_ms": float(np.median(np.abs(values - np.median(values)))),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }


def numkong_cast(image, dtype):
    result = np.empty(image.shape, dtype=dtype)
    nk.astype(image, np.dtype(dtype).name, out=result)
    return result


def format_results(result):
    return (
        re.sub(
            r'("(?:samples_ms|shape|strides)": )\[([^\[\]]*)\]',
            lambda match: match[1] + "[" + " ".join(match[2].split()) + "]",
            json.dumps(result, indent=2),
        )
        + "\n"
    )


def guarded_torch_cast(image, dtype):
    if not _numkong_cast_preserves_numpy_errors(image):
        return image.astype(dtype)
    result = np.empty(image.shape, dtype=dtype)
    torch.from_numpy(result).copy_(torch.from_numpy(image))
    return result


def numpy_roundtrip(operation, image):
    return operation(image.astype(np.float32)).astype(np.float16)


def router_rows(rng, repeats):
    rows = []
    for shape in [(480, 640, 1), (480, 640, 3), (768, 1024, 9)]:
        image = rng.uniform(0.01, 0.99, shape).astype(np.float16)
        height, width = shape[:2]
        operations = {
            "resize_linear": partial(resize, dsize=(width // 2, height // 2)),
            "median3": partial(median_blur, ksize=3),
            "normalize_image": partial(normalize_per_image, normalization="image"),
            "exp": exp,
            "power_opencv": partial(power_opencv, value=1.3),
        }
        for name, operation in operations.items():
            calls = {
                "public_half": partial(operation, image),
                "numpy_roundtrip": partial(numpy_roundtrip, operation, image),
                "public_float32": partial(operation, image.astype(np.float32)),
            }
            np.testing.assert_array_equal(
                calls["public_half"]().view(np.uint16), calls["numpy_roundtrip"]().view(np.uint16)
            )
            rows.append({"op": name, "shape": shape, "timings": measure(calls, repeats)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--contiguous-only", action="store_true")
    parser.add_argument("--with-routers", action="store_true")
    parser.add_argument("--routers-only", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    settings = configure_libraries(torch, cv2, args.threads)
    rng = np.random.default_rng(183)
    shapes = [(1, 16_383, 1), (1, 16_384, 1), (1, 61_439, 1), (1, 61_440, 1)]
    shapes += [(h, w, c) for h, w in [(128, 160), (240, 320), (480, 640), (768, 1024)] for c in [1, 3, 9]]
    shapes += [(16, 128, 160, 1), (32, 128, 160, 3), (32, 240, 320, 1), (1, 32, 384, 384)]
    if args.routers_only:
        shapes = []
    rows = []
    for shape in shapes:
        half = rng.uniform(-1, 1, shape).astype(np.float16)
        for base, dtype in [(half, np.float32), (half.astype(np.float32), np.float16)]:
            views = {"C": base}
            if not args.contiguous_only and shape == (480, 640, 3):
                readonly = base.copy()
                readonly.flags.writeable = False
                views.update(
                    {
                        "XY_transposed": base.swapaxes(-3, -2),
                        "F": np.asfortranarray(base),
                        "negative": base[..., ::-1, :],
                        "slice": base[..., ::2, :],
                        "readonly": readonly,
                    }
                )
            for layout, image in views.items():
                calls = {"numpy": partial(image.astype, dtype), "public_numpy": partial(from_float, image, dtype)}
                if image.flags.c_contiguous:
                    calls["numkong_out"] = partial(numkong_cast, image, dtype)
                if layout != "readonly" and all(s >= 0 for s in image.strides):
                    if image.flags.c_contiguous:
                        calls["guarded_torch_out"] = partial(guarded_torch_cast, image, dtype)
                    tensor = torch.from_numpy(image)
                    target = torch.float32 if dtype == np.float32 else torch.float16
                    calls["torch"] = partial(tensor.to, target)
                    calls["public_tensor"] = partial(from_float, tensor, target)
                reference = image.astype(dtype)
                for name, call in calls.items():
                    result = call()
                    actual = result.numpy() if isinstance(result, torch.Tensor) else result
                    unsigned = np.uint32 if dtype == np.float32 else np.uint16
                    np.testing.assert_array_equal(actual.view(unsigned), reference.view(unsigned))
                    assert not np.shares_memory(actual, image), name
                    assert actual.flags.writeable, name
                row = {
                    "shape": image.shape,
                    "layout": layout,
                    "strides": image.strides,
                    "source_dtype": str(image.dtype),
                    "target_dtype": np.dtype(dtype).name,
                    "timings": measure(calls, args.repeats),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {**row, "timings": {name: round(t["median_ms"], 6) for name, t in row["timings"].items()}}
                    ),
                    flush=True,
                )
    capabilities = nk.get_capabilities()
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "numkong": nk.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
        },
        "capabilities": {
            name: capabilities.get(name, False)
            for name in [
                "serial",
                "neon",
                "haswell",
                "skylake",
                "icelake",
                "sapphire",
                "rvv",
                "powervsx",
                "v128relaxed",
            ]
        },
        "threads": settings,
        "source_sha256": {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / "albucore" / name).read_bytes()).hexdigest()
            for name in [
                "_float_cast.py",
                "arithmetic.py",
                "convert.py",
                "elementwise.py",
                "geometric.py",
                "normalize.py",
                "ops_misc.py",
                "stats.py",
            ]
        },
        "rows": rows,
        "router_rows": router_rows(rng, args.repeats) if args.with_routers or args.routers_only else [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(format_results(result))


if __name__ == "__main__":
    main()
