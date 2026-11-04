"""Paired Core image-kernel and AX Compose comparisons against AX 349a12c."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

FACADE_SOURCE = """def box_blur(img, ksize):
    return albucore.box_blur(img, ksize)

def convolve(img, kernel, border_mode=cv2.BORDER_DEFAULT):
    if kernel.ndim not in (2, 3):
        raise ValueError(f'kernel must have 2 or 3 dimensions, got {kernel.ndim}')
    channels = 1 if img.ndim == 2 else img.shape[-1]
    if kernel.ndim == 3 and kernel.shape[0] != channels:
        raise ValueError(f'Expected one kernel per channel ({channels}), got {kernel.shape[0]}')
    return albucore.convolve(img, kernel, border_mode)

def erode(img, kernel):
    return albucore.erode(img, kernel)

def dilate(img, kernel):
    return albucore.dilate(img, kernel)

def linear_transformation_rgb(img, transformation_matrix):
    if img.ndim not in (3, 4):
        raise ValueError(f'Expected input shape (H, W, 3) or (B, H, W, 3), got {img.shape}')
    return albucore.linear_transformation_rgb(img, transformation_matrix)

def unsharp_mask(image, ksize, sigma, alpha, threshold):
    return albucore.unsharp_mask(image, ksize, sigma, alpha, threshold)"""
OWNER_MODULES = {
    "box_blur": "albumentations.augmentations.blur.functional",
    "convolve": "albumentations.augmentations.pixel._functional_sharpness",
    "erode": "albumentations.augmentations.geometric._functional_images",
    "dilate": "albumentations.augmentations.geometric._functional_images",
    "linear_transformation_rgb": "albumentations.augmentations.pixel._functional_color",
    "unsharp_mask": "albumentations.augmentations.pixel._functional_sharpness",
}


def _facades(core: Any, cv2: Any) -> dict[str, Any]:
    namespace = {"cv2": cv2, "albucore": core}
    exec(compile(FACADE_SOURCE, "<AX Core delegates>", "exec"), namespace)  # noqa: S102 - literal reviewed delegates
    return {name: namespace[name] for name in OWNER_MODULES}


def _swaps(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[tuple[Any, str, Any, Any]]:
    replacements = {id(baseline[name]): candidate[name] for name in baseline}
    result = []
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("albumentations."):
            continue
        for key, value in tuple(vars(module).items()):
            if id(value) in replacements:
                result.append((module, key, value, replacements[id(value)]))
    return result


def _transform(albu: Any, name: str) -> Any:
    return {
        "box_blur": lambda: albu.Blur(blur_range=(3, 3), p=1),
        "convolve": lambda: albu.Sharpen(method="kernel", alpha_range=(0.5, 0.5), lightness_range=(1, 1), p=1),
        "erode": lambda: albu.Morphological(scale=(3, 3), operation="erosion", p=1),
        "dilate": lambda: albu.Morphological(scale=(3, 3), operation="dilation", p=1),
        "linear_transformation_rgb": lambda: albu.ToSepia(p=1),
        "unsharp_mask": lambda: albu.UnsharpMask(blur_range=(3, 3), sigma_range=(1, 1), alpha_range=(0.5, 0.5), p=1),
    }[name]()


def _calls(
    albu: Any, core: Any, np: Any, name: str, layer: str, image: Any, baseline: dict[str, Any]
) -> dict[str, Any]:
    if layer == "core":
        kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]], dtype=np.float32)
        controls = {
            "box_blur": (3,),
            "convolve": (kernel,),
            "erode": (np.ones((3, 3), dtype=np.uint8),),
            "dilate": (np.ones((3, 3), dtype=np.uint8),),
            "linear_transformation_rgb": (
                np.array([[1.125, 0.125, 0], [0, 0.875, 0.125], [-0.125, 0.25, 1]], dtype=np.float32),
            ),
            "unsharp_mask": (3, 1.0, 0.5, 10),
        }[name]
        old, new = baseline[name], getattr(core, name)
        return {"baseline": lambda: old(image, *controls), "candidate": lambda: new(image, *controls)}
    if layer == "direct":
        transform = _transform(albu, name)
        recorded = albu.ReplayCompose([transform], seed=137, strict=True)(image=image)
        payload = recorded["replay"]["transforms"][0]["params"]
        params = dict(payload["params"])
        for group in payload["target_params"]:
            if "image" in group["targets"]:
                params.update(group["params"])
        return dict.fromkeys(("baseline", "candidate"), lambda: transform.apply(image, **params))
    calls = {}
    for variant in ("baseline", "candidate"):
        compose = albu.Compose([_transform(albu, name)], seed=137, strict=True)
        calls[variant] = lambda compose=compose: compose(image=image)["image"]
    return calls


def _cases() -> list[tuple[str, str, int, int, int, str]]:
    result = []
    for name in OWNER_MODULES:
        for height, width in ((128, 160), (240, 320), (480, 640), (768, 1024)):
            for channels in (3,) if name == "linear_transformation_rgb" else (1, 3, 9):
                for dtype in ("uint8", "float16", "float32"):
                    result.append((name, "core", height, width, channels, dtype))
        for size in (256, 512, 1024):
            for channels in (3,) if name == "linear_transformation_rgb" else (1, 3, 5):
                for dtype in ("uint8", "float16", "float32"):
                    for layer in ("direct", "compose"):
                        result.append((name, layer, size, size, channels, dtype))
    return result


def _measure(
    calls: dict[str, Any], select: Any, sample: Any, fingerprint: Any, image: Any, samples_count: int
) -> dict[str, Any]:
    original = fingerprint(image)
    outputs = {}
    pilot = []
    for variant, call in calls.items():
        select(variant)
        outputs[variant] = fingerprint(call())
        for _ in range(10):
            call()
        pilot.append(sample(call, 10))
    if outputs["baseline"] != outputs["candidate"]:
        raise RuntimeError("Bitwise output mismatch")
    count = max(100 if max(pilot) < 50_000 else 1, min(10000, math.ceil(10_000_000 / max(pilot))))
    timings: dict[str, list[float]] = {name: [] for name in calls}
    for iteration in range(samples_count):
        for variant in ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline"):
            select(variant)
            timings[variant].append(sample(calls[variant], count))
    if fingerprint(image) != original:
        raise RuntimeError("Input was modified")
    medians = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "median_ns": medians,
        "samples_ns": timings,
        "calls_per_sample": count,
        "candidate_over_baseline": medians["candidate"] / medians["baseline"],
        "bitwise_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--case-list", type=Path)
    args = parser.parse_args()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    core_root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(core_root), str(args.baseline_ax.resolve())]
    import cv2
    import numkong as nk
    import numpy as np
    import torch
    import albucore as core
    import albumentations as albu

    helper_path = args.baseline_ax / "tools/benchmark_conversion_cleanup.py"
    spec = importlib.util.spec_from_file_location("ax_conversion_benchmark", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("AX benchmark helpers are missing")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    _fingerprint, _sample = helpers._fingerprint, helpers._sample
    _source_hash, _resume_rows = helpers._source_hash, helpers._resume_rows

    if not Path(core.__file__).is_relative_to(core_root) or not Path(albu.__file__).is_relative_to(
        args.baseline_ax.resolve()
    ):
        raise RuntimeError("Imported package does not match selected source")
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    capabilities = nk.get_capabilities()
    if sys.platform == "darwin" and not capabilities.get("neon"):
        raise RuntimeError("Native NEON is required")
    baseline = {name: getattr(importlib.import_module(module), name) for name, module in OWNER_MODULES.items()}
    swaps = _swaps(baseline, _facades(core, cv2))

    def select(variant: str) -> None:
        for module, key, old, new in swaps:
            setattr(module, key, old if variant == "baseline" else new)

    cpu = (
        subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    )
    metadata = {
        "baseline_ax": "349a12c477fb026235981d179db7441fdb3666c3",
        "baseline_source": _source_hash(args.baseline_ax / "albumentations"),
        "candidate_source": _source_hash(core_root / "albucore"),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "facades_source": hashlib.sha256(FACADE_SOURCE.encode()).hexdigest(),
        "cpu": cpu,
        "python": sys.version,
        "platform": platform.platform(),
        "capabilities": capabilities,
        "samples": args.samples,
        "threads": {
            "opencv": cv2.getNumThreads(),
            "torch": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "albucore",
                "numpy",
                "opencv-python-headless",
                "torch",
                "numkong",
                "stringzilla",
                "pydantic",
                "scipy",
            ]
        },
    }
    rows = _resume_rows(args.output, metadata)
    completed = {row["case"] for row in rows}
    selected = set(json.loads(args.case_list.read_text())) if args.case_list else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name, layer, height, width, channels, dtype_name in _cases():
            case = f"{name}|{layer}|{height}x{width}x{channels}|{dtype_name}"
            if case in completed or selected is not None and case not in selected:
                continue
            select("candidate")
            values = np.random.default_rng(137).random((height, width, channels), dtype=np.float32)
            image = (
                (values * 255).astype(np.uint8)
                if dtype_name == "uint8"
                else values.astype(np.float16).astype(np.dtype(dtype_name))
            )
            calls = _calls(albu, core, np, name, layer, image, baseline)
            row = {"case": case, **_measure(calls, select, _sample, _fingerprint, image, args.samples)}
            rows.append(row)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
            temporary.replace(args.output)
            print(case, round(row["candidate_over_baseline"], 4), flush=True)
    finally:
        select("baseline")


if __name__ == "__main__":
    main()
