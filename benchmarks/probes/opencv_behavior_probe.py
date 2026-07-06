"""Probe raw OpenCV behavior that affects Albucore routing workarounds.

This script intentionally avoids Albucore wrappers. It answers whether OpenCV
itself supports shape, channel, interpolation, and border cases that Albucore
currently guards or chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _status(fn: Any) -> dict[str, Any]:
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - this is a behavior probe
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0],
        }

    return {
        "ok": True,
        "shape": list(out.shape) if hasattr(out, "shape") else None,
        "dtype": str(out.dtype) if hasattr(out, "dtype") else None,
    }


def _image(channels: int, dtype: np.dtype[Any] = np.uint8, height: int = 17, width: int = 19) -> np.ndarray:
    rng = np.random.default_rng(0)
    if dtype == np.uint8:
        return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return rng.random((height, width, channels), dtype=np.float32)


def _map_xy(height: int = 17, width: int = 19) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return grid_x + 0.125, grid_y + 0.25


def run_probe() -> dict[str, Any]:
    affine = np.array([[1.0, 0.05, 1.0], [0.03, 1.0, -1.0]], dtype=np.float32)
    perspective = np.array(
        [[1.0, 0.02, 1.0], [0.01, 1.0, -1.0], [0.0005, 0.0002, 1.0]],
        dtype=np.float32,
    )
    map_x, map_y = _map_xy()

    interpolations = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "linear_exact": cv2.INTER_LINEAR_EXACT,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }

    results: dict[str, Any] = {
        "meta": {
            "cv2": cv2.__version__,
            "numpy": np.__version__,
        },
        "lut_single_channel": {},
        "flip": {},
        "resize": {},
        "warp_affine": {},
        "warp_perspective": {},
        "remap": {},
        "copy_make_border": {},
        "median_blur": {},
    }

    one_channel = _image(1)
    lut = np.arange(256, dtype=np.uint8)
    results["lut_single_channel"] = _status(lambda: cv2.LUT(one_channel, lut))

    for channels in [1, 3, 4, 5, 9, 513]:
        img = _image(channels)
        key = f"c{channels}"
        results["flip"][key] = _status(lambda img=img: cv2.flip(img, 1))

    for channels in [1, 3, 4, 5, 9]:
        img = _image(channels)
        key = f"c{channels}"
        results["resize"][key] = {
            "area_down": _status(lambda img=img: cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)),
            "area_up": _status(lambda img=img: cv2.resize(img, (38, 34), interpolation=cv2.INTER_AREA)),
            "linear": _status(lambda img=img: cv2.resize(img, (38, 34), interpolation=cv2.INTER_LINEAR)),
        }

    for channels in [1, 3, 4, 5, 9]:
        img = _image(channels)
        key = f"c{channels}"
        results["warp_affine"][key] = {
            name: _status(lambda img=img, interp=interp: cv2.warpAffine(img, affine, (19, 17), flags=interp))
            for name, interp in interpolations.items()
        }
        results["warp_perspective"][key] = {
            name: _status(lambda img=img, interp=interp: cv2.warpPerspective(img, perspective, (19, 17), flags=interp))
            for name, interp in interpolations.items()
        }
        results["remap"][key] = {
            name: _status(lambda img=img, interp=interp: cv2.remap(img, map_x, map_y, interpolation=interp))
            for name, interp in interpolations.items()
        }

    for channels in [1, 3, 4, 5, 9]:
        img = _image(channels)
        key = f"c{channels}"
        scalar_value = 7
        channel_value = tuple(range(channels))
        results["copy_make_border"][key] = {
            "constant_scalar": _status(
                lambda img=img: cv2.copyMakeBorder(img, 2, 3, 4, 5, cv2.BORDER_CONSTANT, value=scalar_value),
            ),
            "constant_per_channel": _status(
                lambda img=img, value=channel_value: cv2.copyMakeBorder(
                    img,
                    2,
                    3,
                    4,
                    5,
                    cv2.BORDER_CONSTANT,
                    value=value,
                ),
            ),
            "reflect": _status(lambda img=img: cv2.copyMakeBorder(img, 2, 3, 4, 5, cv2.BORDER_REFLECT)),
        }

    for channels in [1, 3, 4, 5, 9]:
        img = _image(channels)
        key = f"c{channels}"
        results["median_blur"][key] = {
            f"k{kernel}": _status(lambda img=img, kernel=kernel: cv2.medianBlur(img, kernel))
            for kernel in [3, 5, 7, 9]
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results = run_probe()
    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
