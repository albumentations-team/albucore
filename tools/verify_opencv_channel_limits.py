"""Verify OpenCV multichannel support for each function.

Run: python tools/verify_opencv_channel_limits.py [--markdown]

Outputs a per-function matrix: channels, dtypes, interpolation, borderValue, inplace.
Use --markdown for markdown table output.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import cv2
import numpy as np

CV_VERSION = cv2.__version__
CHANNEL_COUNTS = [1, 3, 4, 5, 8, 16, 32]
DTYPES = [np.uint8, np.float32]
INTERPOLATIONS = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT,
}


@dataclass
class Result:
    ok: bool
    error: str | None = None
    diff_max: float | None = None  # for resize numerical diff


def _make_img(h: int, w: int, c: int, dtype: type) -> np.ndarray:
    rng = np.random.default_rng(42)
    if dtype == np.uint8:
        return rng.integers(0, 256, (h, w, c), dtype=np.uint8)
    return rng.random((h, w, c), dtype=np.float32)


def test_blur() -> dict[str, Any]:
    """cv2.blur: channels, dtypes, inplace (dst=)."""
    results: dict[str, Any] = {"channels": {}, "inplace": None, "notes": []}
    h, w = 16, 16

    for c in CHANNEL_COUNTS:
        for dtype in DTYPES:
            key = f"{c}ch_{dtype.__name__}"
            try:
                img = _make_img(h, w, c, dtype)
                out = cv2.blur(img, (3, 3))
                results["channels"][key] = Result(ok=True)
            except cv2.error as e:
                results["channels"][key] = Result(ok=False, error=str(e))

    # inplace
    try:
        img = _make_img(h, w, 5, np.uint8)
        cv2.blur(img, (3, 3), dst=img)
        results["inplace"] = Result(ok=True)
    except cv2.error as e:
        results["inplace"] = Result(ok=False, error=str(e))

    return results


def test_median_blur() -> dict[str, Any]:
    """cv2.medianBlur: channels, dtypes, inplace."""
    results: dict[str, Any] = {"channels": {}, "inplace": None, "notes": []}
    h, w = 16, 16

    for c in CHANNEL_COUNTS:
        for dtype in DTYPES:
            key = f"{c}ch_{dtype.__name__}"
            try:
                img = _make_img(h, w, c, dtype)
                out = cv2.medianBlur(img, 3)
                results["channels"][key] = Result(ok=True)
            except cv2.error as e:
                results["channels"][key] = Result(ok=False, error=str(e))

    try:
        img = _make_img(h, w, 5, np.uint8)
        cv2.medianBlur(img, 3, dst=img)
        results["inplace"] = Result(ok=True)
    except cv2.error as e:
        results["inplace"] = Result(ok=False, error=str(e))

    return results


def test_gaussian_blur() -> dict[str, Any]:
    """cv2.GaussianBlur: channels, dtypes, inplace."""
    results: dict[str, Any] = {"channels": {}, "inplace": None}
    h, w = 16, 16

    for c in CHANNEL_COUNTS:
        for dtype in DTYPES:
            key = f"{c}ch_{dtype.__name__}"
            try:
                img = _make_img(h, w, c, dtype)
                out = cv2.GaussianBlur(img, (5, 5), 1.0)
                results["channels"][key] = Result(ok=True)
            except cv2.error as e:
                results["channels"][key] = Result(ok=False, error=str(e))

    try:
        img = _make_img(h, w, 5, np.uint8)
        cv2.GaussianBlur(img, (5, 5), 1.0, dst=img)
        results["inplace"] = Result(ok=True)
    except cv2.error as e:
        results["inplace"] = Result(ok=False, error=str(e))

    return results


def test_resize() -> dict[str, Any]:
    """cv2.resize: channels, dtypes, interpolation. No dst."""
    results: dict[str, Any] = {"channels": {}, "interpolation": {}, "notes": []}
    h, w = 32, 32
    dsize = (16, 16)

    for c in CHANNEL_COUNTS:
        for dtype in DTYPES:
            key = f"{c}ch_{dtype.__name__}"
            try:
                img = _make_img(h, w, c, dtype)
                out = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
                # For >4ch: compare direct vs per-channel (stark0908: ±1 pixel diff for uint8)
                diff_max = None
                if c > 4:
                    ref_chunks = []
                    for i in range(c):
                        ch = cv2.resize(img[:, :, i : i + 1], dsize, interpolation=cv2.INTER_LINEAR)
                        if ch.ndim == 2:
                            ch = np.expand_dims(ch, axis=-1)
                        ref_chunks.append(ch)
                    ref = np.concatenate(ref_chunks, axis=-1)
                    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
                    diff_max = float(np.max(diff))
                results["channels"][key] = Result(ok=True, diff_max=diff_max)
            except cv2.error as e:
                results["channels"][key] = Result(ok=False, error=str(e))

    for interp_name, interp_val in INTERPOLATIONS.items():
        try:
            img = _make_img(16, 16, 8, np.uint8)
            out = cv2.resize(img, (8, 8), interpolation=interp_val)
            results["interpolation"][interp_name] = Result(ok=True)
        except cv2.error as e:
            results["interpolation"][interp_name] = Result(ok=False, error=str(e))

    return results


def test_warp_affine() -> dict[str, Any]:
    """cv2.warpAffine: channels, interpolation, borderValue."""
    results: dict[str, Any] = {"channels": {}, "interpolation": {}, "borderValue": {}, "notes": []}
    h, w = 16, 16
    M = np.float32([[1, 0, 2], [0, 1, 2]])
    dsize = (w, h)

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            out = cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    for interp_name, interp_val in INTERPOLATIONS.items():
        try:
            img = _make_img(h, w, 8, np.uint8)
            out = cv2.warpAffine(img, M, dsize, flags=interp_val, borderValue=0)
            results["interpolation"][interp_name] = Result(ok=True)
        except cv2.error as e:
            results["interpolation"][interp_name] = Result(ok=False, error=str(e))

    # borderValue: scalar, (v,)*4, per-channel len>4, (124,116,104)
    for label, border_val in [
        ("scalar 0", 0),
        ("scalar 128", 128),
        ("(0,0,0,0)", (0, 0, 0, 0)),
        ("(124,116,104) len=3", (124, 116, 104)),
        ("(124,)*8 len=8", (124,) * 8),
    ]:
        try:
            img = _make_img(h, w, 8, np.uint8)
            out = cv2.warpAffine(img, M, dsize, borderValue=border_val)
            results["borderValue"][label] = Result(ok=True)
        except (cv2.error, TypeError) as e:
            results["borderValue"][label] = Result(ok=False, error=str(e))

    return results


def test_copy_make_border() -> dict[str, Any]:
    """cv2.copyMakeBorder: channels, borderValue."""
    results: dict[str, Any] = {"channels": {}, "borderValue": {}}
    h, w = 16, 16

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            out = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    for label, border_val in [
        ("scalar 0", 0),
        ("(0,0,0,0)", (0, 0, 0, 0)),
        ("(124,116,104) len=3", (124, 116, 104)),
        ("(124,)*8 len=8", (124,) * 8),
    ]:
        try:
            img = _make_img(h, w, 8, np.uint8)
            out = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_val)
            results["borderValue"][label] = Result(ok=True)
        except (cv2.error, TypeError) as e:
            results["borderValue"][label] = Result(ok=False, error=str(e))

    return results


def test_warp_perspective() -> dict[str, Any]:
    """cv2.warpPerspective: channels."""
    results: dict[str, Any] = {"channels": {}}
    h, w = 16, 16
    M = np.eye(3, dtype=np.float32)
    M[0, 2] = 1
    M[1, 2] = 1
    dsize = (w, h)

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            out = cv2.warpPerspective(img, M, dsize)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    return results


def test_remap() -> dict[str, Any]:
    """cv2.remap: channels."""
    results: dict[str, Any] = {"channels": {}}
    h, w = 16, 16
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    return results


def test_filter2d() -> dict[str, Any]:
    """cv2.filter2D: channels, inplace."""
    results: dict[str, Any] = {"channels": {}, "inplace": None}
    h, w = 16, 16
    kernel = np.ones((3, 3), np.float32) / 9

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            out = cv2.filter2D(img, -1, kernel)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    try:
        img = _make_img(h, w, 5, np.uint8)
        cv2.filter2D(img, -1, kernel, dst=img)
        results["inplace"] = Result(ok=True)
    except cv2.error as e:
        results["inplace"] = Result(ok=False, error=str(e))

    return results


def test_circle() -> dict[str, Any]:
    """cv2.circle with color (sun flare): channels. Expect cn<=4."""
    results: dict[str, Any] = {"channels": {}}
    h, w = 32, 32

    for c in CHANNEL_COUNTS:
        key = f"{c}ch"
        try:
            img = _make_img(h, w, c, np.uint8)
            cv2.circle(img, (16, 16), 5, (255, 255, 255), -1)
            results["channels"][key] = Result(ok=True)
        except cv2.error as e:
            results["channels"][key] = Result(ok=False, error=str(e))

    return results


def _fmt_result(r: Result) -> str:
    if r.ok:
        s = "Yes"
        if r.diff_max is not None:
            s += f" (diff={r.diff_max:.2f})"
        return s
    return f"No: {r.error[:60]}..." if r.error and len(r.error) > 60 else f"No: {r.error or '?'}"


def _print_section(name: str, data: dict[str, Any], indent: str = "") -> None:
    print(f"{indent}{name}:")
    for k, v in data.items():
        if isinstance(v, Result):
            print(f"{indent}  {k}: {_fmt_result(v)}")
        elif isinstance(v, dict):
            _print_section(k, v, indent + "  ")
        else:
            print(f"{indent}  {k}: {v}")


def main(markdown: bool = False) -> int:
    print(f"OpenCV {CV_VERSION}\n")
    print("=" * 60)

    tests = [
        ("cv2.blur (box_blur)", test_blur),
        ("cv2.medianBlur", test_median_blur),
        ("cv2.GaussianBlur", test_gaussian_blur),
        ("cv2.resize", test_resize),
        ("cv2.warpAffine", test_warp_affine),
        ("cv2.copyMakeBorder (pad)", test_copy_make_border),
        ("cv2.warpPerspective", test_warp_perspective),
        ("cv2.remap", test_remap),
        ("cv2.filter2D (convolve)", test_filter2d),
        ("cv2.circle (color)", test_circle),
    ]

    for name, fn in tests:
        print(f"\n### {name}\n")
        try:
            r = fn()
            _print_section("", r, indent="  ")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    return 0


def main_with_args() -> int:
    parser = argparse.ArgumentParser(description="Verify OpenCV multichannel support")
    parser.add_argument("--markdown", action="store_true", help="Output as markdown")
    args = parser.parse_args()
    return main(markdown=args.markdown)


if __name__ == "__main__":
    sys.exit(main_with_args())
