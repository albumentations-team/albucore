# ruff: noqa: INP001
"""Compare ``match_histograms_uint8`` with the AlbumentationsX per-channel kernel and histogram routes.

Run from the repository root outside any sandbox with:

    uv run python benchmarks/benchmark_match_histograms.py --threads 1 \
      --output-json benchmarks/results/match-histograms.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from functools import partial
from pathlib import Path

import cv2
import numkong as nk
import numpy as np
import torch
from benchmark_threads import THREAD_COUNT, configure_libraries
from timing import bench_paired_wall_ms, median_ms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_output import output_path

from albucore import apply_uint8_lut, match_histograms_uint8
from albucore.histogram import _histogram

CASES = ((256, 256), (512, 512), (1024, 1024))
CHANNELS = (1, 3, 5)


def _ax_channel_histogram(pixels: np.ndarray, channel: int) -> np.ndarray:
    if pixels.shape[1] <= 2**24:
        return cv2.calcHist([pixels], [channel], None, [256], [0, 256]).ravel().astype(np.int64)
    return np.bincount(pixels[..., channel].ravel(), minlength=256)


def _ax_matched_channel_lut(source: np.ndarray, template: np.ndarray, channel: int) -> np.ndarray:
    src_counts = _ax_channel_histogram(source, channel)
    tmpl_counts = _ax_channel_histogram(template, channel)
    tmpl_values = np.nonzero(tmpl_counts)[0]
    tmpl_counts = tmpl_counts[tmpl_values]
    src_quantiles = np.cumsum(src_counts) / source.shape[1]
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.shape[1]
    return np.interp(src_quantiles, tmpl_quantiles, tmpl_values).astype(np.uint8)


def ax_match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """AlbumentationsX ``match_histograms`` kernel for uint8 input, copied for comparison."""
    channels = image.shape[-1]
    image = np.ascontiguousarray(image)
    source_pixels = image.reshape(1, -1, channels)
    template_pixels = np.ascontiguousarray(reference).reshape(1, -1, channels)
    luts = np.empty((256, 1, channels), dtype=np.uint8)
    for channel in range(channels):
        luts[:, 0, channel] = _ax_matched_channel_lut(source_pixels, template_pixels, channel)
    return apply_uint8_lut(image, luts if channels > 1 else luts[:, 0, 0])


def old_ax_match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """AlbumentationsX kernel before the OpenCV histogram and LUT form: bincount plus a float64 gather."""
    result = np.empty(image.shape, dtype=np.uint8)
    for channel in range(image.shape[-1]):
        source = image[..., channel]
        template = reference[..., channel]
        src_counts = np.bincount(source.ravel(), minlength=256)
        tmpl_counts = np.bincount(template.ravel(), minlength=256)
        tmpl_values = np.nonzero(tmpl_counts)[0]
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts[tmpl_values]) / template.size
        interp_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        result[..., channel] = interp_values[source].astype(np.uint8)
    return result


def histograms_slice_calchist(image: np.ndarray) -> list[np.ndarray]:
    return [_histogram(image[..., channel]) for channel in range(image.shape[-1])]


def histograms_whole_calchist(image: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(cv2.calcHist([image], [channel], None, [256], (0, 256)), dtype=np.int64).ravel()
        for channel in range(image.shape[-1])
    ]


def histograms_bincount(image: np.ndarray) -> list[np.ndarray]:
    return [np.bincount(image[..., channel].ravel(), minlength=256) for channel in range(image.shape[-1])]


HISTOGRAM_ROUTES = {
    "slice_calchist": histograms_slice_calchist,
    "whole_calchist": histograms_whole_calchist,
    "bincount": histograms_bincount,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=THREAD_COUNT)
    parser.add_argument("--repeats", type=int, default=41)
    parser.add_argument("--output-json", type=output_path, required=True)
    args = parser.parse_args()
    thread_settings = configure_libraries(torch, cv2, args.threads)
    if platform.system() == "Darwin" and platform.machine() == "arm64" and not nk.get_capabilities().get("neon"):
        msg = "NumKong NEON is unavailable; run outside the sandbox."
        raise RuntimeError(msg)

    rng = np.random.default_rng(260)
    rows = []
    for height, width in CASES:
        for channels in CHANNELS:
            image = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
            reference = rng.integers(20, 230, (height, width, channels), dtype=np.uint8)
            baseline = partial(ax_match_histograms, image, reference)
            np.testing.assert_array_equal(old_ax_match_histograms(image, reference), baseline(), strict=True)
            old_form_ms = median_ms(partial(old_ax_match_histograms, image, reference), repeats=args.repeats, warmup=5)
            current = partial(match_histograms_uint8, image, reference)
            np.testing.assert_array_equal(baseline(), current(), strict=True)
            old, new = bench_paired_wall_ms(baseline, current, repeats=args.repeats, warmup=5)
            row: dict[str, object] = {
                "shape": (height, width, channels),
                "ax_ms": old.median,
                "ax_mad_ms": old.mad,
                "public_ms": new.median,
                "public_mad_ms": new.mad,
                "speedup": old.median / new.median,
                "old_ax_ms": old_form_ms,
                "speedup_vs_old_ax": old_form_ms / new.median,
            }
            expected = histograms_bincount(image)
            for name, route in HISTOGRAM_ROUTES.items():
                for got, want in zip(route(image), expected, strict=True):
                    np.testing.assert_array_equal(got, want)
                row[f"hist_{name}_ms"] = median_ms(partial(route, image), repeats=args.repeats, warmup=5)
            rows.append(row)
            print(  # noqa: T201
                f"{row['shape']}: old {row['old_ax_ms']:.3f} AX {row['ax_ms']:.3f} -> public {row['public_ms']:.3f} ms ({row['speedup']:.2f}x); "
                + " ".join(f"{key[5:-3]}={value:.3f}" for key, value in row.items() if key.startswith("hist_")),
            )

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "numkong_capabilities": nk.get_capabilities(),
        "threads": thread_settings,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
