from __future__ import annotations

import argparse
import os
import time
from typing import Any

import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import simsimd as ss
import json
import stringzilla as sz

import platform
from pathlib import Path
import psutil

# Disable multithreading for fair comparison
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

rng = np.random.default_rng()

def lut_opencv(img: np.ndarray, lut: np.ndarray, inplace: bool = False) -> np.ndarray:
    if inplace:
        cv2.LUT(img, lut, dst=img)
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(cv2.LUT(img, lut))

def lut_stringzilla(img: np.ndarray, lut: np.ndarray, inplace: bool = True) -> np.ndarray:
    if not inplace:
        img = img.copy()
    sz.translate(memoryview(img), memoryview(lut), inplace=True)
    return np.ascontiguousarray(img)


def lut_numpy(img: np.ndarray, lut: np.ndarray, inplace: bool = False) -> np.ndarray:
    if inplace:
        # NumPy doesn't have true inplace LUT, so we need to copy back
        np.take(lut, img, out=img)
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(np.take(lut, img))


def add_weighted_simsimd(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    original_shape = img1.shape
    original_dtype = img1.dtype

    return np.frombuffer(
        ss.wsum(img1.reshape(-1), img2.astype(original_dtype).reshape(-1), alpha=weight1, beta=weight2),
        dtype=original_dtype,
    ).reshape(
        original_shape,
    )


def add_arrays_opencv(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return cv2.add(img1, img2)

def add_arrays_simsimd(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return add_weighted_simsimd(img1, 1, img2, 1)

def add_arrays_numpy(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return np.add(img1, img2)

def add_weighted_opencv(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return cv2.addWeighted(img1, weight1, img2, weight2, 0)

def add_weighted_numpy(img1: np.ndarray, weight1: float, img2: np.ndarray, weight2: float) -> np.ndarray:
    return img1.astype(np.float32) * weight1 + img2.astype(np.float32) * weight2


def get_cpu_info() -> dict[str, str]:
    cpu_info = {}

    # Get CPU name
    if platform.system() == "Windows":
        cpu_info["name"] = platform.processor()
    elif platform.system() == "Darwin":
        import subprocess

        cpu_info["name"] = (
            subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()  # noqa: S603
        )
    elif platform.system() == "Linux":
        with Path("/proc/cpuinfo").open() as f:
            info = f.readlines()
        cpu_info["name"] = next(x.strip().split(":")[1] for x in info if "model name" in x)
    else:
        cpu_info["name"] = "Unknown"

    # Get CPU frequency
    freq = psutil.cpu_freq()
    if freq:
        cpu_info["freq"] = f"Current: {freq.current:.2f} MHz, Min: {freq.min:.2f} MHz, Max: {freq.max:.2f} MHz"
    else:
        cpu_info["freq"] = "Frequency information not available"

    # Get CPU cores
    cpu_info["physical_cores"] = psutil.cpu_count(logical=False)
    cpu_info["total_cores"] = psutil.cpu_count(logical=True)

    return cpu_info

def generate_image(shape: tuple[int, int], channels: int = 1, dtype: type = np.uint8) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, size=(*shape, channels), dtype=dtype)
    else:  # float32
        return rng.random(size=(*shape, channels), dtype=dtype)

def benchmark_operation(
    func: Any,
    operation: str,
    img1: np.ndarray,
    img2: np.ndarray,
    weight1: float | None = None,
    weight2: float | None = None,
    inplace: bool = False,
    number: int = 100,
    warmup: int = 10,
    clear_cache: bool = False,
) -> tuple[float, float]:
    # Warmup runs
    for _ in range(warmup):
        if operation == "add_arrays":
            func(img1, img2)
        elif operation == "add_weighted":
            func(img1, weight1, img2, weight2)
        else:  # LUT operations
            func(img1, img2, inplace=inplace)

    # Ensure garbage collection doesn't interfere
    gc.collect()
    gc.disable()

    try:
        times = []
        for _ in range(3):
            if clear_cache:
                _clear_cpu_cache()

            start = time.perf_counter_ns()
            for _ in range(number):
                if operation == "add_arrays":
                    func(img1, img2)
                elif operation == "add_weighted":
                    func(img1, weight1, img2, weight2)
                else:  # LUT operations
                    func(img1, img2, inplace=inplace)
            end = time.perf_counter_ns()

            total_time = (end - start) / 1e9
            avg_time = total_time / number
            times.append(avg_time)

        median = np.median(times)
        mad = np.median(np.abs(times - median))
        sem = 1.4826 * mad / np.sqrt(len(times))

        return median, sem

    finally:
        gc.enable()

def _clear_cpu_cache() -> None:
    """Optional: Clear CPU cache between runs."""
    # Size that's larger than your CPU cache
    cache_size = 20 * 1024 * 1024  # 20MB
    arr = np.random.rand(cache_size // 8)  # 8 bytes per float64
    arr.sum()  # Force read


def run_benchmarks(num_runs: int) -> pd.DataFrame:
    """Run benchmarks for different operations, data types, and image sizes.

    Args:
        num_runs: Number of iterations for each benchmark

    Returns:
        DataFrame with benchmark results
    """
    # Test configurations
    image_sizes: list[tuple[int, int]] = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    channel_counts: list[int] = [1, 3, 5]
    weight_pairs: list[tuple[float, float]] = [
        (0.5, 0.5),
    ]
    dtypes: list[type] = [np.uint8, np.float32]
    results: list[dict[str, Any]] = []

    operations: list[tuple[str, dict[str, Any]]] = [
        ("add_arrays", {
            "OpenCV": add_arrays_opencv,
            "NumPy": add_arrays_numpy,
            "SimSIMD": add_arrays_simsimd,
        }),
        ("add_weighted", {
            "OpenCV": add_weighted_opencv,
            "NumPy": add_weighted_numpy,
            "SimSIMD": add_weighted_simsimd,
        }),
        ("lut", {
            "OpenCV": lut_opencv,
            "NumPy": lut_numpy,
            "StringZilla": lut_stringzilla,
        }),
    ]

    # Add inplace modes for LUT operations
    inplace_modes = [False, True]

    try:
        for operation_name, implementations in tqdm(operations, desc="Operations"):
            for dtype in tqdm(dtypes, desc="Data types", leave=False):
                if operation_name == "lut" and dtype != np.uint8:
                    continue
                for size in tqdm(image_sizes, desc="Image sizes", leave=False):
                    for channels in tqdm(channel_counts, desc="Channels", leave=False):
                        img1 = generate_image(size, channels, dtype)
                        img1 = np.ascontiguousarray(img1)

                        if operation_name == "lut":
                            # Generate LUT table (256 entries for uint8)
                            lut = np.arange(256, dtype=np.uint8)
                            np.random.shuffle(lut)
                            img2 = lut
                            weight_list = [(0.0, 0.0)]
                            modes = inplace_modes
                        else:
                            img2 = generate_image(size, channels, dtype)
                            img2 = np.ascontiguousarray(img2)
                            weight_list = weight_pairs if operation_name == "add_weighted" else [(0.0, 0.0)]
                            modes = [False]  # No inplace mode for other operations

                        for inplace in modes:
                            for weight1, weight2 in tqdm(weight_list, desc="Weights", leave=False):
                                timings: dict[str, tuple[float, float]] = {}
                                for impl_name, func in implementations.items():
                                    median, sem = benchmark_operation(
                                        func=func,
                                        operation=operation_name,
                                        img1=img1.copy(),  # Copy to ensure fresh data for each implementation
                                        img2=img2.copy() if operation_name != "lut" else img2, # Don't copy LUT
                                        weight1=weight1,
                                        weight2=weight2,
                                        inplace=inplace,
                                        number=num_runs,
                                        warmup=10
                                    )
                                    timings[impl_name] = (median, sem)

                                # Create base result dictionary
                                result: dict[str, Any] = {
                                    "Operation": f"{operation_name}{'_inplace' if inplace else ''}",
                                    "Data Type": str(dtype.__name__),
                                    "Size": f"{size[0]}x{size[1]}",
                                    "Channels": channels,
                                    "Total Pixels": size[0] * size[1] * channels,
                                    "Weights": f"({weight1}, {weight2})" if operation_name == "add_weighted" else None,
                                    "OpenCV": {
                                        "time_ms": f"{timings['OpenCV'][0]*1000:.4f} +- {timings['OpenCV'][1]*1000:.4f}"
                                    },
                                    "NumPy": {
                                        "time_ms": f"{timings['NumPy'][0]*1000:.4f} +- {timings['NumPy'][1]*1000:.4f}"
                                    },
                                    "SimSIMD": None,
                                    "StringZilla": None
                                }

                                # Add SimSIMD or StringZilla results based on operation
                                if operation_name == "lut":
                                    sz_median, sz_sem = timings["StringZilla"]
                                    opencv_median, opencv_sem = timings["OpenCV"]
                                    numpy_median, numpy_sem = timings["NumPy"]

                                    speedup_vs_opencv = opencv_median / sz_median
                                    speedup_vs_numpy = numpy_median / sz_median

                                    speedup_opencv_error = speedup_vs_opencv * np.sqrt(
                                        (opencv_sem / opencv_median) ** 2 +
                                        (sz_sem / sz_median) ** 2
                                    )
                                    speedup_numpy_error = speedup_vs_numpy * np.sqrt(
                                        (numpy_sem / numpy_median) ** 2 +
                                        (sz_sem / sz_median) ** 2
                                    )

                                    result["StringZilla"] = {
                                        "time_ms": f"{sz_median*1000:.4f} +- {sz_sem*1000:.4f}",
                                        "speedup_vs_opencv": f"{speedup_vs_opencv:.4f} +- {speedup_opencv_error:.4f}",
                                        "speedup_vs_numpy": f"{speedup_vs_numpy:.4f} +- {speedup_numpy_error:.4f}"
                                    }
                                else:  # add_arrays or add_weighted
                                    simsimd_median, simsimd_sem = timings["SimSIMD"]
                                    result["SimSIMD"] = {
                                        "time_ms": f"{simsimd_median*1000:.4f} +- {simsimd_sem*1000:.4f}"
                                    }
                                results.append(result)

    finally:
        # Restore process priority
        try:
            os.nice(0)
        except (AttributeError, PermissionError):
            pass

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark array operations")
    parser.add_argument("-r", "--runs", type=int, default=100, help="Number of runs for each benchmark")
    args = parser.parse_args()

    results_df = run_benchmarks(args.runs)

    # Save detailed results in CSV
    results_df.to_csv("array_operations_benchmark_results.csv", index=False)

    # Save results in JSON
    # Convert DataFrame to a more JSON-friendly format
    json_results = {
        "metadata": {
            "cpu_info": get_cpu_info(),
            "num_runs": args.runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": []
    }

    # Convert each row to a dictionary
    for _, row in results_df.iterrows():
        result_dict = {}
        for column in results_df.columns:
            value = row[column]
            # Convert numpy types to native Python types
            if isinstance(value, (np.int32, np.int64)):
                value = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                if np.isnan(value):
                    value = None
                else:
                    value = float(value)
            elif pd.isna(value):  # Handle NaN/None values
                value = None
            result_dict[column] = value
        json_results["results"].append(result_dict) # type: ignore

    # Save to JSON file with nice formatting
    with open("array_operations_benchmark_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
