from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from albucore.functions import multiply_by_array, multiply_by_array_simsimd
from albucore.utils import clipped
from benchmark.utils import get_cpu_info

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

rng = np.random.default_rng()

@clipped
def multiply_array_simsimd_with_clip(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    return multiply_by_array_simsimd(img1, img2)


def generate_image(shape: tuple[int, int], channels: int = 1, dtype: np.dtype = np.uint8) -> np.ndarray:
    if dtype == np.uint8:
        return rng.integers(0, 256, size=(*shape, channels), dtype=dtype)
    else:  # float32
        return rng.random(size=(*shape, channels), dtype=dtype)


def benchmark_multiply_array(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    img1: np.ndarray,
    img2: np.ndarray,
    number: int = 100,
) -> tuple[float, float]:
    times = []
    for _ in range(number):
        start = time.perf_counter()
        func(img1, img2)
        end = time.perf_counter()
        times.append(end - start)
    mean = np.mean(times)
    sem = np.std(times, ddof=1) / np.sqrt(number)
    return mean, sem


def run_benchmarks(num_runs: int) -> pd.DataFrame:
    image_sizes: list[tuple[int, int]] = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    channel_counts: list[int] = [1, 3, 4, 5]
    dtypes: list[np.dtype] = [np.uint8, np.float32]
    results: list[dict[str, int | str | float]] = []

    for dtype in tqdm(dtypes, desc="Data types"):
        for size in tqdm(image_sizes, desc="Image sizes", leave=False):
            for channels in tqdm(channel_counts, desc="Channels", leave=False):
                img1 = generate_image(size, channels, dtype)
                img2 = generate_image(size, channels, dtype)

                numpy_mean, numpy_sem = benchmark_multiply_array(
                    multiply_by_array, img1, img2, num_runs
                )
                simsimd_mean, simsimd_sem = benchmark_multiply_array(
                    multiply_array_simsimd_with_clip, img1, img2, num_runs
                )

                speedup = numpy_mean / simsimd_mean
                speedup_error = speedup * np.sqrt(
                    (numpy_sem / numpy_mean) ** 2 + (simsimd_sem / simsimd_mean) ** 2
                )

                results.append(
                    {
                        "Dtype": str(dtype),
                        "Size": f"{size[0]}x{size[1]}",
                        "Channels": channels,
                        "Total Pixels": size[0] * size[1] * channels,
                        "NumPy (ms)": f"{numpy_mean*1000:.4f} ± {numpy_sem*1000:.4f}",
                        "SimSIMD (ms)": f"{simsimd_mean*1000:.4f} ± {simsimd_sem*1000:.4f}",
                        "Speedup": f"{speedup:.4f} ± {speedup_error:.4f}",
                    }
                )

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multiply_by_array operations")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs for each benchmark")
    args = parser.parse_args()

    results_df: pd.DataFrame = run_benchmarks(args.runs)

    print("Benchmark Results:")
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv("multiply_by_array_benchmark_results.csv", index=False)

    # Calculate and print average speedup
    speedups = results_df["Speedup"].apply(lambda x: float(x.split()[0]))
    avg_speedup: float = speedups.mean()
    speedup_sem: float = speedups.std(ddof=1) / np.sqrt(len(speedups))
    print(f"\nAverage Speedup: {avg_speedup:.4f} ± {speedup_sem:.4f}x")

    # Find best and worst cases
    best_case: pd.Series = results_df.loc[speedups.idxmax()]
    worst_case: pd.Series = results_df.loc[speedups.idxmin()]

    print(f"\nBest Speedup: {best_case['Speedup']}")
    print(f"  Image Size: {best_case['Size']}")
    print(f"  Channels: {best_case['Channels']}")

    print(f"\nWorst Speedup: {worst_case['Speedup']}")
    print(f"  Image Size: {worst_case['Size']}")
    print(f"  Channels: {worst_case['Channels']}")

    # Print CPU info
    cpu_info = get_cpu_info()
    print("\nCPU Information:")
    for key, value in cpu_info.items():
        print(f"  {key.capitalize()}: {value}")
