from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from albucore.functions import sz_lut
from benchmark.utils import get_cpu_info

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Create a global random number generator
rng = np.random.default_rng()


def generate_image(shape: tuple[int, int], channels: int = 1) -> np.ndarray:
    return rng.integers(0, 256, size=(*shape, channels), dtype=np.uint8)


def generate_lut() -> np.ndarray:
    return rng.integers(0, 256, size=256, dtype=np.uint8)


def benchmark_lut(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray], img: np.ndarray, lut: np.ndarray, number: int = 100
) -> tuple[float, float]:
    times = []
    for _ in range(number):
        start = time.perf_counter()
        func(img, lut)
        end = time.perf_counter()
        times.append(end - start)
    mean = np.mean(times)
    sem = np.std(times, ddof=1) / np.sqrt(number)  # Standard Error of the Mean
    return mean, sem


def run_benchmarks(num_runs: int) -> pd.DataFrame:
    image_sizes: list[tuple[int, int]] = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    channel_counts: list[int] = [1, 3, 4]
    results: list[dict[str, int | str | float]] = []

    for size in tqdm(image_sizes, desc="Image sizes"):
        for channels in tqdm(channel_counts, desc="Channels", leave=False):
            img = generate_image(size, channels)
            lut = generate_lut()

            cv2_mean, cv2_sem = benchmark_lut(cv2.LUT, img, lut, num_runs)
            sz_mean, sz_sem = benchmark_lut(sz_lut, img, lut, num_runs)

            speedup = cv2_mean / sz_mean
            speedup_error = speedup * np.sqrt((cv2_sem / cv2_mean) ** 2 + (sz_sem / sz_mean) ** 2)

            results.append(
                {
                    "Size": f"{size[0]}x{size[1]}",
                    "Channels": channels,
                    "Total Pixels": size[0] * size[1] * channels,
                    "cv2.LUT (ms)": f"{cv2_mean*1000:.4f} ± {cv2_sem*1000:.4f}",
                    "sz_lut (ms)": f"{sz_mean*1000:.4f} ± {sz_sem*1000:.4f}",
                    "Speedup": f"{speedup:.4f} ± {speedup_error:.4f}",
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LUT operations")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs for each benchmark")
    args = parser.parse_args()

    results_df: pd.DataFrame = run_benchmarks(args.runs)

    print("Benchmark Results:")
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv("lut_benchmark_results.csv", index=False)

    # Calculate and print average speedup
    speedups = results_df["Speedup"].apply(lambda x: float(x.split()[0]))
    avg_speedup: float = speedups.mean()
    speedup_sem: float = speedups.std(ddof=1) / np.sqrt(len(speedups))
    print(f"\nAverage Speedup: {avg_speedup:.4f} ± {speedup_sem:.4f}x")

    # Find best and worst cases
    best_case: pd.Series = results_df.loc[speedups.idxmax()]
    worst_case: pd.Series = results_df.loc[speedups.idxmin()]

    print(f"\nBest Speedup: {best_case['Speedup']}")
    print(f"  Image Size: {best_case['Size']}, Channels: {best_case['Channels']}")

    print(f"\nWorst Speedup: {worst_case['Speedup']}")
    print(f"  Image Size: {worst_case['Size']}, Channels: {worst_case['Channels']}")

    # Print CPU info
    cpu_info = get_cpu_info()
    print("\nCPU Information:")
    for key, value in cpu_info.items():
        print(f"  {key.capitalize()}: {value}")
