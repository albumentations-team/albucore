import argparse
import copy
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from timeit import Timer
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import albucore
from albucore.utils import MAX_VALUES_BY_DTYPE, NPDTYPE_TO_OPENCV_DTYPE, clip
from benchmark.utils import MarkdownGenerator, format_results, get_markdown_table

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Instantiate the random number generator
rng = np.random.default_rng()


DEFAULT_BENCHMARKING_LIBRARIES = ["albucore", "opencv", "numpy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augmentation libraries performance benchmark")
    parser.add_argument("-n", "--num_images", default=10, type=int, help="number of images to test")
    parser.add_argument("-c", "--num_channels", default=3, type=int, help="number of channels in the images")
    parser.add_argument(
        "-t",
        "--img_type",
        choices=["float32", "float64", "uint8", "uint16"],
        type=str,
        help="image type for benchmarking",
    )
    parser.add_argument("-r", "--runs", default=5, type=int, metavar="N", help="number of runs for each benchmark")
    parser.add_argument(
        "--show-std", dest="show_std", action="store_true", help="show standard deviation for benchmark runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-m", "--markdown", action="store_true", help="print benchmarking results as a markdown table")
    return parser.parse_args()


def get_package_versions() -> Dict[str, str]:
    packages = ["albucore", "opencv-python-headless", "numpy"]
    package_versions = {"Python": sys.version}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "Not installed"
    return package_versions


class BenchmarkTest:
    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels

    def __str__(self) -> str:
        return self.__class__.__name__

    def albucore(self, img: np.ndarray) -> np.ndarray:
        return self.albucore_transform(img)

    def opencv(self, img: np.ndarray) -> np.ndarray:
        return clip(self.opencv_transform(img), img.dtype)

    def numpy(self, img: np.ndarray) -> np.ndarray:
        return clip(self.numpy_transform(img), img.dtype)

    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {"albucore": "albucore_transform", "opencv": "opencv_transform", "numpy": "numpy_transform"}

        # Check if the library is in the map
        if library in library_attr_map:
            attrs = library_attr_map[library]
            # Ensure attrs is a list for uniform processing
            if not isinstance(attrs, list):
                attrs = [attrs]  # type: ignore[list-item]
            # Return True if any of the specified attributes exist
            return any(hasattr(self, attr) for attr in attrs)

        # Fallback: checks if the class has an attribute with the library's name
        return hasattr(self, library)

    def run(self, library: str, imgs: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        transform = getattr(self, library)
        transformed_images = []
        for img in imgs:
            result = transform(img)
            if result is None:
                return None  # If transform returns None, skip this benchmark
            transformed_images.append(result)

        return transformed_images


class MultiplyConstant(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.multiplier = 1.5

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply(img, self.multiplier)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return img * self.multiplier

    def opencv_transform(self, img: np.ndarray) -> Optional[np.ndarray]:
        return cv2.multiply(img, self.multiplier, dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype])


class MultiplyVector(BenchmarkTest):
    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = np.array([1.5] * self.num_channels)
        return albucore.multiply(img, multiplier)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = np.array([1.5] * self.num_channels)

        return img * multiplier

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = np.array([1.5] * self.num_channels)
        return cv2.multiply(img, multiplier, dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype])


def get_images(num_images: int, height: int, width: int, num_channels: int, dtype: str) -> List[np.ndarray]:
    if dtype in {"float32", "float64"}:
        return [rng.random((height, width, num_channels), dtype=np.dtype(dtype)) for _ in range(num_images)]
    if dtype in {"uint8", "uint16"}:
        return [
            rng.integers(0, MAX_VALUES_BY_DTYPE[np.dtype(dtype)] + 1, (height, width, num_channels), dtype=dtype)
            for _ in range(num_images)
        ]
    raise ValueError(f"Invalid image type {dtype}")


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()

    num_channels = args.num_channels
    num_images = args.num_images

    height, width = 512, 512

    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    imgs = get_images(num_images, height, width, num_channels, args.img_type)

    benchmark_class_names = [MultiplyConstant, MultiplyVector]

    libraries = DEFAULT_BENCHMARKING_LIBRARIES

    images_per_second = {lib: {} for lib in libraries}
    to_skip = {lib: {} for lib in libraries}

    for benchmark_class in tqdm(benchmark_class_names, desc="Running benchmarks"):
        benchmark = benchmark_class(num_channels)

        for library in libraries:
            images_per_second[library][str(benchmark)] = []

        for _ in range(args.runs):
            shuffled_libraries = copy.deepcopy(libraries)
            random.shuffle(shuffled_libraries)

            for library in shuffled_libraries:
                if benchmark.is_supported_by(library) and not to_skip[library].get(str(benchmark), False):
                    timer = Timer(lambda lib=library: benchmark.run(lib, imgs))
                    try:
                        run_times = timer.repeat(number=1, repeat=1)
                        benchmark_images_per_second = [1 / (run_time / num_images) for run_time in run_times]
                    except Exception as e:
                        print(f"Error running benchmark for {library}: {e}")
                        benchmark_images_per_second = [None]
                        images_per_second[library][str(benchmark)].extend(benchmark_images_per_second)
                        to_skip[library][str(benchmark)] = True
                else:
                    benchmark_images_per_second = [None]

                images_per_second[library][str(benchmark)].extend(benchmark_images_per_second)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std) if r is not None else None)

    transforms = [str(i(num_channels)) for i in benchmark_class_names]

    df = df.reindex(transforms)
    df = df[DEFAULT_BENCHMARKING_LIBRARIES]

    if args.markdown:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        file_path = results_dir / f"{args.img_type}_{num_channels}.md"
        markdown_generator = MarkdownGenerator(df, package_versions, num_images)
        markdown_generator.save_markdown_table(file_path)
        print(f"Benchmark results saved to {file_path}")


if __name__ == "__main__":
    main()
