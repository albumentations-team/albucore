import argparse
import copy
import os
import random
import sys
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from timeit import Timer
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import albucore
from albucore.utils import MAX_VALUES_BY_DTYPE, NPDTYPE_TO_OPENCV_DTYPE
from benchmark.utils import (
    MarkdownGenerator,
    format_results,
    get_markdown_table,
)

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
    parser.add_argument(
        "-n",
        "--num_images",
        default=100,
        type=int,
        metavar="N",
        help="number of images to test",
    )
    parser.add_argument(
        "-t",
        "--img_type",
        choices=["float32", "uint8"],
        type=str,
        help="image type for benchmarking",
    )
    parser.add_argument(
        "-r",
        "--runs",
        default=5,
        type=int,
        metavar="N",
        help="number of runs for each benchmark",
    )
    parser.add_argument(
        "--show-std",
        dest="show_std",
        action="store_true",
        help="show standard deviation for benchmark runs",
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
    def __str__(self) -> str:
        return self.__class__.__name__

    def albucore(self, img: np.ndarray) -> np.ndarray:
        return self.albucore_transform(img)

    def opencv(self, img: np.ndarray) -> np.ndarray:
        return self.opencv_transform(img)

    def numpy(self, img: np.ndarray) -> np.ndarray:
        return self.numpy_transform(img)

    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {
            "albucore": "albucore_transform",
            "opencv": "opencv_transform",
            "numpy": "numpy_transform",
        }

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

    def run(self, library: str, imgs: List[np.ndarray]) -> None:
        transform = getattr(self, library)
        for img in imgs:
            transform(img)


class MultiplyConstant(BenchmarkTest):
    def __init__(self) -> None:
        self.value = 1.5

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_by_constant(img, self.value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        result = img * self.value
        return np.clip(result, 0, MAX_VALUES_BY_DTYPE[img.dtype]).astype(img.dtype)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return cv2.multiply(img, self.value, dtype=NPDTYPE_TO_OPENCV_DTYPE[img.dtype])


def main() -> None:
    args = parse_args()
    package_versions = get_package_versions()
    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    images_per_second: Dict[str, Dict[str, Any]] = defaultdict(dict)

    if args.img_type == "float32":
        # Using the new Generator to create float32 images
        imgs = [rng.random((256, 256, 3), dtype=np.float32) for _ in range(args.num_images)]
    elif args.img_type == "uint8":
        # Using the new Generator to create uint8 images
        imgs = [rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(args.num_images)]
    else:
        raise ValueError("Invalid image type")

    benchmarks = [
        MultiplyConstant(),
    ]
    pbar = tqdm(total=len(benchmarks))

    libraries = DEFAULT_BENCHMARKING_LIBRARIES

    for benchmark in benchmarks:
        shuffled_libraries = copy.deepcopy(libraries)  # Create a deep copy of the libraries list
        random.shuffle(shuffled_libraries)  # Shuffle the copied list
        pbar.set_description(f"Current benchmark: {benchmark}")

        for library in shuffled_libraries:
            benchmark_images_per_second = None

            if benchmark.is_supported_by(library):
                timer = Timer(lambda: benchmark.run(library, imgs))
                run_times = timer.repeat(number=1, repeat=args.runs)
                benchmark_images_per_second = [1 / (run_time / args.num_images) for run_time in run_times]
            images_per_second[library][str(benchmark)] = benchmark_images_per_second

        pbar.update(1)
    pbar.close()

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std))

    transforms = [str(i) for i in benchmarks]

    df = df.reindex(transforms)
    df = df[DEFAULT_BENCHMARKING_LIBRARIES]
    if args.markdown:
        print(f"Benchmark results for {args.num_images} images of {args.img_type} type:")
        makedown_generator = MarkdownGenerator(df, package_versions)
        makedown_generator.print_markdown_table()
    else:
        pass


if __name__ == "__main__":
    main()
