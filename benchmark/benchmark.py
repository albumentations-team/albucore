import argparse
import copy
import os
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from timeit import Timer
from typing import Any, Dict, List, Optional, Union
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import cv2

import pandas as pd
from tqdm import tqdm

import albucore
from albucore.utils import MAX_VALUES_BY_DTYPE, MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, clip
from benchmark.utils import MarkdownGenerator, format_results, get_markdown_table, torch_clip

import torchvision.transforms.v2.functional as torchf

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np # important to do it after these env variables

# Instantiate the random number generator
rng = np.random.default_rng()


DEFAULT_BENCHMARKING_LIBRARIES = ["albucore", "lut", "opencv", "numpy", "torchvision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augmentation libraries performance benchmark")
    parser.add_argument(
        "-d", "--data-dir", metavar="DIR", required=True, type=Path, help="path to a directory with images"
    )
    parser.add_argument(
        "-n", "--num_images", default=2000, type=int, help="number of images for benchmarking (default: 2000)"
    )
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
    parser.add_argument("-j", "--num_workers", default=1, type=int, help="The number of CPU workers to use")
    return parser.parse_args()


def get_package_versions() -> Dict[str, str]:
    packages = ["albucore", "opencv-python-headless", "numpy", "torchvision"]
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

    def lut(self, img: np.ndarray) -> np.ndarray:
        return clip(self.lut_transform(img), img.dtype)

    def torchvision(self, img: np.ndarray) -> np.ndarray:
        return torch_clip(self.torchvision_transform(img), img.dtype)


    def is_supported_by(self, library: str) -> bool:
        library_attr_map = {
            "albucore": "albucore_transform",
            "opencv": "opencv_transform",
            "numpy": "numpy_transform",
            "lut": "lut_transform",
            "kornia-rs": "kornia_transform",
            "torchvision": "torchvision_transform"
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
        return albucore.multiply_numpy(img, self.multiplier)

    def opencv_transform(self, img: np.ndarray) -> Optional[np.ndarray]:
        return albucore.multiply_opencv(img, self.multiplier)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_lut(img, self.multiplier)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.mul(img, self.multiplier)


class MultiplyVector(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.multiplier = rng.uniform(0.5, 1, num_channels)
        self.torch_multiplier = torch.from_numpy(self.multiplier).view(-1, 1, 1)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply(img, self.multiplier)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_numpy(img, self.multiplier)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_opencv(img, self.multiplier)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_lut(img, self.multiplier)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.mul(img, self.torch_multiplier)


class MultiplyArray(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

        self.boundaries = (0.9, 1.1)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.multiply(img, multiplier)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.multiply_numpy(img, multiplier)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        multiplier = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.multiply_opencv(img, multiplier)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        multiplier = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return torch.mul(img, torch.from_numpy(multiplier))


class AddConstant(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.value = 1.5

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add(img, self.value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_numpy(img, self.value)

    def opencv_transform(self, img: np.ndarray) -> Optional[np.ndarray]:
        return albucore.add_opencv(img, self.value)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_lut(img, self.value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.add(img, self.value)



class AddVector(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.value = rng.uniform(0.5, 1, [num_channels])
        self.torch_value = torch.from_numpy(self.value).view(-1, 1, 1)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add(img, self.value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_numpy(img, self.value)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_opencv(img, self.value)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_lut(img, self.value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.add(img, self.torch_value)


class AddArray(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.boundaries = (0.9, 1.1)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        value = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.add(img, value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        value = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.add_numpy(img, value)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        value = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return albucore.add_opencv(img, value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        value = rng.uniform(self.boundaries[0], self.boundaries[1], img.shape)
        return torch.add(img, torch.from_numpy(value))


class Normalize(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        boundaries = (0, 113.0)
        self.mean = rng.uniform(boundaries[0], boundaries[1], num_channels)
        self.denominator = rng.uniform(boundaries[0], boundaries[1], num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize(img, self.denominator, self.mean)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_numpy(img, self.denominator, self.mean)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_opencv(img, self.denominator, self.mean)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_lut(img, self.denominator, self.mean)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchf.normalize(img.float(), self.mean, self.denominator)


class NormalizePerImage(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image(img, "image")

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_numpy(img, "image")

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_opencv(img, "image")

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_lut(img, "image")

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        std, mean = torch.std_mean(img.float())
        normalized_img = (img - mean) / (std + eps)
        return torch.clamp(normalized_img, min=-20, max=20)


class NormalizePerImagePerChannel(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image(img, "image_per_channel")

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_numpy(img, "image_per_channel")

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_opencv(img, "image_per_channel")

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_lut(img, "image_per_channel")


class NormalizeMinMax(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image(img, "min_max")

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_numpy(img, "min_max")

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_opencv(img, "min_max")

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_lut(img, "min_max")

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        min_max = img.aminmax()
        normalized_image = (img - min_max.min) / (min_max.max - min_max.min + eps)
        return torch.clamp(normalized_image, min=-20, max=20)


class NormalizeMinMaxPerChannel(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image(img, "min_max_per_channel")

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_numpy(img, "min_max_per_channel")

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_opencv(img, "min_max_per_channel")

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.normalize_per_image_lut(img, "min_max_per_channel")


class PowerConstant(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.exponent = 1.1

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.power(img, self.exponent)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.power_numpy(img, self.exponent)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.power_opencv(img, self.exponent)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.power_lut(img, self.exponent)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.pow(img, self.exponent)


class AddWeighted(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.weight1 = 0.4
        self.weight2 = 0.6

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted(img, self.weight1, img, self.weight2)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_numpy(img, self.weight1, img, self.weight2)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_opencv(img, self.weight1, img, self.weight2)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_lut(img, self.weight1, img, self.weight2)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return img * self.weight1 + img * self.weight2


def get_images_from_dir(data_dir: Path, num_images: int, num_channels: int, dtype: str) -> List[np.ndarray]:
    image_paths = list(data_dir.expanduser().absolute().glob("*.*"))[:num_images]
    images = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        if img.ndim == MONO_CHANNEL_DIMENSIONS:  # Single channel image
            img = np.stack([img] * num_channels, axis=-1)
        elif img.ndim == NUM_MULTI_CHANNEL_DIMENSIONS and img.shape[2] != num_channels:
            if img.shape[2] < num_channels:
                repeats = num_channels // img.shape[2] + (num_channels % img.shape[2] > 0)
                img = np.tile(img, (1, 1, repeats))[:, :, :num_channels]
            else:
                img = img[:, :, :num_channels]

        img = img.astype(dtype)
        if dtype in {"float32", "float64"}:
            img /= MAX_VALUES_BY_DTYPE[np.dtype(dtype)]

        images.append(img)

    if len(images) < num_images:
        raise ValueError(f"Only {len(images)} images found in directory {data_dir}, but {num_images} requested.")

    return images


def get_images(num_images: int, height: int, width: int, num_channels: int, dtype: str) -> List[np.ndarray]:
    height, width = 256, 256

    if dtype in {"float32", "float64"}:
        return [rng.random((height, width, num_channels), dtype=np.dtype(dtype)) for _ in range(num_images)]
    if dtype in {"uint8", "uint16"}:
        return [
            rng.integers(0, MAX_VALUES_BY_DTYPE[np.dtype(dtype)] + 1, (height, width, num_channels), dtype=dtype)
            for _ in range(num_images)
        ]
    raise ValueError(f"Invalid image type {dtype}")


def run_single_benchmark(
    library: str,
    benchmark_class: Any,
    num_channels: int,
    runs: int,
    img_type: str,
    torch_imgs: List[torch.Tensor],
    imgs: List[Any],
    num_images: int,
    to_skip: Dict[str, Dict[str, bool]],
    unsupported_types: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[Union[None, float]]]]:
    benchmark = benchmark_class(num_channels)
    library_results: Dict[str, List[Union[None, float]]] = {str(benchmark): []}

    # Skip if library does not support the img_type
    if img_type in unsupported_types.get(library, []):
        to_skip[library][str(benchmark)] = True
        return {library: library_results}

    for _ in range(runs):
        if library == "torchvision":
            images = torch_imgs
        else:
            images = imgs

        if benchmark.is_supported_by(library) and not to_skip[library].get(str(benchmark), False):
            timer = Timer(lambda lib=library: benchmark.run(lib, images))
            try:
                run_times = timer.repeat(number=1, repeat=1)
                benchmark_images_per_second = [1 / (run_time / num_images) for run_time in run_times]
            except Exception as e:
                print(f"Error running benchmark for {library}: {e}")
                benchmark_images_per_second = [None]
                to_skip[library][str(benchmark)] = True
        else:
            benchmark_images_per_second = [None]

        library_results[str(benchmark)].extend(benchmark_images_per_second)

    return {library: library_results}


def run_benchmarks(
    benchmark_class_names: List[Any],
    libraries: List[str],
    torch_imgs: List[torch.Tensor],
    imgs: List[Any],
    num_channels: int,
    num_images: int,
    runs: int,
    img_type: str,
    num_workers: int
) -> Dict[str, Dict[str, List[Union[None, float]]]]:
    images_per_second: Dict[str, Dict[str, List[Union[None, float]]]] = {lib: {} for lib in libraries}
    to_skip: Dict[str, Dict[str, bool]] = {lib: {} for lib in libraries}

    # Define unsupported types for each library
    unsupported_types: Dict[str, List[str]] = {
        "albucore": [],
        "opencv": [],  # Example: opencv does not support float32
        "numpy": [],
        "lut": ["float32"],
        "torchvision": []
    }

    with Manager() as manager:
        to_skip = manager.dict(to_skip)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            total_tasks = len(benchmark_class_names) * len(libraries)
            with tqdm(total=total_tasks, desc="Running benchmarks") as progress_bar:
                for benchmark_class in benchmark_class_names:
                    for library in libraries:
                        futures.append(executor.submit(
                            run_single_benchmark,
                            library,
                            benchmark_class,
                            num_channels,
                            runs,
                            img_type,
                            torch_imgs,
                            imgs,
                            num_images,
                            to_skip,
                            unsupported_types
                        ))

                for future in as_completed(futures):
                    result = future.result()
                    for library, results in result.items():
                        for benchmark_name, benchmark_results in results.items():
                            images_per_second[library].setdefault(benchmark_name, []).extend(benchmark_results)
                    progress_bar.update(1)  # Update progress bar for each completed future

    return images_per_second


def main() -> None:
    benchmark_class_names = [
        MultiplyConstant,
        MultiplyVector,
        MultiplyArray,
        AddConstant,
        AddVector,
        AddArray,
        Normalize,
        NormalizePerImage,
        NormalizePerImagePerChannel,
        NormalizeMinMaxPerChannel,
        NormalizeMinMax,
        PowerConstant,
        AddWeighted,
    ]
    args = parse_args()
    package_versions = get_package_versions()

    num_channels = args.num_channels
    num_images = args.num_images

    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    if args.data_dir is not None:
        imgs = get_images_from_dir(args.data_dir, num_images, num_channels, args.img_type)
    else:
        imgs = get_images(num_images, num_channels, args.img_type)

    torch_imgs = [torch.from_numpy(img.transpose(2, 0, 1).astype(args.img_type)) for img in imgs]

    libraries = DEFAULT_BENCHMARKING_LIBRARIES

    images_per_second = run_benchmarks(benchmark_class_names, libraries, torch_imgs, imgs, num_channels, num_images, args.runs, args.img_type, args.num_workers)

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_std) if r is not None else None)

    transforms = [str(tr(num_channels)) for tr in benchmark_class_names]

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

