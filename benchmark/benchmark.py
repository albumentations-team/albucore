from __future__ import annotations

import argparse
import os
import random
import sys
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from timeit import Timer
from typing import Any

import cv2
import pandas as pd
import torch
import torchvision.transforms.v2.functional as torchf
from tqdm import tqdm

import albucore
from albucore.utils import MAX_VALUES_BY_DTYPE, MONO_CHANNEL_DIMENSIONS, NUM_MULTI_CHANNEL_DIMENSIONS, clip
from benchmark.utils import MarkdownGenerator, format_results, get_markdown_table, torch_clip

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np  # important to do it after these env variables

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
        choices=["float32", "uint8"],
        type=str,
        help="image type for benchmarking",
    )
    parser.add_argument("-r", "--runs", default=5, type=int, metavar="N", help="number of runs for each benchmark")
    parser.add_argument(
        "--show-ste", dest="show_ste", action="store_true", help="show standard error for benchmark runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-m", "--markdown", action="store_true", help="print benchmarking results as a markdown table")
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default=None,
        help="Specific benchmark class to run. If not provided, all benchmarks will be run.",
    )
    return parser.parse_args()


def get_package_versions() -> dict[str, str]:
    packages = ["albucore", "opencv-python-headless", "numpy", "torchvision"]
    package_versions = {"Python": sys.version}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "Not installed"
    return package_versions


class BenchmarkTest(ABC):
    def __init__(self, num_channels: int) -> None:
        self.num_channels = num_channels
        self.img_type = None

    @abstractmethod
    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        pass

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("lut_transform is not implemented for this benchmark")

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("torchvision_transform is not implemented for this benchmark")

    def __str__(self) -> str:
        return self.__class__.__name__

    def albucore(self, img: np.ndarray) -> np.ndarray:
        return clip(self.albucore_transform(img), img.dtype)

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
            "torchvision": "torchvision_transform",
        }

        # Check if the library is in the map
        if library in library_attr_map:
            attrs = library_attr_map[library]
            # Ensure attrs is a list for uniform processing
            if not isinstance(attrs, list):
                attrs = [attrs]  # type: ignore[assignment]
            # Return True if any of the specified attributes exist
            return any(hasattr(self, attr) for attr in attrs)

        # Fallback: checks if the class has an attribute with the library's name
        return hasattr(self, library)

    def run(self, library: str, imgs: list[np.ndarray]) -> list[np.ndarray] | None:
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

    def opencv_transform(self, img: np.ndarray) -> np.ndarray | None:
        return albucore.multiply_opencv(img, self.multiplier)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.multiply_lut(img, self.multiplier)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.mul(img, self.multiplier)


class MultiplyVector(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.multiplier = rng.uniform(0.5, 2, num_channels).astype(np.float32)
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
        self.value = 2.5

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add(img, self.value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_numpy(img, self.value)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray | None:
        return albucore.add_opencv(img, self.value)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_lut(img, self.value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torch.add(img, self.value)


class AddVector(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.value = rng.uniform(0, 255, [num_channels]).astype(np.float32)
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

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        # Calculate mean and std per channel in a vectorized manner
        img_float = img.float()
        mean = img_float.mean(dim=(1, 2), keepdim=True)
        std = img_float.std(dim=(1, 2), keepdim=True) + eps
        # Normalize the image
        normalized_img = (img_float - mean) / std
        # Clamp the values
        return torch.clamp(normalized_img, min=-20, max=20)


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
        return (img - min_max.min) / (min_max.max - min_max.min + eps)


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

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        img_float = img.float()
        min_value = torch.amin(img_float, dim=(1, 2), keepdim=True)
        max_value = torch.amax(img_float, dim=(1, 2), keepdim=True)
        # Normalize the image
        return (img_float - min_value) / (max_value - min_value + eps)


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
        return albucore.add_weighted(img, self.weight1, img.copy(), self.weight2)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_numpy(img, self.weight1, img.copy(), self.weight2)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_opencv(img, self.weight1, img.copy(), self.weight2)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.add_weighted_lut(img, self.weight1, img.copy(), self.weight2)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return img * self.weight1 + img.clone() * self.weight2


class MultiplyAdd(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.factor = 1.2

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        value = MAX_VALUES_BY_DTYPE[img.dtype] / 10.0
        return albucore.multiply_add(img, self.factor, value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        value = MAX_VALUES_BY_DTYPE[img.dtype] / 10.0
        return albucore.multiply_add_numpy(img, self.factor, value)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        value = MAX_VALUES_BY_DTYPE[img.dtype] / 10.0
        return albucore.multiply_add_opencv(img, self.factor, value)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        value = MAX_VALUES_BY_DTYPE[img.dtype] / 10.0
        return albucore.multiply_add_lut(img, self.factor, value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return img * self.factor + 13


class ToFloat(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.max_value = 255.0

    def is_supported_by(self, library: str) -> bool:
        # ToFloat doesn't support float32 images
        if self.img_type == np.float32:
            return False
        return super().is_supported_by(library)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.to_float(img, self.max_value)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.to_float_numpy(img, self.max_value)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.to_float_opencv(img, self.max_value)

    def lut_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.to_float_lut(img, self.max_value)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return (img / self.max_value).to(torch.float32)


class FromFloat(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)
        self.dtype = np.uint8

    def is_supported_by(self, library: str) -> bool:
        # FromFloat doesn't support uint8 images
        if self.img_type == np.uint8:
            return False
        return super().is_supported_by(library)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.from_float(img, self.dtype)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.from_float_numpy(img, self.dtype)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.from_float_opencv(img, self.dtype)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return (img * MAX_VALUES_BY_DTYPE[self.dtype]).to(torch.uint8)


class HorizontalFlip(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.hflip(img)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.hflip_numpy(img)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.hflip_cv2(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchf.hflip(img)


class VerticalFlip(BenchmarkTest):
    def __init__(self, num_channels: int) -> None:
        super().__init__(num_channels)

    def albucore_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.vflip(img)

    def numpy_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.vflip_numpy(img)

    def opencv_transform(self, img: np.ndarray) -> np.ndarray:
        return albucore.vflip_cv2(img)

    def torchvision_transform(self, img: torch.Tensor) -> torch.Tensor:
        return torchf.vflip(img)


def get_images_from_dir(data_dir: Path, num_images: int, num_channels: int, dtype: str) -> list[np.ndarray]:
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


def run_single_benchmark(
    library: str,
    benchmark_class: type[BenchmarkTest],
    num_channels: int,
    runs: int,
    img_type: str,
    torch_imgs: list[torch.Tensor],
    imgs: list[Any],
    num_images: int,
    to_skip: dict[str, dict[str, bool]],
    unsupported_types: dict[str, list[str]],
) -> dict[str, dict[str, list[float | None]]]:
    benchmark = benchmark_class(num_channels)
    library_results: dict[str, list[float | None]] = {str(benchmark): []}

    # Skip if library does not support the img_type
    if img_type in unsupported_types.get(library, []):
        to_skip[library][str(benchmark)] = True
        return {library: library_results}

    for _ in range(runs):
        images = torch_imgs if library == "torchvision" else imgs

        if benchmark.is_supported_by(library) and not to_skip[library].get(str(benchmark), False):
            timer = Timer(lambda lib=library: benchmark.run(lib, images))  # type: ignore[misc]
            try:
                run_times = timer.repeat(number=1, repeat=1)
                benchmark_images_per_second: list[float | None] = [
                    1 / (total_time / num_images) if total_time is not None else None for total_time in run_times
                ]
            except Exception as e:
                print(f"Error running benchmark for {library}: {e}")
                benchmark_images_per_second = [None]
                to_skip[library][str(benchmark)] = True
        else:
            benchmark_images_per_second = [None]

        library_results[str(benchmark)].extend(benchmark_images_per_second)

    return {library: library_results}


def run_benchmarks(
    benchmark_classes: list[type[BenchmarkTest]],
    libraries: list[str],
    torch_imgs: list[torch.Tensor],
    imgs: list[Any],
    num_channels: int,
    num_images: int,
    runs: int,
    img_type: str,
) -> dict[str, dict[str, list[float | None]]]:
    images_per_second: dict[str, dict[str, list[float | None]]] = {lib: {} for lib in libraries}
    to_skip: dict[str, dict[str, bool]] = {lib: {} for lib in libraries}
    unsupported_types: dict[str, list[str]] = {}

    total_tasks = len(benchmark_classes) * len(libraries)
    with tqdm(total=total_tasks, desc="Running benchmarks") as progress_bar:
        random.shuffle(benchmark_classes)
        for benchmark_class in benchmark_classes:
            benchmark = benchmark_class(num_channels)
            benchmark.img_type = np.dtype(img_type)  # Set the image type for the benchmark

            random.shuffle(libraries)

            for library in libraries:
                if not benchmark.is_supported_by(library):
                    progress_bar.update(1)
                    continue

                try:
                    random.shuffle(torch_imgs)
                    random.shuffle(imgs)
                    result = run_single_benchmark(
                        library,
                        benchmark_class,  # Pass the class, not an instance
                        num_channels,
                        runs,
                        img_type,
                        torch_imgs,
                        imgs,
                        num_images,
                        to_skip,
                        unsupported_types,
                    )
                    for lib, results in result.items():
                        for benchmark_name, benchmark_results in results.items():
                            images_per_second[lib].setdefault(benchmark_name, []).extend(benchmark_results)
                except Exception as e:
                    print(f"Exception running benchmark for {library} with {benchmark_class}: {e}")
                progress_bar.update(1)

    return images_per_second


def main() -> None:
    benchmark_classes: list[type[BenchmarkTest]] = [
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
        MultiplyAdd,
        ToFloat,
        FromFloat,
        HorizontalFlip,
        VerticalFlip,
    ]

    args = parse_args()
    package_versions = get_package_versions()

    num_channels = args.num_channels
    num_images = args.num_images

    if args.benchmark:
        # Filter benchmark_classes based on the provided benchmark name
        benchmark_classes = [cls for cls in benchmark_classes if cls.__name__ == args.benchmark]
        if not benchmark_classes:
            raise ValueError(f"No benchmark found with name: {args.benchmark}")

    if args.print_package_versions:
        print(get_markdown_table(package_versions))

    imgs = get_images_from_dir(args.data_dir, num_images, num_channels, args.img_type)

    torch_imgs = [torch.from_numpy(img.transpose(2, 0, 1).astype(args.img_type)) for img in imgs]

    libraries = DEFAULT_BENCHMARKING_LIBRARIES.copy()

    images_per_second = run_benchmarks(
        benchmark_classes, libraries, torch_imgs, imgs, num_channels, num_images, args.runs, args.img_type
    )

    pd.set_option("display.width", 1000)
    df = pd.DataFrame.from_dict(images_per_second)
    df = df.applymap(lambda r: format_results(r, args.show_ste) if r is not None else None)

    transforms = [tr(num_channels) for tr in benchmark_classes]

    df = df.reindex(transforms)
    df = df[DEFAULT_BENCHMARKING_LIBRARIES]

    if args.markdown:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create a subfolder for this specific dtype and num_channels
        subfolder = results_dir / f"{args.img_type}_{num_channels}"
        subfolder.mkdir(parents=True, exist_ok=True)

        for benchmark_class in benchmark_classes:
            benchmark_name = benchmark_class.__name__
            file_path = subfolder / f"{benchmark_name}.md"

            # Check if the benchmark was run
            if str(benchmark_class(num_channels)) not in df.index:
                print(f"Benchmark {benchmark_name} was not run for {args.img_type} images.")
                continue

            # Create a DataFrame for this specific benchmark
            benchmark_df = pd.DataFrame(
                {
                    lib: [df.loc[str(benchmark_class(num_channels)), lib]]
                    for lib in DEFAULT_BENCHMARKING_LIBRARIES
                    if str(benchmark_class(num_channels)) in df.index
                },
                index=[benchmark_name],
            )

            print(f"Debug: Benchmark DataFrame for {benchmark_name}:")
            print(benchmark_df)

            if benchmark_df.empty:
                print(f"No results for {benchmark_name}. Skipping markdown generation.")
                continue

            markdown_generator = MarkdownGenerator(benchmark_df, package_versions, num_images)
            markdown_generator.save_markdown_table(file_path)
            print(f"Benchmark results for {benchmark_name} saved to {file_path}")


if __name__ == "__main__":
    main()
