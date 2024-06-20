import re
from pathlib import Path
from typing import List, Tuple

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy>=1.24.4,<2",
    "tomli>=2.0.1",
    "typing-extensions>=4.9.0"
]

MIN_OPENCV_VERSION = "4.9.0.80"

CHOOSE_INSTALL_REQUIRES = [
    (
        (f"opencv-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python>={MIN_OPENCV_VERSION}", f"opencv-contrib-python-headless>={MIN_OPENCV_VERSION}"),
        f"opencv-python-headless>={MIN_OPENCV_VERSION}",
    ),
]

def get_version() -> str:
    current_dir = Path(__file__).parent
    version_file = current_dir / "albucore" / "__init__.py"
    with open(version_file, encoding="utf-8") as f:
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_long_description() -> str:
    base_dir = Path(__file__).parent
    with open(base_dir / "README.md", encoding="utf-8") as f:
        return f.read()

def choose_requirement(mains: Tuple[str, ...], secondary: str) -> str:
    for main in mains:
        try:
            # Extract the package name from the requirement string
            package_name = re.split(r"[!<>=]", main)[0]
            # Check if the package is already installed
            get_distribution(package_name)
            return main
        except DistributionNotFound:
            continue
    return secondary


def get_install_requirements(install_requires: List[str], choose_install_requires: List[Tuple[Tuple[str, ...], str]]) -> List[str]:
    for mains, secondary in choose_install_requires:
        install_requires.append(choose_requirement(mains, secondary))
    return install_requires

setup(
    name="albucore",
    version=get_version(),
    description='A high-performance image processing library designed to optimize and extend the Albumentations library with specialized functions for advanced image transformations. Perfect for developers working in computer vision who require efficient and scalable image augmentation.',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Vladimir I. Iglovikov",
    license="MIT",
    url="https://github.com/albumentations-team/albucore",
    packages=find_packages(exclude=["tests", "benchmark", ".github"]),
    python_requires=">=3.8",
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
    classifiers=[
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Typing :: Typed"
    ],
    keywords=[
        "Image Processing", "Computer Vision", "Image Augmentation", "Albumentations", "Optimization", "Machine Learning",
        "Deep Learning", "Python Imaging", "Data Augmentation", "Performance", "Efficiency", "High-Performance",
        "CV", "OpenCV", "Automation"
    ],
    zip_safe=False
)
