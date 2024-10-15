from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style


class MarkdownGenerator:
    def __init__(self, df: pd.DataFrame, package_versions: dict[str, str], num_images: int) -> None:
        self.df = df
        self.package_versions = package_versions
        self.num_images = num_images
        self.cpu_info = get_cpu_info()

    def _highlight_best_result(self, results: list[float | str]) -> list[str]:
        parsed_results = []
        for result in results:
            if pd.isna(result):
                parsed_results.append((np.nan, np.nan))
            elif isinstance(result, str):
                try:
                    mean, se = map(float, result.split("±"))
                    parsed_results.append((mean, se))
                except ValueError:
                    parsed_results.append((float(result), 0))
            elif isinstance(result, (int, float)):
                parsed_results.append((float(result), 0))
            else:
                parsed_results.append((np.nan, np.nan))

        if not parsed_results or all(np.isnan(mean) for mean, _ in parsed_results):
            return [str(r) for r in results]

        best_mean = max(mean for mean, _ in parsed_results if not np.isnan(mean))
        highlighted_results = []

        for (mean, _se), original_result in zip(parsed_results, results):
            if mean == best_mean:
                highlighted_results.append(f"**{original_result}**")
            else:
                highlighted_results.append(str(original_result))

        return highlighted_results

    def _make_headers(self) -> list[str]:
        libraries = self.df.columns.to_list()
        columns = []

        for library in libraries:
            key = library
            if "opencv" in key or "lut" in key:
                key = "opencv-python-headless"

            version = self.package_versions[key]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self) -> list[list[str]]:
        value_matrix = []
        for transform, results in self.df.iterrows():
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def generate_markdown_table(self) -> str:
        # Convert DataFrame to markdown
        md_table = self.df.to_markdown()

        # Add header
        header = f"# Benchmark Results: {self.df.index[0]}\n\n"
        header += f"Number of images: {self.num_images}\n\n"
        header += "## CPU Information\n\n"
        header += f"- CPU: {self.cpu_info['name']}\n"
        header += f"- Frequency: {self.cpu_info['freq']}\n"
        header += f"- Physical cores: {self.cpu_info['physical_cores']}\n"
        header += f"- Total cores: {self.cpu_info['total_cores']}\n\n"
        header += "## Package Versions\n\n"
        header += pd.DataFrame([self.package_versions]).to_markdown(index=False) + "\n\n"
        header += "## Performance (images/second)\n\n"
        header += f"Raw data:\n{self.df.to_string()}\n\n"

        return header + md_table

    def _make_versions_text(self) -> str:
        libraries = ["numpy", "opencv-python-headless"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self.package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return f"Python and library versions: {', '.join(libraries_with_versions)}."

    def print_markdown_table(self) -> None:
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()

    def save_markdown_table(self, file_path: Path) -> None:
        with file_path.open("w") as f:
            f.write(self.generate_markdown_table())


def format_results(images_per_second_for_aug: float | list[float], show_std: bool = True) -> str:
    if isinstance(images_per_second_for_aug, (int, float)):
        return f"{images_per_second_for_aug:.0f}"

    if all(x is None for x in images_per_second_for_aug):
        return "N/A"

    filtered_results = [x for x in images_per_second_for_aug if x is not None]

    if not filtered_results:
        return "N/A"

    mean = np.mean(filtered_results)

    if show_std:
        std = np.std(filtered_results)
        n = len(filtered_results)
        se = std / np.sqrt(n)  # Calculate standard error
        return f"{mean:.2f} ± {se:.2f}"  # Changed to 2 decimal places for more precision

    return f"{mean:.2f}"  # Changed to 2 decimal places for consistency


def get_markdown_table(data: dict[str, str]) -> str:
    """Prints a dictionary as a nicely formatted Markdown table.

    Parameters:
        data dict[str, str]: The dictionary to print, with keys as columns and values as rows.

    Returns:
    None

    Example input:
        {'Python': '3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]',
        'numpy': '1.26.4',
        'opencv-python-headless': '4.9.0.80'}
    """
    # Start with the table headers
    markdown_table = "| Library | Version |\n"
    markdown_table += "|---------|---------|\n"

    # Add each dictionary item as a row in the table
    for key, value in data.items():
        markdown_table += f"| {key} | {value} |\n"

    return markdown_table


def torch_clip(img: torch.Tensor, dtype: Any) -> torch.Tensor:
    result = img.clamp(0, 255)

    return result.byte() if dtype == torch.uint8 else result.float()


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
