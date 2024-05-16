import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style


class MarkdownGenerator:
    def __init__(self, df: pd.DataFrame, package_versions: Dict[str, str], num_samples: int) -> None:
        self._df = df
        self._package_versions = package_versions
        self.num_samples = num_samples

    def _highlight_best_result(self, results: List[str]) -> List[str]:
        processed_results = []

        # Extract mean values, standard deviations, and filter out None results
        for result in results:
            if result is None or result == "-":
                processed_results.append(
                    (float("-inf"), float("inf"), "-")
                )  # Use infinities to ignore these in comparisons
                continue
            try:
                mean, std = map(float, result.split("±"))
                processed_results.append((mean, std, result))
            except ValueError:
                processed_results.append((float("-inf"), float("inf"), result))  # Handle malformed inputs

        # Determine the best mean value to compare against
        best_mean, best_std = max(processed_results, key=lambda x: x[0])[:2]

        # Highlight results that are statistically similar to the best result
        highlighted_results = []
        for mean, std, original_result in processed_results:
            if mean == float("-inf"):  # Skip results that are placeholders or malformed
                highlighted_results.append(original_result)
                continue
            if abs(best_mean - mean) < best_std + std:
                highlighted_results.append(f"**{original_result}**")
            else:
                highlighted_results.append(original_result)

        return highlighted_results

    def _make_headers(self) -> List[str]:
        libraries = self._df.columns.to_list()
        columns = []

        for library in libraries:
            key = library
            if "opencv" in key:
                key = "opencv-python-headless"

            version = self._package_versions[key]

            columns.append(f"{library}<br><small>{version}</small>")
        return ["", *columns]

    def _make_value_matrix(self) -> List[List[str]]:
        index = self._df.index.tolist()
        values = self._df.to_numpy().tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform, *self._highlight_best_result(results)]
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self) -> str:
        libraries = ["numpy", "opencv-python-headless"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
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
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        with file_path.open("w") as file:
            file.write(writer.dumps())


def format_results(images_per_second_for_aug: Optional[List[float]], show_std: bool = False) -> str:
    if all(x is None for x in images_per_second_for_aug):
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += f" ± {math.ceil(np.std(images_per_second_for_aug))}"
    return result


def get_markdown_table(data: Dict[str, str]) -> str:
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
