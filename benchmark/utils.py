import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style


class MarkdownGenerator:
    def __init__(self, df: pd.DataFrame, package_versions: Dict[str, str]) -> None:
        self._df = df
        self._package_versions = package_versions

    def _highlight_best_result(self, results: List[str]) -> List[str]:
        processed_results = []

        # Extract mean values and convert to float for comparison
        for result in results:
            try:
                if result is None:
                    processed_results.append((float("-inf"), "-"))
                    continue
                mean_value = float(result.split("±")[0].strip())
                processed_results.append((mean_value, result))
            except (ValueError, IndexError):
                # Handle cases where conversion fails or result doesn't follow expected format
                processed_results.append((float("-inf"), result))

        # Determine the best result based on mean values
        best_mean_value = max([mean for mean, _ in processed_results])

        # Highlight the best result
        return [
            f"**{original_result}**" if mean_value == best_mean_value else original_result
            for mean_value, original_result in processed_results
        ]

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


def format_results(images_per_second_for_aug: Optional[List[float]], show_std: bool = False) -> str:
    if images_per_second_for_aug is None:
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
