"""Output paths for generated benchmark artifacts."""

from __future__ import annotations

from argparse import ArgumentTypeError
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def output_path(value: str) -> Path:
    """Accept external paths or the checkout's ignored results directory."""
    path = Path(value).resolve()
    if path.is_relative_to(REPO_ROOT) and not path.is_relative_to(REPO_ROOT / "benchmarks/results"):
        msg = "Store benchmark output outside the checkout or under benchmarks/results/"
        raise ArgumentTypeError(msg)
    return path
