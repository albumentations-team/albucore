"""Output paths for generated benchmark artifacts."""

from __future__ import annotations

import json
import os
import stat
from argparse import ArgumentTypeError
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]


def output_path(value: str) -> Path:
    """Accept external paths or the checkout's ignored results directory."""
    path = Path(value).resolve()
    if path.is_relative_to(REPO_ROOT) and not path.is_relative_to(REPO_ROOT / "benchmarks/results"):
        msg = "Store benchmark output outside the checkout or under benchmarks/results/"
        raise ArgumentTypeError(msg)
    return path


@contextmanager
def atomic_report(path: Path) -> Iterator[TextIO]:
    """Replace a UTF-8 report on successful close, preserving its access mode."""
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None
    descriptor, name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            yield handle
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON through the shared atomic report writer."""
    with atomic_report(path) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
