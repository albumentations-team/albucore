"""PyTorch packaging and import-contract regression tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_without_torch_names_the_installation_extra() -> None:
    script = """
import builtins
from unittest.mock import patch

original_import = builtins.__import__

def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch":
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original_import(name, globals, locals, fromlist, level)

with patch("builtins.__import__", import_without_torch):
    import albucore
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )

    assert result.returncode != 0
    assert 'pip install "albucore[headless,torch]"' in result.stderr


def test_import_propagates_torch_loader_errors() -> None:
    script = """
import builtins
from unittest.mock import patch

original_import = builtins.__import__

def import_with_broken_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch":
        raise ImportError("simulated Torch loader failure")
    return original_import(name, globals, locals, fromlist, level)

with patch("builtins.__import__", import_with_broken_torch):
    import albucore
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )

    assert result.returncode != 0
    assert "simulated Torch loader failure" in result.stderr
    assert "Install the PyTorch build" not in result.stderr
