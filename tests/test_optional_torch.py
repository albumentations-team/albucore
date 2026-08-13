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
        raise ModuleNotFoundError("No module named 'torch'")
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
    assert "pip install albucore[torch]" in result.stderr
