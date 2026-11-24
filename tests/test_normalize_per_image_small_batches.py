# ruff: noqa: S101
"""Float32 NCHW batches of one to three images use the batched statistics route."""

import importlib

import numpy as np
import pytest
import torch

from albucore import normalize_per_image

normalize_mod = importlib.import_module("albucore.normalize")


@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize(
    ("channels", "mode", "batched"),
    [
        (3, "image", True),
        (3, "image_per_channel", True),
        (3, "min_max", True),
        (3, "min_max_per_channel", True),
        (9, "image", True),
        (9, "min_max_per_channel", True),
        (9, "image_per_channel", False),
        (1, "image_per_channel", True),
        (1, "min_max", False),
    ],
)
def test_nchw_small_batch_route_matches_measured_region(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, channels: int, mode: str, *, batched: bool
) -> None:
    image = torch.from_numpy(np.random.default_rng(batch_size).random((batch_size, channels, 128, 160), dtype=np.float32))
    calls = 0
    original = normalize_mod._adaptive_tensor_batch_float32  # noqa: SLF001

    def spy(batch: torch.Tensor, normalization: str, out: torch.Tensor | None) -> torch.Tensor | None:
        nonlocal calls
        calls += 1
        return original(batch, normalization, out)

    monkeypatch.setattr(normalize_mod, "_adaptive_tensor_batch_float32", spy)
    result = normalize_per_image(image, mode, layout="NCHW")

    assert bool(calls) is batched
    expected = torch.stack([normalize_per_image(item, mode, layout="CHW") for item in image])
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 3, 4])
def test_ncdhw_batches_keep_four_item_minimum(batch_size: int) -> None:
    image = torch.empty((batch_size, 3, 8, 64, 64), dtype=torch.float32)
    selected = normalize_mod._use_tensor_batch_float32_route(  # noqa: SLF001
        image, "min_max", "NCDHW", None)
    assert selected is (batch_size >= 4)
