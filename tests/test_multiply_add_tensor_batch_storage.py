from typing import Any

import numpy as np
import pytest
import torch

from albucore import arithmetic, multiply_add


def _batch(shape: tuple[int, ...]) -> torch.Tensor:
    rng = np.random.default_rng(8)
    values = rng.random(shape, dtype=np.float32) * 4 - 1
    values.reshape(-1)[:4] = [np.nan, np.inf, -0.0, 3e38]
    return torch.from_numpy(values)


def _count_batch_loop(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real = arithmetic._multiply_add_tensor_batch

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(arithmetic, "_multiply_add_tensor_batch", spy)
    return calls


@pytest.mark.parametrize(("layout", "shape"), [("NCHW", (4, 3, 40, 50)), ("NCDHW", (2, 3, 5, 20, 30)), ("NCHW", (3, 1, 33, 17))])
@pytest.mark.parametrize(("factor", "bias"), [(1.3, -0.4), (0.0, 2.5), (2, 3), (1.0, 0.25), (-1.5, 0.0)])
@pytest.mark.parametrize("output", ["allocate", "separate", "alias"])
def test_float32_tensor_batch_scalar_uses_whole_storage(
    monkeypatch: pytest.MonkeyPatch,
    layout: Any,
    shape: tuple[int, ...],
    factor: float,
    bias: float,
    output: str,
) -> None:
    image = _batch(shape)
    with np.errstate(all="ignore"):
        expected = arithmetic._multiply_add_tensor_batch(image.clone(), factor, bias, layout, out=None)
    destination = {"allocate": None, "separate": torch.full_like(image, 7), "alias": image}[output]
    calls = _count_batch_loop(monkeypatch)

    with np.errstate(all="ignore"):
        result = multiply_add(image, factor, bias, layout=layout, out=destination)

    np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    if destination is not None:
        assert result is destination
    assert not calls


@pytest.mark.parametrize("output", ["allocate", "separate"])
def test_float32_tensor_batch_with_large_items_uses_whole_storage(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    """Items above 524,288 elements used to loop over strided channel-last views (3.5x slower on Linux x86)."""
    image = torch.from_numpy(np.random.default_rng(8).random((2, 9, 240, 320), dtype=np.float32) * 4 - 1)
    assert image[0].numel() > 524_288
    destination = torch.empty_like(image) if output == "separate" else None
    calls = _count_batch_loop(monkeypatch)

    result = multiply_add(image, 1.3, -0.4, layout="NCHW", out=destination)

    assert not calls
    expected = image.numpy().astype(np.float64) * 1.3 - 0.4
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("case", ["strided_output", "non_contiguous_input", "vector_control"])
def test_float32_tensor_batch_other_controls_and_storage_keep_the_per_item_loop(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    image = _batch((3, 2, 30, 40))
    factor: Any = 1.3
    destination = None
    if case == "strided_output":
        destination = torch.empty(3, 2, 30, 80)[..., ::2]
    elif case == "non_contiguous_input":
        image = image.permute(0, 1, 3, 2)
    else:
        factor = np.array([1.1, 1.2], dtype=np.float32)
    calls = _count_batch_loop(monkeypatch)

    multiply_add(image, factor, -0.4, layout="NCHW", out=destination)

    assert calls
