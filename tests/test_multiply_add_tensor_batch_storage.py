from typing import Any

import numpy as np
import pytest
import torch

from albucore import arithmetic, multiply_add, torch_backend


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
def test_float32_tensor_batch_scalar_uses_whole_storage_below_the_torch_threshold(
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


def test_float32_tensor_batch_with_large_items_keeps_the_per_item_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _batch((3, 2, 30, 40))
    monkeypatch.setattr(torch_backend, "TORCH_MIN_ELEMENTS", image[0].numel())
    calls = _count_batch_loop(monkeypatch)

    multiply_add(image, 1.3, -0.4, layout="NCHW")

    assert calls


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
