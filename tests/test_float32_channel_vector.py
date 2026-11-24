from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from albucore import add, arithmetic, multiply, multiply_add

SPECIAL_VALUES = np.array([np.nan, np.inf, -np.inf, -0.0, 3e38, -3e38, 0.0, 1e-40], dtype=np.float32)


def _image(shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(9)
    image = rng.random(shape, dtype=np.float32) * 4 - 1
    image.reshape(-1)[: SPECIAL_VALUES.size] = SPECIAL_VALUES
    return image


def _vector(channels: int, offset: float = 0.5) -> np.ndarray:
    return (offset + 0.31 * np.arange(channels) * np.where(np.arange(channels) % 2, -1, 1)).astype(np.float32)


def _destination(output: str, image: np.ndarray) -> np.ndarray | None:
    if output == "separate":
        return np.full_like(image, 3)
    if output == "alias":
        return image
    if output == "strided":
        return np.full((*image.shape[:-1], image.shape[-1] * 2), 3, dtype=np.float32)[..., ::2]
    return None


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("shape", [(17, 23, 2), (17, 23, 3), (17, 23, 4), (3, 17, 23, 3), (17, 23, 5), (2, 17, 23, 9)])
@pytest.mark.parametrize("output", ["allocate", "separate", "alias", "strided"])
def test_float32_channel_vector_matches_the_broadcast_route(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., np.ndarray],
    shape: tuple[int, ...],
    output: str,
) -> None:
    image = _image(shape)
    vector = _vector(shape[-1])
    name = "add" if operation is add else "multiply"
    with np.errstate(all="ignore"):
        expected = arithmetic._numpy_pointwise(image.copy(), vector, name)
    destination = _destination(output, image)
    if output != "strided":
        monkeypatch.setattr(arithmetic, "_numpy_pointwise", lambda *_a, **_k: pytest.fail("fell back to broadcasting"))

    with np.errstate(all="ignore"):
        result = operation(image, vector, out=destination)

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32), strict=True)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
def test_float32_single_channel_vector_and_float64_vector_keep_the_broadcast_route(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., np.ndarray],
) -> None:
    calls: list[int] = []
    real = arithmetic._numpy_pointwise

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(arithmetic, "_numpy_pointwise", spy)

    operation(_image((17, 23, 1)), np.array([0.7], dtype=np.float32))
    operation(_image((17, 23, 3)), _vector(3).astype(np.float64))

    assert len(calls) == 2


@pytest.mark.parametrize("shape", [(17, 23, 2), (17, 23, 3), (3, 17, 23, 4), (17, 23, 9)])
@pytest.mark.parametrize("controls", ["vector_vector", "scalar_vector", "vector_scalar"])
@pytest.mark.parametrize("output", ["allocate", "separate", "alias", "strided"])
def test_float32_multiply_add_channel_vectors_match_the_broadcast_route(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    controls: str,
    output: str,
) -> None:
    image = _image(shape)
    factor: Any = _vector(shape[-1], 1.5) if controls != "scalar_vector" else 1.3
    bias: Any = _vector(shape[-1], 0.25) if controls != "vector_scalar" else -0.4
    with np.errstate(all="ignore"):
        expected = arithmetic._multiply_add_numpy(image.copy(), factor, bias)
    destination = _destination(output, image)
    if output != "strided":
        monkeypatch.setattr(arithmetic, "_multiply_add_numpy", lambda *_a, **_k: pytest.fail("fell back to broadcasting"))

    with np.errstate(all="ignore"):
        result = multiply_add(image, factor, bias, out=destination)

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32), strict=True)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize(
    ("factor", "bias"),
    [(np.zeros(3, np.float32), np.ones(3, np.float32)), (np.ones(3, np.float32), np.zeros(3, np.float32)), (0.0, np.ones(3, np.float32))],
)
def test_float32_multiply_add_zero_controls_keep_the_legacy_shortcuts(factor: Any, bias: Any) -> None:
    image = _image((17, 23, 3))
    with np.errstate(all="ignore"):
        expected = arithmetic._multiply_add_numpy(image.copy(), factor, bias)
        result = multiply_add(image, factor, bias)

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32), strict=True)
