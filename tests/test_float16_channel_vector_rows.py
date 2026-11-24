from collections.abc import Callable

import numpy as np
import pytest

from albucore import add, arithmetic, multiply

SPECIAL_HALVES = np.array([np.nan, np.inf, -np.inf, -0.0, 65504.0, -65504.0, 0.0, 6e-8], dtype=np.float16)


def _special_image(shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(10)
    image = (rng.random(shape, dtype=np.float32) * 8 - 2).astype(np.float16)
    image.reshape(-1)[: SPECIAL_HALVES.size] = SPECIAL_HALVES
    return image


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("shape", [(17, 23, 3), (3, 17, 23, 3), (17, 23, 9), (2, 17, 23, 5)])
@pytest.mark.parametrize("scale", [0.1, 1.1, 300.0])
@pytest.mark.parametrize("output", ["allocate", "separate", "alias", "strided"])
def test_float16_inexact_channel_vector_uses_row_tiles(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., np.ndarray],
    shape: tuple[int, ...],
    scale: float,
    output: str,
) -> None:
    channels = shape[-1]
    vector = (scale * (0.1 + 0.07 * np.arange(channels)) * np.where(np.arange(channels) % 2, -1, 1)).astype(np.float32)
    image = _special_image(shape)
    reference_name = "add" if operation is add else "multiply"
    reference_source = image.copy()
    expected = arithmetic._numpy_pointwise(reference_source, vector, reference_name)
    destination = {
        "allocate": None,
        "separate": np.full_like(image, 3),
        "alias": image,
        "strided": np.full((*shape[:-1], channels * 2), 3, dtype=np.float16)[..., ::2],
    }[output]
    if output != "strided":
        monkeypatch.setattr(arithmetic, "_numpy_pointwise", lambda *_a, **_k: pytest.fail("fell back to broadcasting"))

    result = operation(image, vector, out=destination)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16), strict=True)
    if destination is not None:
        assert result is destination


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
def test_float16_single_channel_vector_keeps_the_current_route(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., np.ndarray],
) -> None:
    image = _special_image((17, 23, 1))
    vector = np.array([1.1], dtype=np.float32)
    reference_name = "add" if operation is add else "multiply"
    expected = arithmetic._numpy_pointwise(image.copy(), vector, reference_name)
    calls: list[int] = []
    real = arithmetic._numpy_pointwise

    def spy(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(arithmetic, "_numpy_pointwise", spy)

    result = operation(image, vector)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16), strict=True)
    assert calls
