import numpy as np
import pytest

from albucore import add, arithmetic, multiply

LINUX_MIN = arithmetic._LINUX_NUMKONG_FLOAT16_SCALAR_MIN_ELEMENTS
LINUX_ADD_MAX = arithmetic._LINUX_NUMKONG_FLOAT16_ADD_MAX_ELEMENTS
LINUX_MULTIPLY_MAX = arithmetic._LINUX_NUMKONG_FLOAT16_MULTIPLY_MAX_ELEMENTS


def _reference(image: np.ndarray, scalar: float, operation: str) -> np.ndarray:
    values = image.astype(np.float32)
    result = values + np.float32(scalar) if operation == "add" else values * np.float32(scalar)
    return result.astype(np.float16)


def _half_image(shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(14)
    image = (rng.random(shape, dtype=np.float32) * 8 - 4).astype(np.float16)
    image.reshape(-1)[:3] = [-0.0, 0.0, 65504.0]
    return image


def _patch_linux_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_SCALAR_MIN_ELEMENTS", LINUX_MIN)
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_ADD_MAX_ELEMENTS", LINUX_ADD_MAX)
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_MULTIPLY_MAX_ELEMENTS", LINUX_MULTIPLY_MAX)


def _count_nk_scale(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    real_scale = arithmetic.nk.scale

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real_scale(*args, **kwargs)

    monkeypatch.setattr(arithmetic.nk, "scale", spy)
    return calls


@pytest.mark.parametrize("operation", ["add", "multiply"])
@pytest.mark.parametrize("scalar", [0.5, -2.0, 1.0, -0.0])
@pytest.mark.parametrize("output", ["allocate", "out", "strided"])
@pytest.mark.parametrize(
    ("shape", "region"),
    [
        ((32, 32, 2), "below"),
        ((32, 32, 3), "inside"),
        ((240, 320, 9), "inside"),
        ((1, 230_401, 3), "above_multiply"),
        ((1, 786_433, 3), "above_both"),
    ],
)
def test_float16_scalar_numkong_region_on_linux_constants(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    scalar: float,
    output: str,
    shape: tuple[int, ...],
    region: str,
) -> None:
    if scalar == 0.0 and operation == "add":
        pytest.skip("zero add is an identity shortcut")
    _patch_linux_region(monkeypatch)
    calls = _count_nk_scale(monkeypatch)
    image = _half_image(shape)
    destination = None
    if output == "out":
        destination = np.full_like(image, 3)
    elif output == "strided":
        destination = np.full((*shape[:-1], shape[-1] * 2), 3, dtype=np.float16)[..., ::2]
    expected = _reference(image, scalar, operation)
    function = add if operation == "add" else multiply

    result = function(image, scalar, out=destination)

    np.testing.assert_array_equal(result.view(np.uint16), expected.view(np.uint16), strict=True)
    if destination is not None:
        assert result is destination
    in_region = region == "inside" or (region == "above_multiply" and operation == "add")
    assert bool(calls) == (in_region and output != "strided")


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_float16_inexact_scalar_never_uses_numkong_scale(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    _patch_linux_region(monkeypatch)
    calls = _count_nk_scale(monkeypatch)
    function = add if operation == "add" else multiply

    function(_half_image((64, 64, 3)), 1.1)

    assert not calls


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_float16_scalar_numkong_is_off_without_a_measured_profile(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_SCALAR_MIN_ELEMENTS", np.iinfo(np.int64).max)
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_ADD_MAX_ELEMENTS", 0)
    monkeypatch.setattr(arithmetic, "_NUMKONG_FLOAT16_MULTIPLY_MAX_ELEMENTS", 0)
    calls = _count_nk_scale(monkeypatch)
    image = _half_image((64, 64, 3))
    function = add if operation == "add" else multiply

    result = function(image, 0.5)

    np.testing.assert_array_equal(result.view(np.uint16), _reference(image, 0.5, operation).view(np.uint16))
    assert not calls


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_float16_single_channel_vector_follows_numkong_region(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    _patch_linux_region(monkeypatch)
    calls = _count_nk_scale(monkeypatch)
    image = _half_image((64, 64, 1))
    control = np.array([0.5], dtype=np.float32)
    function = add if operation == "add" else multiply

    result = function(image, control)

    np.testing.assert_array_equal(result.view(np.uint16), _reference(image, 0.5, operation).view(np.uint16))
    assert calls
