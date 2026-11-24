from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from albucore import add, arithmetic, multiply

LINUX_LIMITS = (76_800, 307_200)
MACOS_LIMITS = (691_200, 691_200)


def _patch_limits(monkeypatch: pytest.MonkeyPatch, limits: tuple[int, int]) -> list[int]:
    monkeypatch.setattr(arithmetic, "_TENSOR_DIRECT_FLOAT32_MAX_ELEMENTS", limits[0])
    monkeypatch.setattr(arithmetic, "_TENSOR_DIRECT_FLOAT32_OUT_MAX_ELEMENTS", limits[1])
    bridge_calls: list[int] = []
    real_bridge = arithmetic._tensor_pointwise

    def spy(*args: Any, **kwargs: Any) -> torch.Tensor:
        bridge_calls.append(1)
        return real_bridge(*args, **kwargs)

    monkeypatch.setattr(arithmetic, "_tensor_pointwise", spy)
    return bridge_calls


def _tensor(shape: tuple[int, ...]) -> torch.Tensor:
    rng = np.random.default_rng(15)
    values = rng.random(shape, dtype=np.float32) * 4 - 1
    values.reshape(-1)[:4] = [np.nan, np.inf, -0.0, 3.4e38]
    return torch.from_numpy(values)


@pytest.mark.parametrize("platform", ["linux", "macos"])
@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
@pytest.mark.parametrize(
    ("layout", "shape"),
    [("CHW", (3, 40, 50)), ("NCHW", (2, 3, 30, 40)), ("CDHW", (3, 4, 20, 30)), ("CHW", (3, 200, 300))],
)
@pytest.mark.parametrize("control", ["scalar", "int", "vector", "readonly_vector", "huge_scalar", "zero"])
@pytest.mark.parametrize("output", ["allocate", "out"])
def test_float32_tensor_scalar_and_vector_use_one_torch_op_within_platform_caps(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    operation: Callable[..., torch.Tensor],
    layout: Any,
    shape: tuple[int, ...],
    control: str,
    output: str,
) -> None:
    limits = LINUX_LIMITS if platform == "linux" else MACOS_LIMITS
    image = _tensor(shape)
    channels = shape[0] if layout in ("CHW", "CDHW") else shape[1]
    vector = np.linspace(-1.5, 2.5, channels, dtype=np.float32)
    value: Any = {
        "scalar": 0.37,
        "int": 3,
        "vector": vector,
        "readonly_vector": vector,
        "huge_scalar": 1e39,
        "zero": 0.0,
    }[control]
    if control == "readonly_vector":
        vector.flags.writeable = False
    with np.errstate(all="ignore"):
        expected = arithmetic._tensor_pointwise(
            image.clone(),
            value,
            "add" if operation is add else "multiply",
            layout,
            out=None,
        )
    bridge_calls = _patch_limits(monkeypatch, limits)
    destination = torch.empty_like(image) if output == "out" else None

    with np.errstate(all="ignore"):
        result = operation(image, value, layout=layout, out=destination)

    np.testing.assert_array_equal(result.numpy().view(np.uint32), expected.numpy().view(np.uint32))
    if destination is not None:
        assert result is destination
    limit = limits[1] if output == "out" else limits[0]
    direct_control = control in ("scalar", "int", "vector") or (control == "zero" and operation is multiply)
    direct = image.numel() <= limit and direct_control
    assert bool(bridge_calls) != direct


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
def test_float32_tensor_direct_route_requires_contiguous_storage(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., torch.Tensor],
) -> None:
    bridge_calls = _patch_limits(monkeypatch, MACOS_LIMITS)
    image = _tensor((3, 40, 50)).permute(0, 2, 1)

    result = operation(image, 0.37, layout="CHW")

    assert result.shape == image.shape
    assert bridge_calls


def test_float16_tensor_scalar_keeps_the_bridge_route(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge_calls = _patch_limits(monkeypatch, MACOS_LIMITS)

    add(_tensor((3, 40, 50)).to(torch.float16), 0.5, layout="CHW")

    assert bridge_calls


@pytest.mark.parametrize("operation", [add, multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("output", ["allocate", "out"])
def test_float32_tensor_vector_with_negative_strides_is_supported(
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[..., torch.Tensor],
    output: str,
) -> None:
    bridge_calls = _patch_limits(monkeypatch, MACOS_LIMITS)
    image = _tensor((3, 40, 50))
    vector = np.array([1, 2, 3], dtype=np.float32)[::-1]
    expected = image.numpy() + vector[:, None, None] if operation is add else image.numpy() * vector[:, None, None]
    destination = torch.empty_like(image) if output == "out" else None

    result = operation(image, vector, layout="CHW", out=destination)

    np.testing.assert_array_equal(result.numpy(), expected)
    assert bridge_calls
