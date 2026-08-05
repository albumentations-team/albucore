import numpy as np
import pytest

# Import the functions to test
from albucore.decorators import (
    reshape_for_channel,
    reshape_for_spatial,
    restore_from_channel,
    restore_from_spatial,
)

# Test shapes
SPATIAL_SHAPES = [
    # (input_shape, expected_shape, has_batch, has_depth)
    ((32, 32, 3), (32, 32, 3), False, False),  # H,W,C
    ((32, 32, 1), (32, 32, 1), False, False),  # H,W,C (grayscale)
    ((5, 32, 32, 3), (32, 32, 15), False, True),  # X,H,W,C (depth)
    ((10, 32, 32, 3), (32, 32, 30), True, False),  # X,H,W,C (batch)
]

CHANNEL_SHAPES = [
    # (input_shape, expected_shape, has_batch, has_depth)
    ((32, 32, 3), (32, 32, 3), False, False),  # H,W,C
    ((32, 32, 1), (32, 32, 1), False, False),  # H,W,C (grayscale)
    ((5, 32, 32, 3), (160, 32, 3), False, True),  # X,H,W,C (depth)
    ((10, 32, 32, 3), (320, 32, 3), True, False),  # X,H,W,C (batch)
]


@pytest.mark.parametrize("input_shape,expected_shape,has_batch,has_depth", SPATIAL_SHAPES)
def test_spatial_reshape(input_shape: tuple, expected_shape: tuple, has_batch: bool, has_depth: bool):
    """Test spatial reshape for various input shapes."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_spatial(data)

    assert reshaped.shape == expected_shape
    assert original_shape == input_shape


@pytest.mark.parametrize("input_shape,expected_shape,has_batch,has_depth", CHANNEL_SHAPES)
def test_channel_reshape(input_shape: tuple, expected_shape: tuple, has_batch: bool, has_depth: bool):
    """Test channel reshape for various input shapes."""
    data = np.random.rand(*input_shape)
    reshaped, original_shape = reshape_for_channel(data)

    assert reshaped.shape == expected_shape
    assert original_shape == input_shape


@pytest.mark.parametrize("input_shape,_,has_batch,has_depth", SPATIAL_SHAPES)
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_spatial_roundtrip(input_shape: tuple, _, has_batch: bool, has_depth: bool, non_contiguous: bool):
    """Test that reshape->restore preserves data for spatial transforms."""
    data = np.arange(np.prod(input_shape)).reshape(input_shape)
    if non_contiguous:
        data = np.asfortranarray(data)
    # Use reshape_for_spatial instead of reshape_3d directly
    reshaped, original_shape = reshape_for_spatial(data)
    restored = restore_from_spatial(reshaped, original_shape)

    assert restored.shape == input_shape
    np.testing.assert_array_equal(data, restored)


@pytest.mark.parametrize("input_shape,_,has_batch,has_depth", CHANNEL_SHAPES)
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_channel_roundtrip(input_shape: tuple, _, has_batch: bool, has_depth: bool, non_contiguous: bool):
    """Test that reshape->restore preserves data for channel transforms."""
    data = np.arange(np.prod(input_shape)).reshape(input_shape)
    if non_contiguous:
        data = np.asfortranarray(data)
    # Use reshape_for_channel instead of reshape_batch
    reshaped, original_shape = reshape_for_channel(data)
    restored = restore_from_channel(reshaped, original_shape)

    assert restored.shape == input_shape
    np.testing.assert_array_equal(data, restored)


@pytest.mark.parametrize("transform_type", ["spatial", "channel"])
def test_non_contiguous_input(transform_type: str):
    """Test that non-contiguous arrays are handled correctly."""
    # Create non-contiguous array by slicing
    data = np.random.rand(10, 32, 32, 3)[::2]
    assert not data.flags["C_CONTIGUOUS"]

    reshape_func = {
        "spatial": reshape_for_spatial,
        "channel": reshape_for_channel,
    }[transform_type]

    reshaped, _ = reshape_func(data)

    expected_shape = (32, 32, 15) if transform_type == "spatial" else (160, 32, 3)
    assert reshaped.shape == expected_shape
