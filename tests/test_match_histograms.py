import warnings

import numpy as np
import pytest

import albucore
import albucore.histogram as histogram_module
from albucore import match_histograms_uint8


def reference_match(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.empty(image.shape, dtype=np.uint8)
    for channel in range(image.shape[-1]):
        source = image[..., channel].ravel()
        template = reference[..., channel].ravel()
        template_levels = np.flatnonzero(np.bincount(template, minlength=256))
        template_counts = np.bincount(template, minlength=256)[template_levels]
        source_quantiles = np.cumsum(np.bincount(source, minlength=256)) / source.size
        template_quantiles = np.cumsum(template_counts) / template.size
        table = np.interp(source_quantiles, template_quantiles, template_levels).astype(np.uint8)
        result[..., channel] = table[image[..., channel]]
    return result


def make_layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "dense":
        return array
    padded = np.zeros((array.shape[0] + 2, array.shape[1] + 4, array.shape[2]), dtype=np.uint8)
    padded[1:-1, 2:-2] = array
    crop = padded[1:-1, 2:-2]
    return crop if layout == "crop" else crop[::-1, ::-1]


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 9])
@pytest.mark.parametrize("image_layout", ["dense", "crop", "flip"])
@pytest.mark.parametrize("reference_layout", ["dense", "crop", "flip"])
def test_match_histograms_uint8_matches_reference(channels: int, image_layout: str, reference_layout: str) -> None:
    rng = np.random.default_rng(channels)
    image = rng.integers(0, 256, (24, 31, channels), dtype=np.uint8)
    image[..., 0] //= 3
    reference = rng.integers(40, 220, (17, 19, channels), dtype=np.uint8)
    source = make_layout(image, image_layout)
    template = make_layout(reference, reference_layout)
    source_before, template_before = source.copy(), template.copy()

    result = match_histograms_uint8(source, template)

    np.testing.assert_array_equal(result, reference_match(source, template), strict=True)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(template, template_before)
    assert result.flags.writeable
    assert not np.shares_memory(result, source)
    assert not np.shares_memory(result, template)


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_match_histograms_uint8_matches_skimage(channels: int) -> None:
    exposure = pytest.importorskip("skimage.exposure")
    rng = np.random.default_rng(10 + channels)
    image = rng.integers(0, 256, (32, 40, channels), dtype=np.uint8)
    reference = rng.integers(10, 200, (21, 25, channels), dtype=np.uint8)

    expected = exposure.match_histograms(image, reference, channel_axis=-1).astype(np.uint8)

    np.testing.assert_array_equal(match_histograms_uint8(image, reference), expected, strict=True)


def test_match_histograms_uint8_constant_and_single_level() -> None:
    rng = np.random.default_rng(0)
    varied = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    constant = np.full((16, 16, 3), 90, dtype=np.uint8)
    single_level = np.full((5, 7, 3), 200, dtype=np.uint8)

    for image, reference in ((constant, varied), (varied, single_level), (constant, single_level)):
        np.testing.assert_array_equal(
            match_histograms_uint8(image, reference), reference_match(image, reference), strict=True
        )
    np.testing.assert_array_equal(match_histograms_uint8(varied, single_level), np.full_like(varied, 200))


def test_match_histograms_uint8_plane_above_exact_cv_histogram_limit() -> None:
    size = histogram_module._MAX_EXACT_HISTOGRAM_PIXELS + 1
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (size, 1, 1), dtype=np.uint8)
    reference = rng.integers(30, 120, (size, 1, 1), dtype=np.uint8)

    result = match_histograms_uint8(image, reference)

    np.testing.assert_array_equal(result, reference_match(image, reference), strict=True)


def test_match_histograms_uint8_is_exported() -> None:
    assert "match_histograms_uint8" in albucore.__all__


def test_match_histograms_uint8_volume_matches_reference() -> None:
    rng = np.random.default_rng(2)
    volume = rng.integers(0, 256, (3, 8, 9, 3), dtype=np.uint8)
    reference = rng.integers(0, 256, (2, 5, 6, 3), dtype=np.uint8)

    np.testing.assert_array_equal(match_histograms_uint8(volume, reference), reference_match(volume, reference))


@pytest.mark.parametrize("shape", [(0, 5, 3), (4, 0, 3), (0, 0, 1)])
def test_match_histograms_uint8_empty_image_returns_independent_copy(shape: tuple[int, int, int]) -> None:
    image = np.empty(shape, dtype=np.uint8)
    reference = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)[..., : shape[-1]]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the unguarded path divides 0 by 0
        result = match_histograms_uint8(image, reference)

    assert result.shape == shape
    assert result.dtype == np.uint8
    assert result.flags.writeable
    assert not np.shares_memory(result, image)
