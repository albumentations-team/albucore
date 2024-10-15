from albucore.decorators import preserve_channel_dim
import pytest
import numpy as np
import cv2
from albucore.functions import sz_lut

@pytest.mark.parametrize("shape", [
    (10, 10),
    (100, 100),
    (224, 224),
    (10, 10, 1),
    (100, 100, 3),
    (224, 224, 4)
])
@pytest.mark.parametrize("lut_type", [
    "identity",
    "invert",
    "random",
    "threshold"
])
def test_serialize_lookup_recover_vs_cv2_lut(shape, lut_type):
    # Generate random uint8 image
    img = np.random.randint(0, 256, size=shape, dtype=np.uint8)

    # Generate LUT based on the specified type
    if lut_type == "identity":
        lut = np.arange(256, dtype=np.uint8)
    elif lut_type == "invert":
        lut = np.arange(255, -1, -1, dtype=np.uint8)
    elif lut_type == "random":
        lut = np.random.randint(0, 256, size=256, dtype=np.uint8)
    elif lut_type == "threshold":
        lut = np.where(np.arange(256) > 127, 255, 0).astype(np.uint8)

    cv2_func = preserve_channel_dim(cv2.LUT)
    # Apply cv2.LUT
    cv2_result = cv2_func(img, lut)

    # Apply serialize_lookup_recover
    custom_result = sz_lut(img, lut)

    # Compare results
    np.testing.assert_array_equal(custom_result, cv2_result)
