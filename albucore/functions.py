__all__ = ["multiply_by_constant"]
from typing import cast

from albucore.utils import FloatOrUIntArray, clipped


@clipped
def multiply_by_constant(img: FloatOrUIntArray, value: float) -> FloatOrUIntArray:
    return cast(FloatOrUIntArray, img * value)
