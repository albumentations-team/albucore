from importlib.metadata import metadata

try:
    _metadata = metadata("albucore")
    __version__ = _metadata["Version"]
    __author__ = _metadata["Author"]
    __maintainer__ = _metadata["Maintainer"]
except Exception:  # noqa: BLE001
    __version__ = "unknown"
    __author__ = "Vladimir Iglovikov"
    __maintainer__ = "Vladimir Iglovikov"

# OpenCV and Torch are installation extras so transitive, non-importing
# consumers do not resolve them. Albucore's current public import graph still
# requires both.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "cv2":
        raise
    msg = (
        "Albucore requires OpenCV but it's not installed.\n\n"
        "Install one of the following:\n"
        "  pip install opencv-python                 # Full version with GUI (cv2.imshow)\n"
        "  pip install opencv-python-headless        # Headless for servers/docker\n"
        "  pip install opencv-contrib-python         # With extra algorithms\n"
        "  pip install opencv-contrib-python-headless # Contrib + headless\n\n"
        "Or use extras:\n"
        "  pip install albucore[headless]            # Installs opencv-python-headless\n"
        "  pip install albucore[gui]                 # Installs opencv-python\n"
        "  pip install albucore[contrib]             # Installs opencv-contrib-python\n"
        "  pip install albucore[contrib-headless]    # Installs opencv-contrib-python-headless"
    )
    raise ImportError(msg) from e

try:
    import torch  # noqa: F401
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
    msg = (
        "Albucore requires PyTorch when it is imported.\n\n"
        "Install the PyTorch build for your platform first. For Linux CPU-only:\n"
        '  pip install "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu\n\n'
        "Then install Albucore with an OpenCV extra and Torch profile:\n"
        '  pip install "albucore[headless,torch]"\n\n'
        "Use PyTorch's platform-specific command for CUDA or MPS, and replace "
        "headless with gui, contrib, or contrib-headless if needed."
    )
    raise ImportError(msg) from e

from ._types import (
    ImageInput,
    ImageInputT,
    ImageType,
    NormalizationType,
    SupportedDType,
    TensorImageLayout,
    TensorLayout,
    ValueType,
)
from .arithmetic import (
    add,
    multiply,
    multiply_add,
    power,
)
from .convert import (
    from_float,
    to_float,
)
from .decorators import (
    preserve_channel_dim,
)
from .elementwise import (
    exp,
    log,
    sqrt,
)
from .geometric import (
    copy_make_border,
    flip_volume,
    gaussian_blur3d,
    pad3d,
    remap,
    remap3d,
    resize,
    resize3d,
    rot90_volume,
    separable_filter3d,
    transpose_volume,
    warp_affine,
    warp_affine3d,
    warp_perspective,
)
from .lut import (
    apply_uint8_lut,
)
from .normalize import (
    normalize,
    normalize_per_image,
)
from .ops_misc import (
    float32_io,
    hflip,
    matmul,
    median_blur,
    pairwise_distances_squared,
    uint8_io,
    vflip,
)
from .stats import (
    mean,
    mean_std,
    reduce_sum,
    std,
)
from .utils import (
    MAX_OPENCV_WORKING_CHANNELS,
    MAX_VALUES_BY_DTYPE,
    NPDTYPE_TO_OPENCV_DTYPE,
    NUM_RGB_CHANNELS,
    TWO,
    clip,
    clipped,
    convert_value,
    get_image_data,
    get_max_value,
    get_num_channels,
    get_opencv_dtype_from_numpy,
    get_opencv_max_channels,
    is_grayscale_image,
    is_multispectral_image,
    is_rgb_image,
    maybe_process_in_chunks,
)
from .weighted import add_weighted

__all__ = [
    "MAX_OPENCV_WORKING_CHANNELS",
    "MAX_VALUES_BY_DTYPE",
    "NPDTYPE_TO_OPENCV_DTYPE",
    "NUM_RGB_CHANNELS",
    "TWO",
    "ImageInput",
    "ImageInputT",
    "ImageType",
    "NormalizationType",
    "SupportedDType",
    "TensorImageLayout",
    "TensorLayout",
    "ValueType",
    "__author__",
    "__maintainer__",
    "__version__",
    "add",
    "add_weighted",
    "apply_uint8_lut",
    "clip",
    "clipped",
    "convert_value",
    "copy_make_border",
    "exp",
    "flip_volume",
    "float32_io",
    "from_float",
    "gaussian_blur3d",
    "get_image_data",
    "get_max_value",
    "get_num_channels",
    "get_opencv_dtype_from_numpy",
    "get_opencv_max_channels",
    "hflip",
    "is_grayscale_image",
    "is_multispectral_image",
    "is_rgb_image",
    "log",
    "matmul",
    "maybe_process_in_chunks",
    "mean",
    "mean_std",
    "median_blur",
    "multiply",
    "multiply_add",
    "normalize",
    "normalize_per_image",
    "pad3d",
    "pairwise_distances_squared",
    "power",
    "preserve_channel_dim",
    "reduce_sum",
    "remap",
    "remap3d",
    "resize",
    "resize3d",
    "rot90_volume",
    "separable_filter3d",
    "sqrt",
    "std",
    "to_float",
    "transpose_volume",
    "uint8_io",
    "vflip",
    "warp_affine",
    "warp_affine3d",
    "warp_perspective",
]
