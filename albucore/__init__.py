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

# Check for OpenCV at import time
try:
    import cv2  # noqa: F401
except ImportError as e:
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

from .decorators import *
from .functions import *
from .geometric import *
from .utils import *

# Export type aliases for public API
from .utils import ImageFloat32, ImageType, ImageUInt8, SupportedDType

__all__ = [
    "ImageFloat32",
    "ImageType",
    "ImageUInt8",
    "SupportedDType",
]
