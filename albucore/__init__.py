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

try:
    import torch  # noqa: F401
except ImportError as e:
    msg = (
        "Albucore requires PyTorch when it is imported.\n\n"
        "Install the PyTorch build for your platform first. For Linux CPU-only:\n"
        '  pip install "torch>=2.13.0" --index-url https://download.pytorch.org/whl/cpu\n\n'
        "Then install Albucore's Torch runtime profile:\n"
        "  pip install albucore[torch]\n\n"
        "Use PyTorch's platform-specific command for CUDA or MPS."
    )
    raise ImportError(msg) from e

from . import decorators as _decorators
from . import functions as _functions
from . import geometric as _geometric
from . import utils as _utils
from .decorators import *
from .functions import *
from .geometric import *
from .utils import *

_meta_names = ("__version__", "__author__", "__maintainer__")
_combined: list[str] = (
    list(_meta_names)
    + list(_functions.__all__)
    + list(_decorators.__all__)
    + list(_geometric.__all__)
    + list(_utils.__all__)
)
__all__ = list(dict.fromkeys(_combined))

del _meta_names, _combined
