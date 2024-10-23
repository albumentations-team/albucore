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

from .decorators import *
from .functions import *
from .utils import *
