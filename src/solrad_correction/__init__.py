"""solrad_correction - WRF diffuse solar radiation bias correction.

ML-based correction of WRF model diffuse radiation (SW_dif) using
observational sensor data.  Supports SVM, LSTM, and Transformer models
with a unified interface for training, evaluation, and comparison.
"""

from importlib.metadata import PackageNotFoundError, version

from solrad_correction.utils.torch_runtime import configure_torch_runtime

configure_torch_runtime()

try:
    __version__ = version("labmim-micrometeorology")
except PackageNotFoundError:  # Support direct source-tree imports before installation.
    __version__ = "0+unknown"
