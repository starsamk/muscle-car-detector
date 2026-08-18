"""Public inference API for Car Spotter AI.

The implementation lives in :mod:`photo_model`; this module preserves the
project's stable ``model.py`` entry point for applications and future APIs.
"""

from photo_model import (
    CarPrediction,
    PhotoInferenceError,
    PhotoModelError,
    PhotoModelLoadError,
    PhotoSpotter,
    PhotoSpotterConfig,
)

__all__: list[str] = [
    "CarPrediction",
    "PhotoInferenceError",
    "PhotoModelError",
    "PhotoModelLoadError",
    "PhotoSpotter",
    "PhotoSpotterConfig",
]
