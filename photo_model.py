"""Two-stage photo inference: vehicle detection then fine-grained classification."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_DETECTOR_WEIGHTS: Final[str] = "yolov8n.pt"
DEFAULT_CLASSIFIER_WEIGHTS: Final[Path] = Path("weights/classifier-best.pt")
DEFAULT_TAXONOMY_PATH: Final[Path] = Path("config/taxonomy.json")


class PhotoModelError(RuntimeError):
    """Base exception for the two-stage photo recognition pipeline."""


class PhotoModelLoadError(PhotoModelError):
    """Raised when detector or classifier weights cannot be loaded."""


class PhotoInferenceError(PhotoModelError):
    """Raised when a photo cannot be processed."""


@dataclass(frozen=True)
class PhotoSpotterConfig:
    """Configuration for photo detection and fine-grained classification."""

    detector_weights: str = DEFAULT_DETECTOR_WEIGHTS
    classifier_weights: Path = DEFAULT_CLASSIFIER_WEIGHTS
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH
    detection_confidence: float = 0.25
    classification_confidence: float = 0.40
    iou_threshold: float = 0.45
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate configured confidence thresholds."""
        for field_name, value in (
            ("detection_confidence", self.detection_confidence),
            ("classification_confidence", self.classification_confidence),
            ("iou_threshold", self.iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")

    @classmethod
    def from_environment(cls) -> "PhotoSpotterConfig":
        """Build configuration from ``CAR_SPOTTER_*`` environment variables."""
        try:
            detection_confidence: float = float(
                os.getenv("CAR_SPOTTER_DETECTION_CONFIDENCE", "0.25")
            )
            classification_confidence: float = float(
                os.getenv("CAR_SPOTTER_CLASSIFICATION_CONFIDENCE", "0.40")
            )
            iou_threshold: float = float(os.getenv("CAR_SPOTTER_IOU", "0.45"))
        except ValueError as error:
            raise ValueError(
                "Configured confidence thresholds must be floats."
            ) from error
        return cls(
            detector_weights=os.getenv(
                "CAR_SPOTTER_DETECTOR_PATH", DEFAULT_DETECTOR_WEIGHTS
            ),
            classifier_weights=Path(
                os.getenv(
                    "CAR_SPOTTER_CLASSIFIER_PATH",
                    str(DEFAULT_CLASSIFIER_WEIGHTS),
                )
            ).expanduser(),
            taxonomy_path=Path(
                os.getenv("CAR_SPOTTER_TAXONOMY_PATH", str(DEFAULT_TAXONOMY_PATH))
            ).expanduser(),
            detection_confidence=detection_confidence,
            classification_confidence=classification_confidence,
            iou_threshold=iou_threshold,
            device=os.getenv("CAR_SPOTTER_DEVICE", "auto"),
        )


@dataclass(frozen=True)
class CarPrediction:
    """One classified vehicle and its bounding box."""

    bounding_box: tuple[int, int, int, int]
    class_slug: str
    display_label: str
    detection_confidence: float
    classification_confidence: float


def resolve_device(requested_device: str) -> str:
    """Resolve an automatic inference device.

    Args:
        requested_device: Explicit device or ``"auto"``.

    Returns:
        Ultralytics-compatible device value.
    """
    if requested_device != "auto":
        return requested_device
    try:
        import torch
    except ImportError as error:
        raise PhotoModelLoadError("PyTorch is not installed.") from error
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


class PhotoSpotter:
    """Detect cars, classify their model/period, and render labeled boxes."""

    def __init__(self, config: PhotoSpotterConfig) -> None:
        """Initialize a lazy two-model inference service.

        Args:
            config: Detection and classification configuration.
        """
        self._config: PhotoSpotterConfig = config
        self._detector: Any | None = None
        self._classifier: Any | None = None
        self._labels: dict[str, str] | None = None
        self._device: str = resolve_device(config.device)

    def _load_taxonomy(self) -> dict[str, str]:
        """Load user-facing labels indexed by classifier folder slug."""
        if self._labels is not None:
            return self._labels
        try:
            payload: object = json.loads(
                self._config.taxonomy_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PhotoModelLoadError(
                f"Unable to read taxonomy '{self._config.taxonomy_path}'."
            ) from error
        if not isinstance(payload, dict) or not isinstance(
            payload.get("classes"), list
        ):
            raise PhotoModelLoadError("Taxonomy must contain a classes list.")
        labels: dict[str, str] = {}
        for class_definition in payload["classes"]:
            if not isinstance(class_definition, dict):
                continue
            slug: str = str(class_definition.get("slug", ""))
            display_name: str = str(class_definition.get("display_name", slug))
            year_start: int = int(class_definition.get("year_start", 0))
            year_end: int = int(class_definition.get("year_end", 0))
            show_year_range: bool = bool(
                class_definition.get("show_year_range", True)
            )
            if show_year_range and year_start > 0 and year_end > 0:
                period: str = (
                    str(year_start)
                    if year_start == year_end
                    else f"{year_start}–{year_end}"
                )
                labels[slug] = f"{display_name} — {period}"
            else:
                labels[slug] = display_name
        self._labels = labels
        return labels

    def _load_models(self) -> tuple[Any, Any]:
        """Load and cache the generic detector and custom classifier."""
        if self._detector is not None and self._classifier is not None:
            return self._detector, self._classifier
        if not self._config.classifier_weights.is_file():
            raise PhotoModelLoadError(
                "Classifier weights not found at "
                f"'{self._config.classifier_weights}'. Train the classifier first."
            )
        try:
            from ultralytics import YOLO

            LOGGER.info("Loading detector %s", self._config.detector_weights)
            self._detector = YOLO(self._config.detector_weights)
            LOGGER.info("Loading classifier %s", self._config.classifier_weights)
            self._classifier = YOLO(str(self._config.classifier_weights))
        except Exception as error:
            LOGGER.exception("Unable to load photo inference models")
            raise PhotoModelLoadError(
                "Unable to load photo inference models."
            ) from error
        return self._detector, self._classifier

    @staticmethod
    def _validate_image(image: Image.Image) -> Image.Image:
        """Validate and normalize an input image to RGB."""
        if not isinstance(image, Image.Image):
            raise PhotoInferenceError("Input must be a PIL image.")
        if image.width <= 0 or image.height <= 0:
            raise PhotoInferenceError("Input image dimensions must be positive.")
        return image.convert("RGB").copy()

    def _classify_crop(self, classifier: Any, crop: Image.Image) -> tuple[str, float]:
        """Classify one detected vehicle crop."""
        classification_results: list[Any] = classifier.predict(
            source=np.asarray(crop), device=self._device, verbose=False
        )
        if not classification_results:
            raise PhotoInferenceError("Classifier returned no result.")
        result: Any = classification_results[0]
        probabilities: Any = result.probs
        class_index: int = int(probabilities.top1)
        confidence: float = float(probabilities.top1conf.item())
        class_slug: str = str(result.names[class_index])
        return class_slug, confidence

    def predict_with_details(
        self, image: Image.Image
    ) -> tuple[Image.Image, list[CarPrediction]]:
        """Detect and classify all cars in an image.

        Args:
            image: Source PIL image.

        Returns:
            Annotated image and structured vehicle predictions.

        Raises:
            PhotoModelError: If model loading or inference fails.
        """
        normalized_image: Image.Image = self._validate_image(image)
        detector, classifier = self._load_models()
        labels: dict[str, str] = self._load_taxonomy()
        try:
            detection_results: list[Any] = detector.predict(
                source=np.asarray(normalized_image),
                classes=[2],
                conf=self._config.detection_confidence,
                iou=self._config.iou_threshold,
                device=self._device,
                verbose=False,
            )
            if not detection_results:
                return normalized_image, []
            boxes: Any = detection_results[0].boxes
            predictions: list[CarPrediction] = []
            for box in boxes:
                coordinates: list[float] = box.xyxy[0].tolist()
                left: int = max(0, int(coordinates[0]))
                top: int = max(0, int(coordinates[1]))
                right: int = min(normalized_image.width, int(coordinates[2]))
                bottom: int = min(normalized_image.height, int(coordinates[3]))
                if right <= left or bottom <= top:
                    continue
                crop: Image.Image = normalized_image.crop((left, top, right, bottom))
                class_slug, classification_confidence = self._classify_crop(
                    classifier, crop
                )
                if (
                    classification_confidence
                    < self._config.classification_confidence
                ):
                    class_slug = "other_car"
                display_label: str = labels.get(class_slug, class_slug)
                predictions.append(
                    CarPrediction(
                        bounding_box=(left, top, right, bottom),
                        class_slug=class_slug,
                        display_label=display_label,
                        detection_confidence=float(box.conf[0].item()),
                        classification_confidence=classification_confidence,
                    )
                )
            return self._render(normalized_image, predictions), predictions
        except PhotoModelError:
            raise
        except Exception as error:
            LOGGER.exception("Photo detection and classification failed")
            raise PhotoInferenceError(
                "Photo detection and classification failed."
            ) from error

    def predict(self, image: Image.Image) -> Image.Image:
        """Return an annotated image for Streamlit compatibility."""
        annotated_image, _ = self.predict_with_details(image)
        return annotated_image

    @staticmethod
    def _render(
        image: Image.Image, predictions: list[CarPrediction]
    ) -> Image.Image:
        """Draw boxes and model/period labels on an image."""
        annotated_image: Image.Image = image.copy()
        draw: ImageDraw.ImageDraw = ImageDraw.Draw(annotated_image)
        font: ImageFont.ImageFont = ImageFont.load_default(size=16)
        for prediction in predictions:
            left, top, right, bottom = prediction.bounding_box
            label: str = (
                f"{prediction.display_label} "
                f"{prediction.classification_confidence:.0%}"
            )
            draw.rectangle((left, top, right, bottom), outline="#00E5FF", width=4)
            label_box: tuple[float, float, float, float] = draw.textbbox(
                (left, top), label, font=font
            )
            label_height: int = int(label_box[3] - label_box[1] + 10)
            label_width: int = int(label_box[2] - label_box[0] + 12)
            label_top: int = max(0, top - label_height)
            draw.rectangle(
                (left, label_top, left + label_width, label_top + label_height),
                fill="#00E5FF",
            )
            draw.text(
                (left + 6, label_top + 5),
                label,
                fill="#001014",
                font=font,
            )
        return annotated_image


__all__: list[str] = [
    "CarPrediction",
    "PhotoInferenceError",
    "PhotoModelError",
    "PhotoModelLoadError",
    "PhotoSpotter",
    "PhotoSpotterConfig",
]
