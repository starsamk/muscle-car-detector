"""Detect and crop the principal car from collected training images."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Sequence

from PIL import Image, UnidentifiedImageError

from dataset_config import (
    DEFAULT_MVP_PROFILE_PATH,
    DEFAULT_TAXONOMY_PATH,
    DatasetConfigError,
    load_profile,
    load_taxonomy,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
COCO_CAR_CLASS_ID: Final[int] = 2


class CropDatasetError(RuntimeError):
    """Raised when automatic dataset cropping cannot complete."""


class Detector(Protocol):
    """Minimal interface required from an Ultralytics detector."""

    def predict(self, **kwargs: Any) -> Sequence[Any]:
        """Run object detection and return result objects."""


@dataclass(frozen=True)
class Detection:
    """One car bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        """Return the non-negative bounding-box area."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Crop the principal YOLO-detected car from raw images."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/raw/manifest.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/cropped"))
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--profile", type=Path, default=DEFAULT_MVP_PROFILE_PATH)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument(
        "--ambiguity-ratio",
        type=float,
        default=0.65,
        help="Reject images whose second car is this large relative to the first.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest into validated object records."""
    if not path.is_file():
        raise CropDatasetError(f"Manifest not found: '{path}'.")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise CropDatasetError(
                    f"Manifest line {line_number} is not a JSON object."
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise CropDatasetError(f"Unable to read manifest '{path}'.") from error
    return records


def resolve_device(requested_device: str) -> str:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""
    if requested_device != "auto":
        return requested_device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_detector(model_path: str) -> Detector:
    """Load an Ultralytics YOLO detector lazily."""
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except (ImportError, OSError, RuntimeError) as error:
        raise CropDatasetError(
            f"Unable to load YOLO detector '{model_path}'."
        ) from error


def extract_detections(result: Any) -> list[Detection]:
    """Convert an Ultralytics result into sorted car detections."""
    boxes: Any = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections: list[Detection] = []
    for box in boxes:
        class_id: int = int(box.cls[0].item())
        if class_id != COCO_CAR_CLASS_ID:
            continue
        coordinates: list[float] = box.xyxy[0].tolist()
        detections.append(
            Detection(
                x1=coordinates[0],
                y1=coordinates[1],
                x2=coordinates[2],
                y2=coordinates[3],
                confidence=float(box.conf[0].item()),
            )
        )
    return sorted(detections, key=lambda detection: detection.area, reverse=True)


def select_detection(
    detections: Sequence[Detection], ambiguity_ratio: float
) -> tuple[Detection | None, str]:
    """Select the largest car unless another similarly large car is present."""
    if not detections:
        return None, "no_detection"
    principal: Detection = detections[0]
    if (
        len(detections) > 1
        and principal.area > 0
        and detections[1].area / principal.area >= ambiguity_ratio
    ):
        return None, "ambiguous"
    return principal, "success"


def padded_box(
    detection: Detection, width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Expand and clamp a detection box to image boundaries."""
    horizontal_padding: float = (detection.x2 - detection.x1) * padding
    vertical_padding: float = (detection.y2 - detection.y1) * padding
    return (
        max(0, int(detection.x1 - horizontal_padding)),
        max(0, int(detection.y1 - vertical_padding)),
        min(width, int(detection.x2 + horizontal_padding)),
        min(height, int(detection.y2 + vertical_padding)),
    )


def stable_record_id(record: dict[str, Any]) -> str:
    """Return a stable identifier derived from source metadata."""
    existing: str = str(record.get("record_id", "")).strip()
    if existing:
        return existing
    identity: str = "|".join(
        (
            str(record.get("class_slug", "")),
            str(record.get("source_page", "")),
            str(record.get("sha256", "")),
            str(record.get("local_path", "")),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def process_record(
    record: dict[str, Any],
    detector: Detector,
    output_directory: Path,
    device: str,
    confidence: float,
    padding: float,
    ambiguity_ratio: float,
) -> dict[str, Any]:
    """Detect, crop, and describe one source image."""
    source_path = Path(str(record.get("local_path", "")))
    record_id: str = stable_record_id(record)
    output_record: dict[str, Any] = {**record, "record_id": record_id}
    if not source_path.is_file():
        return {**output_record, "status": "error", "error": "source_missing"}
    try:
        results: Sequence[Any] = detector.predict(
            source=str(source_path),
            classes=[COCO_CAR_CLASS_ID],
            conf=confidence,
            device=device,
            verbose=False,
        )
        detections: list[Detection] = (
            extract_detections(results[0]) if results else []
        )
        selected, status = select_detection(detections, ambiguity_ratio)
        if selected is None:
            return {
                **output_record,
                "status": status,
                "detection_count": len(detections),
            }
        with Image.open(source_path) as image:
            rgb_image: Image.Image = image.convert("RGB")
            box: tuple[int, int, int, int] = padded_box(
                selected, rgb_image.width, rgb_image.height, padding
            )
            crop: Image.Image = rgb_image.crop(box)
            class_slug: str = str(record.get("class_slug", "unknown"))
            destination: Path = (
                output_directory / "images" / class_slug / f"{record_id}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path = destination.with_suffix(".tmp.jpg")
            crop.save(temporary_path, format="JPEG", quality=95)
            os.replace(temporary_path, destination)
        return {
            **output_record,
            "status": "success",
            "crop_path": str(destination),
            "bounding_box": list(box),
            "detection_confidence": selected.confidence,
            "detection_count": len(detections),
        }
    except (OSError, RuntimeError, UnidentifiedImageError, ValueError) as error:
        LOGGER.exception("Unable to crop %s", source_path)
        return {**output_record, "status": "error", "error": str(error)}


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise CropDatasetError(f"Unable to write manifest '{path}'.") from error


def main() -> None:
    """Run automatic cropping for classes in the selected profile."""
    arguments = parse_arguments()
    if not 0.0 <= arguments.padding <= 0.5:
        raise CropDatasetError("Padding must be between 0 and 0.5.")
    if not 0.0 < arguments.ambiguity_ratio <= 1.0:
        raise CropDatasetError("Ambiguity ratio must be between 0 and 1.")
    output_manifest: Path = arguments.output / "manifest.jsonl"
    if output_manifest.exists() and not arguments.force:
        raise CropDatasetError(
            f"Output manifest already exists: '{output_manifest}'. Use --force."
        )
    taxonomy = load_taxonomy(arguments.taxonomy)
    profile = load_profile(arguments.profile, taxonomy)
    allowed_classes: set[str] = set(profile.class_slugs)
    source_records: list[dict[str, Any]] = [
        record
        for record in load_jsonl(arguments.manifest)
        if str(record.get("class_slug", "")) in allowed_classes
    ]
    detector: Detector = load_detector(arguments.model)
    device: str = resolve_device(arguments.device)
    LOGGER.info("Cropping %d images on %s", len(source_records), device)
    processed_records: list[dict[str, Any]] = [
        process_record(
            record,
            detector,
            arguments.output,
            device,
            arguments.confidence,
            arguments.padding,
            arguments.ambiguity_ratio,
        )
        for record in source_records
    ]
    write_jsonl(output_manifest, processed_records)
    counts: dict[str, int] = {}
    for record in processed_records:
        status: str = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    LOGGER.info("Cropping completed: %s", counts)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (CropDatasetError, DatasetConfigError):
        LOGGER.exception("Dataset cropping failed")
        raise SystemExit(1)
