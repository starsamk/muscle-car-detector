"""Collect reviewed candidates for the broad ``other_car`` class from Open Images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
OPEN_IMAGES_CLASS_NAMES: Final[tuple[str, ...]] = (
    "Bus",
    "Car",
    "Motorcycle",
    "Truck",
    "Van",
)
DEFAULT_VEHICLE_CLASSES: Final[tuple[str, ...]] = ("Car",)
OPEN_IMAGES_SOURCE: Final[str] = "open_images_v5_validation"
USER_AGENT: Final[str] = (
    "CarSpotterAI/0.1 "
    "(https://github.com/starsamk/muscle-car-detector; "
    "open-source dataset research)"
)
ALLOWED_LICENSE_MARKERS: Final[tuple[str, ...]] = (
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/publicdomain/",
    "creativecommons.org/share-your-work/public-domain/",
)


class OpenImagesCollectionError(RuntimeError):
    """Raised when Open Images metadata or downloads are invalid."""


def parse_arguments() -> argparse.Namespace:
    """Parse Open Images collection arguments."""
    parser = argparse.ArgumentParser(
        description="Collect outside-vehicle candidates for other_car."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-class", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--minimum-box-area", type=float, default=0.05)
    parser.add_argument(
        "--vehicle-class",
        action="append",
        choices=OPEN_IMAGES_CLASS_NAMES,
        dest="vehicle_classes",
        help=(
            "Open Images vehicle label to include; repeat to include several. "
            "Defaults to Car only."
        ),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Manifest whose Open Images source image IDs must be skipped; "
            "repeat for multiple manifests."
        ),
    )
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_class_descriptions(path: Path) -> dict[str, str]:
    """Load Open Images MIDs and display names."""
    if not path.is_file():
        raise OpenImagesCollectionError(f"Class descriptions not found: '{path}'.")
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            rows = csv.DictReader(input_file)
            descriptions: dict[str, str] = {}
            for row in rows:
                label_name: str = str(row.get("LabelName", "")).strip()
                display_name: str = str(row.get("DisplayName", "")).strip()
                if label_name and display_name:
                    descriptions[label_name] = display_name
    except (OSError, csv.Error) as error:
        raise OpenImagesCollectionError(
            f"Unable to read class descriptions '{path}'."
        ) from error
    return descriptions


def as_bool(value: object) -> bool:
    """Parse an Open Images boolean cell."""
    return str(value).strip().casefold() in {"1", "true", "yes"}


def box_area(row: dict[str, str]) -> float:
    """Return the normalized area of one Open Images box."""
    try:
        width: float = float(row["XMax"]) - float(row["XMin"])
        height: float = float(row["YMax"]) - float(row["YMin"])
    except (KeyError, TypeError, ValueError) as error:
        raise OpenImagesCollectionError(
            "Invalid Open Images box coordinates."
        ) from error
    return max(0.0, width) * max(0.0, height)


def select_candidates(
    rows: Iterable[dict[str, str]],
    class_names: dict[str, str],
    limit_per_class: int,
    max_images: int,
    minimum_box_area: float,
    vehicle_classes: Sequence[str] = DEFAULT_VEHICLE_CLASSES,
    excluded_image_ids: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """Select balanced, exterior vehicle candidates from annotation rows.

    Args:
        rows: Open Images bounding-box rows.
        class_names: Mapping from Open Images MIDs to display names.
        limit_per_class: Maximum candidates selected per vehicle class.
        max_images: Overall candidate cap.
        minimum_box_area: Minimum normalized box area.
        vehicle_classes: Open Images vehicle labels allowed in the collection.
        excluded_image_ids: Source image IDs already represented elsewhere.

    Returns:
        Candidate records indexed by image ID.
    """
    selected_vehicle_classes: set[str] = set(vehicle_classes)
    if not selected_vehicle_classes:
        raise OpenImagesCollectionError("At least one vehicle class must be selected.")
    unsupported_classes: set[str] = selected_vehicle_classes.difference(
        OPEN_IMAGES_CLASS_NAMES
    )
    if unsupported_classes:
        raise OpenImagesCollectionError(
            "Unsupported Open Images vehicle classes: "
            + ", ".join(sorted(unsupported_classes))
        )
    allowed_labels: dict[str, str] = {
        label_name: class_name
        for label_name, class_name in class_names.items()
        if class_name in selected_vehicle_classes
    }
    counts: Counter[str] = Counter()
    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id: str = str(row.get("ImageID", "")).strip()
        label_name: str = str(row.get("LabelName", "")).strip()
        class_name: str | None = allowed_labels.get(label_name)
        if not image_id or class_name is None or image_id in excluded_image_ids:
            continue
        if (
            as_bool(row.get("IsInside"))
            or as_bool(row.get("IsGroupOf"))
            or as_bool(row.get("IsDepiction"))
            or as_bool(row.get("IsTruncated"))
        ):
            continue
        if box_area(row) < minimum_box_area or counts[class_name] >= limit_per_class:
            continue
        if image_id in candidates:
            continue
        candidates[image_id] = {
            "image_id": image_id,
            "vehicle_annotation": class_name,
            "label_name": label_name,
            "box": {
                "xmin": float(row["XMin"]),
                "xmax": float(row["XMax"]),
                "ymin": float(row["YMin"]),
                "ymax": float(row["YMax"]),
            },
        }
        counts[class_name] += 1
        if len(candidates) >= max_images:
            break
    LOGGER.info("Selected Open Images candidates by class: %s", dict(counts))
    return candidates


def load_excluded_image_ids(manifests: Sequence[Path]) -> frozenset[str]:
    """Load Open Images source IDs from existing manifests.

    Args:
        manifests: JSONL manifests that may already contain Open Images records.

    Returns:
        Immutable set of source image IDs to exclude.

    Raises:
        OpenImagesCollectionError: If one manifest cannot be read as JSONL.
    """
    excluded_ids: set[str] = set()
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise OpenImagesCollectionError(
                f"Excluded manifest not found: '{manifest_path}'."
            )
        try:
            for line_number, line in enumerate(
                manifest_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                value: object = json.loads(line)
                if not isinstance(value, dict):
                    raise OpenImagesCollectionError(
                        f"Invalid record in '{manifest_path}' at line {line_number}."
                    )
                source_image_id: str = str(value.get("source_image_id", "")).strip()
                if source_image_id:
                    excluded_ids.add(source_image_id)
        except (OSError, json.JSONDecodeError) as error:
            raise OpenImagesCollectionError(
                f"Unable to read excluded manifest '{manifest_path}'."
            ) from error
    return frozenset(excluded_ids)


def license_is_allowed(license_url: str) -> bool:
    """Return whether the source license permits this derived dataset."""
    normalized_url: str = license_url.strip().casefold()
    return any(marker in normalized_url for marker in ALLOWED_LICENSE_MARKERS)


def download_image(url: str, destination: Path) -> str:
    """Download one image atomically and return its SHA-256 digest."""
    temporary_path: Path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as output_file:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output_file.write(chunk)
        temporary_path.replace(destination)
    except (OSError, urllib.error.HTTPError) as error:
        temporary_path.unlink(missing_ok=True)
        raise OpenImagesCollectionError(f"Unable to download '{url}'.") from error
    return digest.hexdigest()


def download_image_with_fallback(
    urls: Sequence[str], destination: Path
) -> tuple[str, str]:
    """Try source URLs in order and return the digest and URL that succeeded.

    Open Images metadata can contain historical Flickr originals that are no
    longer available. Its thumbnail URL is a valid lower-resolution fallback
    for a negative candidate and is still retained with attribution metadata.

    Args:
        urls: Ordered non-empty candidate URLs.
        destination: Local image destination.

    Returns:
        SHA-256 digest and the successful source URL.

    Raises:
        OpenImagesCollectionError: If every candidate URL fails.
    """
    errors: list[str] = []
    for url in urls:
        try:
            return download_image(url, destination), url
        except OpenImagesCollectionError as error:
            errors.append(str(error))
            LOGGER.warning("Open Images URL unavailable, trying fallback: %s", url)
    raise OpenImagesCollectionError(
        "All Open Images source URLs failed: " + "; ".join(errors)
    )


def load_metadata(path: Path, selected_ids: set[str]) -> dict[str, dict[str, str]]:
    """Load source metadata only for selected image IDs."""
    if not path.is_file():
        raise OpenImagesCollectionError(f"Image metadata not found: '{path}'.")
    metadata: dict[str, dict[str, str]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            rows = csv.DictReader(input_file)
            for row in rows:
                image_id: str = str(row.get("ImageID", "")).strip()
                if image_id in selected_ids:
                    metadata[image_id] = {
                        key: str(row.get(key, "")).strip()
                        for key in (
                            "OriginalURL",
                            "Thumbnail300KURL",
                            "OriginalLandingURL",
                            "License",
                            "Author",
                            "Title",
                        )
                    }
    except (OSError, csv.Error) as error:
        raise OpenImagesCollectionError(
            f"Unable to read image metadata '{path}'."
        ) from error
    return metadata


def load_annotations(path: Path) -> list[dict[str, str]]:
    """Load bounding-box annotations into memory for the selected split."""
    if not path.is_file():
        raise OpenImagesCollectionError(f"Annotations not found: '{path}'.")
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            return list(csv.DictReader(input_file))
    except (OSError, csv.Error) as error:
        raise OpenImagesCollectionError(
            f"Unable to read annotations '{path}'."
        ) from error


def append_manifest_record(path: Path, record: dict[str, Any]) -> None:
    """Append one manifest record and flush it for resumability."""
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def main() -> None:
    """Select, download, and register Open Images negative candidates."""
    arguments = parse_arguments()
    if arguments.limit_per_class <= 0 or arguments.max_images <= 0:
        raise OpenImagesCollectionError("Image limits must be greater than zero.")
    if not 0.0 < arguments.minimum_box_area <= 1.0:
        raise OpenImagesCollectionError("Minimum box area must be in (0, 1].")
    output_directory: Path = arguments.output
    manifest_path: Path = output_directory / "manifest.jsonl"
    if manifest_path.exists() and not arguments.force:
        raise OpenImagesCollectionError(
            f"Output already exists: '{manifest_path}'. Use --force."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.force:
        manifest_path.unlink(missing_ok=True)

    descriptions = load_class_descriptions(arguments.class_descriptions)
    annotations = load_annotations(arguments.annotations)
    excluded_image_ids: frozenset[str] = load_excluded_image_ids(
        arguments.exclude_manifest
    )
    candidates = select_candidates(
        annotations,
        descriptions,
        arguments.limit_per_class,
        arguments.max_images,
        arguments.minimum_box_area,
        tuple(arguments.vehicle_classes or DEFAULT_VEHICLE_CLASSES),
        excluded_image_ids,
    )
    metadata = load_metadata(arguments.metadata, set(candidates))
    downloaded_hashes: set[str] = set()
    for image_id, candidate in candidates.items():
        source = metadata.get(image_id)
        if source is None or not license_is_allowed(source["License"]):
            LOGGER.warning("Skipping missing or unsupported license: %s", image_id)
            continue
        source_urls: list[str] = [
            url for url in (source["OriginalURL"], source["Thumbnail300KURL"]) if url
        ]
        if not source_urls:
            continue
        image_path: Path = output_directory / "images" / "other_car" / f"{image_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if image_path.exists():
                image_hash: str = hashlib.sha256(image_path.read_bytes()).hexdigest()
                source_url: str = source_urls[0]
            else:
                image_hash, source_url = download_image_with_fallback(
                    source_urls, image_path
                )
        except OpenImagesCollectionError:
            LOGGER.exception("Failed to download Open Images image %s", image_id)
            continue
        if image_hash in downloaded_hashes:
            LOGGER.info("Skipping duplicate image content: %s", image_id)
            continue
        downloaded_hashes.add(image_hash)
        record: dict[str, Any] = {
            "class_slug": "other_car",
            "display_name": "Other car",
            "local_path": str(image_path),
            "sha256": image_hash,
            "source": OPEN_IMAGES_SOURCE,
            "source_split": "validation",
            "source_image_id": image_id,
            "source_title": source["Title"],
            "source_page": source["OriginalLandingURL"],
            "source_image_url": source_url,
            "author": source["Author"],
            "license": source["License"],
            "license_url": source["License"],
            "vehicle_annotation": candidate["vehicle_annotation"],
            "annotation_label_name": candidate["label_name"],
            "annotation_box": candidate["box"],
            "record_id": f"open-images-validation-{image_id}",
            "status": "candidate",
        }
        append_manifest_record(manifest_path, record)
        time.sleep(max(0.0, arguments.delay))
    LOGGER.info("Open Images collection completed: %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except OpenImagesCollectionError:
        LOGGER.exception("Open Images negative collection failed")
        raise SystemExit(1)
