"""Validate, deduplicate, and split raw images into classifier folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from PIL import Image, UnidentifiedImageError

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


class DatasetPreparationError(RuntimeError):
    """Raised when raw data cannot be converted into a training dataset."""


@dataclass(frozen=True)
class ImageRecord:
    """Validated source image and its dataset metadata."""

    class_slug: str
    source_path: Path
    source_title: str
    author: str
    license_name: str
    license_url: str
    source_page: str
    sha256: str
    perceptual_hash: str

    @property
    def group_key(self) -> str:
        """Return a conservative grouping key used to prevent data leakage."""
        if self.author.strip():
            return f"author:{self.author.casefold()}"
        return f"source:{self.source_page or self.source_title}"


def parse_arguments() -> argparse.Namespace:
    """Parse dataset preparation arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prepare train, val, and test folders from raw images."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/raw/manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/classification"),
    )
    parser.add_argument("--minimum-width", type=int, default=320)
    parser.add_argument("--minimum-height", type=int, default=240)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating space-efficient hard links.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing prepared dataset directory.",
    )
    return parser.parse_args()


def difference_hash(image: Image.Image, size: int = 8) -> str:
    """Compute a deterministic perceptual difference hash.

    Args:
        image: Decoded image.
        size: Number of comparison columns.

    Returns:
        Hexadecimal difference hash.
    """
    grayscale_image: Image.Image = image.convert("L").resize((size + 1, size))
    pixels: list[int] = list(grayscale_image.getdata())
    bits: list[bool] = []
    for row_index in range(size):
        row_offset: int = row_index * (size + 1)
        for column_index in range(size):
            bits.append(
                pixels[row_offset + column_index]
                > pixels[row_offset + column_index + 1]
            )
    bit_string: str = "".join("1" if bit else "0" for bit in bits)
    return f"{int(bit_string, 2):0{size * size // 4}x}"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Read JSONL manifest records.

    Args:
        path: Source manifest path.

    Returns:
        Decoded manifest records.

    Raises:
        DatasetPreparationError: If the manifest cannot be decoded.
    """
    if not path.is_file():
        raise DatasetPreparationError(f"Manifest not found: '{path}'.")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise DatasetPreparationError(
                    f"Manifest line {line_number} is not a JSON object."
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetPreparationError(f"Unable to read '{path}'.") from error
    return records


def validate_record(
    raw_record: dict[str, Any], minimum_width: int, minimum_height: int
) -> ImageRecord | None:
    """Decode one source image and normalize its metadata.

    Args:
        raw_record: Raw JSONL record.
        minimum_width: Minimum accepted image width.
        minimum_height: Minimum accepted image height.

    Returns:
        Validated record or ``None`` when the image is unsuitable.
    """
    source_path: Path = Path(str(raw_record.get("local_path", "")))
    class_slug: str = str(raw_record.get("class_slug", "")).strip()
    if not class_slug or not source_path.is_file():
        LOGGER.warning("Skipping missing or unclassified image: %s", source_path)
        return None
    try:
        with Image.open(source_path) as image:
            image.load()
            if image.width < minimum_width or image.height < minimum_height:
                LOGGER.warning("Skipping undersized image: %s", source_path)
                return None
            perceptual_hash: str = difference_hash(image)
    except (OSError, UnidentifiedImageError):
        LOGGER.exception("Skipping unreadable image: %s", source_path)
        return None

    sha256: str = str(raw_record.get("sha256", ""))
    if not sha256:
        sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return ImageRecord(
        class_slug=class_slug,
        source_path=source_path,
        source_title=str(raw_record.get("source_title", "")),
        author=str(raw_record.get("author", "")),
        license_name=str(raw_record.get("license", "")),
        license_url=str(raw_record.get("license_url", "")),
        source_page=str(raw_record.get("source_page", "")),
        sha256=sha256,
        perceptual_hash=perceptual_hash,
    )


def deduplicate(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    """Remove exact and perceptually identical images.

    Args:
        records: Validated source records.

    Returns:
        Deduplicated records in deterministic order.
    """
    unique_records: list[ImageRecord] = []
    seen_sha256: set[str] = set()
    seen_perceptual_hashes: set[tuple[str, str]] = set()
    for record in sorted(records, key=lambda item: str(item.source_path)):
        perceptual_key: tuple[str, str] = (
            record.class_slug,
            record.perceptual_hash,
        )
        if record.sha256 in seen_sha256 or perceptual_key in seen_perceptual_hashes:
            LOGGER.info("Skipping duplicate image: %s", record.source_path)
            continue
        seen_sha256.add(record.sha256)
        seen_perceptual_hashes.add(perceptual_key)
        unique_records.append(record)
    return unique_records


def split_for_group(group_key: str, train_ratio: float, val_ratio: float) -> str:
    """Assign a stable split to a source group.

    Args:
        group_key: Stable author or source identifier.
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.

    Returns:
        One of ``train``, ``val``, or ``test``.
    """
    digest: str = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    value: float = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def place_file(source: Path, destination: Path, copy_files: bool) -> None:
    """Copy or hard-link an image into its prepared split.

    Args:
        source: Raw image path.
        destination: Prepared image path.
        copy_files: Whether to copy instead of hard-linking.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def write_prepared_dataset(
    records: list[ImageRecord],
    output_directory: Path,
    train_ratio: float,
    val_ratio: float,
    copy_files: bool,
) -> None:
    """Materialize classification folders and their attribution manifest.

    Args:
        records: Deduplicated records.
        output_directory: Final dataset root.
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.
        copy_files: Whether images should be copied instead of hard-linked.
    """
    output_directory.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, str]] = []
    split_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        split: str = split_for_group(record.group_key, train_ratio, val_ratio)
        destination_name: str = record.sha256[:20] + record.source_path.suffix.lower()
        destination: Path = (
            output_directory / split / record.class_slug / destination_name
        )
        place_file(record.source_path, destination, copy_files)
        split_counts[record.class_slug][split] += 1
        rows.append(
            {
                "split": split,
                "class_slug": record.class_slug,
                "image": str(destination),
                "source_page": record.source_page,
                "source_title": record.source_title,
                "author": record.author,
                "license": record.license_name,
                "license_url": record.license_url,
                "sha256": record.sha256,
                "perceptual_hash": record.perceptual_hash,
            }
        )

    manifest_path: Path = output_directory / "manifest.csv"
    fieldnames: list[str] = list(rows[0].keys()) if rows else []
    if not fieldnames:
        raise DatasetPreparationError("No valid images were found in the manifest.")
    with manifest_path.open("w", encoding="utf-8", newline="") as output_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            output_file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    for class_slug, counts in sorted(split_counts.items()):
        LOGGER.info(
            "%s: train=%d val=%d test=%d",
            class_slug,
            counts["train"],
            counts["val"],
            counts["test"],
        )


def main() -> None:
    """Prepare a reproducible Ultralytics classification dataset."""
    arguments: argparse.Namespace = parse_arguments()
    train_ratio: float = arguments.train_ratio
    val_ratio: float = arguments.val_ratio
    if train_ratio <= 0.0 or val_ratio <= 0.0:
        raise DatasetPreparationError(
            "Train and validation ratios must be positive."
        )
    if train_ratio + val_ratio >= 1.0:
        raise DatasetPreparationError(
            "Train and validation ratios must total less than 1."
        )
    output_directory: Path = arguments.output
    if output_directory.exists():
        if not arguments.force:
            raise DatasetPreparationError(
                f"Output already exists: '{output_directory}'. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_directory)

    raw_records: list[dict[str, Any]] = load_manifest(arguments.manifest)
    validated_records: list[ImageRecord] = []
    for raw_record in raw_records:
        record: ImageRecord | None = validate_record(
            raw_record, arguments.minimum_width, arguments.minimum_height
        )
        if record is not None:
            validated_records.append(record)
    records: list[ImageRecord] = deduplicate(validated_records)
    write_prepared_dataset(
        records,
        output_directory,
        train_ratio,
        val_ratio,
        arguments.copy,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except DatasetPreparationError:
        LOGGER.exception("Dataset preparation failed")
        raise SystemExit(1)
