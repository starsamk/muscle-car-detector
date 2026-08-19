"""Validate the structure and image quality of a classification dataset."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError

from dataset_config import (
    DEFAULT_MVP_PROFILE_PATH,
    DEFAULT_TAXONOMY_PATH,
    DatasetConfigError,
    load_profile,
    load_taxonomy,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
IMAGE_SUFFIXES: Final[set[str]] = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetValidationError(RuntimeError):
    """Raised when a prepared classification dataset is invalid."""


def parse_arguments() -> argparse.Namespace:
    """Parse validation arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Validate prepared classification folders and images."
    )
    parser.add_argument("--data", type=Path, default=Path("datasets/classification"))
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--profile", type=Path, default=DEFAULT_MVP_PROFILE_PATH)
    parser.add_argument("--minimum-per-split", type=int, default=5)
    return parser.parse_args()


def validate_image(path: Path) -> None:
    """Decode and verify an image file.

    Args:
        path: Image path.

    Raises:
        DatasetValidationError: If Pillow cannot verify the file.
    """
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetValidationError(f"Invalid image: '{path}'.") from error


def validate_dataset(
    data_directory: Path, expected_classes: set[str], minimum_per_split: int
) -> dict[str, dict[str, int]]:
    """Validate split symmetry, image files, and minimum sample counts.

    Args:
        data_directory: Prepared dataset root.
        expected_classes: Classes declared in the taxonomy.
        minimum_per_split: Required images per class and split.

    Returns:
        Nested split/class count summary.
    """
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    errors: list[str] = []
    for split in SPLITS:
        split_directory: Path = data_directory / split
        if not split_directory.is_dir():
            errors.append(f"Missing split directory: {split_directory}")
            continue
        present_classes: set[str] = {
            path.name for path in split_directory.iterdir() if path.is_dir()
        }
        unknown_classes: set[str] = present_classes - expected_classes
        if unknown_classes:
            errors.append(
                f"Unknown classes in {split}: {', '.join(sorted(unknown_classes))}"
            )
        for class_slug in sorted(expected_classes):
            class_directory: Path = split_directory / class_slug
            image_paths: list[Path] = []
            if class_directory.is_dir():
                image_paths = [
                    path
                    for path in class_directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                ]
            counts[split][class_slug] = len(image_paths)
            if len(image_paths) < minimum_per_split:
                errors.append(
                    f"{split}/{class_slug}: {len(image_paths)} image(s), "
                    f"minimum is {minimum_per_split}"
                )
            for image_path in image_paths:
                try:
                    validate_image(image_path)
                except DatasetValidationError as error:
                    errors.append(str(error))

    if errors:
        preview: str = "\n".join(f"- {error}" for error in errors[:50])
        suffix: str = "\n- ..." if len(errors) > 50 else ""
        raise DatasetValidationError(
            f"Dataset validation found {len(errors)} issue(s):\n{preview}{suffix}"
        )
    return dict(counts)


def main() -> None:
    """Run validation and log a concise class distribution summary."""
    arguments: argparse.Namespace = parse_arguments()
    taxonomy = load_taxonomy(arguments.taxonomy)
    profile = load_profile(arguments.profile, taxonomy)
    expected_classes: set[str] = set(profile.class_slugs)
    counts: dict[str, dict[str, int]] = validate_dataset(
        arguments.data, expected_classes, arguments.minimum_per_split
    )
    for split in SPLITS:
        total: int = sum(counts.get(split, {}).values())
        LOGGER.info("%s: %d validated images", split, total)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (DatasetValidationError, DatasetConfigError):
        LOGGER.exception("Dataset validation failed")
        raise SystemExit(1)
