"""Typed loaders for the dataset taxonomy and feature profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DEFAULT_TAXONOMY_PATH: Final[Path] = Path("config/taxonomy.json")
DEFAULT_MVP_PROFILE_PATH: Final[Path] = Path("config/profiles/mustang_mvp.json")
DEFAULT_MUSTANG_BODY_STYLE_TAXONOMY_PATH: Final[Path] = Path(
    "config/taxonomy_mustang_body_style_v2.json"
)
DEFAULT_MUSTANG_BODY_STYLE_PROFILE_PATH: Final[Path] = Path(
    "config/profiles/mustang_body_style_v2.json"
)
DEFAULT_VEHICLE_TAXONOMY_V3_PATH: Final[Path] = Path("config/taxonomy_vehicle_v3.json")
DEFAULT_VEHICLE_TAXONOMY_V3_PROFILE_PATH: Final[Path] = Path(
    "config/profiles/vehicle_taxonomy_v3.json"
)


class DatasetConfigError(RuntimeError):
    """Raised when taxonomy or profile configuration is invalid."""


@dataclass(frozen=True)
class TaxonomyClass:
    """One fine-grained classification target."""

    slug: str
    display_name: str
    make: str
    model: str
    generation: str
    body_style: str
    year_start: int
    year_end: int
    show_year_range: bool
    wikimedia_categories: tuple[str, ...]

    @property
    def label(self) -> str:
        """Return an unambiguous user-facing model and production period."""
        if not self.show_year_range or self.year_start <= 0 or self.year_end <= 0:
            return self.display_name
        period: str = (
            str(self.year_start)
            if self.year_start == self.year_end
            else f"{self.year_start}–{self.year_end}"
        )
        return f"{self.display_name} — {period}"


@dataclass(frozen=True)
class DatasetProfile:
    """Named subset of taxonomy classes used by one experiment."""

    name: str
    description: str
    class_slugs: tuple[str, ...]


def load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk.

    Args:
        path: JSON document path.

    Returns:
        Parsed JSON object.

    Raises:
        DatasetConfigError: If the file is missing, malformed, or not an object.
    """
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetConfigError(f"Unable to read configuration '{path}'.") from error
    if not isinstance(payload, dict):
        raise DatasetConfigError(f"Configuration '{path}' must contain an object.")
    return payload


def required_string(mapping: dict[str, Any], key: str, source: Path) -> str:
    """Read a required non-empty string field."""
    value: object = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigError(f"'{key}' must be a non-empty string in '{source}'.")
    return value.strip()


def required_integer(mapping: dict[str, Any], key: str, source: Path) -> int:
    """Read a required integer field."""
    value: object = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DatasetConfigError(f"'{key}' must be an integer in '{source}'.")
    return value


def optional_boolean(
    mapping: dict[str, Any], key: str, default: bool, source: Path
) -> bool:
    """Read an optional boolean field with strict validation.

    Args:
        mapping: Configuration object containing the value.
        key: Field name to read.
        default: Value used when the field is absent.
        source: Source configuration path used in error messages.

    Returns:
        The configured boolean or the supplied default.

    Raises:
        DatasetConfigError: If the configured value is not a boolean.
    """
    value: object = mapping.get(key, default)
    if not isinstance(value, bool):
        raise DatasetConfigError(f"'{key}' must be a boolean in '{source}'.")
    return value


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> dict[str, TaxonomyClass]:
    """Load and validate all configured taxonomy classes.

    Args:
        path: Taxonomy JSON path.

    Returns:
        Class definitions indexed by slug.
    """
    payload: dict[str, Any] = load_json_object(path)
    raw_classes: object = payload.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise DatasetConfigError(f"Taxonomy '{path}' must contain a classes list.")

    taxonomy: dict[str, TaxonomyClass] = {}
    for index, raw_class in enumerate(raw_classes):
        if not isinstance(raw_class, dict):
            raise DatasetConfigError(
                f"Class at index {index} in '{path}' must be an object."
            )
        slug: str = required_string(raw_class, "slug", path)
        if slug in taxonomy:
            raise DatasetConfigError(f"Duplicate class slug '{slug}' in '{path}'.")
        raw_categories: object = raw_class.get("wikimedia_categories", [])
        if not isinstance(raw_categories, list) or not all(
            isinstance(category, str) and category.strip()
            for category in raw_categories
        ):
            raise DatasetConfigError(
                f"Invalid Wikimedia categories for '{slug}' in '{path}'."
            )
        year_start: int = required_integer(raw_class, "year_start", path)
        year_end: int = required_integer(raw_class, "year_end", path)
        if year_start > year_end:
            raise DatasetConfigError(
                f"Invalid year range for class '{slug}' in '{path}'."
            )
        taxonomy[slug] = TaxonomyClass(
            slug=slug,
            display_name=required_string(raw_class, "display_name", path),
            make=required_string(raw_class, "make", path),
            model=required_string(raw_class, "model", path),
            generation=str(raw_class.get("generation", "")).strip(),
            body_style=str(raw_class.get("body_style", "")).strip(),
            year_start=year_start,
            year_end=year_end,
            show_year_range=optional_boolean(raw_class, "show_year_range", True, path),
            wikimedia_categories=tuple(
                str(category).strip() for category in raw_categories
            ),
        )
    return taxonomy


def load_profile(
    path: Path,
    taxonomy: dict[str, TaxonomyClass],
) -> DatasetProfile:
    """Load a class subset and ensure every slug exists in the taxonomy.

    Args:
        path: Profile JSON path.
        taxonomy: Available taxonomy classes.

    Returns:
        Validated dataset profile.
    """
    payload: dict[str, Any] = load_json_object(path)
    raw_slugs: object = payload.get("class_slugs")
    if not isinstance(raw_slugs, list) or not raw_slugs:
        raise DatasetConfigError(f"Profile '{path}' must contain class_slugs.")
    class_slugs: tuple[str, ...] = tuple(str(slug).strip() for slug in raw_slugs)
    if any(not slug for slug in class_slugs):
        raise DatasetConfigError(f"Profile '{path}' contains an empty class slug.")
    if len(class_slugs) != len(set(class_slugs)):
        raise DatasetConfigError(f"Profile '{path}' contains duplicate classes.")
    unknown_slugs: set[str] = set(class_slugs) - set(taxonomy)
    if unknown_slugs:
        unknown_text: str = ", ".join(sorted(unknown_slugs))
        raise DatasetConfigError(
            f"Profile '{path}' references unknown classes: {unknown_text}"
        )
    return DatasetProfile(
        name=required_string(payload, "name", path),
        description=str(payload.get("description", "")).strip(),
        class_slugs=class_slugs,
    )


def select_taxonomy_classes(
    taxonomy: dict[str, TaxonomyClass],
    profile: DatasetProfile | None,
    explicit_slugs: list[str] | None = None,
) -> list[TaxonomyClass]:
    """Select ordered taxonomy definitions for a profile or explicit slugs.

    Args:
        taxonomy: All available classes indexed by slug.
        profile: Optional named class subset.
        explicit_slugs: Optional class slugs supplied by the caller.

    Returns:
        Ordered selected class definitions.
    """
    requested_slugs: list[str]
    if explicit_slugs:
        requested_slugs = explicit_slugs
    elif profile is not None:
        requested_slugs = list(profile.class_slugs)
    else:
        requested_slugs = list(taxonomy)
    unknown_slugs: set[str] = set(requested_slugs) - set(taxonomy)
    if unknown_slugs:
        unknown_text: str = ", ".join(sorted(unknown_slugs))
        raise DatasetConfigError(f"Unknown class slugs: {unknown_text}")
    return [taxonomy[slug] for slug in requested_slugs]


__all__: list[str] = [
    "DEFAULT_MVP_PROFILE_PATH",
    "DEFAULT_MUSTANG_BODY_STYLE_PROFILE_PATH",
    "DEFAULT_MUSTANG_BODY_STYLE_TAXONOMY_PATH",
    "DEFAULT_TAXONOMY_PATH",
    "DEFAULT_VEHICLE_TAXONOMY_V3_PATH",
    "DEFAULT_VEHICLE_TAXONOMY_V3_PROFILE_PATH",
    "DatasetConfigError",
    "DatasetProfile",
    "TaxonomyClass",
    "load_profile",
    "load_taxonomy",
    "select_taxonomy_classes",
]
