"""Safe utilities for migrating reviewed records to a new taxonomy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Iterable

from dataset_config import DatasetConfigError, load_json_object
from review_store import ReviewDecision

MAPPING_FIELD: Final[str] = "class_mapping"


class TaxonomyMigrationError(RuntimeError):
    """Raised when taxonomy migration inputs are invalid or incomplete."""


def load_class_mapping(path: Path, target_slugs: set[str]) -> dict[str, str]:
    """Load and validate a legacy-to-target class mapping.

    Args:
        path: Mapping configuration JSON path.
        target_slugs: Slugs permitted by the target taxonomy.

    Returns:
        Mapping indexed by legacy class slug.

    Raises:
        TaxonomyMigrationError: If the mapping is malformed or targets unknown
            taxonomy classes.
    """
    try:
        payload: dict[str, Any] = load_json_object(path)
    except DatasetConfigError as error:
        raise TaxonomyMigrationError(
            f"Unable to load class mapping '{path}'."
        ) from error
    raw_mapping: object = payload.get(MAPPING_FIELD)
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise TaxonomyMigrationError(
            f"Mapping '{path}' must contain a non-empty {MAPPING_FIELD} object."
        )
    mapping: dict[str, str] = {}
    for source_slug, target_slug in raw_mapping.items():
        if (
            not isinstance(source_slug, str)
            or not source_slug.strip()
            or not isinstance(target_slug, str)
            or not target_slug.strip()
        ):
            raise TaxonomyMigrationError(
                f"Mapping '{path}' contains an empty or non-string class slug."
            )
        normalized_source: str = source_slug.strip()
        normalized_target: str = target_slug.strip()
        if normalized_target not in target_slugs:
            raise TaxonomyMigrationError(
                f"Mapping target '{normalized_target}' is not in the target taxonomy."
            )
        mapping[normalized_source] = normalized_target
    return mapping


def migrate_class_slug(
    class_slug: str,
    mapping: dict[str, str],
    target_slugs: set[str],
) -> str:
    """Translate one legacy class slug while allowing already-migrated values.

    Args:
        class_slug: Legacy or target taxonomy class slug.
        mapping: Validated legacy-to-target mapping.
        target_slugs: Slugs permitted by the target taxonomy.

    Returns:
        Target taxonomy class slug.

    Raises:
        TaxonomyMigrationError: If the slug cannot be mapped safely.
    """
    normalized_slug: str = class_slug.strip()
    if normalized_slug in target_slugs:
        return normalized_slug
    target_slug: str | None = mapping.get(normalized_slug)
    if target_slug is None:
        raise TaxonomyMigrationError(
            f"No target mapping is configured for legacy class '{normalized_slug}'."
        )
    return target_slug


def migrate_manifest_records(
    records: Iterable[dict[str, Any]],
    mapping: dict[str, str],
    target_slugs: set[str],
) -> list[dict[str, Any]]:
    """Copy manifest records with target taxonomy slugs.

    The source records and image files are never modified. Changed labels retain
    their original value in ``legacy_class_slug`` for auditability.

    Args:
        records: Source manifest records.
        mapping: Validated legacy-to-target mapping.
        target_slugs: Slugs permitted by the target taxonomy.

    Returns:
        Migrated record copies in their original order.
    """
    migrated_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        legacy_slug: str = str(record.get("class_slug", "")).strip()
        if not legacy_slug:
            raise TaxonomyMigrationError(
                f"Manifest record {record_index} has no class_slug."
            )
        target_slug: str = migrate_class_slug(
            legacy_slug, mapping, target_slugs
        )
        migrated_record: dict[str, Any] = {**record, "class_slug": target_slug}
        if target_slug != legacy_slug:
            migrated_record["legacy_class_slug"] = legacy_slug
        migrated_records.append(migrated_record)
    return migrated_records


def migrate_review_decisions(
    decisions: dict[str, ReviewDecision],
    mapping: dict[str, str],
    target_slugs: set[str],
) -> dict[str, ReviewDecision]:
    """Copy persisted review decisions with target taxonomy slugs.

    Args:
        decisions: Source decisions indexed by record identifier.
        mapping: Validated legacy-to-target mapping.
        target_slugs: Slugs permitted by the target taxonomy.

    Returns:
        Equivalent decisions with their class slugs mapped to the target taxonomy.
    """
    migrated_decisions: dict[str, ReviewDecision] = {}
    for record_id, decision in decisions.items():
        target_slug: str = migrate_class_slug(
            decision.class_slug, mapping, target_slugs
        )
        migrated_decisions[record_id] = ReviewDecision(
            record_id=decision.record_id,
            status=decision.status,
            class_slug=target_slug,
            reviewed_at=decision.reviewed_at,
        )
    return migrated_decisions


__all__: list[str] = [
    "TaxonomyMigrationError",
    "load_class_mapping",
    "migrate_class_slug",
    "migrate_manifest_records",
    "migrate_review_decisions",
]
