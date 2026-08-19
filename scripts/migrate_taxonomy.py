"""Create derived dataset metadata from reviewed taxonomy records."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Final, Sequence

from dataset_config import (
    DEFAULT_MUSTANG_BODY_STYLE_TAXONOMY_PATH,
    DatasetConfigError,
    load_taxonomy,
)
from review_store import ReviewStoreError, load_decisions, save_decisions
from taxonomy_migration import (
    TaxonomyMigrationError,
    load_class_mapping,
    migrate_manifest_records,
    migrate_review_decisions,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_MAPPING_PATH: Final[Path] = Path(
    "config/mappings/mustang_mvp_to_body_style_v2.json"
)


class TaxonomyMigrationScriptError(RuntimeError):
    """Raised when migration file input or output handling fails."""


def parse_arguments() -> argparse.Namespace:
    """Parse taxonomy migration command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate reviewed records to a compatible target taxonomy."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/cropped/manifest.jsonl")
    )
    parser.add_argument(
        "--decisions", type=Path, default=Path("datasets/review/decisions.json")
    )
    parser.add_argument(
        "--target-taxonomy",
        type=Path,
        default=DEFAULT_MUSTANG_BODY_STYLE_TAXONOMY_PATH,
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("datasets/mustang_body_style_v2/cropped/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-decisions",
        type=Path,
        default=Path("datasets/mustang_body_style_v2/review/decisions.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace derived metadata outputs, never source data.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest as object records."""
    if not path.is_file():
        raise TaxonomyMigrationScriptError(f"Manifest not found: '{path}'.")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise TaxonomyMigrationScriptError(
                    f"Manifest line {line_number} is not a JSON object."
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise TaxonomyMigrationScriptError(
            f"Unable to read manifest '{path}'."
        ) from error
    return records


def write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically write migrated records to JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise TaxonomyMigrationScriptError(
            f"Unable to write manifest '{path}'."
        ) from error


def ensure_outputs_are_safe(
    output_manifest: Path, output_decisions: Path, force: bool
) -> None:
    """Refuse accidental overwrite of derived v2 metadata."""
    existing_outputs: list[Path] = [
        path for path in (output_manifest, output_decisions) if path.exists()
    ]
    if existing_outputs and not force:
        output_text: str = ", ".join(str(path) for path in existing_outputs)
        raise TaxonomyMigrationScriptError(
            f"Migration output already exists: {output_text}. Use --force."
        )


def main() -> None:
    """Migrate manifest and decisions without changing legacy source files."""
    arguments = parse_arguments()
    ensure_outputs_are_safe(
        arguments.output_manifest, arguments.output_decisions, arguments.force
    )
    taxonomy = load_taxonomy(arguments.target_taxonomy)
    target_slugs: set[str] = set(taxonomy)
    mapping = load_class_mapping(arguments.mapping, target_slugs)
    source_records = load_manifest(arguments.manifest)
    source_decisions = load_decisions(arguments.decisions)
    migrated_records = migrate_manifest_records(
        source_records, mapping, target_slugs
    )
    migrated_decisions = migrate_review_decisions(
        source_decisions, mapping, target_slugs
    )
    write_manifest(arguments.output_manifest, migrated_records)
    save_decisions(arguments.output_decisions, migrated_decisions)
    manifest_counts: Counter[str] = Counter(
        str(record["class_slug"]) for record in migrated_records
    )
    accepted_counts: Counter[str] = Counter(
        decision.class_slug
        for decision in migrated_decisions.values()
        if decision.status == "accepted"
    )
    LOGGER.info("Migrated %d manifest records", len(migrated_records))
    LOGGER.info("Migrated %d review decisions", len(migrated_decisions))
    LOGGER.info("Manifest classes: %s", dict(sorted(manifest_counts.items())))
    LOGGER.info("Accepted classes: %s", dict(sorted(accepted_counts.items())))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (
        DatasetConfigError,
        ReviewStoreError,
        TaxonomyMigrationError,
        TaxonomyMigrationScriptError,
    ):
        LOGGER.exception("Taxonomy migration failed")
        raise SystemExit(1)
