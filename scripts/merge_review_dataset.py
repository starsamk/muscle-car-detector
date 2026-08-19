"""Merge derived manifests and review decisions into one review queue."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final

from review_store import (
    ReviewDecision,
    ReviewStoreError,
    load_decisions,
    save_decisions,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DERIVED_PATH_FIELDS: Final[frozenset[str]] = frozenset({"crop_path"})


class DatasetMergeError(RuntimeError):
    """Raised when derived dataset inputs cannot be merged safely."""


def records_are_idempotent_duplicates(
    first: dict[str, Any], second: dict[str, Any]
) -> bool:
    """Check whether two records differ only in derived local paths.

    Args:
        first: Existing record with precedence.
        second: Record from a later manifest.

    Returns:
        ``True`` when both records describe the same source image and metadata.
    """
    first_source: dict[str, Any] = {
        key: value for key, value in first.items() if key not in DERIVED_PATH_FIELDS
    }
    second_source: dict[str, Any] = {
        key: value for key, value in second.items() if key not in DERIVED_PATH_FIELDS
    }
    return first_source == second_source


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest.

    Args:
        path: Manifest path.

    Returns:
        Manifest records in source order.

    Raises:
        DatasetMergeError: If the file is missing or malformed.
    """
    if not path.is_file():
        raise DatasetMergeError(f"Manifest not found: '{path}'.")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise DatasetMergeError(
                    f"Manifest line {line_number} in '{path}' is not an object."
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetMergeError(f"Unable to read manifest '{path}'.") from error
    return records


def merge_manifest_records(
    manifests: Iterable[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge records while rejecting ID collisions and removing SHA duplicates.

    The first occurrence wins, which keeps the existing reviewed dataset as the
    authoritative record when it is supplied first.

    Args:
        manifests: Manifest sequences in precedence order.

    Returns:
        Merged records in stable order.

    Raises:
        DatasetMergeError: If a record has no ID or collides with another record.
    """
    merged: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    records_by_id: dict[str, dict[str, Any]] = {}
    seen_sha256: set[str] = set()
    for manifest_index, records in enumerate(manifests, start=1):
        for record_index, record in enumerate(records, start=1):
            record_id: str = str(record.get("record_id", "")).strip()
            if not record_id:
                raise DatasetMergeError(
                    f"Manifest {manifest_index}, record {record_index} has no record_id."
                )
            if record_id in seen_record_ids:
                if records_are_idempotent_duplicates(records_by_id[record_id], record):
                    LOGGER.info(
                        "Skipping idempotent duplicate record '%s'",
                        record_id,
                    )
                    continue
                raise DatasetMergeError(
                    f"Duplicate record_id encountered: '{record_id}'."
                )
            seen_record_ids.add(record_id)
            records_by_id[record_id] = record
            sha256: str = str(record.get("sha256", "")).strip()
            if sha256 and sha256 in seen_sha256:
                LOGGER.info("Skipping duplicate image for record '%s'", record_id)
                continue
            if sha256:
                seen_sha256.add(sha256)
            merged.append(dict(record))
    return merged


def merge_decisions(
    decisions: Iterable[dict[str, ReviewDecision]],
) -> dict[str, ReviewDecision]:
    """Merge decisions and reject conflicting decisions for one record ID.

    Args:
        decisions: Decision maps in precedence order.

    Returns:
        Combined decisions.

    Raises:
        DatasetMergeError: If two inputs disagree about a record.
    """
    merged: dict[str, ReviewDecision] = {}
    for decision_map in decisions:
        for record_id, decision in decision_map.items():
            previous: ReviewDecision | None = merged.get(record_id)
            if previous is not None and previous != decision:
                raise DatasetMergeError(
                    f"Conflicting review decisions for record '{record_id}'."
                )
            merged[record_id] = decision
    return merged


def write_manifest(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write a JSONL manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise DatasetMergeError(f"Unable to write manifest '{path}'.") from error


def parse_arguments() -> argparse.Namespace:
    """Parse merge command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge review manifests without mutating their source files."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest to merge; repeat in precedence order.",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        action="append",
        default=[],
        help="Review store to merge; repeat in precedence order.",
    )
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-decisions", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_outputs_are_safe(paths: Sequence[Path], force: bool) -> None:
    """Refuse accidental overwrite of derived outputs."""
    existing: list[Path] = [path for path in paths if path.exists()]
    if existing and not force:
        output_text: str = ", ".join(str(path) for path in existing)
        raise DatasetMergeError(
            f"Merge output already exists: {output_text}. Use --force."
        )


def main() -> None:
    """Merge manifests and review stores into derived outputs."""
    arguments = parse_arguments()
    ensure_outputs_are_safe(
        (arguments.output_manifest, arguments.output_decisions), arguments.force
    )
    manifest_records: list[list[dict[str, Any]]] = [
        load_manifest(path) for path in arguments.manifest
    ]
    decision_maps: list[dict[str, ReviewDecision]] = [
        load_decisions(path) for path in arguments.decisions
    ]
    merged_records: list[dict[str, Any]] = merge_manifest_records(manifest_records)
    merged_decisions: dict[str, ReviewDecision] = merge_decisions(decision_maps)
    write_manifest(arguments.output_manifest, merged_records)
    save_decisions(arguments.output_decisions, merged_decisions)
    LOGGER.info("Merged %d manifest records", len(merged_records))
    LOGGER.info("Merged %d review decisions", len(merged_decisions))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (DatasetMergeError, ReviewStoreError):
        LOGGER.exception("Dataset merge failed")
        raise SystemExit(1)
