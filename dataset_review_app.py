"""Streamlit interface for reviewing automatically cropped dataset images."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Final

import streamlit as st

from dataset_config import (
    DEFAULT_MVP_PROFILE_PATH,
    DEFAULT_TAXONOMY_PATH,
    DatasetConfigError,
    TaxonomyClass,
    load_profile,
    load_taxonomy,
)
from review_store import (
    ReviewDecision,
    ReviewStoreError,
    load_deleted_record_ids,
    load_decisions,
    save_deleted_record_ids,
    save_decisions,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_MANIFEST: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_MANIFEST", "datasets/cropped/manifest.jsonl")
)
DEFAULT_DECISIONS: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_DECISIONS", "datasets/review/decisions.json")
)
DEFAULT_DELETED: Final[Path] = Path(
    os.getenv(
        "CAR_SPOTTER_REVIEW_DELETED",
        str(DEFAULT_DECISIONS.with_name("deleted.json")),
    )
)
REVIEW_TAXONOMY_PATH: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_TAXONOMY_PATH", str(DEFAULT_TAXONOMY_PATH))
)
REVIEW_PROFILE_PATH: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_PROFILE_PATH", str(DEFAULT_MVP_PROFILE_PATH))
)


class ReviewAppError(RuntimeError):
    """Raised when review data cannot be displayed safely."""


@st.cache_data(show_spinner=False)
def load_manifest(path_text: str) -> list[dict[str, Any]]:
    """Load successful crop records from a JSONL manifest."""
    path = Path(path_text)
    if not path.is_file():
        raise ReviewAppError(f"Manifest introuvable : {path}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ReviewAppError(
                    f"La ligne {line_number} du manifeste est invalide."
                )
            if value.get("status") == "success":
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewAppError(f"Lecture impossible : {path}") from error
    return records


def image_path(record: dict[str, Any], field: str) -> Path | None:
    """Return an existing image path from a manifest field."""
    value: str = str(record.get(field, "")).strip()
    path = Path(value) if value else None
    return path if path is not None and path.is_file() else None


def persist_decision(
    decision_path: Path,
    decisions: dict[str, ReviewDecision],
    record: dict[str, Any],
    status: str,
    class_slug: str,
) -> None:
    """Save one accepted or rejected review decision."""
    record_id: str = str(record.get("record_id", "")).strip()
    if not record_id:
        raise ReviewAppError("Cette image ne possède pas de record_id.")
    if status not in {"accepted", "rejected"}:
        raise ReviewAppError(f"Statut de revue invalide : {status}")
    decisions[record_id] = ReviewDecision.create(
        record_id,
        "accepted" if status == "accepted" else "rejected",
        class_slug,
    )
    save_decisions(decision_path, decisions)


def persist_deleted_record(
    deletion_path: Path,
    deleted_record_ids: set[str],
    record: dict[str, Any],
) -> None:
    """Exclude one record from review without changing review decisions.

    Args:
        deletion_path: JSON deletion store path.
        deleted_record_ids: Current deleted record identifiers.
        record: Manifest record to exclude.

    Raises:
        ReviewAppError: If the record has no stable identifier.
    """
    record_id: str = str(record.get("record_id", "")).strip()
    if not record_id:
        raise ReviewAppError("Cette image ne possède pas de record_id.")
    deleted_record_ids.add(record_id)
    save_deleted_record_ids(deletion_path, deleted_record_ids)


def render_record(
    record: dict[str, Any], taxonomy: dict[str, TaxonomyClass]
) -> None:
    """Render source, crop, and attribution metadata for one record."""
    original_path: Path | None = image_path(record, "local_path")
    crop_path: Path | None = image_path(record, "crop_path")
    original_column, crop_column = st.columns(2)
    with original_column:
        st.caption("Photo source")
        if original_path is not None:
            st.image(str(original_path), use_column_width=True)
        else:
            st.warning("Image source absente")
    with crop_column:
        st.caption("Recadrage proposé")
        if crop_path is not None:
            st.image(str(crop_path), use_column_width=True)
        else:
            st.warning("Recadrage absent")

    proposed_slug: str = str(record.get("class_slug", ""))
    proposed = taxonomy.get(proposed_slug)
    st.write(
        "Classe proposée : "
        + (proposed.label if proposed is not None else proposed_slug)
    )
    confidence: object = record.get("detection_confidence")
    if isinstance(confidence, (int, float)):
        st.caption(f"Confiance de détection : {float(confidence):.1%}")
    source_page: str = str(record.get("source_page", "")).strip()
    metadata: str = " · ".join(
        value
        for value in (
            str(record.get("author", "")).strip(),
            str(record.get("license", "")).strip(),
        )
        if value
    )
    if metadata:
        st.caption(metadata)
    if source_page:
        st.link_button("Voir la source Wikimedia", source_page)


def main() -> None:
    """Run the interactive dataset review workflow."""
    st.set_page_config(page_title="Car Spotter — Dataset Review", layout="wide")
    st.title("Validation du dataset Car Spotter")
    st.caption(
        "Acceptez, corrigez ou rejetez chaque recadrage avant l'entraînement."
    )

    try:
        taxonomy = load_taxonomy(REVIEW_TAXONOMY_PATH)
        profile = load_profile(REVIEW_PROFILE_PATH, taxonomy)
        records: list[dict[str, Any]] = load_manifest(str(DEFAULT_MANIFEST))
        decisions: dict[str, ReviewDecision] = load_decisions(DEFAULT_DECISIONS)
        deleted_record_ids: set[str] = load_deleted_record_ids(DEFAULT_DELETED)
    except (DatasetConfigError, ReviewAppError, ReviewStoreError) as error:
        LOGGER.exception("Unable to initialize the review interface")
        st.error(str(error))
        st.stop()

    profile_records: list[dict[str, Any]] = [
        record
        for record in records
        if str(record.get("class_slug", "")) in profile.class_slugs
    ]
    class_filter: str = st.sidebar.selectbox(
        "Filtrer par classe", ("Toutes", *profile.class_slugs)
    )
    show_reviewed: bool = st.sidebar.checkbox("Afficher les images déjà revues")
    filtered_records: list[dict[str, Any]] = [
        record
        for record in profile_records
        if str(record.get("record_id", "")) not in deleted_record_ids
        and (class_filter == "Toutes" or record.get("class_slug") == class_filter)
    ]
    pending_records: list[dict[str, Any]] = [
        record
        for record in filtered_records
        if str(record.get("record_id", "")) not in decisions
    ]
    reviewable_profile_records: list[dict[str, Any]] = [
        record
        for record in profile_records
        if str(record.get("record_id", "")) not in deleted_record_ids
    ]
    reviewed_count: int = sum(
        str(record.get("record_id", "")) in decisions
        for record in reviewable_profile_records
    )
    total_reviewable: int = len(reviewable_profile_records)
    pending_count: int = total_reviewable - reviewed_count
    st.sidebar.metric(
        "Progression globale",
        f"{reviewed_count}/{total_reviewable}",
    )
    st.sidebar.metric("Images à revoir", pending_count)
    if deleted_record_ids:
        st.sidebar.caption(
            f"{len(deleted_record_ids)} image(s) supprimée(s) de la revue."
        )
    if total_reviewable:
        st.sidebar.progress(reviewed_count / total_reviewable)

    if not filtered_records:
        st.info("Aucune image ne correspond à ce filtre.")
        return
    current_record_id: str = str(
        st.session_state.get("review_record_id", "")
    )
    records_by_id: dict[str, dict[str, Any]] = {
        str(record.get("record_id", "")): record for record in filtered_records
    }
    current_record: dict[str, Any] | None = records_by_id.get(current_record_id)
    if current_record is None or (
        not show_reviewed
        and current_record_id in decisions
        and not st.session_state.get("review_explicit_navigation", False)
    ):
        current_record = pending_records[0] if pending_records else None
    if current_record is None:
        st.success("Aucune image ne reste à valider avec ce filtre.")
        return
    record: dict[str, Any] = current_record
    current_record_id = str(record.get("record_id", ""))
    st.session_state.review_record_id = current_record_id
    current_index: int = next(
        index
        for index, candidate in enumerate(filtered_records)
        if str(candidate.get("record_id", "")) == current_record_id
    )
    review_state: str = "déjà revue" if current_record_id in decisions else "à revoir"
    st.caption(
        f"Image {current_index + 1} sur {len(filtered_records)} · {review_state}"
    )
    render_record(record, taxonomy)

    display_names: dict[str, str] = {
        slug: taxonomy[slug].label for slug in profile.class_slugs
    }
    proposed_slug: str = str(record.get("class_slug", profile.class_slugs[0]))
    default_index: int = (
        profile.class_slugs.index(proposed_slug)
        if proposed_slug in profile.class_slugs
        else 0
    )
    selected_slug: str = st.selectbox(
        "Classe correcte",
        profile.class_slugs,
        index=default_index,
        format_func=lambda slug: display_names[slug],
    )
    previous_column, accept_column, reject_column, next_column = st.columns(4)
    if previous_column.button("← Précédente", use_container_width=True):
        previous_index: int = max(0, current_index - 1)
        st.session_state.review_record_id = str(
            filtered_records[previous_index].get("record_id", "")
        )
        st.session_state.review_explicit_navigation = True
        st.rerun()
    if accept_column.button(
        "Accepter", type="primary", use_container_width=True
    ):
        persist_decision(
            DEFAULT_DECISIONS, decisions, record, "accepted", selected_slug
        )
        remaining_records: list[dict[str, Any]] = [
            candidate
            for candidate in filtered_records[current_index + 1 :]
            if str(candidate.get("record_id", "")) not in decisions
        ]
        st.session_state.review_record_id = (
            str(remaining_records[0].get("record_id", ""))
            if remaining_records
            else ""
        )
        st.session_state.review_explicit_navigation = False
        st.rerun()
    if reject_column.button("Rejeter", use_container_width=True):
        persist_decision(
            DEFAULT_DECISIONS, decisions, record, "rejected", selected_slug
        )
        remaining_records = [
            candidate
            for candidate in filtered_records[current_index + 1 :]
            if str(candidate.get("record_id", "")) not in decisions
        ]
        st.session_state.review_record_id = (
            str(remaining_records[0].get("record_id", ""))
            if remaining_records
            else ""
        )
        st.session_state.review_explicit_navigation = False
        st.rerun()
    if next_column.button("Suivante →", use_container_width=True):
        following_records: list[dict[str, Any]] = (
            filtered_records[current_index + 1 :]
            if show_reviewed
            else [
                candidate
                for candidate in filtered_records[current_index + 1 :]
                if str(candidate.get("record_id", "")) not in decisions
            ]
        )
        if following_records:
            st.session_state.review_record_id = str(
                following_records[0].get("record_id", "")
            )
        st.session_state.review_explicit_navigation = show_reviewed
        st.rerun()
    if st.button(
        "Supprimer cette image de la revue",
        use_container_width=True,
        help="Masque cette image de la revue et de la préparation du dataset. "
        "Les décisions existantes ne sont pas modifiées.",
    ):
        persist_deleted_record(DEFAULT_DELETED, deleted_record_ids, record)
        following_records = [
            candidate
            for candidate in filtered_records[current_index + 1 :]
            if str(candidate.get("record_id", "")) not in deleted_record_ids
            and (show_reviewed or str(candidate.get("record_id", "")) not in decisions)
        ]
        st.session_state.review_record_id = (
            str(following_records[0].get("record_id", ""))
            if following_records
            else ""
        )
        st.session_state.review_explicit_navigation = False
        st.rerun()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
