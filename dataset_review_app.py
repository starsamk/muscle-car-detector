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
    load_decisions,
    save_decisions,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
DEFAULT_MANIFEST: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_MANIFEST", "datasets/cropped/manifest.jsonl")
)
DEFAULT_DECISIONS: Final[Path] = Path(
    os.getenv("CAR_SPOTTER_REVIEW_DECISIONS", "datasets/review/decisions.json")
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
        + (proposed.display_name if proposed is not None else proposed_slug)
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
    st.title("Validation du dataset Mustang MVP")
    st.caption(
        "Acceptez, corrigez ou rejetez chaque recadrage avant l'entraînement."
    )

    try:
        taxonomy = load_taxonomy(DEFAULT_TAXONOMY_PATH)
        profile = load_profile(DEFAULT_MVP_PROFILE_PATH, taxonomy)
        records: list[dict[str, Any]] = load_manifest(str(DEFAULT_MANIFEST))
        decisions: dict[str, ReviewDecision] = load_decisions(DEFAULT_DECISIONS)
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
    visible_records: list[dict[str, Any]] = [
        record
        for record in profile_records
        if (class_filter == "Toutes" or record.get("class_slug") == class_filter)
        and (
            show_reviewed
            or str(record.get("record_id", "")) not in decisions
        )
    ]
    reviewed_count: int = sum(
        str(record.get("record_id", "")) in decisions for record in profile_records
    )
    st.sidebar.metric("Progression", f"{reviewed_count}/{len(profile_records)}")
    if profile_records:
        st.sidebar.progress(reviewed_count / len(profile_records))

    if not visible_records:
        st.success("Aucune image ne reste à valider avec ce filtre.")
        return
    current_index: int = min(
        int(st.session_state.get("review_index", 0)), len(visible_records) - 1
    )
    record: dict[str, Any] = visible_records[current_index]
    st.caption(f"Image {current_index + 1} sur {len(visible_records)}")
    render_record(record, taxonomy)

    display_names: dict[str, str] = {
        slug: taxonomy[slug].display_name for slug in profile.class_slugs
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
        st.session_state.review_index = max(0, current_index - 1)
        st.rerun()
    if accept_column.button(
        "Accepter", type="primary", use_container_width=True
    ):
        persist_decision(
            DEFAULT_DECISIONS, decisions, record, "accepted", selected_slug
        )
        st.session_state.review_index = current_index
        st.rerun()
    if reject_column.button("Rejeter", use_container_width=True):
        persist_decision(
            DEFAULT_DECISIONS, decisions, record, "rejected", selected_slug
        )
        st.session_state.review_index = current_index
        st.rerun()
    if next_column.button("Suivante →", use_container_width=True):
        st.session_state.review_index = min(
            len(visible_records) - 1, current_index + 1
        )
        st.rerun()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()
