"""Streamlit user interface for Car Spotter AI."""

from __future__ import annotations

import logging
from typing import Final

import streamlit as st
from PIL import Image

from model import (
    CarPrediction,
    PhotoModelError,
    PhotoSpotter,
    PhotoSpotterConfig,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)

PAGE_TITLE: Final[str] = "Car Spotter AI"
SUPPORTED_IMAGE_TYPES: Final[list[str]] = ["jpg", "jpeg", "png", "webp"]


@st.cache_resource(show_spinner=False)
def get_detector() -> PhotoSpotter:
    """Create and cache the two-stage photo spotter.

    Returns:
        A cached :class:`PhotoSpotter` instance.

    Raises:
        ValueError: If environment-based configuration is invalid.
    """
    config: PhotoSpotterConfig = PhotoSpotterConfig.from_environment()
    return PhotoSpotter(config)


def render_sidebar() -> None:
    """Render configuration and usage information in the sidebar."""
    with st.sidebar:
        st.header("Configuration")
        config: PhotoSpotterConfig = PhotoSpotterConfig.from_environment()
        st.caption(f"Détecteur : `{config.detector_weights}`")
        st.caption(f"Classifieur : `{config.classifier_weights}`")
        st.markdown(
            "**Utilisation**\n"
            "1. Importez une image.\n"
            "2. Cliquez sur **Détecter les véhicules**.\n"
            "3. Comparez l’image originale et le résultat annoté."
        )


def render_results(original: Image.Image, annotated: Image.Image) -> None:
    """Display the source image and the annotated prediction side by side.

    Args:
        original: Image uploaded by the user.
        annotated: Image returned by the inference service.
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Image originale")
        st.image(original, use_column_width=True)
    with right_column:
        st.subheader("Détection")
        st.image(annotated, use_column_width=True)


def render_prediction_summary(predictions: list[CarPrediction]) -> None:
    """Render a concise table of recognized vehicles.

    Args:
        predictions: Structured predictions returned by the photo spotter.
    """
    if not predictions:
        st.warning("Aucune voiture n’a été détectée dans cette image.")
        return
    rows: list[dict[str, str]] = [
        {
            "Modèle et période": prediction.display_label,
            "Confiance": f"{prediction.classification_confidence:.1%}",
        }
        for prediction in predictions
    ]
    st.subheader("Véhicules reconnus")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the Car Spotter application and handle user interactions."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🚗", layout="wide")
    st.title("🚗 Car Spotter AI")
    st.write(
        "Importez une image pour détecter les véhicules avec votre modèle YOLOv8."
    )

    try:
        render_sidebar()
    except ValueError as error:
        LOGGER.exception("Invalid detector configuration")
        st.error(f"Configuration invalide : {error}")
        return

    uploaded_file = st.file_uploader(
        "Choisissez une image",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Ajoutez une image pour commencer.")
        return

    try:
        original_image: Image.Image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        LOGGER.exception("Unable to read uploaded file %s", uploaded_file.name)
        st.error(f"Impossible de lire l’image importée : {error}")
        return

    st.image(original_image, caption="Aperçu", use_column_width=True)

    should_predict: bool = st.button(
        "Détecter les véhicules", type="primary", use_container_width=True
    )
    if not should_predict:
        return

    try:
        detector: PhotoSpotter = get_detector()
        with st.spinner("Analyse de l’image en cours…"):
            annotated_image, predictions = detector.predict_with_details(
                original_image
            )
        render_results(original_image, annotated_image)
        render_prediction_summary(predictions)
    except PhotoModelError as error:
        LOGGER.exception("Detection failed for uploaded file %s", uploaded_file.name)
        st.error(f"La détection a échoué : {error}")
    except ValueError as error:
        LOGGER.exception("Invalid detector configuration during detection")
        st.error(f"Configuration invalide : {error}")


if __name__ == "__main__":
    main()
