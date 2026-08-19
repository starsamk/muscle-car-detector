"""Streamlit user interface for Car Spotter AI."""

from __future__ import annotations

import html
import logging
from typing import Final

import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

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

PAGE_TITLE: Final[str] = "Car Spotter — Classic car intelligence"
SUPPORTED_IMAGE_TYPES: Final[list[str]] = ["jpg", "jpeg", "png", "webp"]


def inject_styles() -> None:
    """Apply the visual system used by the Car Spotter interface."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #111827;
            --muted: #667085;
            --line: #e5e7eb;
            --surface: #ffffff;
            --canvas: #f7f8fa;
            --accent: #c7f36b;
            --navy: #132033;
        }

        [data-testid="stAppViewContainer"] { background: var(--canvas); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1260px; padding: 2.25rem 3rem 4rem; }

        .topbar {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 4.5rem;
        }

        .brand {
            align-items: center;
            color: var(--ink);
            display: flex;
            font-size: 0.83rem;
            font-weight: 800;
            gap: 0.65rem;
            letter-spacing: 0.13em;
        }

        .brand-mark {
            align-items: center;
            background: var(--navy);
            border-radius: 10px;
            color: var(--accent);
            display: inline-flex;
            font-size: 0.84rem;
            height: 31px;
            justify-content: center;
            letter-spacing: 0;
            width: 31px;
        }

        .topbar-status {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.68rem;
            font-weight: 700;
            gap: 0.45rem;
            letter-spacing: 0.12em;
        }

        .status-dot {
            background: #74b816;
            border-radius: 999px;
            box-shadow: 0 0 0 4px #e9f8cc;
            height: 7px;
            width: 7px;
        }

        .hero-eyebrow,
        .section-label {
            color: #7c8b5b;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2.7rem, 6vw, 5.5rem);
            font-weight: 800;
            letter-spacing: -0.075em;
            line-height: 0.97;
            margin: 0.95rem 0 1.4rem;
            max-width: 820px;
        }

        .hero-title span { color: #89936f; }

        .hero-copy {
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.65;
            margin-bottom: 3.2rem;
            max-width: 590px;
        }

        .upload-heading {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin: 0.4rem 0 0.35rem;
        }

        .upload-subheading,
        .muted-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.55;
        }

        div[data-testid="stFileUploader"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(17, 24, 39, 0.06);
            margin-top: 1.15rem;
            padding: 0.55rem;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: #fbfcf9;
            border: 1px dashed #b9c797;
            border-radius: 16px;
            min-height: 190px;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--muted); }
        div[data-testid="stFileUploaderDropzoneInstructions"] svg { color: #829b4b; }

        div[data-testid="stFileUploaderDropzone"] button {
            background: var(--navy);
            border: 0;
            border-radius: 8px;
            color: white;
            font-weight: 700;
        }

        .feature-panel {
            background: var(--navy);
            border-radius: 22px;
            color: white;
            min-height: 255px;
            padding: 1.75rem;
        }

        .feature-kicker {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }

        .feature-title {
            font-size: 1.35rem;
            font-weight: 750;
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin: 1rem 0 1.45rem;
            max-width: 210px;
        }

        .feature-row {
            align-items: center;
            border-top: 1px solid rgba(255, 255, 255, 0.13);
            display: flex;
            font-size: 0.78rem;
            gap: 0.75rem;
            padding: 0.8rem 0;
        }

        .feature-number {
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.1em;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 15px;
            min-height: 100px;
            padding: 1rem 1.15rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.67rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-top: 0.45rem;
        }

        .result-heading {
            align-items: end;
            display: flex;
            justify-content: space-between;
            margin: 3rem 0 1.2rem;
        }

        .result-title {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0.4rem 0 0;
        }

        .result-status {
            background: #eef8dc;
            border-radius: 999px;
            color: #55721e;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            padding: 0.5rem 0.75rem;
            text-transform: uppercase;
        }

        .image-frame {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            padding: 0.45rem;
        }

        .image-caption {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            margin: 0.65rem 0 0.25rem 0.25rem;
            text-transform: uppercase;
        }

        .prediction-row {
            align-items: center;
            border-bottom: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            padding: 0.95rem 0;
        }

        .prediction-name { color: var(--ink); font-size: 0.92rem; font-weight: 700; }
        .prediction-confidence { color: #637d28; font-size: 0.78rem; font-weight: 800; }

        div.stButton > button { border-radius: 10px; font-weight: 750; min-height: 46px; }
        div.stButton > button[kind="primary"] { background: var(--navy); border: 1px solid var(--navy); }
        div.stButton > button[kind="primary"]:hover { background: #263953; border-color: #263953; }

        [data-testid="stSidebar"] { background: #f1f3ed; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] .block-container { padding: 2rem 1.35rem; }

        .sidebar-note {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 12px;
            color: var(--muted);
            font-size: 0.75rem;
            line-height: 1.5;
            margin-top: 2rem;
            padding: 0.85rem;
        }

        @media (max-width: 740px) {
            .block-container { padding: 1.5rem 1.15rem 3rem; }
            .topbar { margin-bottom: 3rem; }
            .topbar-status { display: none; }
            .hero-title { font-size: 3.25rem; }
            .hero-copy { font-size: 0.95rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def render_brand_header() -> None:
    """Render the compact product header."""
    st.markdown(
        """
        <div class="topbar">
            <div class="brand"><span class="brand-mark">CS</span> CAR SPOTTER</div>
            <div class="topbar-status"><span class="status-dot"></span> LOCAL INFERENCE · V5</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render model status and usage notes in the sidebar."""
    with st.sidebar:
        st.markdown("### Car Spotter")
        st.caption("Classic car intelligence")
        st.divider()
        config: PhotoSpotterConfig = PhotoSpotterConfig.from_environment()
        st.markdown("**Modèle actif**")
        st.code(str(config.classifier_weights), language="text")
        st.markdown("**Périmètre**")
        st.caption(
            "Mustang Fastback, Hardtop, Convertible et autres véhicules classiques."
        )
        st.markdown(
            '<div class="sidebar-note">Les résultats sont indicatifs. Une photo nette,<br>'
            "avec une vue extérieure de la voiture, améliore la fiabilité.</div>",
            unsafe_allow_html=True,
        )


def render_empty_state() -> None:
    """Render the product explanation shown before an image is selected."""
    st.markdown(
        """
        <div class="feature-panel">
            <div class="feature-kicker">Built for car people</div>
            <div class="feature-title">De la silhouette à l'identité.</div>
            <div class="feature-row"><span class="feature-number">01</span> Détection automatique du véhicule</div>
            <div class="feature-row"><span class="feature-number">02</span> Classification de la carrosserie</div>
            <div class="feature-row"><span class="feature-number">03</span> Résultat annoté en quelques secondes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_section() -> UploadedFile | None:
    """Render the image uploader and return the selected image file."""
    st.markdown(
        '<div class="section-label">01 / Source image</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="upload-heading">Importez une photo de voiture</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-subheading">JPG, PNG ou WEBP · vue extérieure recommandée</div>',
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Choisissez une image",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=False,
        label_visibility="collapsed",
    )


def render_image_preview(image: Image.Image, caption: str) -> None:
    """Render an image inside the application's framed preview surface.

    Args:
        image: Image to display.
        caption: Small label displayed above the image.
    """
    st.markdown(
        f'<div class="image-caption">{html.escape(caption)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_summary(predictions: list[CarPrediction]) -> None:
    """Render structured prediction cards beneath the annotated image.

    Args:
        predictions: Structured predictions returned by the photo spotter.
    """
    if not predictions:
        st.warning("Aucune voiture n’a été détectée dans cette image.")
        return

    st.markdown(
        '<div class="section-label">03 / Identification</div>', unsafe_allow_html=True
    )
    for prediction in predictions:
        label: str = html.escape(prediction.display_label)
        confidence: str = f"{prediction.classification_confidence:.1%}"
        st.markdown(
            f'<div class="prediction-row"><span class="prediction-name">{label}</span>'
            f'<span class="prediction-confidence">{confidence}</span></div>',
            unsafe_allow_html=True,
        )


def render_results(
    original: Image.Image,
    annotated: Image.Image,
    predictions: list[CarPrediction],
) -> None:
    """Render the annotated result and its compact confidence summary.

    Args:
        original: Uploaded source image.
        annotated: Image returned by the inference service.
        predictions: Structured predictions returned by the photo spotter.
    """
    average_confidence: float = (
        sum(item.classification_confidence for item in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    st.markdown(
        '<div class="result-heading"><div><div class="section-label">02 / Analysis result</div>'
        '<div class="result-title">Votre identification</div></div>'
        '<div class="result-status">Analyse terminée</div></div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(2)
    with metric_columns[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Véhicules détectés</div>'
            f'<div class="metric-value">{len(predictions)}</div></div>',
            unsafe_allow_html=True,
        )
    with metric_columns[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Confiance moyenne</div>'
            f'<div class="metric-value">{average_confidence:.0%}</div></div>',
            unsafe_allow_html=True,
        )

    image_columns = st.columns(2)
    with image_columns[0]:
        render_image_preview(original, "Source")
    with image_columns[1]:
        render_image_preview(annotated, "Détection annotée")
    render_prediction_summary(predictions)


def main() -> None:
    """Render the Car Spotter application and handle user interactions."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="CS", layout="wide")
    inject_styles()

    try:
        render_sidebar()
    except ValueError as error:
        LOGGER.exception("Invalid detector configuration")
        st.error(f"Configuration invalide : {error}")
        return

    render_brand_header()
    st.markdown(
        '<div class="hero-eyebrow">Visual intelligence for classic cars</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="hero-title">Détectez la voiture.<br><span>Retrouvez son histoire.</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Importez une photo et laissez Car Spotter localiser le véhicule, '
        "identifier sa carrosserie et afficher le résultat directement sur l’image.</p>",
        unsafe_allow_html=True,
    )

    upload_columns = st.columns([1.55, 0.85], gap="large")
    with upload_columns[0]:
        uploaded_file: UploadedFile | None = render_upload_section()
    with upload_columns[1]:
        render_empty_state()

    if uploaded_file is None:
        return

    try:
        original_image: Image.Image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        LOGGER.exception("Unable to read uploaded file %s", uploaded_file.name)
        st.error(f"Impossible de lire l’image importée : {error}")
        return

    st.markdown(
        f'<div class="muted-copy" style="margin-top: 1rem;">Image prête · '
        f"<strong>{html.escape(uploaded_file.name)}</strong></div>",
        unsafe_allow_html=True,
    )
    should_predict: bool = st.button(
        "Analyser la photo",
        type="primary",
        use_container_width=True,
    )
    if not should_predict:
        return

    try:
        detector: PhotoSpotter = get_detector()
        with st.spinner("Analyse de l’image en cours…"):
            annotated_image, predictions = detector.predict_with_details(original_image)
        render_results(original_image, annotated_image, predictions)
    except PhotoModelError as error:
        LOGGER.exception("Detection failed for uploaded file %s", uploaded_file.name)
        st.error(f"La détection a échoué : {error}")
    except ValueError as error:
        LOGGER.exception("Invalid detector configuration during detection")
        st.error(f"Configuration invalide : {error}")


if __name__ == "__main__":
    main()
