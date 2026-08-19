"""Evaluate the end-to-end photo spotter on an independent image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

from PIL import Image, UnidentifiedImageError

from photo_model import CarPrediction, PhotoModelError, PhotoSpotter, PhotoSpotterConfig

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
NO_DETECTION: Final[str] = "__no_detection__"
EVALUATED_THRESHOLDS: Final[tuple[float, ...]] = (
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
)


class FieldEvaluationError(RuntimeError):
    """Raised when an independent field evaluation cannot be completed."""


@dataclass(frozen=True)
class FieldPrediction:
    """Prediction and source metadata for one evaluation photo."""

    expected_class: str
    predicted_class: str
    is_correct: bool
    is_false_positive: bool
    detection_count: int
    detection_confidence: float
    classification_confidence: float
    local_path: str
    source_category: str
    source_page: str
    source_title: str


def parse_arguments() -> argparse.Namespace:
    """Parse field-evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Car Spotter on an independent labeled manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--weights", type=Path, default=Path("weights/classifier-best.pt")
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=Path("config/taxonomy_vehicle_v3.json")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    parser.add_argument("--classification-confidence", type=float, default=0.50)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/model_v3_field_predictions.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/model_v3_field_summary.json"),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load labeled image records from a JSONL manifest.

    Args:
        path: Independent evaluation manifest.

    Returns:
        Parsed manifest records.

    Raises:
        FieldEvaluationError: If the manifest is missing or malformed.
    """
    if not path.is_file():
        raise FieldEvaluationError(f"Evaluation manifest not found: '{path}'.")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise FieldEvaluationError(
                    f"Manifest line {line_number} is not an object."
                )
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise FieldEvaluationError(f"Unable to read manifest '{path}'.") from error
    if not records:
        raise FieldEvaluationError("Evaluation manifest contains no records.")
    return records


def prediction_area(prediction: CarPrediction) -> int:
    """Return the bounding-box area for primary-car selection."""
    left, top, right, bottom = prediction.bounding_box
    return max(0, right - left) * max(0, bottom - top)


def select_primary_prediction(
    predictions: Sequence[CarPrediction],
) -> CarPrediction | None:
    """Select the largest detected vehicle as the photo subject."""
    return max(predictions, key=prediction_area) if predictions else None


def build_field_prediction(
    record: dict[str, Any], predictions: Sequence[CarPrediction]
) -> FieldPrediction:
    """Build one auditable evaluation row from model predictions."""
    expected_class: str = str(record.get("class_slug", "")).strip()
    primary: CarPrediction | None = select_primary_prediction(predictions)
    predicted_class: str = primary.class_slug if primary else NO_DETECTION
    is_false_positive: bool = (
        expected_class == "other_car"
        and predicted_class not in {"other_car", NO_DETECTION}
    )
    return FieldPrediction(
        expected_class=expected_class,
        predicted_class=predicted_class,
        is_correct=predicted_class == expected_class,
        is_false_positive=is_false_positive,
        detection_count=len(predictions),
        detection_confidence=(primary.detection_confidence if primary else 0.0),
        classification_confidence=(
            primary.classification_confidence if primary else 0.0
        ),
        local_path=str(record.get("local_path", "")),
        source_category=str(record.get("source_category", "")),
        source_page=str(record.get("source_page", "")),
        source_title=str(record.get("source_title", "")),
    )


def summarize_predictions(
    predictions: Iterable[FieldPrediction],
) -> dict[str, Any]:
    """Calculate overall, per-class, and negative false-positive metrics."""
    prediction_list: list[FieldPrediction] = list(predictions)
    if not prediction_list:
        raise FieldEvaluationError("No field predictions were produced.")
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()
    per_class_no_detection: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction in prediction_list:
        per_class_total[prediction.expected_class] += 1
        per_class_correct[prediction.expected_class] += int(prediction.is_correct)
        per_class_no_detection[prediction.expected_class] += int(
            prediction.predicted_class == NO_DETECTION
        )
        confusion[prediction.expected_class][prediction.predicted_class] += 1

    negative_predictions: list[FieldPrediction] = [
        prediction
        for prediction in prediction_list
        if prediction.expected_class == "other_car"
    ]
    false_positive_count: int = sum(
        prediction.is_false_positive for prediction in negative_predictions
    )
    negative_by_category: dict[str, dict[str, Any]] = {}
    negative_categories: dict[str, list[FieldPrediction]] = defaultdict(list)
    for prediction in negative_predictions:
        negative_categories[prediction.source_category].append(prediction)
    for category, category_predictions in sorted(negative_categories.items()):
        category_false_positives: int = sum(
            prediction.is_false_positive for prediction in category_predictions
        )
        negative_by_category[category] = {
            "total": len(category_predictions),
            "false_positive_count": category_false_positives,
            "false_positive_rate": (
                category_false_positives / len(category_predictions)
            ),
            "predictions": dict(
                sorted(
                    Counter(
                        prediction.predicted_class
                        for prediction in category_predictions
                    ).items()
                )
            ),
        }

    threshold_sweep: list[dict[str, float]] = []
    for threshold in EVALUATED_THRESHOLDS:
        threshold_predictions: list[tuple[str, str]] = []
        for prediction in prediction_list:
            predicted_class: str = prediction.predicted_class
            if (
                predicted_class not in {"other_car", NO_DETECTION}
                and prediction.classification_confidence < threshold
            ):
                predicted_class = "other_car"
            threshold_predictions.append(
                (prediction.expected_class, predicted_class)
            )
        target_predictions: list[tuple[str, str]] = [
            result
            for result in threshold_predictions
            if result[0] != "other_car"
        ]
        negative_threshold_predictions: list[tuple[str, str]] = [
            result
            for result in threshold_predictions
            if result[0] == "other_car"
        ]
        threshold_sweep.append(
            {
                "threshold": threshold,
                "accuracy": sum(
                    expected == predicted
                    for expected, predicted in threshold_predictions
                )
                / len(threshold_predictions),
                "target_accuracy": sum(
                    expected == predicted
                    for expected, predicted in target_predictions
                )
                / len(target_predictions)
                if target_predictions
                else 0.0,
                "false_positive_rate": sum(
                    predicted not in {"other_car", NO_DETECTION}
                    for _, predicted in negative_threshold_predictions
                )
                / len(negative_threshold_predictions)
                if negative_threshold_predictions
                else 0.0,
            }
        )
    class_metrics: dict[str, dict[str, Any]] = {}
    for class_slug in sorted(per_class_total):
        total: int = per_class_total[class_slug]
        class_metrics[class_slug] = {
            "total": total,
            "correct": per_class_correct[class_slug],
            "accuracy": per_class_correct[class_slug] / total,
            "no_detection": per_class_no_detection[class_slug],
        }
    return {
        "total": len(prediction_list),
        "correct": sum(prediction.is_correct for prediction in prediction_list),
        "accuracy": sum(
            prediction.is_correct for prediction in prediction_list
        )
        / len(prediction_list),
        "no_detection": sum(
            prediction.predicted_class == NO_DETECTION
            for prediction in prediction_list
        ),
        "negative_total": len(negative_predictions),
        "false_positive_count": false_positive_count,
        "false_positive_rate": (
            false_positive_count / len(negative_predictions)
            if negative_predictions
            else 0.0
        ),
        "negative_by_category": negative_by_category,
        "threshold_sweep": threshold_sweep,
        "per_class": class_metrics,
        "confusion": {
            expected: dict(sorted(predicted.items()))
            for expected, predicted in sorted(confusion.items())
        },
    }


def write_results(
    predictions: Sequence[FieldPrediction],
    summary: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> None:
    """Write detailed predictions and aggregate metrics."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [asdict(prediction) for prediction in predictions]
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer: csv.DictWriter[str] = csv.DictWriter(
            output_file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the independent end-to-end evaluation."""
    arguments: argparse.Namespace = parse_arguments()
    records: list[dict[str, Any]] = load_manifest(arguments.manifest)
    config = PhotoSpotterConfig(
        classifier_weights=arguments.weights,
        taxonomy_path=arguments.taxonomy,
        detection_confidence=arguments.detection_confidence,
        classification_confidence=arguments.classification_confidence,
        device=arguments.device,
    )
    spotter = PhotoSpotter(config)
    predictions: list[FieldPrediction] = []
    for index, record in enumerate(records, start=1):
        local_path: Path = Path(str(record.get("local_path", "")))
        try:
            with Image.open(local_path) as image:
                _, car_predictions = spotter.predict_with_details(
                    image.convert("RGB")
                )
        except (OSError, UnidentifiedImageError, PhotoModelError) as error:
            LOGGER.warning("Unable to evaluate %s: %s", local_path, error)
            car_predictions = []
        predictions.append(build_field_prediction(record, car_predictions))
        LOGGER.info("Evaluated %d/%d: %s", index, len(records), local_path)

    summary: dict[str, Any] = summarize_predictions(predictions)
    write_results(
        predictions,
        summary,
        arguments.output_csv,
        arguments.output_json,
    )
    LOGGER.info("Field accuracy: %.2f%%", 100.0 * float(summary["accuracy"]))
    LOGGER.info(
        "Negative false-positive rate: %.2f%%",
        100.0 * float(summary["false_positive_rate"]),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except FieldEvaluationError:
        LOGGER.exception("Field evaluation failed")
        raise SystemExit(1)
