"""Train a YOLOv8 image classifier for fine-grained classic car recognition."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse classifier training arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fine-tune a YOLOv8 classifier on the prepared dataset."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("datasets/classification")
    )
    parser.add_argument("--model", default="yolov8s-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="classic-car-classifier-v1")
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    """Resolve the best available PyTorch training device.

    Args:
        requested_device: Explicit Ultralytics device or ``"auto"``.

    Returns:
        ``"mps"``, ``"0"``, ``"cpu"`` or the explicit requested value.
    """
    if requested_device != "auto":
        return requested_device
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to resolve the device.") from error
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    """Validate inputs and launch Ultralytics classifier fine-tuning."""
    arguments: argparse.Namespace = parse_arguments()
    data_directory: Path = arguments.data.resolve()
    if not data_directory.is_dir():
        raise FileNotFoundError(
            f"Prepared classification dataset not found: '{data_directory}'."
        )
    required_splits: tuple[str, ...] = ("train", "val", "test")
    missing_splits: list[str] = [
        split for split in required_splits if not (data_directory / split).is_dir()
    ]
    if missing_splits:
        raise ValueError(f"Missing dataset splits: {', '.join(missing_splits)}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before training.") from error

    device: str = resolve_device(arguments.device)
    LOGGER.info(
        "Training %s on %s with device=%s",
        arguments.model,
        data_directory,
        device,
    )
    model: object = YOLO(arguments.model)
    model.train(  # type: ignore[attr-defined]
        data=str(data_directory),
        epochs=arguments.epochs,
        imgsz=arguments.image_size,
        batch=arguments.batch_size,
        device=device,
        workers=arguments.workers,
        project="runs/classify",
        name=arguments.name,
        patience=20,
        seed=42,
        deterministic=True,
        plots=True,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError):
        LOGGER.exception("Classifier training failed")
        raise SystemExit(1)
