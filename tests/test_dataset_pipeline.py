"""Unit tests for deterministic dataset-pipeline components."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_config import DatasetConfigError, load_profile, load_taxonomy
from review_store import ReviewDecision, load_decisions, save_decisions
from scripts.crop_dataset import Detection, padded_box, select_detection
from scripts.prepare_classification_dataset import split_for_group


class DatasetConfigurationTests(unittest.TestCase):
    """Verify taxonomy/profile consistency."""

    def test_mustang_profile_references_existing_classes(self) -> None:
        """The committed MVP profile must be a valid taxonomy subset."""
        taxonomy = load_taxonomy(Path("config/taxonomy.json"))
        profile = load_profile(Path("config/profiles/mustang_mvp.json"), taxonomy)

        self.assertEqual(len(profile.class_slugs), 5)
        self.assertIn("ford_mustang_fastback_1967_1968", profile.class_slugs)

    def test_profile_rejects_an_unknown_class(self) -> None:
        """A typo in an experiment profile must fail immediately."""
        taxonomy = load_taxonomy(Path("config/taxonomy.json"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(
                json.dumps({"name": "invalid", "class_slugs": ["missing"]}),
                encoding="utf-8",
            )
            with self.assertRaises(DatasetConfigError):
                load_profile(path, taxonomy)


class CroppingTests(unittest.TestCase):
    """Verify principal-car selection and crop geometry."""

    def test_largest_detection_is_selected(self) -> None:
        """A clearly dominant car should be selected."""
        large = Detection(10, 10, 210, 110, 0.9)
        small = Detection(0, 0, 50, 50, 0.8)

        selected, status = select_detection([large, small], 0.65)

        self.assertEqual(status, "success")
        self.assertEqual(selected, large)

    def test_similarly_sized_cars_are_ambiguous(self) -> None:
        """Comparable cars should not produce a potentially wrong label."""
        first = Detection(0, 0, 100, 100, 0.9)
        second = Detection(0, 0, 90, 90, 0.8)

        selected, status = select_detection([first, second], 0.65)

        self.assertIsNone(selected)
        self.assertEqual(status, "ambiguous")

    def test_padding_is_clamped_to_image_bounds(self) -> None:
        """Padding must never create coordinates outside the source image."""
        detection = Detection(5, 10, 95, 90, 0.9)

        self.assertEqual(padded_box(detection, 100, 100, 0.1), (0, 2, 100, 98))


class ReviewStoreTests(unittest.TestCase):
    """Verify durable review decisions and stable data splits."""

    def test_decisions_round_trip(self) -> None:
        """A saved correction must be restored without information loss."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.json"
            decision = ReviewDecision.create("record-1", "accepted", "corrected")
            save_decisions(path, {decision.record_id: decision})

            self.assertEqual(load_decisions(path), {"record-1": decision})

    def test_split_assignment_is_deterministic(self) -> None:
        """The same source group must always remain in the same split."""
        first = split_for_group("author:example", 0.70, 0.15)
        second = split_for_group("author:example", 0.70, 0.15)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
