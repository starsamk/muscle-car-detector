"""Unit tests for deterministic dataset-pipeline components."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_config import (
    DatasetConfigError,
    load_profile,
    load_taxonomy,
)
from review_store import ReviewDecision, load_decisions, save_decisions
from scripts.crop_dataset import Detection, padded_box, select_detection
from scripts.prepare_classification_dataset import split_for_group
from taxonomy_migration import (
    TaxonomyMigrationError,
    load_class_mapping,
    migrate_manifest_records,
    migrate_review_decisions,
)


class DatasetConfigurationTests(unittest.TestCase):
    """Verify taxonomy/profile consistency."""

    def test_mustang_profile_references_existing_classes(self) -> None:
        """The committed MVP profile must be a valid taxonomy subset."""
        taxonomy = load_taxonomy(Path("config/taxonomy.json"))
        profile = load_profile(Path("config/profiles/mustang_mvp.json"), taxonomy)

        self.assertEqual(len(profile.class_slugs), 5)
        self.assertIn("ford_mustang_fastback_1967_1968", profile.class_slugs)
        self.assertEqual(
            taxonomy["ford_mustang_fastback_1967_1968"].label,
            "Ford Mustang Fastback — 1967–1968",
        )
        self.assertEqual(taxonomy["other_car"].label, "Other car")

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

    def test_body_style_v2_has_unambiguous_labels(self) -> None:
        """V2 labels must not expose an unsupported production-year prediction."""
        taxonomy = load_taxonomy(Path("config/taxonomy_mustang_body_style_v2.json"))
        profile = load_profile(
            Path("config/profiles/mustang_body_style_v2.json"), taxonomy
        )

        self.assertEqual(len(profile.class_slugs), 4)
        self.assertEqual(
            taxonomy["ford_mustang_fastback_classic"].label,
            "Ford Mustang Fastback (classic)",
        )

    def test_vehicle_v3_declares_target_and_bootstrap_profiles(self) -> None:
        """V3 keeps future classes explicit while bootstrap uses only available data."""
        taxonomy = load_taxonomy(Path("config/taxonomy_vehicle_v3.json"))
        full_profile = load_profile(
            Path("config/profiles/vehicle_taxonomy_v3.json"), taxonomy
        )
        bootstrap_profile = load_profile(
            Path("config/profiles/vehicle_taxonomy_v3_bootstrap.json"), taxonomy
        )

        self.assertEqual(len(full_profile.class_slugs), 7)
        self.assertEqual(
            set(bootstrap_profile.class_slugs),
            {
                "ford_mustang_fastback_classic",
                "ford_mustang_hardtop_classic",
                "other_car",
            },
        )
        self.assertEqual(
            taxonomy["chevrolet_camaro_classic"].label,
            "Chevrolet Camaro (classic)",
        )


class TaxonomyMigrationTests(unittest.TestCase):
    """Verify that legacy reviewed data migrates without source mutations."""

    def setUp(self) -> None:
        """Load the committed target taxonomy and mapping."""
        self.taxonomy = load_taxonomy(
            Path("config/taxonomy_mustang_body_style_v2.json")
        )
        self.target_slugs = set(self.taxonomy)
        self.mapping = load_class_mapping(
            Path("config/mappings/mustang_mvp_to_body_style_v2.json"),
            self.target_slugs,
        )

    def test_manifest_migration_preserves_legacy_class(self) -> None:
        """Changed manifest labels must retain their prior class for traceability."""
        records = [
            {
                "record_id": "record-1",
                "class_slug": "ford_mustang_fastback_1967_1968",
            }
        ]

        migrated = migrate_manifest_records(
            records, self.mapping, self.target_slugs
        )

        self.assertEqual(
            records[0]["class_slug"], "ford_mustang_fastback_1967_1968"
        )
        self.assertEqual(
            migrated[0]["class_slug"], "ford_mustang_fastback_classic"
        )
        self.assertEqual(
            migrated[0]["legacy_class_slug"],
            "ford_mustang_fastback_1967_1968",
        )

    def test_decision_migration_keeps_review_metadata(self) -> None:
        """Migrated decisions must retain the original status and timestamp."""
        decision = ReviewDecision(
            record_id="record-1",
            status="accepted",
            class_slug="ford_mustang_hardtop_1964_1966",
            reviewed_at="2026-08-19T00:00:00+00:00",
        )

        migrated = migrate_review_decisions(
            {decision.record_id: decision}, self.mapping, self.target_slugs
        )

        self.assertEqual(migrated["record-1"].status, "accepted")
        self.assertEqual(
            migrated["record-1"].class_slug,
            "ford_mustang_hardtop_classic",
        )
        self.assertEqual(
            migrated["record-1"].reviewed_at, decision.reviewed_at
        )

    def test_unknown_legacy_class_fails_migration(self) -> None:
        """Unmapped labels must stop migration instead of silently changing data."""
        with self.assertRaises(TaxonomyMigrationError):
            migrate_manifest_records(
                [{"class_slug": "unknown_legacy_class"}],
                self.mapping,
                self.target_slugs,
            )


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
