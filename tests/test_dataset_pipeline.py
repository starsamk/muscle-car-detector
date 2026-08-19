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
from review_store import (
    ReviewDecision,
    load_deleted_record_ids,
    load_decisions,
    save_deleted_record_ids,
    save_decisions,
)
from scripts.crop_dataset import Detection, padded_box, select_detection
from scripts.prepare_classification_dataset import (
    ImageRecord,
    assign_group_splits,
    split_for_group,
)
from scripts.merge_review_dataset import (
    DatasetMergeError,
    merge_decisions,
    merge_manifest_records,
)
from scripts.collect_open_images_negatives import select_candidates
from scripts.download_wikimedia import load_manifest_exclusions
from scripts.evaluate_photo_spotter import (
    NO_DETECTION,
    FieldPrediction,
    summarize_predictions,
)
from train_classifier import build_training_options, parse_arguments
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

    def test_mustang_generalization_profile_keeps_body_styles_separate(self) -> None:
        """The V5 positive collection must target only the two Mustang body styles."""
        taxonomy = load_taxonomy(Path("config/taxonomy_vehicle_v3.json"))
        profile = load_profile(
            Path("config/profiles/vehicle_taxonomy_v3_mustang_generalization.json"),
            taxonomy,
        )

        self.assertEqual(
            profile.class_slugs,
            (
                "ford_mustang_fastback_classic",
                "ford_mustang_hardtop_classic",
            ),
        )


class TrainingConfigurationTests(unittest.TestCase):
    """Verify safe fine-tuning training settings."""

    def test_explicit_learning_rate_is_passed_to_ultralytics(self) -> None:
        """A fine-tuning learning rate must become the Ultralytics ``lr0`` option."""
        arguments = parse_arguments(["--learning-rate", "0.001"])

        options = build_training_options(
            arguments,
            Path("/tmp/classification"),
            "mps",
        )

        self.assertEqual(options["lr0"], 0.001)
        self.assertEqual(options["optimizer"], "AdamW")

    def test_non_positive_learning_rate_is_rejected(self) -> None:
        """A zero learning rate must stop before starting a training run."""
        arguments = parse_arguments(["--learning-rate", "0"])

        with self.assertRaises(ValueError):
            build_training_options(arguments, Path("/tmp/classification"), "mps")

    def test_freeze_and_patience_are_forwarded_to_ultralytics(self) -> None:
        """The final V5 recipe must preserve its anti-forgetting controls."""
        arguments = parse_arguments(
            ["--freeze", "9", "--patience", "10", "--learning-rate", "0.0003"]
        )

        options = build_training_options(
            arguments,
            Path("/tmp/classification"),
            "mps",
        )

        self.assertEqual(options["freeze"], 9)
        self.assertEqual(options["patience"], 10)
        self.assertEqual(options["lr0"], 0.0003)

    def test_invalid_freeze_is_rejected(self) -> None:
        """A negative layer count must stop before starting a training run."""
        arguments = parse_arguments(["--freeze", "-1"])

        with self.assertRaises(ValueError):
            build_training_options(arguments, Path("/tmp/classification"), "mps")


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

    def test_deleted_records_round_trip(self) -> None:
        """Deleted records persist independently from review decisions."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deleted.json"
            save_deleted_record_ids(path, {"record-2", "record-1"})

            self.assertEqual(
                load_deleted_record_ids(path), {"record-1", "record-2"}
            )

    def test_wikimedia_exclusions_load_source_identities(self) -> None:
        """Independent collection excludes known titles, pages, and hashes."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_title": "File:known.jpg",
                        "source_page": "https://example.test/known",
                        "sha256": "known-sha",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            titles, pages, hashes = load_manifest_exclusions([path])

            self.assertEqual(titles, {"File:known.jpg"})
            self.assertEqual(pages, {"https://example.test/known"})
            self.assertEqual(hashes, {"known-sha"})

    def test_field_summary_measures_negative_false_positives(self) -> None:
        """A target prediction on other_car increments the false-positive rate."""
        predictions = [
            FieldPrediction(
                expected_class="other_car",
                predicted_class="ford_mustang_fastback_classic",
                is_correct=False,
                is_false_positive=True,
                detection_count=1,
                detection_confidence=0.9,
                classification_confidence=0.8,
                local_path="negative.jpg",
                source_category="",
                source_page="",
                source_title="",
            ),
            FieldPrediction(
                expected_class="other_car",
                predicted_class=NO_DETECTION,
                is_correct=False,
                is_false_positive=False,
                detection_count=0,
                detection_confidence=0.0,
                classification_confidence=0.0,
                local_path="missed.jpg",
                source_category="",
                source_page="",
                source_title="",
            ),
        ]

        summary = summarize_predictions(predictions)

        self.assertEqual(summary["false_positive_count"], 1)
        self.assertEqual(summary["false_positive_rate"], 0.5)
        self.assertEqual(
            summary["negative_by_category"][""]["false_positive_count"], 1
        )
        self.assertEqual(summary["threshold_sweep"][0]["threshold"], 0.4)

    def test_split_assignment_is_deterministic(self) -> None:
        """The same source group must always remain in the same split."""
        first = split_for_group("author:example", 0.70, 0.15)
        second = split_for_group("author:example", 0.70, 0.15)

        self.assertEqual(first, second)

    def test_stratified_split_covers_small_classes(self) -> None:
        """Small classes with enough groups must reach validation and test."""
        records: list[ImageRecord] = []
        for class_slug in ("camaro", "charger"):
            for group_index in range(3):
                records.append(
                    ImageRecord(
                        class_slug=class_slug,
                        source_path=Path(f"{class_slug}-{group_index}.jpg"),
                        source_title="",
                        author=f"author-{class_slug}-{group_index}",
                        license_name="",
                        license_url="",
                        source_page="",
                        sha256=f"{class_slug}-{group_index}",
                        perceptual_hash=f"{class_slug}-{group_index}",
                        record_id=f"{class_slug}-{group_index}",
                        review_status="accepted",
                    )
                )

        assignments = assign_group_splits(records, 0.70, 0.15)
        class_splits = {
            class_slug: {
                assignments[record.group_key]
                for record in records
                if record.class_slug == class_slug
            }
            for class_slug in ("camaro", "charger")
        }

        self.assertEqual(class_splits["camaro"], {"train", "val", "test"})
        self.assertEqual(class_splits["charger"], {"train", "val", "test"})

    def test_manifest_merge_deduplicates_by_sha256(self) -> None:
        """The first source record wins when two manifests contain one image."""
        merged = merge_manifest_records(
            [
                [{"record_id": "first", "sha256": "same"}],
                [{"record_id": "second", "sha256": "same"}],
            ]
        )

        self.assertEqual([record["record_id"] for record in merged], ["first"])

    def test_manifest_merge_rejects_record_id_collision(self) -> None:
        """A reused record ID must fail instead of silently losing metadata."""
        with self.assertRaises(DatasetMergeError):
            merge_manifest_records(
                [
                    [{"record_id": "same", "sha256": "first"}],
                    [{"record_id": "same", "sha256": "second"}],
                ]
            )

    def test_manifest_merge_allows_idempotent_crop_rerun(self) -> None:
        """A crop rerun may change only the derived crop path."""
        merged = merge_manifest_records(
            [
                [{"record_id": "same", "sha256": "same", "crop_path": "old.jpg"}],
                [{"record_id": "same", "sha256": "same", "crop_path": "new.jpg"}],
            ]
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["crop_path"], "old.jpg")

    def test_decision_merge_rejects_conflicts(self) -> None:
        """Conflicting human labels must never be resolved implicitly."""
        first = ReviewDecision.create("record-1", "accepted", "other_car")
        second = ReviewDecision.create(
            "record-1", "accepted", "ford_mustang_fastback_classic"
        )

        with self.assertRaises(DatasetMergeError):
            merge_decisions([{first.record_id: first}, {second.record_id: second}])

    def test_open_images_candidates_exclude_inside_and_group_boxes(self) -> None:
        """Negative candidates must represent visible individual vehicles."""
        rows = [
            {
                "ImageID": "outside",
                "LabelName": "/m/car",
                "XMin": "0.1",
                "XMax": "0.9",
                "YMin": "0.1",
                "YMax": "0.9",
                "IsInside": "0",
                "IsGroupOf": "0",
            },
            {
                "ImageID": "inside",
                "LabelName": "/m/car",
                "XMin": "0.1",
                "XMax": "0.9",
                "YMin": "0.1",
                "YMax": "0.9",
                "IsInside": "1",
                "IsGroupOf": "0",
            },
            {
                "ImageID": "truncated",
                "LabelName": "/m/car",
                "XMin": "0.1",
                "XMax": "0.9",
                "YMin": "0.1",
                "YMax": "0.9",
                "IsInside": "0",
                "IsGroupOf": "0",
                "IsTruncated": "1",
            },
        ]

        candidates = select_candidates(
            rows,
            {"/m/car": "Car"},
            limit_per_class=5,
            max_images=5,
            minimum_box_area=0.05,
        )

        self.assertEqual(set(candidates), {"outside"})

    def test_open_images_candidates_exclude_existing_source_ids(self) -> None:
        """Existing Open Images sources must not re-enter the review queue."""
        rows = [
            {
                "ImageID": "already-reviewed",
                "LabelName": "/m/car",
                "XMin": "0.1",
                "XMax": "0.9",
                "YMin": "0.1",
                "YMax": "0.9",
                "IsInside": "0",
                "IsGroupOf": "0",
            },
            {
                "ImageID": "new-candidate",
                "LabelName": "/m/car",
                "XMin": "0.1",
                "XMax": "0.9",
                "YMin": "0.1",
                "YMax": "0.9",
                "IsInside": "0",
                "IsGroupOf": "0",
            },
        ]

        candidates = select_candidates(
            rows,
            {"/m/car": "Car"},
            limit_per_class=5,
            max_images=5,
            minimum_box_area=0.05,
            excluded_image_ids=frozenset({"already-reviewed"}),
        )

        self.assertEqual(set(candidates), {"new-candidate"})

if __name__ == "__main__":
    unittest.main()
