"""Persistent review decisions shared by the review UI and dataset builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ReviewStatus = Literal["accepted", "rejected"]


class ReviewStoreError(RuntimeError):
    """Raised when review decisions cannot be read or persisted."""


@dataclass(frozen=True)
class ReviewDecision:
    """Human decision for one automatically cropped source image."""

    record_id: str
    status: ReviewStatus
    class_slug: str
    reviewed_at: str

    @classmethod
    def create(
        cls,
        record_id: str,
        status: ReviewStatus,
        class_slug: str,
    ) -> "ReviewDecision":
        """Create a timestamped review decision."""
        return cls(
            record_id=record_id,
            status=status,
            class_slug=class_slug,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )


def load_decisions(path: Path) -> dict[str, ReviewDecision]:
    """Load review decisions indexed by record identifier.

    Args:
        path: JSON decision store path.

    Returns:
        Decisions indexed by stable record ID. A missing file yields an empty map.
    """
    if not path.exists():
        return {}
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewStoreError(f"Unable to read review store '{path}'.") from error
    if not isinstance(payload, dict):
        raise ReviewStoreError(f"Review store '{path}' must contain an object.")
    raw_decisions: object = payload.get("decisions", {})
    if not isinstance(raw_decisions, dict):
        raise ReviewStoreError(f"Invalid decisions object in '{path}'.")

    decisions: dict[str, ReviewDecision] = {}
    for record_id, raw_decision in raw_decisions.items():
        if not isinstance(record_id, str) or not isinstance(raw_decision, dict):
            raise ReviewStoreError(f"Invalid review decision in '{path}'.")
        status: object = raw_decision.get("status")
        if status not in {"accepted", "rejected"}:
            raise ReviewStoreError(
                f"Invalid status for record '{record_id}' in '{path}'."
            )
        decisions[record_id] = ReviewDecision(
            record_id=record_id,
            status=status,
            class_slug=str(raw_decision.get("class_slug", "")),
            reviewed_at=str(raw_decision.get("reviewed_at", "")),
        )
    return decisions


def save_decisions(path: Path, decisions: dict[str, ReviewDecision]) -> None:
    """Atomically persist the complete decision map.

    Args:
        path: Destination JSON file.
        decisions: Decisions indexed by record ID.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decisions": {
            record_id: asdict(decision)
            for record_id, decision in sorted(decisions.items())
        },
    }
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise ReviewStoreError(f"Unable to write review store '{path}'.") from error


__all__: list[str] = [
    "ReviewDecision",
    "ReviewStatus",
    "ReviewStoreError",
    "load_decisions",
    "save_decisions",
]
