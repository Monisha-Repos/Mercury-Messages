"""output.csv writing and contract validation.

Validation runs before anything is written, because a malformed submission
scores zero regardless of how good the reasoning was. The checks mirror
problem_statement.md exactly: column order, one row per input message, allowed
values for action and message_type, confidence in [0, 1], and a non-empty
evidence field.
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import config
from .dataset import Message
from .fusion import Decision


class ContractError(ValueError):
    """Raised when predictions would violate the submission contract."""


def validate(decisions: list[Decision], messages: list[Message]) -> None:
    expected_ids = [m.message_id for m in messages]
    actual_ids = [d.message_id for d in decisions]

    if len(actual_ids) != len(expected_ids):
        raise ContractError(
            f"expected {len(expected_ids)} predictions, got {len(actual_ids)}"
        )

    missing = set(expected_ids) - set(actual_ids)
    if missing:
        raise ContractError(f"missing predictions for: {sorted(missing)[:5]}")

    duplicates = len(actual_ids) - len(set(actual_ids))
    if duplicates:
        raise ContractError(f"{duplicates} duplicate message_id rows")

    if actual_ids != expected_ids:
        raise ContractError("prediction order does not match messages.csv")

    for decision in decisions:
        if decision.action not in config.ACTIONS:
            raise ContractError(f"{decision.message_id}: bad action {decision.action!r}")
        if decision.message_type not in config.MESSAGE_TYPES:
            raise ContractError(
                f"{decision.message_id}: bad message_type {decision.message_type!r}"
            )
        if not 0.0 <= decision.confidence <= 1.0:
            raise ContractError(
                f"{decision.message_id}: confidence {decision.confidence} out of range"
            )
        if not decision.reason.strip():
            raise ContractError(f"{decision.message_id}: empty reason")
        if not decision.evidence_message_ids.strip():
            raise ContractError(
                f"{decision.message_id}: empty evidence (use 'none' when absent)"
            )


def write(decisions: list[Decision], paths: list[Path] | None = None) -> list[Path]:
    targets = [Path(p) for p in (paths or config.OUTPUT_PATHS)]
    written: list[Path] = []

    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=config.OUTPUT_COLUMNS)
            writer.writeheader()
            for decision in decisions:
                writer.writerow(decision.as_row())
        written.append(target)

    return written


def write_trace(decisions: list[Decision], path: Path) -> Path:
    """Per-message score breakdown - the input to evaluation and the UI.

    Kept separate from output.csv so the submission stays exactly on-contract
    while the debugging detail remains available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "message_id", "action", "message_type", "confidence",
        "content_score", "pattern_score", "blended_score", "risk_score",
        "overrides", "backend", "signals", "evidence_message_ids", "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for d in decisions:
            writer.writerow({
                "message_id": d.message_id,
                "action": d.action,
                "message_type": d.message_type,
                "confidence": f"{d.confidence:.2f}",
                "content_score": d.content_score,
                "pattern_score": d.pattern_score,
                "blended_score": d.blended_score,
                "risk_score": d.risk_score,
                "overrides": "|".join(d.overrides) or "none",
                "backend": d.backend,
                "signals": "|".join(d.signals) or "none",
                "evidence_message_ids": d.evidence_message_ids,
                "reason": d.reason,
            })
    return path
