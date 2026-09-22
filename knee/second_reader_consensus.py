"""Strict-agreement consensus for deterministic v2 and the second reader.

Model-only predictions are never training labels. Uncertain and not_mentioned
remain unknown unless a verified global-normal negative rule applies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .report_labels import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN
from .second_reader_schema import STATUS_TO_LABEL

VERIFIED_NEGATIVE_EVIDENCE = frozenset({"global_normal"})


@dataclass(frozen=True)
class ConsensusCall:
    label: str
    reason: str
    source: str


def _as_status(value: str | None) -> str:
    if value in STATUS_TO_LABEL:
        return str(value)
    if value == LABEL_POSITIVE:
        return "positive"
    if value == LABEL_NEGATIVE:
        return "negative"
    return "not_mentioned"


def strict_agreement(
    deterministic_label: str,
    reader_status: str | None,
    *,
    deterministic_evidence: str = "",
    global_normal: bool = False,
    reader_enabled: bool = True,
    reader_malformed: bool = False,
) -> ConsensusCall:
    """Combine one target from both systems.

    positive + positive = candidate positive
    negative + negative = candidate negative
    any disagreement = unknown
    uncertain or not_mentioned = unknown unless verified global-normal
    supports a negative call
    """
    if reader_malformed:
        return ConsensusCall(LABEL_UNKNOWN, "reader_malformed", "unknown")
    status = _as_status(reader_status)
    reader_label = STATUS_TO_LABEL.get(status, LABEL_UNKNOWN)
    verified_negative = bool(global_normal) or deterministic_evidence in VERIFIED_NEGATIVE_EVIDENCE

    if not reader_enabled:
        return ConsensusCall(LABEL_UNKNOWN, "second_reader_disabled", "unknown")

    if status in {"uncertain", "not_mentioned"}:
        if deterministic_label == LABEL_NEGATIVE and verified_negative:
            return ConsensusCall(
                LABEL_NEGATIVE,
                "verified_global_normal_negative",
                "strict_agreement",
            )
        return ConsensusCall(LABEL_UNKNOWN, f"reader_{status}", "unknown")

    if deterministic_label == LABEL_UNKNOWN:
        return ConsensusCall(LABEL_UNKNOWN, "model_only_not_training_label", "unknown")

    if deterministic_label == LABEL_POSITIVE and reader_label == LABEL_POSITIVE:
        return ConsensusCall(LABEL_POSITIVE, "strict_positive_agreement", "strict_agreement")
    if deterministic_label == LABEL_NEGATIVE and reader_label == LABEL_NEGATIVE:
        return ConsensusCall(LABEL_NEGATIVE, "strict_negative_agreement", "strict_agreement")
    return ConsensusCall(LABEL_UNKNOWN, "disagreement", "unknown")


def consensus_row(
    *,
    deterministic_labels: dict[str, str],
    reader_statuses: dict[str, str],
    deterministic_evidence: dict[str, str] | None = None,
    global_normal: bool = False,
    reader_enabled: bool = True,
    reader_malformed: bool = False,
) -> dict[str, ConsensusCall]:
    evidence = deterministic_evidence or {}
    return {
        key: strict_agreement(
            deterministic_labels.get(key, LABEL_UNKNOWN),
            reader_statuses.get(key),
            deterministic_evidence=evidence.get(key, ""),
            global_normal=global_normal,
            reader_enabled=reader_enabled,
            reader_malformed=reader_malformed,
        )
        for key in deterministic_labels
    }
