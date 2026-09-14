"""Closed-vocabulary error categories. Never stores report text or identifiers."""

from __future__ import annotations

from typing import Any

# Coordinator-required taxonomy. Values are aggregate codes only.
ERROR_CLASSES = (
    "negation_scope_failure",
    "generic_meniscal_degeneration_as_injury",
    "postoperative_or_reconstructed",
    "historical_finding",
    "uncertain_or_differential",
    "compartment_unspecified_oa",
    "generic_degeneration_as_oa",
    "effusion_versus_synovitis",
    "popliteal_bakers_cyst_terminology",
    "bone_marrow_signal_versus_contusion",
    "fracture_exclusion_statement",
    "language_detection_failure",
    "unsupported_terminology",
    "section_boundary_failure",
    "insufficient_evidence",
    "other_call_error",
)

OA_TARGETS = {"medial_compartment_oa", "lateral_compartment_oa", "patellofemoral_oa"}
MENISCUS_TARGETS = {"medial_meniscus_injury", "lateral_meniscus_injury"}


def diagnostic_class_for(
    *,
    target_key: str,
    evidence_category: str,
    rule_id: str,
    label: str,
) -> str:
    """Map extractor evidence to a closed diagnostic class. No report text."""
    evidence = (evidence_category or "").lower()
    rule = (rule_id or "").lower()
    blob = f"{evidence} {rule}"

    if "history" in blob:
        return "historical_finding"
    if "postoperative" in blob or "graft" in blob:
        return "postoperative_or_reconstructed"
    if "combined_effusion" in blob or "combined_construct" in blob:
        return "effusion_versus_synovitis"
    if "anatomy_unspecified" in blob and target_key in OA_TARGETS:
        return "compartment_unspecified_oa"
    if "anatomy_unspecified" in blob and target_key in MENISCUS_TARGETS:
        return "generic_meniscal_degeneration_as_injury"
    if "meniscus_grade" in blob or "degeneration" in blob and target_key in MENISCUS_TARGETS:
        return "generic_meniscal_degeneration_as_injury"
    if target_key in OA_TARGETS and (
        "degenerative" in blob or "oa_kl1" in blob or "severity_below" in blob
    ):
        return "generic_degeneration_as_oa"
    if "bme_without_contusion" in blob or (
        target_key == "bone_contusion" and "insufficient" in blob
    ):
        return "bone_marrow_signal_versus_contusion"
    if target_key == "fracture" and (
        "differential" in blob or "exclusion" in blob or "cannot_exclude" in blob
    ):
        return "fracture_exclusion_statement"
    if "unsupported_language" in blob or "language_detection" in blob:
        return "language_detection_failure"
    if "technique" in blob or "section" in blob or "comparison" in blob:
        return "section_boundary_failure"
    if "uncertain" in blob or "hedged" in blob or "differential" in blob:
        return "uncertain_or_differential"
    if target_key == "baker_cyst" and (
        "morphology_without_anatomy" in blob or "cyst" in blob and "insufficient" in blob
    ):
        return "popliteal_bakers_cyst_terminology"
    if "unsupported" in blob or "no_decisive" in blob:
        return "unsupported_terminology"
    if "no_mention" in blob or "insufficient" in blob:
        return "insufficient_evidence"
    if label == "unknown":
        return "insufficient_evidence"
    return "other_call_error"


def classify_evaluation_error(
    *,
    target_key: str,
    gold: float,
    pred: float,
    evidence_category: str,
    rule_id: str,
    diagnostic_class: str | None = None,
) -> str | None:
    """Return an aggregate error/abstention bucket. Never includes quotations."""
    if gold != gold:  # NaN gold
        return None
    gold_pos = gold == 1
    called = pred == pred
    base = diagnostic_class or diagnostic_class_for(
        target_key=target_key,
        evidence_category=evidence_category,
        rule_id=rule_id,
        label="unknown" if not called else ("positive" if pred == 1 else "negative"),
    )
    if not called:
        outcome = "abstain_on_positive" if gold_pos else "abstain_on_negative"
        return f"{outcome}:{base}"
    if gold_pos and pred == 1:
        return "tp"
    if (not gold_pos) and pred == 0:
        return "tn"
    if (not gold_pos) and pred == 1:
        # False positive: prefer the diagnostic class, with negation-scope if
        # the extractor treated a negative statement as a positive finding.
        if "negat" in (evidence_category or "").lower():
            return "fp:negation_scope_failure"
        return f"fp:{base}"
    if gold_pos and pred == 0:
        if "negat" in (evidence_category or "").lower() or "normal_structure" in (
            evidence_category or ""
        ).lower():
            return "fn:negation_scope_failure"
        if "fracture" == target_key:
            return "fn:fracture_exclusion_statement"
        return f"fn:{base}"
    return "other_call_error"


def aggregate_error_counts(buckets: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bucket in buckets:
        if not bucket:
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def taxonomy_payload(counts: dict[str, int]) -> dict[str, Any]:
    """Sanitized error-category document. Codes only."""
    known = {cls: 0 for cls in ERROR_CLASSES}
    outcomes: dict[str, int] = {}
    for bucket, n in counts.items():
        outcomes[bucket] = int(n)
        if ":" in bucket:
            _, cls = bucket.split(":", 1)
        else:
            cls = bucket
        if cls in known:
            known[cls] += int(n)
    return {
        "taxonomy": list(ERROR_CLASSES),
        "contains_report_text": False,
        "contains_identifiers": False,
        "outcome_counts": outcomes,
        "class_counts": known,
        "note": (
            "Counts only. Categories never include report quotations or "
            "StudyInstanceUID / PatientID values."
        ),
    }
