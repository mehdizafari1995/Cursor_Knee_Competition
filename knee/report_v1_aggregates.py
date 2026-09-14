"""Frozen Phase-4 v1 gold-count aggregates.

Source: reports/report_labeling_summary.md from the internet-off Kaggle run
(extractor report-labeler-v1.0.0, eval gold-eval-v1.0.0, n_gold_studies=58).
These are committed counts, not report text. Used to emit the corrected
coverage-aware evaluation of the version-1 extractor.
"""

from __future__ import annotations

# Official column -> published TP, TN, FP, FN, gold positives, expert pos rate.
# n_gold_known is 58 for every target (completely labelled studies).
V1_N_GOLD = 58

V1_GOLD_COUNTS: dict[str, dict[str, int | float]] = {
    "ACL": {"tp": 17, "tn": 10, "fp": 4, "fn": 0, "n_gold_positive": 24, "expert_pos_rate": 0.414},
    "MCL": {"tp": 6, "tn": 12, "fp": 4, "fn": 0, "n_gold_positive": 9, "expert_pos_rate": 0.155},
    "Medial Meniscus": {"tp": 11, "tn": 3, "fp": 9, "fn": 0, "n_gold_positive": 26, "expert_pos_rate": 0.448},
    "Lateral Meniscus": {"tp": 14, "tn": 4, "fp": 6, "fn": 0, "n_gold_positive": 23, "expert_pos_rate": 0.397},
    "Medial OA": {"tp": 4, "tn": 2, "fp": 1, "fn": 0, "n_gold_positive": 15, "expert_pos_rate": 0.259},
    "Lateral OA": {"tp": 2, "tn": 1, "fp": 4, "fn": 0, "n_gold_positive": 11, "expert_pos_rate": 0.190},
    "PF OA": {"tp": 3, "tn": 0, "fp": 4, "fn": 0, "n_gold_positive": 21, "expert_pos_rate": 0.362},
    "Effusion": {"tp": 16, "tn": 5, "fp": 11, "fn": 1, "n_gold_positive": 35, "expert_pos_rate": 0.603},
    "Synovitis": {"tp": 8, "tn": 0, "fp": 4, "fn": 0, "n_gold_positive": 27, "expert_pos_rate": 0.466},
    "Baker's": {"tp": 6, "tn": 2, "fp": 5, "fn": 0, "n_gold_positive": 12, "expert_pos_rate": 0.207},
    "Contusion": {"tp": 7, "tn": 4, "fp": 3, "fn": 1, "n_gold_positive": 19, "expert_pos_rate": 0.328},
    "Fracture": {"tp": 8, "tn": 3, "fp": 3, "fn": 0, "n_gold_positive": 18, "expert_pos_rate": 0.310},
}

# Weak-label positive rate among known (report-only n=4349), from the same summary.
V1_WEAK_POS_RATE_AMONG_KNOWN: dict[str, float] = {
    "ACL": 0.308,
    "MCL": 0.215,
    "Medial Meniscus": 0.704,
    "Lateral Meniscus": 0.461,
    "Medial OA": 0.703,
    "Lateral OA": 0.673,
    "PF OA": 0.796,
    "Effusion": 0.619,
    "Synovitis": 0.809,
    "Baker's": 0.650,
    "Contusion": 0.286,
    "Fracture": 0.224,
}

V1_MACRO_CONDITIONAL = {
    "conditional_sensitivity": 0.985,
    "conditional_specificity": 0.388,
    "conditional_precision": 0.621,
    "coverage": 0.299,
    "positive_capture_approx": 0.422,
}

INFLATED_PREVALENCE_TARGETS = (
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Synovitis",
    "Baker's",
)
