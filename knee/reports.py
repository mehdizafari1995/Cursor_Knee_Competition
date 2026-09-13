"""Multilingual radiology reports for weak supervision.

RSNA Knee Abnormality Detection allows the *training* set to be weakly
supervised from free-text radiology reports, which may be written in several
languages. Ground-truth expert labels are not assumed at train time; instead a
label vector is *extracted* from report text. Crucially, this text is available
for training only — **test-time inference must rely on MRI images alone** and
never consumes report text.

This module (a) synthesizes multilingual reports from a finding vector and
(b) extracts a 12-dim binary weak-label vector back out of report text via
multilingual keyword matching. The synthetic reports keep the environment
self-contained; the extraction logic mirrors how a real weak-supervision
pipeline would parse multilingual reports.
"""

from __future__ import annotations

import numpy as np

from .config import TARGET_KEYS

# For each target, a short, mutually non-overlapping phrase in several
# languages (English, Spanish, French). Extraction is a case-insensitive
# substring search across every language variant.
MULTILINGUAL_KEYWORDS: dict[str, list[str]] = {
    "acl_injury": ["acl tear", "rotura del lca", "rupture du lca"],
    "mcl_injury": ["mcl sprain", "esguince del lcm", "entorse du llm"],
    "medial_meniscus_injury": [
        "medial meniscus tear",
        "rotura del menisco medial",
        "dechirure du menisque medial",
    ],
    "lateral_meniscus_injury": [
        "lateral meniscus tear",
        "rotura del menisco lateral",
        "dechirure du menisque lateral",
    ],
    "medial_compartment_oa": [
        "medial compartment osteoarthritis",
        "artrosis del compartimento medial",
        "arthrose du compartiment medial",
    ],
    "lateral_compartment_oa": [
        "lateral compartment osteoarthritis",
        "artrosis del compartimento lateral",
        "arthrose du compartiment lateral",
    ],
    "patellofemoral_oa": [
        "patellofemoral osteoarthritis",
        "artrosis patelofemoral",
        "arthrose femoro-patellaire",
    ],
    "joint_effusion": ["joint effusion", "derrame articular", "epanchement articulaire"],
    "synovitis": ["synovitis", "sinovitis", "synovite"],
    "baker_cyst": ["baker cyst", "quiste de baker", "kyste de baker"],
    "bone_contusion": ["bone contusion", "contusion osea", "contusion osseuse"],
    "fracture": ["fracture", "fractura", "fracture osseuse"],
}

# Multilingual "no abnormality" fillers for negative studies.
_NEGATIVE_FILLERS = (
    "No significant abnormality.",
    "Sin hallazgos significativos.",
    "Pas d'anomalie significative.",
)


def generate_report(labels: np.ndarray, rng: np.random.Generator) -> str:
    """Build a mixed-language report string describing the positive findings.

    Each positive finding contributes one sentence in a randomly chosen
    language. This produces genuinely multilingual free text from which weak
    labels can later be extracted.
    """
    present = [TARGET_KEYS[i] for i in range(len(labels)) if labels[i] > 0.5]
    if not present:
        return str(rng.choice(_NEGATIVE_FILLERS))

    rng.shuffle(present)
    sentences = []
    for key in present:
        phrase = str(rng.choice(MULTILINGUAL_KEYWORDS[key]))
        sentences.append(f"Findings: {phrase}.")
    return " ".join(sentences)


def extract_weak_labels(report: str, num_targets: int | None = None) -> np.ndarray:
    """Extract a binary weak-label vector from a (multilingual) report string."""
    text = report.lower()
    n = num_targets if num_targets is not None else len(TARGET_KEYS)
    labels = np.zeros(n, dtype=np.float32)
    for i, key in enumerate(TARGET_KEYS[:n]):
        if any(phrase in text for phrase in MULTILINGUAL_KEYWORDS[key]):
            labels[i] = 1.0
    return labels


def extract_weak_label_matrix(reports: list[str], num_targets: int) -> np.ndarray:
    """Vectorized weak-label extraction for a list of reports."""
    return np.stack([extract_weak_labels(r, num_targets) for r in reports])
