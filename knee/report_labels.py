"""Deterministic high-precision report-label extractor and gold evaluation.

The extractor never accepts gold/expert labels. Unknown is a first-class
output and is never imputed as negative. Report text is processed in memory
and is not written to sanitized artefacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
import unicodedata

import numpy as np
import pandas as pd
import yaml

from .config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    PROJECT_ROOT,
    REPORT_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEY_TO_OFFICIAL,
    TARGET_KEYS,
)
from .kaggle_audit import (
    FORBIDDEN_TEXT_COLUMNS,
    ReportTextLeakError,
    assert_no_report_text,
    collect_report_snippets,
    jsonable,
)
from .kaggle_bundle import sha256_file
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    DATA_SOURCE_SOFTWARE_FIXTURE,
    KaggleLayout,
    PreflightError,
    PreflightResult,
    ReadOnlyInputGuard,
    assert_not_kaggle_input,
    preflight,
    scientific_evaluation_for,
)
from .coverage_metrics import (
    BOOTSTRAP_METRIC_NAMES,
    bootstrap_coverage_intervals,
    coverage_confusion,
    coverage_error_curve,
    macro_mean,
)
from .labels import extract_label_matrix, prevalence
from .report_error_taxonomy import classify_evaluation_error, diagnostic_class_for, taxonomy_payload
from .report_language import (
    LanguageResult,
    aggregate_language_audit,
    audit_report_format,
    detect_language,
)
from .report_negation import (
    Modifiers,
    classify_modifiers,
    clause_containing,
    normalize_text,
    sentence_containing,
    split_sentences,
)

EXTRACTOR_FORBIDDEN_PARAMS = {
    "gold",
    "labels",
    "y_true",
    "expert",
    "expert_labels",
    "target_labels",
    "train_labels",
    "gold_labels",
}

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_UNKNOWN = "unknown"

SMALL_N_DEFAULT = ("mcl_injury", "lateral_compartment_oa", "baker_cyst")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "report_labeler.yaml"

# Coordinator gates after the real gold-label evaluation (n=58). These are not
# extractor config keys and must not be flipped by the Kaggle notebook.
HARD_WEAK_LABEL_TRAINING_VERDICT = "NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING"
SELECTIVE_LABEL_REFINEMENT_VERDICT = "READY_FOR_SELECTIVE_LABEL_REFINEMENT"
HARD_WEAK_LABEL_TRAINING_ALLOWED = False
SELECTIVE_LABEL_REFINEMENT_ALLOWED = True


class HardWeakLabelTrainingError(RuntimeError):
    """Raised when image training would treat report weak labels as hard 0/1."""


def assert_hard_weak_label_training_forbidden() -> None:
    """Refuse image-model training on hard-imputed report weak labels."""
    if HARD_WEAK_LABEL_TRAINING_ALLOWED:
        return
    raise HardWeakLabelTrainingError(
        f"{HARD_WEAK_LABEL_TRAINING_VERDICT}: report weak labels must not be "
        "used as hard 0/1 training targets. Unknown stays unknown. "
        f"{SELECTIVE_LABEL_REFINEMENT_VERDICT}."
    )

PRIVATE_WEAK_FILENAME = "weak_labels.csv"
PRIVATE_GOLD_PRED_FILENAME = "gold_predictions.csv"
PRIVATE_LANGUAGE_FILENAME = "language_assignments.csv"
SANITIZED_SUMMARY_FILENAME = "report_labeling_summary.md"
SANITIZED_METRICS_FILENAME = "gold_metrics.json"
SANITIZED_LANGUAGE_FILENAME = "language_audit.json"
SANITIZED_WEAK_FILENAME = "weak_label_aggregates.json"
SANITIZED_ERRORS_FILENAME = "error_categories.json"
SANITIZED_MANIFEST_FILENAME = "report_label_manifest.json"
SANITIZED_HASHES_FILENAME = "private_output_hashes.json"

KAGGLE_REPORT_OUTPUT_DIRNAME = "rsna_report_label_outputs"

COMPARTMENT_CUES = {
    "medial": (
        "medial", "inner", "mediale", "médial", "medialen", "medialny",
        "έσω", "медиал", "iç ", "medial kompart",
    ),
    "lateral": (
        "lateral", "outer", "laterale", "latéral", "lateralen",
        "έξω", "латерал", "dış ", "lateral kompart",
    ),
    "patellofemoral": (
        "patellofemoral", "patello-femoral", "femoropatellar", "femoro-patellar",
        "femoro-patellaire", "retropatellar", "pfj", "pf joint", "pf oa",
        "trochlea", "patellofemoral", "επιγονατιδομηρια", "пателофеморал",
    ),
}

GLOBAL_OA_CUES = (
    "tricompartmental", "tri-compartmental", "pancompartmental",
    "all three compartments", "three-compartment", "tres compartimentos",
    "trois compartiments", "drei kompartimente",
)

BOTH_MENISCI_CUES = (
    "both menisci", "medial and lateral menisc", "lateral and medial menisc",
    "ambos meniscos", "los dos meniscos", "deux menisques", "beide menisken",
    "her iki menisk", "και οι δύο μηνίσκοι",
)

GLOBAL_LIGAMENT_INTACT = (
    "cruciate ligaments are intact",
    "cruciate and collateral ligaments are intact",
    "acl, pcl, mcl, and lcl are intact",
    "acl pcl mcl lcl intact",
)

IDENTIFIER_COLUMN_CANDIDATES = (
    STUDY_ID_COLUMN,
    "SeriesInstanceUID",
    "PatientID",
    "patient_id",
    "patientId",
)


class PrivacyLeakError(RuntimeError):
    """Sanitized output contained report text or row-level identifiers."""


class ExtractorConfigError(ValueError):
    """The report-labeler config cannot be used."""


@dataclass(frozen=True)
class TargetPrediction:
    target_key: str
    official_column: str
    label: str
    confidence: float
    evidence_category: str
    rule_id: str
    extractor_version: str
    diagnostic_class: str = "insufficient_evidence"


@dataclass
class StudyExtraction:
    language: str
    language_uncertain: bool
    language_reason: str
    language_supported: bool
    heuristic_language: str
    ngram_language: str
    detectors_agree: bool
    quality_flags: tuple[str, ...]
    extractor_version: str
    config_sha256: str
    predictions: dict[str, TargetPrediction]
    format_audit: dict[str, Any]


@dataclass(frozen=True)
class _Term:
    pattern: re.Pattern[str]
    text: str
    source: str  # ontology_verified | clinical_standard | language_agnostic
    kind: str  # synonym | abbreviation | morphology_positive | ...


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_sha256(path: Path) -> str:
    return sha256_file(path)


def load_extractor_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExtractorConfigError(f"{config_path} must contain a mapping")
    targets = payload.get("targets") or {}
    missing = [key for key in TARGET_KEYS if key not in targets]
    extra = [key for key in targets if key not in TARGET_KEYS]
    if missing or extra:
        raise ExtractorConfigError(
            f"report labeler targets drifted from TARGET_KEYS (missing={missing}, extra={extra})"
        )
    payload["_path"] = str(config_path)
    payload["_sha256"] = config_sha256(config_path)
    return payload


def extractor_version_of(config: dict) -> str:
    return str(config.get("extractor_version") or "report-labeler-unversioned")


def eval_protocol_version_of(config: dict) -> str:
    return str(config.get("eval_protocol_version") or "gold-eval-unversioned")


def _compile_term(term: str) -> re.Pattern[str]:
    escaped = re.escape(unicodedata.normalize("NFKC", term).lower())
    escaped = escaped.replace(r"\ ", r"[\s\-]+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def compiled_terms(config: dict) -> dict[str, list[_Term]]:
    cached = config.get("_compiled_terms")
    if cached:
        return cached
    compiled = {
        key: _terms_for_target((config.get("targets") or {}).get(key) or {})
        for key in TARGET_KEYS
    }
    config["_compiled_terms"] = compiled
    return compiled


KIND_ALIASES = {
    "synonyms": "synonym",
    "abbreviations": "abbreviation",
    "morphology_positive": "morphology_positive",
    "morphology_negative": "morphology_negative",
    "morphology_unknown": "morphology_unknown",
    "postoperative": "postoperative",
}


def _terms_for_target(target_cfg: dict) -> list[_Term]:
    terms: list[_Term] = []
    for source in ("ontology_verified", "clinical_standard"):
        block = target_cfg.get(source) or {}
        for raw_kind in (
            "synonyms",
            "abbreviations",
            "morphology_positive",
            "morphology_negative",
            "morphology_unknown",
            "postoperative",
        ):
            kind = KIND_ALIASES[raw_kind]
            for raw in block.get(raw_kind) or []:
                text = str(raw).strip()
                if not text:
                    continue
                if kind == "abbreviation" and text.lower() in {"oa", "pf", "mm", "lm"}:
                    continue
                terms.append(
                    _Term(
                        pattern=_compile_term(text),
                        text=text,
                        source=source,
                        kind=kind,
                    )
                )
    for raw in target_cfg.get("language_agnostic_abbreviations") or []:
        text = str(raw).strip()
        if not text:
            continue
        terms.append(
            _Term(
                pattern=_compile_term(text),
                text=text,
                source="language_agnostic",
                kind="abbreviation",
            )
        )
    # Longest first so "anterior cruciate ligament rupture" wins over "anterior cruciate".
    terms.sort(key=lambda item: (-len(item.text), item.text, item.source))
    return terms


def _cue_present(text: str, cues: Iterable[str]) -> bool:
    lowered = text.lower()
    for cue in cues:
        if cue.lower() in lowered:
            return True
    return False


def _compartment_of(sentence: str) -> set[str]:
    found: set[str] = set()
    lowered = sentence.lower()
    if _cue_present(lowered, GLOBAL_OA_CUES):
        return {"medial", "lateral", "patellofemoral"}
    for name, cues in COMPARTMENT_CUES.items():
        if _cue_present(lowered, cues):
            found.add(name)
    return found


@dataclass
class _Match:
    start: int
    end: int
    term: _Term
    sentence_index: int
    section: str
    modifiers: Modifiers


def _find_matches(text: str, terms: list[_Term], sentences) -> list[_Match]:
    lowered = text.lower()
    matches: list[_Match] = []
    occupied: list[tuple[int, int]] = []
    for term in terms:
        for hit in term.pattern.finditer(lowered):
            start, end = hit.start(), hit.end()
            if any(not (end <= a or start >= b) for a, b in occupied):
                continue
            sentence = sentence_containing(sentences, start, end)
            if sentence is None:
                continue
            local_start = start - sentence.start
            local_end = end - sentence.start
            local_start = max(0, local_start)
            local_end = min(len(sentence.text), max(local_start + 1, local_end))
            modifiers = classify_modifiers(
                sentence.text,
                match_start=local_start,
                match_end=local_end,
                section=sentence.section,
            )
            matches.append(
                _Match(
                    start=start,
                    end=end,
                    term=term,
                    sentence_index=sentence.index,
                    section=sentence.section,
                    modifiers=modifiers,
                )
            )
            occupied.append((start, end))
    return matches


def _confidence(config: dict, key: str) -> float:
    return float((config.get("confidence") or {}).get(key, 0.0))


def _prediction(
    *,
    target_key: str,
    label: str,
    evidence_category: str,
    rule_id: str,
    config: dict,
    confidence_key: str,
) -> TargetPrediction:
    conf = _confidence(config, confidence_key)
    if label == LABEL_UNKNOWN:
        conf = min(conf, _confidence(config, "unknown") or 0.0) if confidence_key == "unknown" else (
            _confidence(config, "uncertain") if label == LABEL_UNKNOWN and confidence_key == "uncertain" else conf
        )
        if confidence_key == "unknown":
            conf = _confidence(config, "unknown")
    diag = diagnostic_class_for(
        target_key=target_key,
        evidence_category=evidence_category,
        rule_id=rule_id,
        label=label,
    )
    return TargetPrediction(
        target_key=target_key,
        official_column=TARGET_KEY_TO_OFFICIAL[target_key],
        label=label,
        confidence=float(conf),
        evidence_category=evidence_category,
        rule_id=f"{extractor_version_of(config)}:{target_key}:{rule_id}",
        extractor_version=extractor_version_of(config),
        diagnostic_class=diag,
    )


def _unknown(target_key: str, config: dict, category: str, rule_id: str, conf_key: str = "unknown") -> TargetPrediction:
    return _prediction(
        target_key=target_key,
        label=LABEL_UNKNOWN,
        evidence_category=category,
        rule_id=rule_id,
        config=config,
        confidence_key=conf_key,
    )


def _kl_decision(sentence: str, config: dict) -> str | None:
    oa_cfg = config.get("oa_severity") or {}
    text = sentence.lower()
    for grade in oa_cfg.get("negative_kl") or [0]:
        if re.search(rf"\b(?:kl|kellgren(?:-lawrence)?)\s*[=:]?\s*{grade}\b", text):
            return LABEL_NEGATIVE
    for grade in oa_cfg.get("unknown_kl") or [1]:
        if re.search(rf"\b(?:kl|kellgren(?:-lawrence)?)\s*[=:]?\s*{grade}\b", text):
            return LABEL_UNKNOWN
    for grade in oa_cfg.get("positive_kl") or [2, 3, 4]:
        if re.search(rf"\b(?:kl|kellgren(?:-lawrence)?)\s*[=:]?\s*{grade}\b", text):
            return LABEL_POSITIVE
    for grade in oa_cfg.get("outerbridge_positive") or [3, 4]:
        if re.search(rf"\bouterbridge\s*{grade}\b", text):
            return LABEL_POSITIVE
    for grade in oa_cfg.get("outerbridge_unknown") or [1, 2]:
        if re.search(rf"\bouterbridge\s*{grade}\b", text):
            return LABEL_UNKNOWN
    return None


def _effusion_severity(sentence: str, config: dict, negated: bool) -> str:
    cfg = config.get("effusion_severity") or {}
    text = sentence.lower()
    if negated or any(p in text for p in (cfg.get("negative_phrases") or [])):
        return LABEL_NEGATIVE
    if any(q in text for q in (cfg.get("unknown_qualifiers") or [])):
        return LABEL_UNKNOWN
    if any(q in text for q in (cfg.get("positive_qualifiers") or [])):
        return LABEL_POSITIVE
    # Unqualified "effusion" is treated as positive (named finding).
    return LABEL_POSITIVE


def _decide_target(
    *,
    target_key: str,
    matches: list[_Match],
    sentences,
    config: dict,
    language: LanguageResult,
    global_normal: bool,
) -> TargetPrediction:
    target_cfg = (config.get("targets") or {}).get(target_key) or {}
    required_compartment = target_cfg.get("compartment")
    required_laterality = target_cfg.get("laterality")

    if not language.supported and language.language not in {"en", "und"}:
        agnostic = [m for m in matches if m.term.source == "language_agnostic"]
        if not agnostic and not matches:
            return _unknown(target_key, config, "unsupported_language", "unsupported_language")

    if not matches:
        if global_normal:
            return _prediction(
                target_key=target_key,
                label=LABEL_NEGATIVE,
                evidence_category="global_normal",
                rule_id="global_normal",
                config=config,
                confidence_key="global_normal",
            )
        return _unknown(target_key, config, "insufficient_evidence", "no_mention")

    current = [
        m for m in matches
        if m.section not in {"technique", "comparison"}
    ]
    if not current:
        return _unknown(target_key, config, "insufficient_evidence", "technique_only")

    # Morphology such as "tear" or "cyst" is only usable next to an anatomy anchor.
    anatomy_clauses: set[tuple[int, str]] = set()
    for m in current:
        if m.term.kind in {"synonym", "abbreviation"}:
            sentence = sentences[m.sentence_index]
            local_start = m.start - sentence.start
            local_end = m.end - sentence.start
            anatomy_clauses.add((sentence.index, clause_containing(sentence, local_start, local_end)))
    if required_compartment:
        for sentence in sentences:
            comps = _compartment_of(sentence.text)
            if required_compartment in comps or comps >= {"medial", "lateral", "patellofemoral"}:
                anatomy_clauses.add((sentence.index, sentence.text))
    filtered: list[_Match] = []
    for m in current:
        if m.term.kind in {"synonym", "abbreviation"}:
            filtered.append(m)
            continue
        sentence = sentences[m.sentence_index]
        local_start = m.start - sentence.start
        local_end = m.end - sentence.start
        clause = clause_containing(sentence, local_start, local_end)
        if (sentence.index, clause) in anatomy_clauses or any(
            clause in existing or existing in clause
            for idx, existing in anatomy_clauses
            if idx == sentence.index
        ):
            filtered.append(m)
    current = filtered
    if not current:
        if global_normal:
            return _prediction(
                target_key=target_key,
                label=LABEL_NEGATIVE,
                evidence_category="global_normal",
                rule_id="global_normal",
                config=config,
                confidence_key="global_normal",
            )
        return _unknown(target_key, config, "insufficient_evidence", "morphology_without_anatomy")

    # History-only mentions of pathology are unknown (not current findings).
    non_history = [m for m in current if m.section != "history" and not m.modifiers.historical]
    if not non_history:
        return _unknown(target_key, config, "history_only", "history_only", "uncertain")

    usable = non_history

    # Combined effusion-synovitis construct: unknown for both unless distinct.
    if target_key in {"joint_effusion", "synovitis"}:
        combined = [
            m for m in usable
            if "effusion-synovitis" in m.term.text.lower() or "effusion synovitis" in m.term.text.lower()
        ]
        distinct = [m for m in usable if m not in combined]
        sentence_has_combined = any(
            "effusion-synovitis" in sentences[m.sentence_index].text.lower()
            or "effusion synovitis" in sentences[m.sentence_index].text.lower()
            for m in usable
        )
        if sentence_has_combined:
            extra = False
            for match in usable:
                stripped = (
                    sentences[match.sentence_index].text.lower()
                    .replace("effusion-synovitis", " ")
                    .replace("effusion synovitis", " ")
                )
                if target_key == "synovitis" and any(
                    token in stripped for token in ("synovial", "synovitis", "sinovitis", "synovite")
                ):
                    extra = True
                if target_key == "joint_effusion" and any(
                    token in stripped for token in ("joint effusion", "joint fluid", "hydrarthrosis", "derrame")
                ):
                    extra = True
            if not extra:
                return _unknown(
                    target_key, config, "combined_effusion_synovitis", "combined_construct", "uncertain"
                )
        if combined and not distinct:
            return _unknown(target_key, config, "combined_effusion_synovitis", "combined_construct", "uncertain")
        usable = distinct or usable

    # Compartment / laterality gates.
    if required_compartment or required_laterality:
        gated: list[_Match] = []
        unspecified = False
        for match in usable:
            sentence = sentences[match.sentence_index].text if match.sentence_index < len(sentences) else ""
            compartments = _compartment_of(sentence)
            both_menisci = _cue_present(sentence, BOTH_MENISCI_CUES)
            if required_laterality == "medial":
                if both_menisci or "medial" in compartments or "medial" in match.term.text.lower():
                    gated.append(match)
                elif "lateral" in compartments or "lateral" in match.term.text.lower():
                    continue
                elif "menisc" in match.term.text.lower() or "μηνίσ" in match.term.text.lower():
                    unspecified = True
                else:
                    unspecified = True
            elif required_laterality == "lateral":
                if both_menisci or "lateral" in compartments or "lateral" in match.term.text.lower():
                    gated.append(match)
                elif "medial" in compartments or "medial" in match.term.text.lower():
                    continue
                elif "menisc" in match.term.text.lower() or "μηνίσ" in match.term.text.lower():
                    unspecified = True
                else:
                    unspecified = True
            elif required_compartment:
                if required_compartment in compartments or required_compartment in match.term.text.lower():
                    gated.append(match)
                elif compartments and required_compartment not in compartments:
                    continue
                else:
                    unspecified = True
        if not gated:
            category = "anatomy_unspecified" if unspecified else "wrong_compartment"
            return _unknown(target_key, config, category, category, "uncertain")
        usable = gated
        # OA: compartment cue must share the clause with an OA term, not a meniscus laterality.
        if required_compartment and target_key.endswith("_oa"):
            oa_tokens = (
                "osteoarthr", "oa", "artros", "arthrose", "artrose", "chondros",
                "gonarthro", "kireçlenme", "артроз", "οστεοαρθρ", "αarthρίτιδα", "αρθρίτιδα",
            )
            clause_ok: list[_Match] = []
            for match in usable:
                sentence = sentences[match.sentence_index].text if match.sentence_index < len(sentences) else ""
                dummy = sentence
                parts = [p.strip() for p in re.split(r"\s*(?:,|/|;| - | – | — )\s*", dummy) if p and p.strip()]
                found = False
                for part in parts or [dummy]:
                    low = part.lower()
                    comps = _compartment_of(part)
                    if required_compartment in comps or required_compartment in match.term.text.lower():
                        if any(tok in low or tok in match.term.text.lower() for tok in oa_tokens):
                            found = True
                            break
                if found:
                    clause_ok.append(match)
            if not clause_ok:
                return _unknown(
                    target_key, config, "anatomy_unspecified", "oa_compartment_not_in_clause", "uncertain"
                )
            usable = clause_ok

    # Meniscus grade 1-2 / degeneration without tear -> unknown.
    if target_key in {"medial_meniscus_injury", "lateral_meniscus_injury"}:
        grade_unknown = [
            m for m in usable
            if m.term.kind == "morphology_unknown"
            or re.search(r"\bgrade\s*[12]\b", sentences[m.sentence_index].text.lower())
        ]
        tear_pos = [
            m for m in usable
            if m.term.kind in {"synonym", "abbreviation", "morphology_positive"}
            and not m.modifiers.negated
        ]
        if grade_unknown and not tear_pos:
            return _unknown(target_key, config, "severity_below_threshold", "meniscus_grade_1_2", "uncertain")

    # OA severity.
    if target_key.endswith("_oa"):
        for match in usable:
            sentence = sentences[match.sentence_index].text
            kl = _kl_decision(sentence, config)
            if kl == LABEL_NEGATIVE and match.modifiers.negated:
                pass
            if kl == LABEL_UNKNOWN:
                return _unknown(target_key, config, "severity_below_threshold", "oa_kl1", "uncertain")
            if match.term.kind == "morphology_unknown" and not match.modifiers.negated:
                return _unknown(target_key, config, "severity_below_threshold", "oa_degenerative_nonspecific", "uncertain")

    # Effusion severity.
    if target_key == "joint_effusion":
        labels = []
        for match in usable:
            sentence = sentences[match.sentence_index].text
            labels.append(_effusion_severity(sentence, config, match.modifiers.negated))
        if LABEL_POSITIVE in labels and LABEL_NEGATIVE not in labels:
            return _prediction(
                target_key=target_key,
                label=LABEL_POSITIVE,
                evidence_category="explicit_positive",
                rule_id="effusion_named",
                config=config,
                confidence_key="explicit_positive",
            )
        if labels and all(lab == LABEL_NEGATIVE for lab in labels):
            return _prediction(
                target_key=target_key,
                label=LABEL_NEGATIVE,
                evidence_category="negated_finding",
                rule_id="effusion_negated",
                config=config,
                confidence_key="explicit_negative",
            )
        if LABEL_UNKNOWN in labels and LABEL_POSITIVE not in labels:
            return _unknown(target_key, config, "severity_below_threshold", "effusion_trace", "uncertain")

    # Bone marrow edema without contusion word.
    if target_key == "bone_contusion":
        only_bme = usable and all(m.term.kind == "morphology_unknown" for m in usable)
        if only_bme:
            return _unknown(target_key, config, "insufficient_evidence", "bme_without_contusion", "uncertain")

    # Fracture differentials.
    if target_key == "fracture":
        only_diff = usable and all(m.term.kind == "morphology_unknown" for m in usable)
        if only_diff:
            return _unknown(target_key, config, "insufficient_evidence", "fracture_differential", "uncertain")

    # Baker / popliteal: generic "cyst" is not enough.
    if target_key == "baker_cyst":
        anchors = (
            "baker", "popliteal cyst", "poplitealzyste", "gastrocnemius-semimembranosus",
            "quiste de baker", "quiste poplite", "kyste de baker", "bakerzyste",
            "cisti di baker", "cisti poplitea", "baker kisti", "κύστη baker",
            "ιγνυακή", "киста на бейкър", "киста бейкера", "torbiel bakera",
            "popliteal",
        )
        generic_cyst = {
            "cyst", "cystic", "quiste", "kyste", "zyste", "cisti", "cisto",
            "kisti", "κύστη", "киста", "torbiel",
        }
        anchored: list[_Match] = []
        for match in usable:
            sentence = (
                sentences[match.sentence_index].text.lower()
                if match.sentence_index < len(sentences) else ""
            )
            term_l = match.term.text.lower()
            has_anchor = any(a in sentence or a in term_l for a in anchors)
            if match.term.kind == "morphology_positive" and term_l in generic_cyst and not has_anchor:
                continue
            if match.term.kind in {"synonym", "abbreviation"} or has_anchor:
                anchored.append(match)
        if not anchored:
            return _unknown(
                target_key, config, "insufficient_evidence",
                "baker_without_popliteal_anchor", "uncertain",
            )
        usable = anchored

    # Postoperative without explicit current graft/native tear.
    postop = [m for m in usable if m.modifiers.postoperative]
    if postop and not any(
        (not m.modifiers.negated and m.term.kind in {"synonym", "morphology_positive"} and not m.modifiers.postoperative)
        for m in usable
    ):
        # Intact graft -> unknown for native injury (not a current native-ACL statement).
        if any(m.modifiers.normal_structure or m.modifiers.negated for m in postop):
            return _unknown(target_key, config, "postoperative", "postoperative_intact", "uncertain")
        # Reconstruction mentioned with tear language may still be current graft injury.
        if any(not m.modifiers.negated and m.term.kind in {"morphology_positive", "synonym"} for m in usable):
            pass
        else:
            return _unknown(target_key, config, "postoperative", "postoperative_only", "uncertain")

    positives = [
        m for m in usable
        if not m.modifiers.negated
        and not m.modifiers.uncertain
        and m.term.kind in {"synonym", "abbreviation", "morphology_positive"}
        and not m.modifiers.normal_structure
    ]
    uncertain_pos = [
        m for m in usable
        if not m.modifiers.negated
        and m.modifiers.uncertain
        and m.term.kind in {"synonym", "abbreviation", "morphology_positive"}
    ]
    negatives = [
        m for m in usable
        if (m.modifiers.negated or m.modifiers.normal_structure)
        and not m.modifiers.uncertain
    ]

    if positives:
        require_morph = bool(target_cfg.get("require_pathology_morphology"))
        has_named_finding = any(m.term.kind == "synonym" for m in positives)
        has_injury_morphology = any(
            m.term.kind == "morphology_positive" and not m.modifiers.negated for m in usable
        ) or any(
            token in m.term.text.lower()
            for m in positives
            for token in (
                "tear", "torn", "rupture", "rotura", "injury", "sprain",
                "osteoarthr", "artros", "arthrose", "effusion", "synov",
                "cyst", "contusion", "bruise", "fracture", "fraktur",
                "derrame", "kyste", "quiste", "thickening", "baker",
            )
        )
        if require_morph and not has_injury_morphology:
            positives = []
        elif not has_named_finding and not has_injury_morphology:
            positives = []
        else:
            source = positives[0].term.source
            conf_key = "explicit_positive"
            if source == "language_agnostic":
                conf_key = "abbreviation_positive"
            elif source == "clinical_standard":
                conf_key = "multilingual_standard"
            return _prediction(
                target_key=target_key,
                label=LABEL_POSITIVE,
                evidence_category="explicit_positive",
                rule_id="current_finding",
                config=config,
                confidence_key=conf_key,
            )
    if uncertain_pos:
        return _unknown(target_key, config, "uncertain_language", "hedged_finding", "uncertain")
    if negatives:
        return _prediction(
            target_key=target_key,
            label=LABEL_NEGATIVE,
            evidence_category="negated_finding" if any(m.modifiers.negated for m in negatives) else "normal_structure",
            rule_id="negated_or_normal",
            config=config,
            confidence_key="explicit_negative",
        )
    return _unknown(target_key, config, "insufficient_evidence", "no_decisive_match")


def extract_labels_from_report(
    report: str,
    *,
    language_result: LanguageResult | None = None,
    config: dict | None = None,
) -> StudyExtraction:
    """Extract 12 ternary labels from one report.

    Gold / expert labels are not parameters and must never be supplied.
    """
    illegal = EXTRACTOR_FORBIDDEN_PARAMS.intersection(inspect.signature(extract_labels_from_report).parameters)
    if illegal:  # pragma: no cover - defensive
        raise TypeError(f"extractor signature must not include {sorted(illegal)}")

    cfg = config if config is not None else load_extractor_config()
    text = normalize_text(report)
    lang = language_result or detect_language(
        text,
        supported=tuple((cfg.get("languages") or {}).get("supported") or []),
        uncertain_margin=float((cfg.get("languages") or {}).get("uncertain_margin") or 0.15),
        min_score=float((cfg.get("languages") or {}).get("min_score") or 2.0),
    )
    fmt = audit_report_format(text)
    sentences = split_sentences(text)
    global_normal = _cue_present(text, cfg.get("global_normal_cues") or [])
    if _cue_present(text, GLOBAL_LIGAMENT_INTACT):
        # Ligament intact statement is used per-target via normal-structure too.
        pass

    predictions: dict[str, TargetPrediction] = {}
    terms_by_target = compiled_terms(cfg)
    for key in TARGET_KEYS:
        terms = terms_by_target[key]
        matches = _find_matches(text, terms, sentences)
        predictions[key] = _decide_target(
            target_key=key,
            matches=matches,
            sentences=sentences,
            config=cfg,
            language=lang,
            global_normal=global_normal,
        )

    flags = list(fmt.quality_flags)
    if lang.uncertain:
        flags.append("uncertain_language")
    if not lang.supported:
        flags.append("unsupported_language")
    if lang.mixed_scripts:
        flags.append("mixed_language")

    return StudyExtraction(
        language=lang.language,
        language_uncertain=lang.uncertain,
        language_reason=lang.reason,
        language_supported=lang.supported,
        heuristic_language=str(lang.heuristic_language or lang.language),
        ngram_language=str(lang.ngram_language or ""),
        detectors_agree=bool(lang.detectors_agree),
        quality_flags=tuple(sorted(set(flags))),
        extractor_version=extractor_version_of(cfg),
        config_sha256=str(cfg.get("_sha256") or ""),
        predictions=predictions,
        format_audit={
            "n_chars": fmt.n_chars,
            "n_section_headers": fmt.n_section_headers,
            "n_empty_sections": fmt.n_empty_sections,
            "length_bucket": fmt.length_bucket,
            "boilerplate_hits": fmt.boilerplate_hits,
        },
    )


def label_to_float(label: str) -> float:
    if label == LABEL_POSITIVE:
        return 1.0
    if label == LABEL_NEGATIVE:
        return 0.0
    return float("nan")


def extraction_to_row(study_id: str, extraction: StudyExtraction) -> dict[str, Any]:
    row: dict[str, Any] = {
        STUDY_ID_COLUMN: study_id,
        "extractor_version": extraction.extractor_version,
        "language_code": extraction.language,
        "language_uncertain": extraction.language_uncertain,
        "heuristic_language": extraction.heuristic_language,
        "ngram_language": extraction.ngram_language,
        "language_detectors_agree": extraction.detectors_agree,
        "quality_flags": "|".join(extraction.quality_flags),
    }
    for key in TARGET_KEYS:
        pred = extraction.predictions[key]
        official = TARGET_KEY_TO_OFFICIAL[key]
        row[official] = label_to_float(pred.label)
        row[f"{key}_confidence"] = pred.confidence
        row[f"{key}_evidence"] = pred.evidence_category
        row[f"{key}_rule"] = pred.rule_id
        row[f"{key}_diagnostic"] = pred.diagnostic_class
    return row


def assert_extractor_has_no_gold_parameter() -> None:
    params = set(inspect.signature(extract_labels_from_report).parameters)
    leaked = params & EXTRACTOR_FORBIDDEN_PARAMS
    if leaked:
        raise TypeError(f"extractor must not accept gold labels: {sorted(leaked)}")


def split_gold_and_report_only(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold = all 12 official labels known. Report-only = all 12 unknown."""
    labels, mask = extract_label_matrix(frame, require_all=False)
    gold_idx = mask.all(axis=1)
    report_only_idx = ~mask.any(axis=1)
    return frame.loc[gold_idx].copy(), frame.loc[report_only_idx].copy()


def extract_frame_reports(
    frame: pd.DataFrame,
    *,
    config: dict | None = None,
) -> pd.DataFrame:
    """Run the extractor on each row's Report column. Gold columns are ignored."""
    assert_extractor_has_no_gold_parameter()
    cfg = config if config is not None else load_extractor_config()
    if REPORT_COLUMN not in frame.columns:
        raise KeyError(f"frame is missing {REPORT_COLUMN}")
    if STUDY_ID_COLUMN not in frame.columns:
        raise KeyError(f"frame is missing {STUDY_ID_COLUMN}")
    rows = []
    reports = frame[REPORT_COLUMN].tolist()
    ids = frame[STUDY_ID_COLUMN].astype(str).tolist()
    for study_id, report in zip(ids, reports):
        extraction = extract_labels_from_report(report, config=cfg)
        rows.append(extraction_to_row(study_id, extraction))
    return pd.DataFrame(rows)


def _safe_div(n: float, d: float) -> float:
    if d == 0:
        return float("nan")
    return n / d


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = sensitivity
    f1 = _safe_div(2 * precision * recall, precision + recall) if (
        precision == precision and recall == recall
    ) else float("nan")
    balanced = (
        (sensitivity + specificity) / 2.0
        if sensitivity == sensitivity and specificity == specificity
        else float("nan")
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced,
    }


def _confusion_for_target(
    gold: np.ndarray,
    pred: np.ndarray,
) -> dict[str, int | float]:
    """Unknown gold is skipped. Unknown predictions are abstentions, not negatives."""
    gold_known = np.isfinite(gold)
    pred_known = np.isfinite(pred)
    covered = gold_known & pred_known
    g = gold[covered]
    p = pred[covered]
    tp = int(((g == 1) & (p == 1)).sum())
    tn = int(((g == 0) & (p == 0)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    n_gold_known = int(gold_known.sum())
    n_pred_unknown = int((gold_known & ~pred_known).sum())
    n_gold_unknown = int((~gold_known).sum())
    metrics = _binary_metrics(tp, tn, fp, fn)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_gold_known": n_gold_known,
        "n_gold_unknown": n_gold_unknown,
        "n_pred_unknown_on_gold_known": n_pred_unknown,
        "n_covered": int(covered.sum()),
        "coverage": _safe_div(int(covered.sum()), n_gold_known) if n_gold_known else float("nan"),
        "n_gold_positive": int(((gold == 1) & gold_known).sum()),
        "n_gold_negative": int(((gold == 0) & gold_known).sum()),
        **metrics,
    }


def bootstrap_confidence_intervals(
    gold: np.ndarray,
    pred: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, dict[str, float]]:
    """Study-level percentile bootstrap. Rows are studies."""
    rng = np.random.default_rng(seed)
    n = gold.shape[0]
    metric_names = (
        "sensitivity", "specificity", "precision", "recall", "f1", "balanced_accuracy"
    )
    samples = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats = _confusion_for_target(gold[idx], pred[idx])
        for name in metric_names:
            value = stats[name]
            if value == value:
                samples[name].append(float(value))
    alpha = (1.0 - ci_level) / 2.0
    out: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray(samples[name], dtype=np.float64)
        if values.size == 0:
            out[name] = {"low": float("nan"), "high": float("nan"), "n_valid_resamples": 0}
            continue
        out[name] = {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1.0 - alpha)),
            "n_valid_resamples": int(values.size),
        }
    return out


def evaluate_gold_predictions(
    gold_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    *,
    config: dict,
) -> dict:
    """Compare extractor outputs to expert labels. Unknown gold is never a negative."""
    if STUDY_ID_COLUMN not in gold_frame.columns or STUDY_ID_COLUMN not in pred_frame.columns:
        raise KeyError(f"{STUDY_ID_COLUMN} is required on gold and prediction frames")
    gold_ids = gold_frame[STUDY_ID_COLUMN].astype(str)
    pred_indexed = pred_frame.copy()
    pred_indexed[STUDY_ID_COLUMN] = pred_indexed[STUDY_ID_COLUMN].astype(str)
    missing = [sid for sid in gold_ids if sid not in set(pred_indexed[STUDY_ID_COLUMN])]
    if missing:
        raise ValueError(f"predictions missing {len(missing)} gold StudyInstanceUID values")
    aligned = pred_indexed.set_index(STUDY_ID_COLUMN).loc[list(gold_ids)].reset_index()
    gold_labels, _gold_mask = extract_label_matrix(gold_frame, require_all=True)
    pred_labels, _pred_mask = extract_label_matrix(aligned, require_all=False)
    pred_frame = aligned
    if gold_labels.shape != pred_labels.shape:
        raise ValueError(f"shape mismatch gold {gold_labels.shape} vs pred {pred_labels.shape}")
    boot_cfg = config.get("bootstrap") or {}
    n_resamples = int(boot_cfg.get("n_resamples") or 1000)
    seed = int(boot_cfg.get("seed") or 42)
    ci_level = float(boot_cfg.get("ci_level") or 0.95)
    small_n = tuple(config.get("small_n_targets") or SMALL_N_DEFAULT)

    per_target: dict[str, dict] = {}
    error_categories: dict[str, int] = {}
    taxonomy_buckets: list[str] = []
    is_v2 = str(eval_protocol_version_of(config)).startswith("gold-eval-v2")
    for i, key in enumerate(TARGET_KEYS):
        official = OFFICIAL_TARGET_COLUMNS[i]
        stats = coverage_confusion(gold_labels[:, i], pred_labels[:, i])
        v1_stats = _confusion_for_target(gold_labels[:, i], pred_labels[:, i])
        stats["v1_conditional"] = {
            "sensitivity": v1_stats["sensitivity"],
            "specificity": v1_stats["specificity"],
            "precision": v1_stats["precision"],
            "recall": v1_stats["recall"],
            "f1": v1_stats["f1"],
            "balanced_accuracy": v1_stats["balanced_accuracy"],
            "coverage": v1_stats["coverage"],
        }
        stats["ci"] = bootstrap_coverage_intervals(
            gold_labels[:, i],
            pred_labels[:, i],
            n_resamples=n_resamples,
            seed=seed + i,
            ci_level=ci_level,
            metric_names=BOOTSTRAP_METRIC_NAMES,
        )
        # Preserve v1 CI keys under v1_conditional for direct comparison.
        stats["v1_conditional"]["ci"] = bootstrap_confidence_intervals(
            gold_labels[:, i],
            pred_labels[:, i],
            n_resamples=n_resamples,
            seed=seed + i,
            ci_level=ci_level,
        )
        conf_col = f"{key}_confidence"
        if conf_col in pred_frame.columns:
            conf = pd.to_numeric(pred_frame[conf_col], errors="coerce").to_numpy(dtype=np.float64)
            stats["coverage_error_curve_joint"] = coverage_error_curve(
                gold_labels[:, i], pred_labels[:, i], conf, polarity="joint"
            )
            stats["coverage_error_curve_positive"] = coverage_error_curve(
                gold_labels[:, i], pred_labels[:, i], conf, polarity="positive"
            )
            stats["coverage_error_curve_negative"] = coverage_error_curve(
                gold_labels[:, i], pred_labels[:, i], conf, polarity="negative"
            )
        stats["small_n_flag"] = key in small_n
        stats["target_key"] = key
        stats["official_column"] = official
        per_target[official] = stats

        gold_col = gold_labels[:, i]
        pred_col = pred_labels[:, i]
        evidence_col = pred_frame.get(f"{key}_evidence")
        rule_col = pred_frame.get(f"{key}_rule")
        diag_col = pred_frame.get(f"{key}_diagnostic")
        for row_i in range(len(gold_col)):
            g, p = gold_col[row_i], pred_col[row_i]
            if not np.isfinite(g):
                continue
            evidence = str(evidence_col.iloc[row_i]) if evidence_col is not None else ""
            rule = str(rule_col.iloc[row_i]) if rule_col is not None else ""
            diag = str(diag_col.iloc[row_i]) if diag_col is not None else None
            bucket = classify_evaluation_error(
                target_key=key,
                gold=float(g),
                pred=float(p) if np.isfinite(p) else float("nan"),
                evidence_category=evidence,
                rule_id=rule,
                diagnostic_class=diag,
            )
            if bucket:
                error_categories[bucket] = error_categories.get(bucket, 0) + 1
                taxonomy_buckets.append(bucket)

    def _macro(name: str) -> float:
        values = [per_target[col][name] for col in OFFICIAL_TARGET_COLUMNS]
        finite = [v for v in values if v == v]
        return float(np.mean(finite)) if finite else float("nan")

    macro_names = (
        "coverage", "precision", "npv", "positive_capture", "negative_capture",
        "error_rate_all_gold", "selective_accuracy",
        "conditional_sensitivity", "conditional_specificity",
        "sensitivity", "specificity", "recall", "f1", "balanced_accuracy",
    )
    macro = {name: _macro(name) for name in macro_names}
    return {
        "extractor_version": extractor_version_of(config),
        "eval_protocol_version": eval_protocol_version_of(config),
        "config_sha256": config.get("_sha256"),
        "n_gold_studies": int(len(gold_frame)),
        "unknown_imputed_as_negative": False,
        "conditional_sensitivity_is_not_full_set_sensitivity": True,
        "bootstrap": {
            "n_resamples": n_resamples,
            "ci_level": ci_level,
            "seed": seed,
            "unit": boot_cfg.get("unit") or "study",
            "metrics": list(BOOTSTRAP_METRIC_NAMES),
        },
        "per_target": per_target,
        "macro": macro,
        "small_n_targets": {
            TARGET_KEY_TO_OFFICIAL[key]: per_target[TARGET_KEY_TO_OFFICIAL[key]]
            for key in small_n
            if TARGET_KEY_TO_OFFICIAL[key] in per_target
        },
        "error_categories": error_categories,
        "error_taxonomy": taxonomy_payload(error_categories),
        "eval_protocol_is_v2": is_v2,
        "note": (
            "Unknown extractor outputs are abstentions and are excluded from "
            "TP/TN/FP/FN. Unknown gold labels are never treated as negative. "
            "conditional_sensitivity is TP/(TP+FN) among called cases only; "
            "positive_capture is TP/all gold positives. Do not treat "
            "conditional_sensitivity as recall on every gold-positive study."
        ),
    }


def weak_label_aggregates(pred_frame: pd.DataFrame) -> dict:
    labels, mask = extract_label_matrix(pred_frame, require_all=False)
    prev = prevalence(labels)
    conf_cols = {}
    for key in TARGET_KEYS:
        col = f"{key}_confidence"
        if col in pred_frame.columns:
            values = pd.to_numeric(pred_frame[col], errors="coerce").to_numpy(dtype=np.float64)
            conf_cols[key] = {
                "mean": float(np.nanmean(values)) if values.size else None,
                "p25": float(np.nanpercentile(values, 25)) if values.size else None,
                "p50": float(np.nanpercentile(values, 50)) if values.size else None,
                "p75": float(np.nanpercentile(values, 75)) if values.size else None,
            }
    language_counts = (
        pred_frame["language_code"].astype(str).value_counts().to_dict()
        if "language_code" in pred_frame.columns
        else {}
    )
    return {
        "n_studies": int(len(pred_frame)),
        "prevalence": prev,
        "confidence": conf_cols,
        "language_counts": language_counts,
        "unknown_imputed_as_negative": False,
    }


def new_evaluation_dir(base: Path, config: dict) -> Path:
    """Versioned directory that refuses to overwrite a previous run."""
    stamp = _utc_stamp()
    parent = (
        Path(base)
        / "evaluations"
        / extractor_version_of(config)
        / eval_protocol_version_of(config)
    )
    path = parent / stamp
    suffix = 0
    while path.exists():
        suffix += 1
        path = parent / f"{stamp}-{suffix:02d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def collect_identifiers(frame: pd.DataFrame) -> list[str]:
    values: list[str] = []
    for column in IDENTIFIER_COLUMN_CANDIDATES:
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                continue
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "nat"}:
                values.append(text)
    unique = sorted(set(values), key=len, reverse=True)
    return unique


def assert_no_identifiers(paths: list[Path], identifiers: list[str]) -> None:
    if not identifiers:
        return
    hits: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        blob = path.read_text(encoding="utf-8")
        for ident in identifiers:
            if ident and ident in blob:
                hits.append(f"{path.name}: identifier")
                break
    if hits:
        raise PrivacyLeakError(
            "sanitized outputs contain row-level identifiers: " + "; ".join(hits[:8])
        )


def write_private_csv(frame: pd.DataFrame, path: Path) -> None:
    assert_not_kaggle_input(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    drop = [col for col in frame.columns if str(col) in FORBIDDEN_TEXT_COLUMNS]
    out = frame.drop(columns=drop, errors="ignore")
    if REPORT_COLUMN in out.columns:
        raise PrivacyLeakError("refusing to write Report text to private label tables")
    out.to_csv(path, index=False)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.3f}"


def _fmt_ci(ci: dict | None, name: str) -> str:
    if not ci or name not in ci:
        return "n/a"
    low, high = ci[name].get("low"), ci[name].get("high")
    return f"{_fmt_metric(low)}–{_fmt_metric(high)}"


def _render_language_level_section(lang_level: dict) -> list[str]:
    """Aggregate language-level gold metrics. Never includes report text."""
    lines = [
        "### Language-level gold performance",
        "",
        "Reported only when a language has at least 8 gold studies. Other",
        "languages are listed as underpowered. Counts are study counts, not",
        "report quotations.",
        "",
        "| language | n_gold | reported |",
        "| --- | ---: | --- |",
    ]
    if not lang_level:
        lines.extend(["| n/a | 0 | no |", ""])
        return lines
    ranked = sorted(
        lang_level.items(),
        key=lambda item: (-int((item[1] or {}).get("n") or 0), str(item[0])),
    )
    for lang, payload in ranked:
        payload = payload or {}
        if payload.get("reported"):
            reported = "yes"
        else:
            reported = f"no ({payload.get('reason') or 'underpowered'})"
        lines.append(f"| {lang} | {payload.get('n', 0)} | {reported} |")
    for lang, payload in ranked:
        payload = payload or {}
        if not payload.get("reported"):
            continue
        lines.extend(
            [
                "",
                f"#### Language `{lang}` (n={payload.get('n')})",
                "",
                "| Official column | TP | TN | FP | FN | coverage | cond_sens | cond_spec | prec | f1 | bal acc |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        per_target = payload.get("per_target") or {}
        for official in OFFICIAL_TARGET_COLUMNS:
            stats = per_target.get(official) or {}
            lines.append(
                "| {col} | {tp} | {tn} | {fp} | {fn} | {cov} | {sens} | {spec} | {prec} | {f1} | {bal} |".format(
                    col=official,
                    tp=stats.get("tp", "n/a"),
                    tn=stats.get("tn", "n/a"),
                    fp=stats.get("fp", "n/a"),
                    fn=stats.get("fn", "n/a"),
                    cov=_fmt_metric(stats.get("coverage")),
                    sens=_fmt_metric(stats.get("conditional_sensitivity") or stats.get("sensitivity")),
                    spec=_fmt_metric(stats.get("conditional_specificity") or stats.get("specificity")),
                    prec=_fmt_metric(stats.get("precision")),
                    f1=_fmt_metric(stats.get("f1")),
                    bal=_fmt_metric(stats.get("balanced_accuracy")),
                )
            )
    lines.append("")
    return lines


def _render_error_category_section(errors: dict) -> list[str]:
    """Aggregate error-category counts. Categories never include quotations."""
    lines = [
        "### Aggregate error categories",
        "",
        "Counts only. Categories never include report quotations.",
        "",
        "| category | n |",
        "| --- | ---: |",
    ]
    if not errors:
        lines.extend(["| none | 0 |", ""])
        return lines
    for name in sorted(errors):
        lines.append(f"| {name} | {errors[name]} |")
    lines.append("")
    return lines


def render_sanitized_summary(
    *,
    language_audit: dict,
    gold_eval: dict | None,
    weak_agg: dict | None,
    gold_prevalence: dict | None,
    data_source: str,
    scientific_evaluation: bool,
    warnings: list[str],
    hashes: dict,
    config: dict,
) -> str:
    lines = [
        "# Report labeling summary",
        "",
    ]
    if not scientific_evaluation:
        lines.extend(
            [
                "**THIS RUN USED A SOFTWARE FIXTURE, NOT THE KAGGLE COMPETITION DATA.**",
                "These numbers are software verification only and must not be quoted",
                "as scientific gold-label performance.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Scientific evaluation on the expert-labelled training subset from the",
                "real Kaggle mount. Report text, StudyInstanceUID values, and PatientID",
                "values are not included.",
                "",
            ]
        )
    lines.extend(
        [
            f"- extractor_version: `{extractor_version_of(config)}`",
            f"- eval_protocol_version: `{eval_protocol_version_of(config)}`",
            f"- config_sha256: `{config.get('_sha256')}`",
            f"- data_source: `{data_source}`",
            f"- scientific_evaluation: `{str(scientific_evaluation).lower()}`",
            f"- unknown_imputed_as_negative: `false`",
            f"- external_language_model: `false`",
            f"- internet_used: `false`",
            f"- gpu_used: `false`",
            f"- training_run: `false`",
            "",
            "## Language audit (aggregate only)",
            "",
            f"- n_reports: {language_audit.get('n_reports')}",
            f"- n_empty: {language_audit.get('n_empty')}",
            f"- n_uncertain_language: {language_audit.get('n_uncertain_language')}",
            f"- n_unsupported_language: {language_audit.get('n_unsupported_language')}",
            f"- n_mixed_scripts: {language_audit.get('n_mixed_scripts')}",
            "",
            "| language | n | n_uncertain |",
            "| --- | ---: | ---: |",
        ]
    )
    lang_counts = language_audit.get("language_counts") or {}
    uncertain_counts = language_audit.get("uncertain_language_counts") or {}
    for lang in sorted(lang_counts, key=lambda item: (-int(lang_counts[item]), item)):
        lines.append(
            f"| {lang} | {lang_counts[lang]} | {uncertain_counts.get(lang, 0)} |"
        )
    lines.extend(
        [
            "",
            f"Scripts observed: `{json.dumps(language_audit.get('scripts_observed') or {}, sort_keys=True)}`",
            f"Quality flags: `{json.dumps(language_audit.get('quality_flag_counts') or {}, sort_keys=True)}`",
            "",
            language_audit.get("note") or "",
            "",
        ]
    )

    if gold_eval is None:
        lines.extend(["## Gold-label evaluation", "", "Not computed.", ""])
    else:
        lines.extend(
            [
                "## Gold-label evaluation",
                "",
                f"- n_gold_studies: {gold_eval.get('n_gold_studies')}",
                f"- bootstrap resamples: {(gold_eval.get('bootstrap') or {}).get('n_resamples')}",
                f"- CI level: {(gold_eval.get('bootstrap') or {}).get('ci_level')}",
                f"- unknown_imputed_as_negative: `false`",
                f"- conditional_sensitivity_is_not_full_set_sensitivity: `true`",
                "",
                "Conditional sensitivity/specificity are computed among called cases only.",
                "positive_capture = TP / all gold positives. negative_capture = TN / all gold negatives.",
                "Do not treat conditional_sensitivity as recall on every gold-positive study.",
                "",
                "| Official column | n_pos | n_neg | pos_calls | neg_calls | abstain | TP | TN | FP | FN | abst_pos | abst_neg | pos_capture | neg_capture | err_all | sel_acc | prec | NPV | coverage | cond_sens | cond_spec |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for official in OFFICIAL_TARGET_COLUMNS:
            stats = (gold_eval.get("per_target") or {}).get(official) or {}
            ci = stats.get("ci") or {}
            lines.append(
                "| {col} | {npos} | {nneg} | {pc} | {nc} | {ab} | {tp} | {tn} | {fp} | {fn} | {ap} | {an} | {pcap} | {ncap} | {err} | {sel} | {prec} | {npv} | {cov} | {csens} | {cspec} |".format(
                    col=official,
                    npos=stats.get("n_gold_positive", "n/a"),
                    nneg=stats.get("n_gold_negative", "n/a"),
                    pc=stats.get("n_positive_calls", "n/a"),
                    nc=stats.get("n_negative_calls", "n/a"),
                    ab=stats.get("n_abstentions", "n/a"),
                    tp=stats.get("tp", "n/a"),
                    tn=stats.get("tn", "n/a"),
                    fp=stats.get("fp", "n/a"),
                    fn=stats.get("fn", "n/a"),
                    ap=stats.get("n_abstained_positives", "n/a"),
                    an=stats.get("n_abstained_negatives", "n/a"),
                    pcap=_fmt_metric(stats.get("positive_capture")),
                    ncap=_fmt_metric(stats.get("negative_capture")),
                    err=_fmt_metric(stats.get("error_rate_all_gold")),
                    sel=_fmt_metric(stats.get("selective_accuracy")),
                    prec=_fmt_metric(stats.get("precision")),
                    npv=_fmt_metric(stats.get("npv")),
                    cov=_fmt_metric(stats.get("coverage")),
                    csens=_fmt_metric(stats.get("conditional_sensitivity")),
                    cspec=_fmt_metric(stats.get("conditional_specificity")),
                )
            )
        lines.extend(
            [
                "",
                "| Official column | coverage CI | precision CI | NPV CI | pos_capture CI | neg_capture CI |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for official in OFFICIAL_TARGET_COLUMNS:
            stats = (gold_eval.get("per_target") or {}).get(official) or {}
            ci = stats.get("ci") or {}
            lines.append(
                "| {col} | {cov} | {prec} | {npv} | {pcap} | {ncap} |".format(
                    col=official,
                    cov=_fmt_ci(ci, "coverage"),
                    prec=_fmt_ci(ci, "precision"),
                    npv=_fmt_ci(ci, "npv"),
                    pcap=_fmt_ci(ci, "positive_capture"),
                    ncap=_fmt_ci(ci, "negative_capture"),
                )
            )
        macro = gold_eval.get("macro") or {}
        lines.extend(
            [
                "",
                "Macro (unweighted mean over targets with defined metrics):",
                "",
            ]
        )
        for name in (
            "coverage", "positive_capture", "negative_capture",
            "error_rate_all_gold", "selective_accuracy",
            "precision", "npv",
            "conditional_sensitivity", "conditional_specificity",
        ):
            lines.append(f"- {name}: {_fmt_metric(macro.get(name))}")
        lines.extend(
            [
                "",
                "v1-style conditional aliases (among called cases only, not full-set):",
                "",
            ]
        )
        for name in ("sensitivity", "specificity", "recall", "f1", "balanced_accuracy"):
            lines.append(f"- v1_alias_{name}: {_fmt_metric(macro.get(name))}")
        lines.extend(["", "### Small-n targets (MCL, Lateral OA, Baker's)", ""])
        for official, stats in (gold_eval.get("small_n_targets") or {}).items():
            lines.append(
                f"- **{official}**: gold positives={stats.get('n_gold_positive')}, "
                f"coverage={_fmt_metric(stats.get('coverage'))}, "
                f"conditional_sensitivity={_fmt_metric(stats.get('conditional_sensitivity'))} "
                f"(not full-set), "
                f"conditional_specificity={_fmt_metric(stats.get('conditional_specificity'))}, "
                f"positive_capture={_fmt_metric(stats.get('positive_capture'))}, "
                f"NPV={_fmt_metric(stats.get('npv'))}. "
                "n is small; intervals are wide."
            )
        lines.extend(["", gold_eval.get("note") or "", ""])
        lines.extend(_render_language_level_section(gold_eval.get("language_level") or {}))
        lines.extend(_render_error_category_section(gold_eval.get("error_categories") or {}))
        if gold_eval.get("error_taxonomy"):
            lines.extend(
                [
                    "### Error taxonomy class counts",
                    "",
                    "Closed vocabulary. No quotations.",
                    "",
                    "| class | n |",
                    "| --- | ---: |",
                ]
            )
            for name, n in (gold_eval["error_taxonomy"].get("class_counts") or {}).items():
                lines.append(f"| {name} | {n} |")
            lines.append("")
        if gold_eval.get("trainable_status"):
            lines.extend(
                [
                    "### Per-target trainable status",
                    "",
                    "Statuses use precision, NPV, coverage, confidence intervals and",
                    "prevalence plausibility. There is no single macro threshold.",
                    "",
                    "| Official column | status | precision | NPV | coverage | pos_capture | inflated |",
                    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for official in OFFICIAL_TARGET_COLUMNS:
                row = (gold_eval.get("trainable_status") or {}).get(official) or {}
                lines.append(
                    "| {col} | {st} | {p} | {n} | {c} | {pc} | {inf} |".format(
                        col=official,
                        st=row.get("status", "n/a"),
                        p=_fmt_metric(row.get("precision")),
                        n=_fmt_metric(row.get("npv")),
                        c=_fmt_metric(row.get("coverage")),
                        pc=_fmt_metric(row.get("positive_capture")),
                        inf=str(row.get("inflated_prevalence")).lower(),
                    )
                )
            lines.append("")
        if gold_eval.get("second_reader"):
            sr = gold_eval["second_reader"]
            lines.extend(
                [
                    "### Second reader",
                    "",
                    f"- enabled: `{sr.get('enabled')}`",
                    f"- skipped_reason: `{sr.get('skipped_reason')}`",
                    f"- weights_downloaded: `{sr.get('weights_downloaded')}`",
                    "",
                ]
            )

    if weak_agg is None:
        lines.extend(["## Weak labels (report-only studies)", "", "Not generated.", ""])
    else:
        lines.extend(
            [
                "## Weak labels (report-only studies)",
                "",
                f"- n_report_only_studies: {weak_agg.get('n_studies')}",
                "",
                "| Official column | n_known | n_positive | n_negative | n_unknown | unknown rate | pos rate among known |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        prev = weak_agg.get("prevalence") or {}
        n_studies = int(weak_agg.get("n_studies") or 0)
        for official in OFFICIAL_TARGET_COLUMNS:
            stats = prev.get(official) or {}
            n_unknown = stats.get("n_unknown")
            unknown_rate = _safe_div(n_unknown, n_studies) if n_studies else float("nan")
            lines.append(
                f"| {official} | {stats.get('n_known', 'n/a')} | {stats.get('n_positive', 'n/a')} | "
                f"{stats.get('n_negative', 'n/a')} | {n_unknown} | {_fmt_metric(unknown_rate)} | "
                f"{_fmt_metric(stats.get('positive_rate_among_known'))} |"
            )
        lines.extend(["", "Confidence distribution (mean / p50):", ""])
        conf = weak_agg.get("confidence") or {}
        for key in TARGET_KEYS:
            item = conf.get(key) or {}
            lines.append(
                f"- {TARGET_KEY_TO_OFFICIAL[key]}: mean={_fmt_metric(item.get('mean'))}, "
                f"p50={_fmt_metric(item.get('p50'))}"
            )
        lines.append("")

    if gold_prevalence and weak_agg:
        lines.extend(
            [
                "## Expert vs weak-label prevalence",
                "",
                "| Official column | expert pos rate (n=gold known) | weak pos rate (among known weak) |",
                "| --- | ---: | ---: |",
            ]
        )
        weak_prev = weak_agg.get("prevalence") or {}
        for official in OFFICIAL_TARGET_COLUMNS:
            g = (gold_prevalence.get(official) or {}).get("positive_rate_among_known")
            w = (weak_prev.get(official) or {}).get("positive_rate_among_known")
            lines.append(f"| {official} | {_fmt_metric(g)} | {_fmt_metric(w)} |")
        lines.append("")

    lines.extend(["## Warnings", ""])
    if not warnings:
        lines.append("- None recorded.")
    else:
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Private output hashes (contents not committed)",
            "",
            "| file | sha256 | bytes |",
            "| --- | --- | ---: |",
        ]
    )
    for name, meta in hashes.items():
        lines.append(f"| `{name}` | `{meta.get('sha256')}` | {meta.get('bytes')} |")
    lines.extend(
        [
            "",
            "Row-level weak labels, gold predictions, language assignments, and report",
            "text remain in gitignored / Kaggle working outputs only.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def hash_private_outputs(paths: dict[str, Path]) -> dict:
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if path.is_file():
            out[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            out[name] = {"sha256": None, "bytes": 0}
    return out


def _not_available_result(pre: PreflightResult) -> dict:
    return {
        "status": PreflightError.status,
        "data_source": pre.data_source,
        "scientific_evaluation": False,
        "internet_used": False,
        "gpu_required": False,
        "gpu_used": False,
        "training_run": False,
        "folds_created": False,
        "preflight": pre.to_dict(),
        "gold_eval": None,
        "weak_aggregates": None,
        "language_audit": None,
        "second_reader_enabled": False,
        "second_reader_weights_downloaded": False,
        "note": "DATA_NOT_AVAILABLE: competition files were not found.",
    }


def run_notebook_report_labeling(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    layout: KaggleLayout | None = None,
    data_source: str | None = None,
    config_path: str | Path | None = None,
    extract_source: bool = False,
    source_root: str | Path | None = None,
) -> dict:
    """Kaggle/fixture entry point. Never trains an image model. Never prints reports."""
    layout = layout or KaggleLayout(output_dirname=KAGGLE_REPORT_OUTPUT_DIRNAME)
    pre = preflight(
        layout=layout,
        data_root=data_root,
        source_root=source_root,
        output_dir=output_dir,
        data_source=data_source,
        extract_source=extract_source,
    )
    out_dir = Path(output_dir) if output_dir is not None else pre.output_dir
    assert_not_kaggle_input(out_dir, pre.layout)
    private_dir = out_dir / "private"
    sanitized_dir = out_dir / "sanitized"
    eval_root = out_dir / "eval_archive"
    private_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    config = load_extractor_config(config_path)
    source = data_source or pre.data_source
    scientific = scientific_evaluation_for(source) and source == DATA_SOURCE_REAL_KAGGLE
    if source == DATA_SOURCE_SOFTWARE_FIXTURE:
        scientific = False

    snippets: list[str] = []
    identifiers: list[str] = []
    warnings: list[str] = []

    if not pre.ok or pre.data_root is None:
        result = _not_available_result(pre)
        result["scientific_evaluation"] = False
        summary = render_sanitized_summary(
            language_audit={"n_reports": 0, "note": "not computed"},
            gold_eval=None,
            weak_agg=None,
            gold_prevalence=None,
            data_source=source,
            scientific_evaluation=False,
            warnings=["AUDIT STOPPED: required inputs missing."],
            hashes={},
            config=config,
        )
        (sanitized_dir / SANITIZED_SUMMARY_FILENAME).write_text(summary, encoding="utf-8")
        result["output_dir"] = str(out_dir)
        result["output_files"] = {"summary": str(sanitized_dir / SANITIZED_SUMMARY_FILENAME)}
        return result

    root = Path(pre.data_root)
    guard = ReadOnlyInputGuard([root])
    train_path = root / "train.csv"
    train = pd.read_csv(train_path)
    snippets = collect_report_snippets(train)
    identifiers = collect_identifiers(train)

    gold_frame, report_only = split_gold_and_report_only(train)
    reports = train[REPORT_COLUMN].tolist() if REPORT_COLUMN in train.columns else []
    language_audit = aggregate_language_audit(
        reports,
        supported=tuple((config.get("languages") or {}).get("supported") or []),
        uncertain_margin=float((config.get("languages") or {}).get("uncertain_margin") or 0.15),
        min_score=float((config.get("languages") or {}).get("min_score") or 2.0),
    )

    all_pred = extract_frame_reports(train, config=config)
    gold_ids = set(gold_frame[STUDY_ID_COLUMN].astype(str)) if len(gold_frame) else set()
    report_only_ids = set(report_only[STUDY_ID_COLUMN].astype(str)) if len(report_only) else set()
    gold_pred = (
        all_pred[all_pred[STUDY_ID_COLUMN].astype(str).isin(gold_ids)].copy()
        if len(all_pred) and gold_ids
        else pd.DataFrame()
    )
    weak_pred = (
        all_pred[all_pred[STUDY_ID_COLUMN].astype(str).isin(report_only_ids)].copy()
        if len(all_pred) and report_only_ids
        else pd.DataFrame()
    )

    gold_eval = None
    gold_prev = None
    operational_thresholds: dict[str, dict[str, float]] = {}
    if len(gold_frame) and len(gold_pred):
        gold_eval = evaluate_gold_predictions(gold_frame, gold_pred, config=config)
        gold_labels, _ = extract_label_matrix(gold_frame)
        gold_prev = prevalence(gold_labels)
        lang_perf = language_level_gold_performance(gold_frame, gold_pred, gold_eval)
        gold_eval["language_level"] = lang_perf
        if str(eval_protocol_version_of(config)).startswith("gold-eval-v2"):
            from .report_confidence import evaluate_oof_all_targets
            from .report_acceptance import decide_all_targets

            pred_labels, _ = extract_label_matrix(gold_pred, require_all=False)
            conf_map = {}
            for key in TARGET_KEYS:
                col = f"{key}_confidence"
                conf_map[key] = (
                    pd.to_numeric(gold_pred[col], errors="coerce").to_numpy(dtype=np.float64)
                    if col in gold_pred.columns
                    else np.zeros(len(gold_pred), dtype=np.float64)
                )
            aligned_gold = gold_pred.set_index(gold_pred[STUDY_ID_COLUMN].astype(str))
            ordered = aligned_gold.loc[list(gold_frame[STUDY_ID_COLUMN].astype(str))]
            pred_labels, _ = extract_label_matrix(ordered.reset_index(drop=True), require_all=False)
            for key in TARGET_KEYS:
                col = f"{key}_confidence"
                conf_map[key] = (
                    pd.to_numeric(ordered[col], errors="coerce").to_numpy(dtype=np.float64)
                    if col in ordered.columns
                    else np.zeros(len(ordered), dtype=np.float64)
                )
            oof_sanitized, operational_thresholds = evaluate_oof_all_targets(
                gold_labels,
                pred_labels,
                conf_map,
                gold_frame[STUDY_ID_COLUMN].astype(str).tolist(),
                config=config,
            )
            gold_eval["oof_confidence_policies"] = oof_sanitized
            gold_eval["operational_thresholds_for_weak_labels_only"] = operational_thresholds
            gold_eval["trainable_status"] = decide_all_targets(
                gold_eval.get("per_target") or {},
                weak_prev=(weak_label_aggregates(weak_pred).get("prevalence") if len(weak_pred) else None),
            )
            from .report_second_reader import run_second_reader

            reader_probe = run_second_reader(
                "", config=config, language_supported=True
            )
            gold_eval["second_reader"] = {
                "enabled": reader_probe.enabled,
                "skipped_reason": reader_probe.skipped_reason,
                "weights_downloaded": False,
            }

    weak_agg = weak_label_aggregates(weak_pred) if len(weak_pred) else None

    if language_audit.get("n_unsupported_language"):
        warnings.append(
            f"{language_audit['n_unsupported_language']} reports had unsupported/undetermined "
            "language codes; those studies abstain except language-agnostic abbreviations."
        )
    if language_audit.get("n_uncertain_language"):
        warnings.append(
            f"{language_audit['n_uncertain_language']} reports had uncertain language classification."
        )
    if gold_eval:
        for official, stats in (gold_eval.get("small_n_targets") or {}).items():
            warnings.append(
                f"{official} has only {stats.get('n_gold_positive')} gold positives; "
                "intervals are wide and performance is unstable."
            )
        for official, stats in (gold_eval.get("per_target") or {}).items():
            unknown_rate = 1.0 - float(stats.get("coverage") or 0.0) if stats.get("coverage") == stats.get("coverage") else 1.0
            if unknown_rate >= 0.5:
                warnings.append(
                    f"{official} abstained on {unknown_rate:.0%} of gold studies (high uncertainty)."
                )

    private_paths = {
        PRIVATE_WEAK_FILENAME: private_dir / PRIVATE_WEAK_FILENAME,
        PRIVATE_GOLD_PRED_FILENAME: private_dir / PRIVATE_GOLD_PRED_FILENAME,
        PRIVATE_LANGUAGE_FILENAME: private_dir / PRIVATE_LANGUAGE_FILENAME,
    }
    if len(weak_pred):
        weak_out = weak_pred.drop(
            columns=[c for c in weak_pred.columns if c.endswith("_evidence") or c.endswith("_rule") or c.endswith("_diagnostic")],
            errors="ignore",
        )
        write_private_csv(weak_out, private_paths[PRIVATE_WEAK_FILENAME])
    if len(gold_pred):
        write_private_csv(gold_pred, private_paths[PRIVATE_GOLD_PRED_FILENAME])
    if len(all_pred):
        lang_cols = [
            STUDY_ID_COLUMN,
            "language_code",
            "language_uncertain",
            "heuristic_language",
            "ngram_language",
            "language_detectors_agree",
            "quality_flags",
            "extractor_version",
        ]
        present = [col for col in lang_cols if col in all_pred.columns]
        write_private_csv(all_pred.loc[:, present], private_paths[PRIVATE_LANGUAGE_FILENAME])

    if operational_thresholds:
        from .report_masks import MASK_FILENAMES, build_trainable_masks, write_mask_csvs

        pos_t = {k: v["positive_threshold"] for k, v in operational_thresholds.items()}
        neg_t = {k: v["negative_threshold"] for k, v in operational_thresholds.items()}
        mask_source = weak_pred if len(weak_pred) else all_pred
        if len(mask_source):
            masks = build_trainable_masks(
                mask_source,
                positive_thresholds=pos_t,
                negative_thresholds=neg_t,
            )
            written = write_mask_csvs(masks, private_dir)
            for name, path in written.items():
                private_paths[name] = path

    hashes = hash_private_outputs(private_paths)
    eval_dir = new_evaluation_dir(eval_root, config)
    eval_payload = jsonable(gold_eval) if gold_eval else None
    (eval_dir / SANITIZED_METRICS_FILENAME).write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (eval_dir / SANITIZED_ERRORS_FILENAME).write_text(
        json.dumps((gold_eval or {}).get("error_categories") or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = render_sanitized_summary(
        language_audit=language_audit,
        gold_eval=gold_eval,
        weak_agg=weak_agg,
        gold_prevalence=gold_prev,
        data_source=source,
        scientific_evaluation=scientific,
        warnings=warnings,
        hashes=hashes,
        config=config,
    )
    sanitized_paths = {
        SANITIZED_SUMMARY_FILENAME: sanitized_dir / SANITIZED_SUMMARY_FILENAME,
        SANITIZED_METRICS_FILENAME: sanitized_dir / SANITIZED_METRICS_FILENAME,
        SANITIZED_LANGUAGE_FILENAME: sanitized_dir / SANITIZED_LANGUAGE_FILENAME,
        SANITIZED_WEAK_FILENAME: sanitized_dir / SANITIZED_WEAK_FILENAME,
        SANITIZED_ERRORS_FILENAME: sanitized_dir / SANITIZED_ERRORS_FILENAME,
        SANITIZED_HASHES_FILENAME: sanitized_dir / SANITIZED_HASHES_FILENAME,
        SANITIZED_MANIFEST_FILENAME: sanitized_dir / SANITIZED_MANIFEST_FILENAME,
    }
    sanitized_paths[SANITIZED_SUMMARY_FILENAME].write_text(summary, encoding="utf-8")
    sanitized_paths[SANITIZED_METRICS_FILENAME].write_text(
        json.dumps(jsonable(gold_eval), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sanitized_paths[SANITIZED_LANGUAGE_FILENAME].write_text(
        json.dumps(jsonable(language_audit), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sanitized_paths[SANITIZED_WEAK_FILENAME].write_text(
        json.dumps(jsonable(weak_agg), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sanitized_paths[SANITIZED_ERRORS_FILENAME].write_text(
        json.dumps(jsonable((gold_eval or {}).get("error_categories")), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sanitized_paths[SANITIZED_HASHES_FILENAME].write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extractor_version": extractor_version_of(config),
        "eval_protocol_version": eval_protocol_version_of(config),
        "config_sha256": config.get("_sha256"),
        "data_source": source,
        "scientific_evaluation": scientific,
        "internet_used": False,
        "gpu_used": False,
        "training_run": False,
        "folds_created": False,
        "n_train_studies": int(len(train)),
        "n_gold_studies": int(len(gold_frame)),
        "n_report_only_studies": int(len(report_only)),
        "eval_archive_dir": str(eval_dir),
        "private_outputs": {name: str(path) for name, path in private_paths.items()},
        "sanitized_outputs": {name: str(path) for name, path in sanitized_paths.items()},
        "private_hashes": hashes,
    }
    sanitized_paths[SANITIZED_MANIFEST_FILENAME].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sanitized_list = list(sanitized_paths.values()) + [eval_dir / SANITIZED_METRICS_FILENAME]
    assert_no_report_text(sanitized_list, snippets)
    assert_no_identifiers(sanitized_list, identifiers)

    guard.assert_unchanged()
    result = {
        "status": "LABELLED" if scientific or source == DATA_SOURCE_SOFTWARE_FIXTURE else "LABELLED",
        "data_source": source,
        "scientific_evaluation": scientific,
        "internet_used": False,
        "gpu_required": False,
        "gpu_used": False,
        "training_run": False,
        "folds_created": False,
        "n_train_studies": int(len(train)),
        "n_gold_studies": int(len(gold_frame)),
        "n_report_only_studies": int(len(report_only)),
        "gold_eval": jsonable(gold_eval),
        "weak_aggregates": jsonable(weak_agg),
        "language_audit": jsonable(language_audit),
        "output_dir": str(out_dir),
        "eval_archive_dir": str(eval_dir),
        "output_files": {path.name: str(path) for path in sanitized_list if path.is_file()},
        "private_hashes": hashes,
        "warnings": warnings,
        "extractor_version": extractor_version_of(config),
        "eval_protocol_version": eval_protocol_version_of(config),
        "second_reader_enabled": bool((config.get("second_reader") or {}).get("enabled")),
        "second_reader_weights_downloaded": bool(
            ((gold_eval or {}).get("second_reader") or {}).get("weights_downloaded")
        ),
        "preflight": pre.to_dict(),
        "note": (
            "THIS RUN USED A SOFTWARE FIXTURE, NOT THE KAGGLE COMPETITION DATA."
            if not scientific
            else "Real Kaggle mount report labeling. Row-level outputs are private."
        ),
    }
    if not scientific:
        result["claim_scientific_performance"] = False
    else:
        result["claim_scientific_performance"] = True
    return result


def language_level_gold_performance(
    gold_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    gold_eval: dict,
    *,
    min_n: int = 8,
) -> dict:
    """Per-language gold metrics where sample size permits. No report text."""
    if "language_code" not in pred_frame.columns:
        return {}
    merged = pred_frame[[STUDY_ID_COLUMN, "language_code"]].copy()
    gold_ids = gold_frame[STUDY_ID_COLUMN].astype(str).tolist()
    merged = merged.set_index(STUDY_ID_COLUMN)
    gold_labels, _ = extract_label_matrix(gold_frame)
    pred_labels, _ = extract_label_matrix(pred_frame)
    out: dict[str, dict] = {}
    for lang, group in pred_frame.groupby("language_code"):
        n = int(len(group))
        if n < min_n:
            out[str(lang)] = {"n": n, "reported": False, "reason": f"n<{min_n}"}
            continue
        idx = [
            gold_ids.index(str(sid))
            for sid in group[STUDY_ID_COLUMN].astype(str).tolist()
            if str(sid) in gold_ids
        ]
        if len(idx) < min_n:
            out[str(lang)] = {"n": len(idx), "reported": False, "reason": f"n<{min_n}"}
            continue
        per_target = {}
        for i, official in enumerate(OFFICIAL_TARGET_COLUMNS):
            per_target[official] = _confusion_for_target(gold_labels[idx, i], pred_labels[idx, i])
        out[str(lang)] = {"n": len(idx), "reported": True, "per_target": per_target}
    return out
