"""High-precision report-label extractor, gold evaluation, and privacy controls."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    REPORT_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEYS,
)
from knee.report_labels import (
    EXTRACTOR_FORBIDDEN_PARAMS,
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    HARD_WEAK_LABEL_TRAINING_VERDICT,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_UNKNOWN,
    PrivacyLeakError,
    SELECTIVE_LABEL_REFINEMENT_ALLOWED,
    SELECTIVE_LABEL_REFINEMENT_VERDICT,
    HardWeakLabelTrainingError,
    assert_extractor_has_no_gold_parameter,
    assert_hard_weak_label_training_forbidden,
    assert_no_identifiers,
    collect_identifiers,
    evaluate_gold_predictions,
    extract_labels_from_report,
    load_extractor_config,
    new_evaluation_dir,
    run_notebook_report_labeling,
    split_gold_and_report_only,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    make_report_labeling_fixture,
)


def _extract(text: str):
    config = load_extractor_config()
    return extract_labels_from_report(text, config=config)


def test_config_targets_match_target_keys_in_order():
    config = load_extractor_config()
    assert tuple(config["targets"]) == TARGET_KEYS
    assert config["unknown_policy"] == "never_impute_as_negative"
    assert config["gold_labels_in_extractor"] == "forbidden"
    assert config["no_external_language_model"] is True
    assert NUM_TARGETS == 12


def test_extractor_signature_has_no_gold_parameters():
    assert_extractor_has_no_gold_parameter()
    params = set(inspect.signature(extract_labels_from_report).parameters)
    assert not (params & EXTRACTOR_FORBIDDEN_PARAMS)
    assert REPORT_COLUMN.lower() not in params or "report" in params


def test_acl_tear_positive_and_intact_negative():
    pos = _extract("FINDINGS: Complete ACL tear.")
    assert pos.predictions["acl_injury"].label == LABEL_POSITIVE
    neg = _extract("FINDINGS: The ACL is intact. No MCL injury.")
    assert neg.predictions["acl_injury"].label == LABEL_NEGATIVE
    assert neg.predictions["mcl_injury"].label == LABEL_NEGATIVE


def test_spanish_acl_and_french_effusion():
    es = _extract("Hallazgos: Rotura del LCA. Sin fractura.")
    assert es.predictions["acl_injury"].label == LABEL_POSITIVE
    assert es.predictions["fracture"].label == LABEL_NEGATIVE
    fr = _extract("Constatations: Rupture du LCA. Epanchement articulaire.")
    assert fr.predictions["acl_injury"].label == LABEL_POSITIVE
    assert fr.predictions["joint_effusion"].label == LABEL_POSITIVE


def test_uncertainty_and_history_are_unknown_not_negative():
    unc = _extract("FINDINGS: Possible ACL tear.")
    assert unc.predictions["acl_injury"].label == LABEL_UNKNOWN
    hist = _extract("HISTORY: prior ACL tear. FINDINGS: ACL graft is intact.")
    assert hist.predictions["acl_injury"].label == LABEL_UNKNOWN


def test_meniscus_laterality_and_unspecified_meniscus():
    both = _extract("FINDINGS: Medial meniscus tear. Lateral meniscus is intact.")
    assert both.predictions["medial_meniscus_injury"].label == LABEL_POSITIVE
    assert both.predictions["lateral_meniscus_injury"].label == LABEL_NEGATIVE
    unspecified = _extract("FINDINGS: Meniscus tear.")
    assert unspecified.predictions["medial_meniscus_injury"].label == LABEL_UNKNOWN
    assert unspecified.predictions["lateral_meniscus_injury"].label == LABEL_UNKNOWN


def test_tricompartmental_oa_and_unspecified_oa():
    tri = _extract("FINDINGS: Tricompartmental osteoarthritis.")
    assert tri.predictions["medial_compartment_oa"].label == LABEL_POSITIVE
    assert tri.predictions["lateral_compartment_oa"].label == LABEL_POSITIVE
    assert tri.predictions["patellofemoral_oa"].label == LABEL_POSITIVE
    unspecified = _extract("FINDINGS: Osteoarthritis of the knee.")
    assert unspecified.predictions["medial_compartment_oa"].label == LABEL_UNKNOWN


def test_effusion_synovitis_and_baker_are_distinct():
    baker = _extract("FINDINGS: Baker's cyst. No joint effusion.")
    assert baker.predictions["baker_cyst"].label == LABEL_POSITIVE
    assert baker.predictions["joint_effusion"].label == LABEL_NEGATIVE
    assert baker.predictions["synovitis"].label != LABEL_POSITIVE
    syn = _extract("FINDINGS: Synovial thickening without joint effusion.")
    assert syn.predictions["synovitis"].label == LABEL_POSITIVE
    assert syn.predictions["joint_effusion"].label == LABEL_NEGATIVE
    combined = _extract("FINDINGS: effusion-synovitis.")
    assert combined.predictions["joint_effusion"].label == LABEL_UNKNOWN
    assert combined.predictions["synovitis"].label == LABEL_UNKNOWN


def test_trace_effusion_is_unknown_large_is_positive():
    trace = _extract("FINDINGS: Trace joint effusion.")
    assert trace.predictions["joint_effusion"].label == LABEL_UNKNOWN
    large = _extract("FINDINGS: Large joint effusion.")
    assert large.predictions["joint_effusion"].label == LABEL_POSITIVE


def test_no_mention_is_unknown_not_negative():
    result = _extract("FINDINGS: The patellar tendon is unremarkable.")
    assert result.predictions["acl_injury"].label == LABEL_UNKNOWN
    assert result.predictions["fracture"].label == LABEL_UNKNOWN


def test_global_normal_is_negative():
    result = _extract("IMPRESSION: Normal MRI of the knee. No significant abnormality.")
    assert result.predictions["acl_injury"].label == LABEL_NEGATIVE
    assert result.predictions["baker_cyst"].label == LABEL_NEGATIVE


def test_each_prediction_has_required_fields():
    result = _extract("FINDINGS: ACL tear.")
    pred = result.predictions["acl_injury"]
    assert pred.label in {LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNKNOWN}
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.evidence_category
    assert pred.rule_id
    assert pred.extractor_version
    assert REPORT_COLUMN not in pred.evidence_category


def test_gold_split_does_not_feed_labels_to_extractor(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "data")
    train = pd.read_csv(root / "train.csv")
    gold, weak = split_gold_and_report_only(train)
    assert len(gold) == 12
    assert len(weak) == 4
    assert gold.loc[:, list(OFFICIAL_TARGET_COLUMNS)].notna().all().all()
    assert weak.loc[:, list(OFFICIAL_TARGET_COLUMNS)].isna().all().all()
    # Extractor is called with report text only.
    report = gold.iloc[0][REPORT_COLUMN]
    extract_labels_from_report(report)


def test_evaluate_never_imputes_unknown_as_negative():
    config = load_extractor_config()
    config["bootstrap"] = {**config["bootstrap"], "n_resamples": 30}
    gold = pd.DataFrame(
        {
            STUDY_ID_COLUMN: ["s1", "s2"],
            "ACL": [1, 0],
            **{col: 0 for col in OFFICIAL_TARGET_COLUMNS if col != "ACL"},
        }
    )
    pred = pd.DataFrame(
        {
            STUDY_ID_COLUMN: ["s1", "s2"],
            "ACL": [np.nan, 0],
            **{col: np.nan for col in OFFICIAL_TARGET_COLUMNS if col != "ACL"},
        }
    )
    metrics = evaluate_gold_predictions(gold, pred, config=config)
    acl = metrics["per_target"]["ACL"]
    assert metrics["unknown_imputed_as_negative"] is False
    assert acl["fn"] == 0  # abstention is not a false negative
    assert acl["n_pred_unknown_on_gold_known"] == 1
    assert acl["tp"] + acl["tn"] + acl["fp"] + acl["fn"] == acl["n_covered"]


def test_versioned_evaluation_dir_does_not_overwrite(tmp_path):
    config = load_extractor_config()
    first = new_evaluation_dir(tmp_path, config)
    second = new_evaluation_dir(tmp_path, config)
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert config["extractor_version"] in str(first)
    assert config["eval_protocol_version"] in str(first)


def test_assert_no_identifiers_raises(tmp_path):
    path = tmp_path / "sanitized.md"
    path.write_text("study " + REPORT_LABEL_GOLD_UIDS[0] + "\n")
    with pytest.raises(PrivacyLeakError):
        assert_no_identifiers([path], list(REPORT_LABEL_GOLD_UIDS))


def test_hard_weak_label_training_is_forbidden():
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SELECTIVE_LABEL_REFINEMENT_ALLOWED is True
    assert HARD_WEAK_LABEL_TRAINING_VERDICT == "NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING"
    assert SELECTIVE_LABEL_REFINEMENT_VERDICT == "READY_FOR_SELECTIVE_LABEL_REFINEMENT"
    with pytest.raises(HardWeakLabelTrainingError, match="NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING"):
        assert_hard_weak_label_training_forbidden()


def test_committed_readiness_report_has_both_verdicts():
    path = Path(__file__).resolve().parents[1] / "reports" / "weak_label_training_readiness.md"
    text = path.read_text(encoding="utf-8")
    assert "`NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`" in text
    assert "`READY_FOR_SELECTIVE_LABEL_REFINEMENT`" in text
    assert "`READY_FOR_HARD_WEAK_LABEL_TRAINING`" not in text
    assert "NOT_READY_FOR_SELECTIVE_LABEL_REFINEMENT" not in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
