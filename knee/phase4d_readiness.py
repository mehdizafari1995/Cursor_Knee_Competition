"""Phase 4D read-only competition-readiness review.

Maps the score-bearing image pipeline, inventories existing evidence, and
selects one next experiment. Never trains, never infers, never launches a
Kaggle kernel, and never writes a competition submission.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .metrics import macro_roc_auc
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    HARD_WEAK_LABEL_TRAINING_VERDICT,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_VERDICT,
)
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha
from .submission import align_to_sample, load_sample_submission

PROTOCOL = "phase-4d-competition-readiness-review"
STATUS_COMPLETE = "READINESS_REVIEW_COMPLETE"
SELECTED_EXPERIMENT_ID = "gold58-image-resnet18-grouped-oof"

SECOND_READER_SCORE_BEARING_ALLOWED = False
SECOND_READER_PARKED_VERDICT = "SECOND_READER_PARKED_NOT_SCORE_BEARING"
PHASE4C_CLOSED = True
PHASE4D_EXPERIMENT_EXECUTED = False

FROZEN_PHASE4C_KERNEL = "mahdimzafari/rsna-knee-second-reader-gold-constrained-v14"
FROZEN_PHASE4C_KERNEL_VERSION = 1
FROZEN_V14_CONFIG_SHA256 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"

SAVED_PREDICTION_CANDIDATES = (
    PROJECT_ROOT / "outputs" / "baseline" / "predictions.csv",
    PROJECT_ROOT / "outputs" / "submission.csv",
    PROJECT_ROOT / "outputs" / "baseline" / "oof_predictions.csv",
)

SCORE_BEARING_STEPS = (
    "RSNA_DATA_ROOT competition DICOMs (image-only at test time)",
    "knee.dataset.StudySliceDataset (one series per study, linspace slices)",
    "knee.baseline_model.BaselineStudyModel (ImageNet ResNet-18, mean slice pool, 12 logits)",
    "sigmoid probabilities",
    "scripts/predict_baseline.py predictions.csv",
    "scripts/make_submission.py + knee.submission.write_competition_submission",
    "submission.csv with StudyInstanceUID + OFFICIAL_TARGET_COLUMNS, sample row order",
)

NON_SCORE_BEARING = (
    "Phase 4C second reader (parked; report-time only; not image inference)",
    "synthetic python -m knee.pipeline write_submission (study_id + TARGET_KEYS)",
    "report-labeler weak labels used as hard 0/1",
    "CPU smoke TinySliceEncoder metrics",
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def reproduce_metrics_from_predictions(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
) -> dict[str, Any]:
    """Recompute macro ROC AUC from saved prediction and gold tables. No model."""
    pred = predictions.copy()
    gold_frame = gold.copy()
    pred[STUDY_ID_COLUMN] = pred[STUDY_ID_COLUMN].astype(str)
    gold_frame[STUDY_ID_COLUMN] = gold_frame[STUDY_ID_COLUMN].astype(str)
    aligned = pred.set_index(STUDY_ID_COLUMN).loc[list(gold_frame[STUDY_ID_COLUMN])].reset_index()
    y_true = gold_frame.loc[:, list(OFFICIAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    y_prob = aligned.loc[:, list(OFFICIAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    macro, per_target = macro_roc_auc(y_true, y_prob)
    return {
        "n_studies": int(len(gold_frame)),
        "macro_roc_auc": None if macro != macro else float(macro),
        "per_target_roc_auc": {
            official: (None if value != value else float(value))
            for official, value in zip(OFFICIAL_TARGET_COLUMNS, per_target)
        },
        "unknown_imputed_as_negative": False,
        "model_inference_run": False,
        "scientific_evaluation": False,
        "note": "Recomputed from caller-supplied tables only. Not a leaderboard score.",
    }


def inventory_saved_score_artifacts(output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root or (PROJECT_ROOT / "outputs"))
    found: list[dict[str, Any]] = []
    for path in SAVED_PREDICTION_CANDIDATES:
        if path.is_file():
            found.append(
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    extra_smoke = list(root.glob("**/smoke_metrics.json"))
    return {
        "competition_prediction_files": found,
        "n_competition_prediction_files": len(found),
        "n_smoke_metric_files_under_outputs": int(len(extra_smoke)),
        "real_image_checkpoints_present": False,
        "ensembles_present": False,
        "public_leaderboard_score_present": False,
        "private_leaderboard_score_present": False,
        "fold_tables_committed": False,
    }


def ranked_backlog() -> list[dict[str, Any]]:
    """Highest expected leaderboard value per unit compute first.

    Selection does not inspect Phase 4C second-reader gold error patterns.
    """
    return [
        {
            "rank": 1,
            "id": SELECTED_EXPERIMENT_ID,
            "selected": True,
            "component": (
                "scripts/train_baseline.py + knee.baseline_model.BaselineStudyModel "
                "+ knee.folds.assign_grouped_folds; gold-only 58-study image OOF"
            ),
            "why_selected": (
                "No image checkpoint exists. The only permitted labelled image "
                "training set is the 58 fully labelled studies. Weak labels and "
                "the second reader are gated off. This is the cheapest path to a "
                "non-random image-only scorer."
            ),
            "leakage_safe_validation": (
                "5-fold grouped CV. Group by PatientID if every selected study "
                "has one, else StudyInstanceUID. assert_no_group_leakage. Train "
                "only on labelled studies. Never use report-only rows, weak 0/1 "
                "imputation, or second-reader outputs. Image-only inputs."
            ),
            "expected_score_mechanism": (
                "OOF macro ROC AUC on the 58 labelled studies is a noisy "
                "internal estimate. It is not a public leaderboard score. A "
                "later submission (not this experiment) would fit on all 58 "
                "and predict the hidden test inside a Kaggle scoring notebook."
            ),
            "compute_estimate": (
                "58 studies, 8 slices, 224px, ResNet-18, 5 folds × a few epochs. "
                "Tesla T4: about 20–90 minutes. CPU: several hours. Dataset "
                "access is Kaggle-side (~570 GB); this VM cannot hold the tree."
            ),
            "failure_and_rollback": (
                "Abort on group leakage, report-only rows, unknown-as-0, or "
                "second-reader inputs. If OOF AUC is undefined for most targets, "
                "stop. Delete gitignored checkpoints. Do not submit. Keep "
                "HARD_WEAK_LABEL_TRAINING_ALLOWED=false."
            ),
            "consumes_submission_slot": False,
            "datasets": (
                "Competition train DICOMs and the 58 fully labelled train.csv "
                "target columns. No report-only labels. No hidden-test labels. "
                "No second-reader files."
            ),
            "authorized_to_run_in_phase4d": False,
        },
        {
            "rank": 2,
            "id": "multi-series-3plane-fusion-on-gold58",
            "selected": False,
            "component": "knee.dataset series selection; fuse sagittal/coronal/axial",
            "why_selected": "Mean 5.53 series/study; current loader uses one series.",
            "leakage_safe_validation": "Same 5-fold grouped OOF as rank 1.",
            "expected_score_mechanism": "Better anatomy coverage vs single-series baseline.",
            "compute_estimate": "Roughly 3× rank 1.",
            "failure_and_rollback": "Keep single-series checkpoint; do not submit.",
            "consumes_submission_slot": False,
            "datasets": "Same 58 labelled studies plus series tables.",
            "authorized_to_run_in_phase4d": False,
        },
        {
            "rank": 3,
            "id": "stronger-encoder-on-gold58",
            "selected": False,
            "component": "ResNet-50 / EfficientNet encoder swap in BaselineStudyModel",
            "why_selected": "Capacity after a working ResNet-18 OOF exists.",
            "leakage_safe_validation": "Same grouped OOF.",
            "expected_score_mechanism": "Possible small AUC gain; overfit risk at n=58.",
            "compute_estimate": "Higher than rank 1; still T4-hours, not days.",
            "failure_and_rollback": "Revert encoder; keep ResNet-18.",
            "consumes_submission_slot": False,
            "datasets": "Same 58 labelled studies.",
            "authorized_to_run_in_phase4d": False,
        },
        {
            "rank": 4,
            "id": "slice-count-and-tta",
            "selected": False,
            "component": "num_slices and test-time flips in predict_baseline.py",
            "why_selected": "Cheap once a checkpoint exists.",
            "leakage_safe_validation": "Apply TTA only at OOF/test; no extra labels.",
            "expected_score_mechanism": "Variance reduction of slice sampling.",
            "compute_estimate": "2–8× inference cost.",
            "failure_and_rollback": "Disable TTA.",
            "consumes_submission_slot": False,
            "datasets": "Same images as the parent checkpoint.",
            "authorized_to_run_in_phase4d": False,
        },
        {
            "rank": 5,
            "id": "selective-weak-label-refinement-then-masked-training",
            "selected": False,
            "component": "report-labeler v2 high-confidence masks + masked BCE",
            "why_selected": "4349 report-only studies exist, but hard 0/1 training is gated off.",
            "leakage_safe_validation": "Requires a new coordinator gate; still never impute unknown as 0.",
            "expected_score_mechanism": "More labelled cells if precision/NPV recover.",
            "compute_estimate": "Full train tree on Kaggle GPU; hours.",
            "failure_and_rollback": "Keep HARD_WEAK_LABEL_TRAINING_ALLOWED=false.",
            "consumes_submission_slot": False,
            "datasets": "Reports + images; blocked until the hard-weak gate lifts.",
            "authorized_to_run_in_phase4d": False,
            "blocked_by": HARD_WEAK_LABEL_TRAINING_VERDICT,
        },
        {
            "rank": 6,
            "id": "public-external-mri-pretrain",
            "selected": False,
            "component": "rules-compliant public MRI checkpoint attached offline",
            "why_selected": "Rules allow public pretrained models; licence/access not verified here.",
            "leakage_safe_validation": "Pretrain off competition patients; fine-tune with grouped OOF.",
            "expected_score_mechanism": "Better features than ImageNet for MRI.",
            "compute_estimate": "Days if training from scratch; hours if attaching a public checkpoint.",
            "failure_and_rollback": "Fall back to ImageNet ResNet-18.",
            "consumes_submission_slot": False,
            "datasets": "Public external MRI only; competition DICOMs for fine-tune.",
            "authorized_to_run_in_phase4d": False,
        },
    ]


def selected_experiment() -> dict[str, Any]:
    return next(item for item in ranked_backlog() if item["selected"])


def proposed_execution_command() -> dict[str, Any]:
    return {
        "executed": PHASE4D_EXPERIMENT_EXECUTED,
        "implemented": False,
        "do_not_run_in_phase4d": True,
        "consumes_submission_slot": False,
        "command": (
            "python scripts/train_gold58_image_oof.py "
            "--config configs/baseline.yaml --folds 5 --seed 42 "
            "--labelled-only --no-weak-labels --no-second-reader"
        ),
        "current_blocker": (
            "scripts/train_baseline.py refuses full training even with "
            "--allow-training because HARD_WEAK_LABEL_TRAINING_ALLOWED is false. "
            "A gold-only trainer that never touches weak labels or the second "
            "reader is not implemented. Phase 4D does not add or run it."
        ),
        "preflight": [
            "HARD_WEAK_LABEL_TRAINING_ALLOWED is false",
            "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is false",
            "SECOND_READER_SCORE_BEARING_ALLOWED is false",
            "select exactly the 58 fully labelled studies; report-only count stays 0",
            "group folds and assert_no_group_leakage",
            "unknown labels stay NaN; masked BCE; never impute 0",
            "image-only tensors; no Report column in the model",
            "official submission path unused in this experiment",
            "internet off if executed on Kaggle; no new kernel in Phase 4D",
            "STOP: do not run the command in this phase",
        ],
    }


def leakage_and_contract_audit() -> dict[str, Any]:
    return {
        "train_vs_example_test_study_overlap": 0,
        "duplicate_train_study_ids": 0,
        "patient_id_complete_on_train_headers": True,
        "unique_patient_id_equals_unique_study_id": True,
        "patient_may_be_exam_scoped": True,
        "patient_grouped_folds_allowed_gate": False,
        "real_fold_tables_created": False,
        "example_test_n_studies": 3,
        "hidden_test_size_documented_not_local": "~1300",
        "test_reports_available": False,
        "inference_must_be_image_only": True,
        "official_vs_internal_spelling_matches": False,
        "official_vs_internal_order_matches": True,
        "competition_submission_rewrites_internal_keys": True,
        "synthetic_pipeline_submission_is_not_competition_contract": True,
        "metric_implementation": (
            "sklearn roc_auc_score per target, skip single-class and NaN, "
            "macro mean of defined targets"
        ),
        "official_kaggle_evaluator_source_not_in_repo": True,
        "second_reader_on_score_path": False,
        "unknown_imputed_as_negative_in_score_path": False,
        "nondeterminism_notes": (
            "baseline.yaml seed 42; synthetic train_model shuffles batches; "
            "DICOM slice order is deterministic (IOP normal, InstanceNumber, name)"
        ),
        "series_selection_policy": "caller-supplied; no default plane/sequence chooser",
        "local_vm_cannot_hold_full_dicom_tree": True,
    }


def existing_validation_evidence() -> dict[str, Any]:
    return {
        "public_leaderboard_score": None,
        "private_leaderboard_score": None,
        "real_image_oof_macro_auc": None,
        "real_image_checkpoint": None,
        "ensemble": None,
        "postprocessing": "sigmoid only on the unimplemented real predict path",
        "software_smoke_macro_auc": "present only when scripts/train_baseline.py --smoke is run; scientific_evaluation false",
        "synthetic_pipeline_macro_auc": "CI fixture only; not competition data",
        "phase3_audit": "AUDITED; 4407 train studies; 58 fully labelled; 4349 report-only; folds_created false; training_run false",
        "phase4_weak_labels": HARD_WEAK_LABEL_TRAINING_VERDICT,
        "phase4c_second_reader": SECOND_READER_PARKED_VERDICT,
        "submission_slots_used_this_repo": 0,
    }


def run_phase4d_readiness_review(
    *,
    predictions_path: str | Path | None = None,
    gold_path: str | Path | None = None,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only review. Optional saved tables may be scored; no generate()."""
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert PHASE4C_CLOSED is True
    assert PHASE4D_EXPERIMENT_EXECUTED is False

    artifacts = inventory_saved_score_artifacts()
    reproduced = None
    reproduce_status = "NO_SAVED_COMPETITION_PREDICTIONS"
    if predictions_path and gold_path:
        pred = pd.read_csv(predictions_path)
        gold = pd.read_csv(gold_path)
        if sample_path:
            sample = load_sample_submission(Path(sample_path))
            pred = align_to_sample(pred, sample)
        reproduced = reproduce_metrics_from_predictions(pred, gold)
        reproduce_status = "REPRODUCED_FROM_SAVED_TABLES"
    elif artifacts["n_competition_prediction_files"] == 0:
        reproduce_status = "NO_SAVED_COMPETITION_PREDICTIONS"

    selected = selected_experiment()
    payload = {
        "protocol": PROTOCOL,
        "status": STATUS_COMPLETE,
        "ok": True,
        "phase4c_closed": PHASE4C_CLOSED,
        "second_reader_parked_verdict": SECOND_READER_PARKED_VERDICT,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "training_run": False,
        "inference_run": False,
        "kaggle_kernel_launched": False,
        "competition_submission_written": False,
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "n_report_only_studies_processed": 0,
        "gold_rerun_authorized": False,
        "phase4d_experiment_executed": PHASE4D_EXPERIMENT_EXECUTED,
        "score_bearing_system": "image-only BaselineStudyModel (ResNet-18); not the second reader",
        "score_bearing_steps": list(SCORE_BEARING_STEPS),
        "non_score_bearing": list(NON_SCORE_BEARING),
        "frozen_phase4c": {
            "kernel": FROZEN_PHASE4C_KERNEL,
            "version": FROZEN_PHASE4C_KERNEL_VERSION,
            "v14_config_sha256": FROZEN_V14_CONFIG_SHA256,
            "medial_meniscus_pilot_pass_is_exploratory_only": True,
            "not_sufficient_for_deployment": True,
        },
        "existing_validation_evidence": existing_validation_evidence(),
        "saved_artifacts": artifacts,
        "metric_reproduction_status": reproduce_status,
        "reproduced_metrics": reproduced,
        "leakage_and_contract": leakage_and_contract_audit(),
        "ranked_backlog": ranked_backlog(),
        "selected_experiment": selected,
        "proposed_execution": proposed_execution_command(),
        "readiness_blockers": [
            "No real image checkpoint or OOF score exists",
            "Full training loop is not implemented; train_baseline.py returns 3 even with --allow-training",
            "HARD_WEAK_LABEL_TRAINING_ALLOWED=false (keep); do not train on 4349 weak labels as hard 0/1",
            "SECOND_READER_SCORE_BEARING_ALLOWED=false; 11/12 second-reader targets FAIL/RESTRICTED",
            "Medial Meniscus PILOT_PASS is exploratory: coverage 10/58, NPV denom=2",
            "patient_grouped_folds_allowed remains a closed coordinator gate; PatientID is 1:1 with studies",
            "No default series-selection policy (one series per study, caller-supplied)",
            "Example test is 3 rows; hidden test is not local; full DICOM tree does not fit this VM",
            "Synthetic write_submission uses study_id + TARGET_KEYS and must not be uploaded",
            "Official Kaggle evaluator source is not in the repo",
        ],
        "utc": _utc(),
        "git_head_sha": git_head_sha(),
    }
    return jsonable(payload)


def render_parked_decision_markdown(payload: dict[str, Any] | None = None) -> str:
    payload = payload or {}
    return "\n".join(
        [
            "# Phase 4C closed: second reader parked",
            "",
            "Phase 4C format engineering succeeded. The second reader is **not**",
            "eligible for hard weak-label training, report-only scaling, production",
            "routing, or score-bearing submissions.",
            "",
            f"- verdict: `{SECOND_READER_PARKED_VERDICT}`",
            f"- `SECOND_READER_SCORE_BEARING_ALLOWED`: `{str(SECOND_READER_SCORE_BEARING_ALLOWED).lower()}`",
            f"- `HARD_WEAK_LABEL_TRAINING_ALLOWED`: `{str(HARD_WEAK_LABEL_TRAINING_ALLOWED).lower()}`",
            f"- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED`: `{str(SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED).lower()}`",
            f"- frozen kernel: `{FROZEN_PHASE4C_KERNEL}` version {FROZEN_PHASE4C_KERNEL_VERSION}",
            f"- frozen v1.4 config SHA-256: `{FROZEN_V14_CONFIG_SHA256}`",
            "- 11 of 12 targets are `PILOT_FAIL` or `PILOT_RESTRICTED` under strict agreement",
            "- Medial Meniscus remains a formal `PILOT_PASS`, but coverage is 10/58",
            "  and the negative decision denominator is 2. Record it as exploratory",
            "  evidence only, not sufficient validation for deployment.",
            "- No additional gold inference or gold-driven tuning is authorized",
            "- Notebook 07 stays a historical blocked stub; notebook 09 is not re-run",
            "- Do not process the 4,349 report-only studies",
            f"- Forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
            "Phase 4D reviews the **image-only** baseline as the score-bearing system.",
            "",
        ]
    )


def render_readiness_markdown(payload: dict[str, Any]) -> str:
    selected = payload.get("selected_experiment") or {}
    proposed = payload.get("proposed_execution") or {}
    lines = [
        "# Phase 4D competition-readiness review",
        "",
        "Read-only. No training, inference, Kaggle kernel, or submission.",
        "The second reader is excluded from score-bearing behaviour.",
        "",
        f"- status: `{payload.get('status')}`",
        f"- score-bearing system: {payload.get('score_bearing_system')}",
        f"- second reader: `{payload.get('second_reader_parked_verdict')}`",
        f"- metric_reproduction_status: `{payload.get('metric_reproduction_status')}`",
        f"- phase4d_experiment_executed: `{str(payload.get('phase4d_experiment_executed')).lower()}`",
        f"- training_run / inference_run / submission: `false` / `false` / `false`",
        "",
        "## Score-bearing path",
        "",
    ]
    for step in payload.get("score_bearing_steps") or []:
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "## Why the second reader is parked",
            "",
            "See `reports/phase4c_second_reader_parked.md`. Eleven of twelve",
            "targets are FAIL/RESTRICTED. Medial Meniscus `PILOT_PASS` is",
            "exploratory (coverage 10/58, NPV denom=2), not deployment evidence.",
            "",
            "## Existing validation evidence",
            "",
        ]
    )
    evidence = payload.get("existing_validation_evidence") or {}
    for key, value in evidence.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Leakage and contract audit", ""])
    leak = payload.get("leakage_and_contract") or {}
    for key, value in leak.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Readiness blockers", ""])
    for item in payload.get("readiness_blockers") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Ranked improvement backlog", ""])
    for row in payload.get("ranked_backlog") or []:
        mark = "SELECTED" if row.get("selected") else f"rank {row.get('rank')}"
        lines.extend(
            [
                f"### {row.get('id')} ({mark})",
                "",
                f"- component: {row.get('component')}",
                f"- leakage-safe validation: {row.get('leakage_safe_validation')}",
                f"- expected score mechanism: {row.get('expected_score_mechanism')}",
                f"- compute: {row.get('compute_estimate')}",
                f"- failure/rollback: {row.get('failure_and_rollback')}",
                f"- consumes submission slot: `{str(row.get('consumes_submission_slot')).lower()}`",
                f"- datasets: {row.get('datasets')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Selected next experiment (not run)",
            "",
            f"- id: `{selected.get('id')}`",
            f"- why: {selected.get('why_selected')}",
            f"- authorized_to_run_in_phase4d: `{str(selected.get('authorized_to_run_in_phase4d')).lower()}`",
            f"- consumes_submission_slot: `{str(selected.get('consumes_submission_slot')).lower()}`",
            "",
            "## Preflight checklist and proposed command",
            "",
            f"- executed: `{str(proposed.get('executed')).lower()}`",
            f"- implemented: `{str(proposed.get('implemented')).lower()}`",
            f"- command (do not run): `{proposed.get('command')}`",
            f"- current blocker: {proposed.get('current_blocker')}",
            "",
        ]
    )
    for item in proposed.get("preflight") or []:
        lines.append(f"- [ ] {item}")
    lines.extend(
        [
            "",
            "## Coordinator gates",
            "",
            f"- `{HARD_WEAK_LABEL_TRAINING_VERDICT}`",
            f"- `{SCALE_TO_REPORT_ONLY_SECOND_READER_VERDICT}`",
            f"- `{SECOND_READER_PARKED_VERDICT}`",
            "- do not merge PR #10",
        ]
    )
    return "\n".join(lines) + "\n"
