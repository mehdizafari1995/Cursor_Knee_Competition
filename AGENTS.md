# Agent Instructions

RSNA Knee Abnormality Detection repository. The main Cursor Agent acts as
**coordinator**: it plans work, delegates to the project subagents in
`.cursor/agents/`, merges their output, and is the only party that grants approvals.

## The task

Study-level multi-label knee MRI: each study is scored for **twelve independent
binary findings**, evaluated by macro-averaged ROC AUC. Training labels may be
weakly supervised from free-text multilingual radiology reports; inference is
image-only.

`TARGET_KEYS` in `knee/config.py` is the single source of truth for the twelve
targets and their internal order. `docs/clinical_ontology_schema.md`
mirrors it, and the test suite fails if the two drift apart. No agent adds, drops or
renames a target. Official Kaggle `sample_submission.csv` headers use different
spelling (`OFFICIAL_TARGET_COLUMNS` in `knee/config.py`); competition submissions
must use that spelling without renaming `TARGET_KEYS`.

## Subagents

| Agent | Role | Read-only | Background |
| --- | --- | --- | --- |
| `literature-searcher` | PubMed/PMC retrieval with reproducible, logged queries | no | yes |
| `clinical-ontology` | Structured, fully cited ontology for the twelve targets | no | yes |
| `evidence-verifier` | Independent audit of citations, claims and licences | yes | no |
| `competition-data-auditor` | CSV/DICOM schema, missingness and leakage audit | yes | no |
| `repository-verifier` | Runs tests and confirms claimed code exists and works | yes | no |

Delegate discovery and writing work to the background agents; block on the
verifiers before accepting any result.

## Policies

These are binding on the coordinator and every subagent. Full text in
`docs/literature_protocol.md` and `docs/citation_and_figure_policy.md`.

1. Literature retrieval uses the PubMed E-utilities and PMC APIs, with every query,
   date and hit count logged so the search is reproducible.
2. "All papers" means all records returned by the documented search strategies on
   the documented dates. It is never a claim that every publication was found.
3. Article figures are retrieved only when full-text and figure access is legal.
4. Every retrieved figure records its licence and canonical source URL.
5. Literature figures live in `literature/figures/` and never mix with Kaggle
   training images under `data/`.
6. Literature figures are not used for model training without licence review,
   patient-level label verification and explicit coordinator approval.
7. PDFs, figures, DICOM files, reports, credentials and model checkpoints are never
   committed. `.gitignore` enforces this.
8. Parallel writing work uses isolated git worktrees or separate branches.
9. Every agent returns: evidence, file paths, tests run with results, and unresolved
   questions.

## Development

```bash
bash .cursor/install.sh        # creates .venv and installs pinned dependencies
source .venv/bin/activate
pytest                         # pyproject.toml points this at tests/
python -m knee.pipeline        # synthetic end-to-end run, writes to outputs/

# Competition data (not shipped). Never hard-code a machine path.
export RSNA_DATA_ROOT=/path/to/rsna-knee-abnormality-detection
python scripts/audit_competition_data.py
python scripts/create_folds.py
python scripts/train_baseline.py --smoke    # software verification only
python scripts/predict_baseline.py --smoke
python scripts/make_submission.py --predictions ... --sample-submission ...

# Phase 3: Kaggle real-data audit notebook (no API token; internet off on Kaggle)
python scripts/build_kaggle_bundle.py   # allowlisted zip under outputs/ (gitignored)
# Upload notebooks/01_kaggle_real_data_audit.ipynb on Kaggle with the
# competition dataset and the private source zip attached. See
# reports/kaggle_notebook_instructions.md

# Phase 4: private multilingual report-label generation (no image training)
python scripts/evaluate_report_labeler.py   # 58 gold studies when RSNA_DATA_ROOT is set
python scripts/generate_weak_labels.py      # 4349 report-only studies; gitignored CSV
# Upload notebooks/02_kaggle_report_labeling.ipynb (internet off, GPU off).
# Retrieve sanitized aggregates only. See reports/report_labeling_protocol.md
# Coordinator gates: NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING,
# READY_FOR_SELECTIVE_LABEL_REFINEMENT
# (reports/weak_label_training_readiness.md). Do not train on hard 0/1 weak labels.

# Phase 4B: coverage-aware evaluation + report-labeler v2 (no image training)
# Fixture tests in Cursor. Real gold/weak evaluation on Kaggle:
# notebooks/03_kaggle_report_labeler_v2.ipynb (internet off). See
# reports/report_labeler_v2_protocol.md

# Phase 4C: offline multilingual second-reader gold pilot (no image training)
# Fixture tests in Cursor. Real 58-study GPU run on Kaggle, internet off:
# notebooks/04_kaggle_second_reader_gold_pilot.ipynb
# Do not process the 4,349 report-only studies. See
# reports/second_reader_pilot_protocol.md
# Phase 4C.1: diagnose v7 PILOT_EXECUTION_FAIL, v1.1 transport/parser,
# notebooks/05_kaggle_second_reader_synthetic_canary.ipynb before any gold rerun.
```

Run the whole suite, not one file. `tests/test_pipeline.py` covers the modelling
pipeline; `tests/test_agent_configuration.py` validates the agent definitions, the
policy docs and the target registry. Run both after editing anything under
`.cursor/agents/`, `docs/` or `knee/config.py`. Phase 2 adds tests for audit,
folds/leakage, labels, metrics, DICOM loading, submission format, and CPU smoke.
Phase 3 adds a Kaggle internet-off audit notebook, allowlisted source packaging,
and tests for path detection, provenance, deterministic sampling, report-text
exclusion, bundle allowlists, and missing-data failure. Phase 4 adds a
deterministic report-label extractor, gold-label evaluation with bootstrap
intervals, weak-label generation for report-only studies, and privacy tests
that fail if report text or identifiers enter sanitized artefacts. After the
real gold evaluation the coordinator gates are
`NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING` and
`READY_FOR_SELECTIVE_LABEL_REFINEMENT`. Phase 4B
adds coverage-aware capture/abstention metrics, target-specific OOF confidence
policies, an independent offline language detector, private trainable masks,
and per-target TRAINABLE / NOT_TRAINABLE statuses. It does not train an image
model. Phase 4C adds an offline multilingual instruction-model second reader
on the 58 gold studies only, with a frozen prompt/schema, structured-output
parser, strict-agreement consensus, and PILOT_PASS / PILOT_RESTRICTED /
PILOT_FAIL statuses. Kernel version 7 is `PILOT_EXECUTION_FAIL` (all
second-reader outputs malformed JSON). Protocol v1.1 repairs transport/parsing
only. It does not process the 4,349 report-only studies and
does not train an image model.
