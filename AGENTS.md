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
```

Run the whole suite, not one file. `tests/test_pipeline.py` covers the modelling
pipeline; `tests/test_agent_configuration.py` validates the agent definitions, the
policy docs and the target registry. Run both after editing anything under
`.cursor/agents/`, `docs/` or `knee/config.py`. Phase 2 adds tests for audit,
folds/leakage, labels, metrics, DICOM loading, submission format, and CPU smoke.
