# Clinical Ontology Schema

Structure that the `clinical-ontology` agent must produce, one record per competition
target. The `evidence-verifier` agent validates records against this schema and
against `docs/citation_and_figure_policy.md`.

## 1. Target registry

RSNA Knee Abnormality Detection is a **study-level multi-label** problem: each knee
MRI study is scored for **twelve independent binary findings**, and the leaderboard
metric is the macro-averaged ROC AUC across them.

The single source of truth for the twelve targets is `TARGET_KEYS` and `TARGET_NAMES`
in `knee/config.py`, which also define the `submission.csv` column order. This table
mirrors that tuple, in the same order. It is never edited independently: change
`knee/config.py` first, then this table.
`tests/test_agent_configuration.py` fails if the two drift apart.

| Slot | `target_id` (submission column) | Label | Ontology record |
| --- | --- | --- | --- |
| 1 | `acl_injury` | ACL injury | pending |
| 2 | `mcl_injury` | MCL injury | pending |
| 3 | `medial_meniscus_injury` | Medial meniscus injury | pending |
| 4 | `lateral_meniscus_injury` | Lateral meniscus injury | pending |
| 5 | `medial_compartment_oa` | Medial-compartment osteoarthritis | pending |
| 6 | `lateral_compartment_oa` | Lateral-compartment osteoarthritis | pending |
| 7 | `patellofemoral_oa` | Patellofemoral osteoarthritis | pending |
| 8 | `joint_effusion` | Joint effusion | pending |
| 9 | `synovitis` | Synovitis | pending |
| 10 | `baker_cyst` | Baker cyst | pending |
| 11 | `bone_contusion` | Bone contusion | pending |
| 12 | `fracture` | Fracture | pending |

`clinical-ontology` populates one record per row and flips `Ontology record` to
`drafted`, then to `verified` once `evidence-verifier` returns a clean report. It
must never add, drop or rename a target: a target that is not in `knee/config.py`
does not exist for this project, and any mismatch against the official competition
data dictionary is escalated to the coordinator rather than fixed in this table.

Two scope notes that bound the ontology work:

- **Weak supervision.** Training labels may be derived from free-text multilingual
  radiology reports, so the `negation_cues` and `uncertainty_terms` fields directly
  affect label quality and are not merely descriptive. Collect them in every report
  language the data contains.
- **Image-only inference.** Test-time prediction consumes MRI images only, with no
  report text. Ontology terminology therefore supports label extraction and error
  analysis, never an inference-time feature.

## 2. Files

```
docs/ontology/targets/<target_id>.json   machine-readable record, source of truth
docs/ontology/targets/<target_id>.md     human-readable rendering of the same record
docs/ontology/index.json                 registry of target_id -> file, with status
```

## 3. Record structure

```json
{
  "target_id": "string, lowercase snake_case, matches the registry",
  "competition_column": "string, verbatim label column name",
  "definition": {"text": "string", "citations": ["<citation>"]},
  "synonyms": [{"term": "string", "citations": ["<citation>"]}],
  "abbreviations": [{"short": "string", "long": "string", "citations": ["<citation>"]}],
  "anatomy": [{"structure": "string", "compartment": "string|null",
               "laterality_relevant": true, "citations": ["<citation>"]}],
  "mri_findings": [{"finding": "string",
                    "signal": "string|null",
                    "morphology": "string|null",
                    "secondary_signs": ["string"],
                    "citations": ["<citation>"]}],
  "sequences": [{"sequence": "string", "plane": "axial|coronal|sagittal|any",
                 "why_informative": "string", "citations": ["<citation>"]}],
  "negation_cues": [{"phrase": "string", "scope": "string", "citations": ["<citation>"]}],
  "uncertainty_terms": [{"phrase": "string",
                         "strength": "low|moderate|high",
                         "citations": ["<citation>"]}],
  "postoperative_terms": [{"term": "string",
                           "procedure": "string|null",
                           "expected_appearance": "string|null",
                           "citations": ["<citation>"]}],
  "grading_or_severity": [{"scale": "string", "levels": ["string"],
                           "citations": ["<citation>"]}],
  "differentials": [{"entity": "string", "distinguishing_feature": "string",
                     "citations": ["<citation>"]}],
  "disagreements": [{"issue": "string",
                     "positions": [{"position": "string", "citations": ["<citation>"]}]}],
  "notes": [{"text": "string", "type": "inference|caveat|todo"}],
  "provenance": {"created_at_utc": "ISO-8601",
                 "updated_at_utc": "ISO-8601",
                 "corpus_snapshot": "path to literature/records.jsonl revision"},
  "verification": {"status": "unverified|verified|rejected",
                   "verified_at_utc": "ISO-8601|null",
                   "report": "path|null"}
}
```

`<citation>` is the object defined in `docs/citation_and_figure_policy.md`
(`pmid`, `pmcid`, `doi`, `source_url`, `licence`, `quote`, `locator`).

## 4. Field rules

- **Citations are per claim.** Every entry in every list carries its own non-empty
  `citations` array. An entry whose citations array is empty is invalid and is
  deleted rather than kept.
- **No invented citations.** Identifiers must exist in `literature/records.jsonl`.
- **Inference is labelled.** Anything not stated by a source goes in `notes` with
  `"type": "inference"` and never into a cited field.
- **Conflicts are preserved.** Contradictory sources go in `disagreements` with both
  positions cited. Do not silently choose.
- **Negation vs uncertainty are distinct.** `negation_cues` assert absence ("no
  evidence of", "intact", "without"). `uncertainty_terms` hedge ("possible",
  "cannot exclude", "equivocal", "suspected"). A term never appears in both.
- **Postoperative terminology is separate** from native-anatomy findings, because
  expected postoperative appearance can mimic pathology.
- **Completeness language is bounded.** Field coverage is described as "terms found
  by the documented searches", never as "all terms".
- **No figures.** Ontology records reference literature by identifier only. Figures
  are governed by `docs/citation_and_figure_policy.md` and are not embedded here.

## 5. Lifecycle

1. `literature-searcher` populates `literature/` under `docs/literature_protocol.md`.
2. `competition-data-auditor` reconciles the twelve `TARGET_KEYS` in `knee/config.py`
   against the label columns of the real competition data and reports any mismatch
   to the coordinator.
3. `clinical-ontology` drafts records; `verification.status` starts as `unverified`.
4. `evidence-verifier` audits and returns per-claim verdicts; unsupported claims are
   removed by `clinical-ontology`, not by the verifier.
5. Coordinator promotes records to `verified` only after a clean verification report.

## 6. What an agent returns

Evidence (per-target field coverage and citation counts), file paths written, tests
run with results, and unresolved questions.
