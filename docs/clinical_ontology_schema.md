# Clinical Ontology Schema

Structure that the `clinical-ontology` agent must produce, one record per competition
target. The `evidence-verifier` agent validates records against this schema and
against `docs/citation_and_figure_policy.md`.

## 1. Target registry

The competition defines **twelve** prediction targets. They are transcribed verbatim
from the official competition data dictionary and confirmed against the label columns
of the training CSV by the `competition-data-auditor` agent. They are not invented,
guessed or renamed here.

| Slot | `target_id` | Competition column name | Source of truth | Status |
| --- | --- | --- | --- | --- |
| 1 | TBD | TBD | competition data dictionary | unpopulated |
| 2 | TBD | TBD | competition data dictionary | unpopulated |
| 3 | TBD | TBD | competition data dictionary | unpopulated |
| 4 | TBD | TBD | competition data dictionary | unpopulated |
| 5 | TBD | TBD | competition data dictionary | unpopulated |
| 6 | TBD | TBD | competition data dictionary | unpopulated |
| 7 | TBD | TBD | competition data dictionary | unpopulated |
| 8 | TBD | TBD | competition data dictionary | unpopulated |
| 9 | TBD | TBD | competition data dictionary | unpopulated |
| 10 | TBD | TBD | competition data dictionary | unpopulated |
| 11 | TBD | TBD | competition data dictionary | unpopulated |
| 12 | TBD | TBD | competition data dictionary | unpopulated |

Populating this table is a prerequisite for ontology work. Until a slot is filled,
`clinical-ontology` must stop and ask the coordinator rather than assume a target.

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
2. `competition-data-auditor` confirms the twelve target column names from the data.
3. Coordinator fills the target registry in section 1.
4. `clinical-ontology` drafts records; `verification.status` starts as `unverified`.
5. `evidence-verifier` audits and returns per-claim verdicts; unsupported claims are
   removed by `clinical-ontology`, not by the verifier.
6. Coordinator promotes records to `verified` only after a clean verification report.

## 6. What an agent returns

Evidence (per-target field coverage and citation counts), file paths written, tests
run with results, and unresolved questions.
