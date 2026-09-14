# Independent Evidence Verification — Remediated Set

Audit date: 2026-09-14
Evidence-content commit: `07511e1853c205cee24defe63f3c0894dbffe520`
Final repository verification commit: `e366d7945baf8c7c5f5a967b144d33225ab8ddf1`

## Overall verdict

**PASS**

The independent `evidence-verifier` found that all 83 active claims are directly
supported by their cited abstracts or licensed full text. All identifiers
resolve, the rejected-claim ledger is intact, every target exceeds the minimum
coverage threshold, and all runtime and safety checks pass.

## Claim results

- Verified (`supported`): 83
- Active `partially_supported`: 0
- Active `unsupported`: 0
- Unresolved (`not_checkable`): 0
- Fabricated citations: 0
- Rejected claims preserved in history: 75
- Rejected IDs remaining active: 0
- Active citation quotations: 90/90 contiguous, verbatim, and at most 25 words

## Claim lineage reconciliation

The immutable `237372c` baseline contains 154 unique originals:
79 remain active supported and 75 are rejected removed. The final active ledger
contains 83 claims: those 79 originals plus four active supported replacement
claims (`KO-PILOT-0155` through `KO-PILOT-0158`). Thus the active and rejected
ledgers contain 158 rows because they span 154 originals plus four replacements,
not because historical versions are distinct claims. There are no duplicate or
versioned rejection-history records, active/rejected overlaps, or originals
without a final disposition. The complete mapping is recorded in
`literature/claim_lineage.json` and explained in
`reports/claim_lineage_audit.md`. The overall verdict remains **PASS**.

## Identifier resolution

- All 19 uniquely cited PMIDs resolve with matching titles, years, and DOI
  pairings.
- All 12 uniquely cited PMCIDs resolve.
- All 19 uniquely cited DOIs resolve: 18 through Crossref and
  `10.26355/eurrev_202212_30537` through the DOI Handle API.
- Fabricated or mismatched citation identifiers: 0.

The linked correction PMID `39325448`, PMCID `PMC11581473`, and DOI
`10.1001/jama.2024.20532` also resolve. It is recorded in `papers.csv` and
`records.jsonl`; active claim `KO-PILOT-0108` carries structured correction
metadata. PMID `34968719` consistently uses bibliographic year 2022.

## Corpus checks

- Unique records: 92
- Stored open-access status: 23 true, 6 false, 63 unknown
- Ten duplicate retrieval clusters were consolidated; no duplicate remains in
  the final paper inventory.
- Deduplication follows the user-required PMID, DOI, normalized-title order.
  This intentionally differs from the older repository protocol's DOI-first
  order.
- Three corpus records have verbatim licence values: two `CC BY` and one
  `CC BY-NC-ND`; 89 remain unknown. No figure or training reuse is permissible
  from records without a reviewed licence.
- No PDF, figure, DICOM, checkpoint, `data/` file, or
  `literature/figures/` file was found.
- No unqualified exhaustive-literature claim was found.

## Papers per target

| Target | Papers |
| --- | ---: |
| ACL injury | 8 |
| MCL injury | 7 |
| Medial meniscus injury | 7 |
| Lateral meniscus injury | 7 |
| Medial-compartment osteoarthritis | 7 |
| Lateral-compartment osteoarthritis | 7 |
| Patellofemoral osteoarthritis | 7 |
| Joint effusion | 5 |
| Synovitis | 7 |
| Baker cyst | 4 |
| Bone contusion | 5 |
| Fracture | 7 |

These are supported active-claim counts. All twelve canonical targets exceed the
required minimum of three. Five additional active claims cover cross-cutting
methods, producing 83 active claims total.

## Verification methods and limitations

The verifier checked PubMed, PMC, Crossref, and DOI Handle resolution and
compared every active claim with cited abstracts or licensed PMC text. It found
zero resolver failures, metadata mismatches, quote violations, PDFs, figures,
DICOM files, model checkpoints, or unqualified completeness claims.

An independent read-only runtime verifier then executed:

- Full `pytest`: **147 passed**, 14 warnings, 3.01 seconds.
- `git diff --check`: passed with no output.
- Baseline history comparison: 75/75 `original_claim` values exactly match
  commit `237372c`; no rejected ID remains among 83 active rows.
- Report tracking check: all three required Markdown reports are tracked.

After the PASS verdict, the coordinator changed only verification statuses,
reports, and their assertions before commit `e366d79`; no clinical claim text or
citation changed. The project `repository-verifier` inspected that final commit
and returned PASS against the captured full-suite evidence.

## Blocking rejections

None.

## Unresolved questions

None.
