# Claim Evidence Remediation

Remediation timestamp: 2026-09-13T23:52:41.069171+00:00

## Outcome

Before remediation, the active ledger contained 154 claims: 99 independently
verified as `supported`, 53 rejected as `partially_supported`, and 2 rejected as
`unsupported`. Remediation action (d), removal, was applied exactly once to each
of the 55 rejected claims. The active ledger now contains 99 claims and the
historical rejection ledger contains 55 claims.

Removal was selected because each rejected claim exceeded what its cited excerpt
or abstract established. Narrowing or splitting could preserve unsupported
implications or create new uncited claims. Removed claims remain traceable in
`literature/rejected_claims.jsonl` but are absent from both active evidence files.

## Active claims by target

| Target ID | Active claims |
| --- | ---: |
| `acl_injury` | 8 |
| `mcl_injury` | 7 |
| `medial_meniscus_injury` | 8 |
| `lateral_meniscus_injury` | 8 |
| `medial_compartment_oa` | 9 |
| `lateral_compartment_oa` | 9 |
| `patellofemoral_oa` | 10 |
| `joint_effusion` | 7 |
| `synovitis` | 7 |
| `baker_cyst` | 7 |
| `bone_contusion` | 4 |
| `fracture` | 10 |

The 12 canonical targets each retain at least 3 active claims. Target-linked
claims total 94; 5 retained cross-cutting method claims
bring the active total to 99.

## Removed claim IDs

`KO-PILOT-0005`, `KO-PILOT-0008`, `KO-PILOT-0009`, `KO-PILOT-0010`, `KO-PILOT-0019`, `KO-PILOT-0020`, `KO-PILOT-0021`, `KO-PILOT-0029`, `KO-PILOT-0030`, `KO-PILOT-0031`, `KO-PILOT-0032`, `KO-PILOT-0033`, `KO-PILOT-0034`, `KO-PILOT-0043`, `KO-PILOT-0044`, `KO-PILOT-0045`, `KO-PILOT-0046`, `KO-PILOT-0047`, `KO-PILOT-0048`, `KO-PILOT-0051`, `KO-PILOT-0052`, `KO-PILOT-0054`, `KO-PILOT-0059`, `KO-PILOT-0060`, `KO-PILOT-0061`, `KO-PILOT-0066`, `KO-PILOT-0067`, `KO-PILOT-0069`, `KO-PILOT-0074`, `KO-PILOT-0075`, `KO-PILOT-0076`, `KO-PILOT-0088`, `KO-PILOT-0089`, `KO-PILOT-0092`, `KO-PILOT-0094`, `KO-PILOT-0098`, `KO-PILOT-0099`, `KO-PILOT-0101`, `KO-PILOT-0102`, `KO-PILOT-0103`, `KO-PILOT-0110`, `KO-PILOT-0112`, `KO-PILOT-0113`, `KO-PILOT-0114`, `KO-PILOT-0123`, `KO-PILOT-0124`, `KO-PILOT-0125`, `KO-PILOT-0130`, `KO-PILOT-0132`, `KO-PILOT-0133`, `KO-PILOT-0134`, `KO-PILOT-0136`, `KO-PILOT-0144`, `KO-PILOT-0145`, `KO-PILOT-0146`

## Correction handling

Remaining uses `KO-PILOT-0095` and `KO-PILOT-0108` of PMID `38776073` now carry
structured correction metadata for authoritative erratum PMID `39325448`, DOI
`10.1001/jama.2024.20532`, and its PubMed URL. Removed uses
`KO-PILOT-0098` and `KO-PILOT-0099` are represented only in the rejection ledger.
The existing correction notice for PMID `30118392` remains in every active use.

## Identifier and quotation evidence

Every active citation retains at least one PMID, PMCID, or DOI represented in
`literature/papers.csv`. Unknown open-access and licence values remain explicit
as null/blank rather than inferred. Retained supporting excerpts are contiguous,
verbatim, and no longer than 25 words.

## Safety

No PDF, article figure, DICOM object, report corpus, model checkpoint, or other
prohibited binary was downloaded. The target registry was not changed.

## Audit state

All ontology target records are `unverified` with
`review_state: pending_fresh_evidence_audit`. This remediation does not promote
the retained claims to verified; an independent fresh evidence audit is pending.
