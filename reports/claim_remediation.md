# Claim Evidence Remediation

First remediation timestamp: 2026-09-13T23:52:41.069171+00:00
Second remediation timestamp: 2026-09-14T00:12:31.975603+00:00

## Outcome

In round one, the active ledger contained 154 claims: 99 independently verified
as `supported`, 53 rejected as `partially_supported`, and 2 rejected as
`unsupported`. Remediation action (d), removal, was applied exactly once to each
of those 55 rejected claims. Round one therefore changed the active ledger from
154 to 99 claims and created a 55-claim historical rejection ledger.

In the second independent audit, 20 of the 99 active claims were newly rejected:
19 as `partially_supported` and 1 as `unsupported`. All 20 were removed once from
active evidence and appended to the historical ledger. Four directly supported
`bone_contusion` claims (`KO-PILOT-0155` through `KO-PILOT-0158`) were then added.
Round two therefore changed the active ledger from 99 to 83 claims
(`99 - 20 + 4`) and the historical rejection ledger from 55 to 75 claims.

Removal was selected because each rejected claim exceeded what its cited excerpt
or abstract established. Narrowing or splitting could preserve unsupported
implications or create new uncited claims. Removed claims remain traceable in
`literature/rejected_claims.jsonl` but are absent from both active evidence files.

## Active claims by target

| Target ID | Active claims |
| --- | ---: |
| `acl_injury` | 8 |
| `mcl_injury` | 7 |
| `medial_meniscus_injury` | 7 |
| `lateral_meniscus_injury` | 7 |
| `medial_compartment_oa` | 7 |
| `lateral_compartment_oa` | 7 |
| `patellofemoral_oa` | 7 |
| `joint_effusion` | 5 |
| `synovitis` | 7 |
| `baker_cyst` | 4 |
| `bone_contusion` | 5 |
| `fracture` | 7 |

The 12 canonical targets each retain at least 3 active claims. Target-linked
claims total 78; 5 retained cross-cutting method claims bring the active total
to 83.

## Round-one removed claim IDs

`KO-PILOT-0005`, `KO-PILOT-0008`, `KO-PILOT-0009`, `KO-PILOT-0010`, `KO-PILOT-0019`, `KO-PILOT-0020`, `KO-PILOT-0021`, `KO-PILOT-0029`, `KO-PILOT-0030`, `KO-PILOT-0031`, `KO-PILOT-0032`, `KO-PILOT-0033`, `KO-PILOT-0034`, `KO-PILOT-0043`, `KO-PILOT-0044`, `KO-PILOT-0045`, `KO-PILOT-0046`, `KO-PILOT-0047`, `KO-PILOT-0048`, `KO-PILOT-0051`, `KO-PILOT-0052`, `KO-PILOT-0054`, `KO-PILOT-0059`, `KO-PILOT-0060`, `KO-PILOT-0061`, `KO-PILOT-0066`, `KO-PILOT-0067`, `KO-PILOT-0069`, `KO-PILOT-0074`, `KO-PILOT-0075`, `KO-PILOT-0076`, `KO-PILOT-0088`, `KO-PILOT-0089`, `KO-PILOT-0092`, `KO-PILOT-0094`, `KO-PILOT-0098`, `KO-PILOT-0099`, `KO-PILOT-0101`, `KO-PILOT-0102`, `KO-PILOT-0103`, `KO-PILOT-0110`, `KO-PILOT-0112`, `KO-PILOT-0113`, `KO-PILOT-0114`, `KO-PILOT-0123`, `KO-PILOT-0124`, `KO-PILOT-0125`, `KO-PILOT-0130`, `KO-PILOT-0132`, `KO-PILOT-0133`, `KO-PILOT-0134`, `KO-PILOT-0136`, `KO-PILOT-0144`, `KO-PILOT-0145`, `KO-PILOT-0146`

## Round-two removed claim IDs

`KO-PILOT-0025`, `KO-PILOT-0039`, `KO-PILOT-0055`, `KO-PILOT-0056`, `KO-PILOT-0070`, `KO-PILOT-0071`, `KO-PILOT-0086`, `KO-PILOT-0090`, `KO-PILOT-0091`, `KO-PILOT-0095`, `KO-PILOT-0096`, `KO-PILOT-0120`, `KO-PILOT-0121`, `KO-PILOT-0122`, `KO-PILOT-0128`, `KO-PILOT-0129`, `KO-PILOT-0135`, `KO-PILOT-0137`, `KO-PILOT-0138`, `KO-PILOT-0148`

The five requested round-one records (`KO-PILOT-0029`, `KO-PILOT-0043`,
`KO-PILOT-0088`, `KO-PILOT-0092`, and `KO-PILOT-0094`) retain their original
verdict, reason, action, and timestamp. Their added `followup_audit_note` records
that later licensed-full-text review found more support than the initial
abstract-level audit.

## Correction handling

The remaining use `KO-PILOT-0108` of PMID `38776073` carries structured correction
metadata for authoritative erratum PMID `39325448`, DOI
`10.1001/jama.2024.20532`, and its PubMed URL. Removed uses `KO-PILOT-0095`,
`KO-PILOT-0098`, and `KO-PILOT-0099` are represented only in the rejection
ledger. The existing correction notice for PMID `30118392` remains in every
active use.

## Identifier and quotation evidence

Every active citation retains at least one PMID, PMCID, or DOI represented in
`literature/papers.csv`. Unknown open-access and licence values remain explicit
as null/blank rather than inferred. Retained supporting excerpts are contiguous,
verbatim, and no longer than 25 words.

## Safety

No PDF, article figure, DICOM object, report corpus, model checkpoint, or other
prohibited binary was downloaded. The target registry was not changed.

## Audit state

All ontology target records remain `unverified` with
`review_state: pending_fresh_evidence_audit`. The second-round removals do not
promote the retained or newly added claims; ontology audit state remains pending.
