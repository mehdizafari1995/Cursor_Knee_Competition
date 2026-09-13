# Independent Evidence Verification

Audit date: 2026-09-13

## Overall verdict

**FAIL — clinical claim support requires revision.**

No fabricated citation was detected. The pilot therefore did not trigger the
specific fabricated-citation failure condition, but it fails closed because 55
of 154 claims were not fully supported as written and four claims omitted a
known erratum flag.

The custom `evidence-verifier` performed the audit. Its read-only execution mode
prevented it from writing this permitted report file, so the coordinator
transcribed the verifier's returned findings without changing its verdict.

## Claim results

- Verified (`supported`): 99
- Rejected: 55
  - `partially_supported`: 53
  - `unsupported`: 2
- Unresolved (`not_checkable`): 0
- Fabricated citations: 0

`partially_supported` claim suffixes:

`0005, 0008, 0009, 0010, 0019, 0020, 0021, 0029, 0030, 0031, 0032,
0033, 0034, 0043, 0044, 0045, 0046, 0047, 0048, 0051, 0052, 0054,
0059, 0060, 0061, 0066, 0067, 0069, 0074, 0075, 0076, 0088, 0089,
0092, 0099, 0101, 0102, 0103, 0110, 0112, 0113, 0114, 0123, 0124,
0125, 0130, 0132, 0133, 0134, 0136, 0144, 0145, 0146`

`unsupported`:

- `KO-PILOT-0094`: the cited source does not establish the asserted
  pain-without-chondral-abnormality distinction.
- `KO-PILOT-0098`: “knee joint cavity” is not stated by the cited record.

Every other claim from `KO-PILOT-0001` through `KO-PILOT-0154` is `supported`.
No claim is `not_checkable`.

## Identifier resolution

- All 18 uniquely cited PMIDs resolve with matching titles, years, and DOI
  pairings.
- All 10 uniquely cited PMCIDs resolve.
- All 18 uniquely cited DOIs resolve: 17 through Crossref and
  `10.26355/eurrev_202212_30537` through the DOI Handle API.
- Fabricated or mismatched citation identifiers: 0.

PMID `38776073` has erratum PMID `39325448`. Claims `KO-PILOT-0095`,
`KO-PILOT-0098`, `KO-PILOT-0099`, and `KO-PILOT-0108` do not carry the required
correction flag.

## Corpus checks

- Unique records: 89
- Review records: 24
- Primary/non-review records: 65
- Stored open-access status: 23 true, 6 false, 60 unknown
- Ten duplicate retrieval clusters were consolidated; no duplicate remains in
  the final paper inventory.
- Deduplication follows the user-required PMID, DOI, normalized-title order.
  This intentionally differs from the older repository protocol's DOI-first
  order.
- The initial report incorrectly stated that no licences were known.
  `records.jsonl` contains three verbatim licence values: two `CC BY` and one
  `CC BY-NC-ND`. Fourteen of the 18 uniquely cited records still have unknown
  citation-level licences. No figure or training reuse is permissible from
  records without a reviewed licence.
- No PDF, figure, DICOM, checkpoint, `data/` file, or
  `literature/figures/` file was found.
- No unqualified exhaustive-literature claim was found.

## Papers per target

| Target | Papers |
| --- | ---: |
| ACL injury | 12 |
| MCL injury | 11 |
| Medial meniscus injury | 12 |
| Lateral meniscus injury | 12 |
| Medial-compartment osteoarthritis | 14 |
| Lateral-compartment osteoarthritis | 14 |
| Patellofemoral osteoarthritis | 10 |
| Joint effusion | 11 |
| Synovitis | 11 |
| Baker cyst | 11 |
| Bone contusion | 11 |
| Fracture | 12 |

All twelve canonical targets are covered. Counts are non-exclusive.

## Verification methods and limitations

The verifier checked PubMed and PMC ESummary, PMC BioC, Crossref REST, and DOI
Handle resolution. It compared claims with cited abstracts or legally available
PMC text and did not retrieve paywalled full text, PDFs, or figures.

One identifier-resolution batch was parallelized, so the verifier could not
certify that its own audit requests stayed below three requests per second.
The audited retrieval logs document throttled pilot collection; this limitation
applies to the independent re-resolution pass.

## Blocking rejections

1. Remove or rewrite the 53 partially supported claims to match their cited
   evidence.
2. Remove claims `KO-PILOT-0094` and `KO-PILOT-0098` unless new supporting
   evidence is retrieved.
3. Add the PMID `39325448` correction notice to every use of PMID `38776073`.
4. Treat unknown licences as prohibiting reuse until reviewed.
5. Document the normalized-title algorithm for reproducible deduplication.

## Unresolved questions

- Which narrower wording would preserve useful content in each partially
  supported claim?
- Should the repository protocol be updated in a separate change to match the
  user-mandated PMID-first pilot deduplication order?
- Which supported sources, if any, provide legally reusable full text under a
  licence suitable for future non-figure text review?
