# First Knee-MRI Literature Pilot

Retrieval date: 2026-09-13

## Scope and method

This bounded pilot retained 92 unique records returned by the documented PubMed
and PubMed Central API searches in `literature/search_queries.json`. It is not an
exhaustive search. Requests used the NCBI E-utilities and PMC APIs without an API
key and remained below three requests per second. Records were deduplicated by
PMID, then DOI, then normalized title.

No Google Scholar scraping, Kaggle test-data search, PDF or figure download, or
paywalled full-text access was performed. Abstracts and metadata came from NCBI
API responses. The retired PMC OA Web Service returned HTTP 404 during the
licence follow-up; licences were therefore left unknown rather than inferred.

## Corpus summary

- Unique papers: 92
- Primary/non-review records: 68
- Review records: 24
- Open-access status: 23 true, 6 false, 63 unknown
- Records with a known licence: 3 (two `CC BY`, one `CC BY-NC-ND`)
- Active verified claims: 83
- Rejected claims retained in audit history: 75

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
| Bone contusion | 13 |
| Fracture | 12 |

Counts are non-exclusive because a paper may cover multiple targets.

## Coverage and limitations

The searches covered MRI diagnostic characteristics, planes and sequences,
report terminology, synonyms and abbreviations, negation and uncertainty,
postoperative language, diagnostic mimics, knee-MRI machine learning,
report-derived weak supervision, multi-instance learning, and 2.5D
classification. Search coverage does not imply that every retained article
supports an ontology claim.

The draft ontology covers all twelve targets. Unsupported fields were left empty.
Evidence was thin for ACL/MCL sequence detail, Baker-cyst anatomy,
bone-contusion terminology, several postoperative and grading fields, and
non-English report cues. The ontology found no direct support in its staging
corpus for explicit multi-instance or 2.5D methods; those topics remain represented
in the broader paper search only.

## Outputs

- `literature/search_queries.json`
- `literature/papers.csv`
- `literature/papers.ris`
- `literature/ontology.yaml`
- `literature/claims.jsonl`
- `literature/searches.jsonl`
- `literature/records.jsonl`
- `literature/api_responses.jsonl`
- `literature/search_log.md`
- `literature/ontology_source_records.jsonl`
- `reports/literature_pilot.md`
- `reports/evidence_verification.md` (written by the independent audit)

The independent evidence-verification report is the authority on identifier
resolution and claim support.
