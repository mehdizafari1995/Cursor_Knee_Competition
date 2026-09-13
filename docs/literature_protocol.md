# Literature Protocol

Binding protocol for all literature work in this repository. The `literature-searcher`
agent must follow it; the `evidence-verifier` agent audits against it.

## 1. Scope

Knee MRI: anatomy, pathology, MRI acquisition and sequence selection, radiology
reporting language (including negation, uncertainty and postoperative terminology),
and machine-learning work on knee MRI. Searches are driven by the twelve competition
targets registered in `docs/clinical_ontology_schema.md`.

## 2. Reproducible searches only

All retrieval goes through documented APIs:

| Source | API | Typical use |
| --- | --- | --- |
| PubMed | NCBI E-utilities (`esearch`, `esummary`, `efetch`) | Discovery, metadata, abstracts |
| PMC | NCBI E-utilities, PMC OA Web Service, BioC API | Open-access full text |
| DOI | `doi.org` content negotiation / Crossref REST | DOI resolution and metadata cross-check |

Rules:

- Scraping the PubMed or PMC HTML sites is prohibited. Use the APIs.
- A general web search may be used to orient, never as a citation source. Anything
  cited must be re-resolved through the APIs above.
- Set a descriptive `tool` and `email` parameter on every E-utilities request.
- Respect NCBI rate limits: 3 requests/second without an API key, 10 with one.
- Any API key comes from an environment variable. Keys are never committed.

## 3. Search strategy construction

Build each query from explicit concept blocks combined with `AND`, with synonyms
inside a block combined with `OR`:

1. **Anatomy / target** — MeSH terms plus free-text synonyms and abbreviations.
2. **Modality** — `"Magnetic Resonance Imaging"[MeSH]` plus `MRI`, `MR imaging`.
3. **Aspect** — diagnosis, accuracy, sequence/protocol, reporting language, or
   deep learning / segmentation / classification, depending on the question.
4. **Limits** — language, date range, publication type. State every limit applied.

Record both the query as written and the query as PubMed translated it, when the
API returns a translation.

## 4. What must be logged

Every executed search is logged with:

- `query`: the exact query string sent
- `database`: `pubmed` or `pmc`
- `executed_at_utc`: ISO-8601 timestamp
- `limits`: date range, language, publication type, or `none`
- `hit_count`: total records reported by the API
- `retained_count`: records kept after screening
- `screening_rule`: the inclusion/exclusion rule applied
- `tool_version`: the script or command used

Every retained record is logged with: `pmid`, `pmcid`, `doi`, `title`, `journal`,
`year`, `publication_type`, `licence`, `open_access` and `source_url`. Unknown
fields are `null`. Missing identifiers are never inferred or reconstructed.

## 5. Output layout

```
literature/
  search_log.md            human-readable log, newest entries appended
  searches.jsonl           one JSON object per executed search
  records.jsonl            one JSON object per retained record (deduplicated)
  notes/                   per-topic Markdown notes, each claim cited
```

Deduplicate by DOI, then PMID, then normalised title. Keep the first record and
record the merged identifiers.

## 6. The "all papers" rule

The phrase **"all papers"** in this project means **all records returned by the
documented search strategies, on the documented dates, subject to the documented
limits**. It never means that every publication on a subject was found.

Prohibited without qualification: "all papers", "every publication", "the complete
literature", "exhaustive", "the full body of evidence". Required form:

> All 214 records returned by query `<query>` against PubMed on 2026-09-13, limited
> to English-language human studies.

The `evidence-verifier` agent rejects unqualified universality language.

## 7. Access limits

- Abstracts and metadata returned by the APIs may be stored and quoted briefly.
- Open-access full text from the PMC OA subset may be read and paraphrased under
  its licence.
- Paywalled full text must not be downloaded, stored or reproduced. A claim that
  needs paywalled text is recorded as `not_checkable`, not asserted.
- Figures are governed by `docs/citation_and_figure_policy.md` and are never
  retrieved as part of routine searching.

## 8. What is never committed

PDFs, article figures, DICOM files, generated reports, credentials and model
checkpoints are never committed. `.gitignore` enforces this; violating the rule by
force-adding a file is a policy breach, not a workaround.

## 9. Parallel work

Concurrent writing agents must each work in an isolated git worktree or on a
separate branch. `literature/` files are append-only within a worktree, and the
coordinator merges. Read-only agents may share the main worktree.

## 10. What an agent returns

Every literature task returns: evidence (the search log), file paths written,
tests run with their results, and unresolved questions. A task with no unresolved
questions must say so explicitly.
