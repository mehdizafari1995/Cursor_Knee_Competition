# Citation and Figure Policy

Binding for every agent in this repository. The `evidence-verifier` agent enforces it.

## 1. Citation integrity

- **Do not invent citations.** Every citation must trace to an API response received
  during the work, recorded in `literature/records.jsonl`.
- Identifiers are copied, never reconstructed. A PMID, PMCID or DOI that was not
  returned by an API is `null`.
- Every clinical claim carries its own citation. One citation at the end of a
  paragraph does not cover the claims inside it.
- Paraphrase by default. Quotations are short, marked with quotation marks, and
  carry a locator.
- Statements derived by reasoning rather than stated by a source are labelled as
  inference and kept separate from cited claims.
- Retracted or corrected articles are flagged as such next to every use.

### Citation format

Inline: `[PMID:12345678]`, `[PMCID:PMC1234567]` or `[DOI:10.1000/xyz123]`, using the
identifiers present in the record. Machine-readable claims carry:

```json
{
  "claim": "…",
  "citations": [
    {"pmid": "12345678", "pmcid": "PMC1234567", "doi": "10.1000/xyz123",
     "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
     "licence": "CC BY 4.0", "quote": null, "locator": "Abstract"}
  ]
}
```

## 2. Figure retrieval

Figures are retrieved **only** when full-text and figure access is legal for this
project: the article is in the PMC open-access subset (or equivalently licensed) and
the licence permits reuse for the intended purpose. Paywalled figures, figures behind
a publisher agreement this project does not hold, and figures scraped from article
HTML pages are all out of bounds.

Retrieval requires, before download:

1. Confirmation of open-access or otherwise permissive status from the API.
2. The licence identifier recorded verbatim (for example `CC BY 4.0`, `CC BY-NC 4.0`,
   `CC0`, or a publisher-specific string).
3. The canonical source URL recorded.

## 3. Figure provenance record

Every retrieved figure gets a row in `literature/figures/manifest.csv` (git-ignored
alongside the images) with at minimum:

`figure_id, pmcid, doi, figure_label, caption, licence, licence_url, source_url,
retrieved_at_utc, attribution_text, permitted_uses, local_path, sha256`

A figure without a recorded licence and source URL must be deleted, not kept
pending. Attribution accompanies any display of the figure.

## 4. Separation from competition data

Literature figures and Kaggle competition images live in separate trees and never
mix:

```
literature/figures/    published figures, licence-tracked, never training data by default
data/                  Kaggle competition DICOM and CSVs, never modified
```

No script may read from `literature/figures/` and write into the competition data
tree, or vice versa. The `competition-data-auditor` agent checks this separation and
the `evidence-verifier` agent flags violations.

## 5. Literature figures are not training data

Literature figures must not be used for model training, fine-tuning, pretraining,
augmentation or validation unless **all three** of the following are recorded first:

1. **Licence review** — a written determination that the specific licence permits
   machine-learning use, including any non-commercial or share-alike consequences.
2. **Patient-level label verification** — confirmation of what the figure actually
   shows and that its label is correct and at the right level, plus confirmation
   that the case is not also present in the competition data.
3. **Explicit coordinator approval** — a dated, written approval from the
   coordinator referencing items 1 and 2.

Absent any one of these, the answer is no. An agent may propose the review; no agent
may perform the training use on its own authority.

## 6. Never committed

The following are never committed to git, and `.gitignore` enforces it:

- PDFs and any article full text
- Figures and extracted images
- DICOM files and other medical imaging data
- Generated reports and audit artefacts under `reports/`
- Credentials, API keys, tokens and Kaggle configuration
- Model checkpoints and serialised weights

Tracked placeholders (`reports/.gitkeep`, `literature/.gitkeep`) keep the directories
in the tree. Force-adding an ignored artefact is a policy breach.

## 7. Language limits

Do not claim completeness of the literature. See the "all papers" rule in
`docs/literature_protocol.md`. Do not describe a finding as proven, definitive or
established unless the cited source uses equivalent language, and say which source
does.
