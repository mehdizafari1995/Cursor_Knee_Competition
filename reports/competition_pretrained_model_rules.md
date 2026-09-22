# Competition rules: external data and pretrained models

Inspected before implementing the optional Phase 4B second reader.
No model weights were downloaded or loaded.

## Sources

| Source | URL |
| --- | --- |
| Competition overview (submission constraints) | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview |
| Official competition rules §2.6 External Data and Tools | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules |
| Data tab (report text not available at test time) | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data |
| Internal inventory of the same constraints | `reports/kaggle_data_inventory.md` |

Retrieved 2026-09-14.

## What the rules allow

1. **Public external data and pretrained models are allowed** on the
   competition overview, with the scoring notebook constraints:
   - submissions are notebooks
   - CPU or GPU runtime ≤ 9 hours
   - **internet disabled** at scoring
   - `submission.csv` is the submission artefact
2. Rules §2.6.a: External Data may be used if it is **publicly available and
   equally accessible** to all participants at no cost, or it meets the
   Reasonableness criteria in §2.6.b.
3. Rules §2.6.b: external data **and models** are acceptable unless the host
   prohibits them. They must be **reasonably accessible to all** and of
   **minimal cost**. Expensive proprietary LLM subscriptions or datasets that
   cost more than a prize are examples of what may fail Reasonableness.
4. Winner licence is CC-BY-NC 4.0. Incompatible pretrained-model licences do
   not have to be re-licensed by the winner, but the model itself must still
   have been legally usable during the competition.
5. Test-time `Report` text is **not** provided. Any second reader that consumes
   reports can run only on training reports, never as part of image-only
   inference.

## What this repository will and will not do

Permitted design (not activated in `report-labeler-v2.0.0`):

- Attach a **freely and publicly available** multilingual encoder (for example
  XLM-RoBERTa base, MIT licence, or multilingual BERT, Apache-2.0) as a
  **versioned Kaggle dataset**.
- Record SHA-256 of the weight files in config.
- Load with `local_files_only=True` inside a private internet-off notebook.
- Never export report text.
- Never pass gold labels into the reader.
- Combine with the deterministic extractor conservatively (agreement may raise
  confidence; disagreement, unsupported language, and no-mention stay unknown).

Not done in this revision:

- No weight download
- No `transformers` / Hugging Face hub call
- `second_reader.enabled: false` in `configs/report_labeler.yaml`
- `download_weights: false` is enforced in code

A future coordinator approval is required before attaching checksummed weights.

Phase 4C records that approval for a **gold-only** instruction-model pilot in
`reports/second_reader_model_compliance.md`. Weights are still never
downloaded. The selected snapshot is attached locally and restricted to the
58 gold studies. The 4,349 report-only studies remain out of scope.

## Second-reader contract (when enabled)

| Field | Requirement |
| --- | --- |
| Runtime | Private Kaggle notebook, internet off, GPU on for the 4C gold pilot |
| Input | Report text in memory only; gold studies only in Phase 4C |
| Output | Per-target `{status, confidence, evidence_category, uncertainty_or_negation_category}` |
| Evidence class | Closed vocabulary; never a quotation |
| Gold labels | Forbidden in the function signature |
| Combiner | Strict agreement for the gold pilot (`knee/second_reader_consensus.py`) |
| Failure modes | missing weights, sha256 mismatch, unsupported language, malformed JSON → unknown |
