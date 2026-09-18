# Phase 4C.6 corrected Phi GPU intervention replay

Synthetic reports only. No gold studies. No report-only studies.
Report text, completions, and full logits are not included.

- protocol: `second-reader-phase-4c.6-corrected-phi-gpu-replay`
- kernel slug (new, do not overwrite frozen v3): `rsna-knee-second-reader-phase46-replay`
- frozen kernel left untouched: `mahdimzafari/rsna-knee-second-reader-constrained-canary` v3
- transport: `second-reader-transport-v1.4` (constrained) vs v1.2 (unconstrained)
- prompt: `second-reader-prompt-v1.2.0`
- schema: `second-reader-schema-v1.0.0`
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- gold_rerun_authorized: false
- gold_rerun_executed: false
- notebook 07 executed: false
- gpu_replay_executed: false

Immutable prior artefacts remain: gold kernel v7, canary v1, smoke v2,
smoke/canary v1.3, constrained kernel versions 1–3, prompts v1.0–v1.2,
transport configs v1.0–v1.4.

## Outcome

`INTERVENTION_DIAGNOSTIC_FAIL`

This Cursor environment has no CUDA and no Transformers install, so Gates 2
and 3 (Phi Tesla T4 paired replay) were not executed. Gate 1 CPU proofs
passed. Frozen kernel v3 still records unconstrained 11/14 vs constrained
28/28 with `intervention_rate=0.0`. Completions were not retained. Gold
stays blocked.

Do not record `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`. No predeclared
synthetic-accuracy threshold exists (`semantic_gate_status: NOT_PREDECLARED`).

## Gate 1 — runtime identity and processor-order proof

Shared deterministic generate settings (schema enforcement excluded):
`max_new_tokens=2048`, `do_sample=false`, temperature 0, `top_p=1.0`,
JSON-object stop, Phi EOS IDs `[32000, 32007, 32001]`, eager attention.

- `model.generate()` does not receive `prefix_allowed_tokens_fn`
- unconstrained: zero schema-masking logits processors
- constrained: exactly one schema-masking logits processor
  (`ConstraintInterventionRecorder` with `schema_constraint_mask=True`)
- recorder observes unconstrained top-1, then masks
- schema / prompt / v1.4 config / LMFE wheel hashes match the frozen pins
- tokenize-once clone of `input_ids` and `attention_mask` is required for
  paired calls

v1.4 config SHA-256 unchanged:
`992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`.

## Gate 2 — paired focus replay (07, 12, 13)

Not executed. Requires pinned Phi-3.5-mini-instruct weights on a Tesla T4
with Transformers, internet off. Notebook
`notebooks/08_kaggle_second_reader_phase46_replay.ipynb` is the execution
vehicle. Do not reuse or overwrite constrained-canary kernel v3.

## Gate 3 — corrected 14-example paired canary

Not executed. Same skip reason as Gate 2.

## Semantic classification

- metric: `synthetic_status_accuracy` (descriptive only)
- predeclared_threshold: none
- semantic_gate_status: `NOT_PREDECLARED`
- code never maps `uncertain` / `not_mentioned` / `unknown` to `negative`

## Authorization

- gold_rerun_authorized: false
- gold_rerun_executed: false
- notebook 07 remains blocked until a later authorized
  `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS` with reconciled interventions
- HARD_WEAK_LABEL_TRAINING_ALLOWED remains false
- PR #10 remains draft and unmerged
