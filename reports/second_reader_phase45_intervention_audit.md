# Phase 4C.5 constrained-decoding provenance and intervention audit

Synthetic reports only. No gold studies. No report-only studies.
Report text, completions, and full logits are not included.

- protocol: `second-reader-phase-4c.5-intervention-audit`
- transport: `second-reader-transport-v1.4` (constrained) vs v1.2 (unconstrained capture)
- prompt: `second-reader-prompt-v1.2.0`
- schema: `second-reader-schema-v1.0.0`
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- gold_rerun_authorized: false
- gold_rerun_executed: false
- gpu_replay_executed: false

Immutable prior artefacts remain: gold kernel v7, canary v1, smoke v2,
smoke/canary v1.3, constrained kernel versions 1–3, prompts v1.0–v1.2,
transport configs v1.0–v1.4.

## Outcome

`INTERVENTION_DIAGNOSTIC_FAIL`

Kernel v3 recorded `CONSTRAINED_CANARY_PASS` with 28/28 constrained
schema-valid outputs after an unconstrained 11/14 capture and
`intervention_rate=0.0`. Completions were not retained, so token identity
cannot be proven. Format-valid counts differ with zero recorded
interventions. Gold stays blocked.

## A — paired-run identity

Shared deterministic generate settings (schema enforcement excluded):
`max_new_tokens=2048`, `do_sample=false`, temperature 0, `top_p=1.0`,
JSON-object stop, Phi EOS IDs `[32000, 32007, 32001]`, eager attention.

Tests clone `input_ids` and `attention_mask` once and fail if any
non-schema setting differs. Sanitized SHA-256 values are recorded for all
14 synthetic examples (input and rendered prompt). Completions are not
stored. Focus examples: `CANARY_SYNTHETIC_07`, `CANARY_SYNTHETIC_12`,
`CANARY_SYNTHETIC_13`.

This environment has no CUDA and no Transformers install, so a real Phi
tokenize-once GPU pair was not executed.

## B — malformed-sequence replay

Production-schema-invalid JSON was rebuilt from the compact skeleton
(invalid enum `suspected_not_confirmed` in `evidence_category`, missing
`joint_effusion`, unbalanced braces, extra text). Sequences were walked
through LMFE 0.11.3 `JsonSchemaParser` and the pinned Phi tokenizer prefix
tree.

- Valid compact skeleton is accepted token-by-token; EOS is allowed after
  `can_end()`.
- EOS is forbidden after `{` (incomplete object).
- Invalid variants are either rejected at a first forbidden token or cannot
  end. No variant of a malformed production object was accepted as a
  complete schema-valid object.
- `SCHEMA_ENFORCEMENT_MISMATCH` was not recorded.

Frozen GPU token IDs for 07/12/13 were not stored, so those exact prefixes
cannot be replayed.

## C — intervention instrumentation

Cause of the zero-intervention GPU record: `model.generate` was passed both
`prefix_allowed_tokens_fn` and a recorder `LogitsProcessor`. Transformers
wraps the callback as `PrefixConstrainedLogitsProcessor`, which can mask
scores before the recorder. Unconstrained top-1 is then read from already
masked logits (`intervention_rate=0.0` on 16,304 tokens).

Fix: the recorder now observes unconstrained top-1, then masks. Generate
must not also receive `prefix_allowed_tokens_fn`. Conservation asserts:

- `allowed_top1 + blocked_top1 = inspected generated steps`
- structural + clinical subtype counts = total interventions
- zero interventions implies identical generated token IDs and identical
  termination
- first unconstrained/constrained token divergence is an intervention
  unless an earlier state difference is proven

CPU tests confirm a blocked top-1 is counted and changes argmax. Frozen
kernel v3 still fails those sequence-identity rules.

## D — semantic gate

- metric: `synthetic_status_accuracy`
- predeclared_threshold: none
- semantic_gate_status: `NOT_PREDECLARED`
- internal degrade floor 0.50 lives in `knee/second_reader_constrained.py`
  commit `ff4709e476268866213e10af2c762dd02b26e9e0` (v1.4 implementation)
- that floor existed before GPU record commit
  `cf8278edcf3888eb061a6417db7246fb591bc05b`
- it is not an immutable pre-run canary pass threshold
- GPU accuracy 0.815 (137/168) must not be used to invent a threshold
- fixture software canary status is now
  `CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED`

## E — corrected synthetic replay

Not executed. Protocol C records `INTERVENTION_DIAGNOSTIC_FAIL` on the
frozen GPU aggregates and stops. This environment also lacks CUDA and
Transformers, so examples 07/12/13 were not regenerated on Phi.

## F — gates

- gold_rerun_authorized: false
- gold_rerun_executed: false
- notebook 07 remains blocked until
  `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
- HARD_WEAK_LABEL_TRAINING_ALLOWED remains false
