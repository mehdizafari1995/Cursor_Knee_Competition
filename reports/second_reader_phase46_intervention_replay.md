# Phase 4C.6 corrected Phi GPU intervention replay

Synthetic reports only. No gold studies. No report-only studies.
Report text, completions, and full logits are not included.

- protocol: `second-reader-phase-4c.6-corrected-phi-gpu-replay`
- kernel slug (new, do not overwrite frozen v3): `mahdimzafari/rsna-knee-second-reader-phase46-replay`
- kernel version: 2 (`KernelWorkerStatus.COMPLETE`; version 1 ERROR on missing `reports/` files)
- notebook: `notebooks/08_kaggle_second_reader_phase46_replay.ipynb`
- frozen kernel left untouched: `mahdimzafari/rsna-knee-second-reader-constrained-canary` v3
- GPU: Tesla T4, internet off, private
- source HEAD packaged for the run: `3e8e0d40929a13f0ed6f2fb120823eab36166fdc`
- transport: `second-reader-transport-v1.4` (constrained) vs v1.2 (unconstrained)
- prompt: `second-reader-prompt-v1.2.0`
- schema: `second-reader-schema-v1.0.0`
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- gold_rerun_authorized: false
- gold_rerun_executed: false
- notebook 07 executed: false
- gpu_replay_executed: true
- utc: `2026-09-19T23:35:05Z`
- inference_seconds: `1682.6`
- peak_gpu_memory_bytes: `5237321216`

Immutable prior artefacts remain: gold kernel v7, canary v1, smoke v2,
smoke/canary v1.3, constrained kernel versions 1–3, prompts v1.0–v1.2,
transport configs v1.0–v1.4.

## Outcome

`CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED`

Do not record `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`. No predeclared
synthetic-accuracy threshold exists (`semantic_gate_status: NOT_PREDECLARED`).

Runtime: Python 3.12.13, torch `2.10.0+cu128`, transformers `5.0.0`,
tokenizers `0.22.2`, lm-format-enforcer `0.11.3`, interegular `0.3.3`.
Pinned Phi-3.5-mini-instruct revision
`2fe192450127e6a83f7441aef6e3ca586c338b77`. Weight and tokenizer hashes
matched the frozen table.

## Preflight hardening

`capture_isolated_recorder_stats` reads
`second_reader_constrained.LAST_CONSTRAINT_STATS` and
`CONSTRAINT_STAT_LOG` through the module namespace after each isolated
constrained generation. After requiring exactly one log summary it requires
`LAST_CONSTRAINT_STATS` to be that same object, or to match its sanitized
payload. Mismatch is `INTERVENTION_DIAGNOSTIC_FAIL` with
`missing_recorder:<role>:last_stats_log_mismatch`. Missing, stale, empty, or
fixture recorder output does not substitute `empty_intervention_payload`.
Aggregated `n_calls` is the sum of per-run `n_calls`.

`semantic_gate_audit()` stays `NOT_PREDECLARED` when `reports/` files are
absent from the allowlisted Kaggle source zip. Kernel v1 hit that path;
kernel v2 continued.

## Gate 1 — runtime identity and processor-order proof

Shared deterministic generate settings (schema enforcement excluded):
`max_new_tokens=2048`, `do_sample=false`, temperature 0, `top_p=1.0`,
JSON-object stop, Phi EOS IDs `[32000, 32007, 32001]`, eager attention,
float16.

- `model.generate()` does not receive `prefix_allowed_tokens_fn`
- unconstrained: zero schema-masking logits processors
- constrained: exactly one schema-masking logits processor
  (`ConstraintInterventionRecorder` with `schema_constraint_mask=True`)
- recorder observes unconstrained top-1, then masks
- schema / prompt / v1.4 config / LMFE / interegular hashes match the frozen pins
- tokenize-once clone of `input_ids` and `attention_mask`

v1.4 config SHA-256 unchanged:
`992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`.

Schema hashes:

| Hash | SHA-256 |
| --- | --- |
| schema_raw_bytes_sha256 | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| schema_canonical_json_sha256 | `c39e6236e509c466850e501969e7e8525bd40a189ab05ed3d591e99bd30d3b05` |

Prompt SHA-256: `5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289`.
LMFE wheel: `cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f`.
interegular wheel: `b0c07007d48c89d6d19f7204972d369b2a77222722e126b6aa63aa721dc3b19c`.

## Gate 2 — paired focus replay (07, 12, 13)

All three constrained outputs schema-valid. Constrained repeats matched
normalized JSON SHA-256 and generated-token SHA-256. Recorder
`source=transformers_logits_processor`, `fixture_not_a_model_result=false`,
`n_calls == n_generated_tokens > 0`. First unconstrained/constrained
divergence was counted. Zero-intervention identity was not applicable
because every focus example had blocked top-1 tokens.

| Example | un schema | un term | con/repeat token SHA-256 | n_calls | allowed | blocked | first divergence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CANARY_SYNTHETIC_07 | false | json_close | `c1dadb7f…a5d853b2` | 721 | 707 | 14 | pos 103, counted |
| CANARY_SYNTHETIC_12 | false | eos | `e3829e5c…112ab39c` | 564 | 561 | 3 | pos 27, counted |
| CANARY_SYNTHETIC_13 | false | json_close | `2db019ae…4367d2ea` | 581 | 580 | 1 | pos 555, counted |

## Gate 3 — corrected 14-example paired canary

- unconstrained schema-valid: 11/14
- constrained schema-valid: 28/28 (two constrained passes)
- no truncation, missing targets, invalid enums, additional fields,
  prompt leakage, or extra text on constrained outputs
- total inspected generated steps / recorder calls: 16304
- allowed top-1: 16268
- blocked top-1: 36
- allowed + blocked = 16304
- intervention_rate: 0.002208047105004907
- structural interventions: 18
- target-key interventions: 6
- property-name interventions: 12
- clinical-enum interventions: 18
- other / JSON-structure interventions: 0
- structural + clinical-enum + other = 36 blocked tokens
- interventions_by_field: `targets` 12, `uncertainty_or_negation_category` 14,
  `target_key` 6, `evidence_category` 2, `status` 2
- mean selected rank: 0.015947006869479884
- mean logit/prob gap: 0.001327206549477175
- n_outputs_clinical_fields_changed: 0
- n_target_any_field_changed: 0
- field agreement vs every schema-valid unconstrained field: 132/132
  (confidence, evidence_category, status, uncertainty_or_negation_category)
- code_never_coerces_unknown_to_negative: true

Synthetic-truth status accuracy is descriptive only: 0.8154761904761905
(same overall figure as frozen v3). It is not a pass threshold.

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
