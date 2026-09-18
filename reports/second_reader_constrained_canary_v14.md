# Phase 4C.4 schema-constrained synthetic validation

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

- protocol: `second-reader-pilot-v1.4.0`
- transport: `second-reader-transport-v1.4`
- prompt: `second-reader-prompt-v1.2.0` (frozen hash `5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289`)
- schema: `second-reader-schema-v1.0.0` (frozen hash `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20`)
- v1.4 config hash: `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- gold_rerun_executed: false
- gold_rerun_authorized: false
- parser_aliases: false
- post_generation_clinical_repair: false

Immutable prior artefacts remain: gold kernel v7, canary v1, smoke v2,
smoke/canary v1.3, prompts v1.0–v1.2, transport configs v1.0–v1.3.

## Phase A — dependency and licence lock

- package: `lm-format-enforcer==0.11.3`
- licence: MIT
- upstream tag: `v0.11.3`
- tag object SHA: `29514a40ccc64bdb7fc07b970cd31e5f53d1564e`
- commit SHA: `05c7dc3a46305e39c1a67c8622c433d8deda48a2`
- wheel: `lm_format_enforcer-0.11.3-py3-none-any.whl`
- wheel SHA-256: `cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f`
- vendored dependency: `interegular==0.3.3` MIT,
  `interegular-0.3.3-py37-none-any.whl`
  SHA-256 `b0c07007d48c89d6d19f7204972d369b2a77222722e126b6aa63aa721dc3b19c`
- not vendored (Kaggle preinstalled, ABI-specific): pydantic, packaging, PyYAML
- Kaggle install: hashed local wheels with `--no-index --no-deps`
- wheels are gitignored and copied into the private source zip
- network/telemetry scan: no runtime downloads in `JsonSchemaParser`,
  `TokenEnforcer`, or `integrations/transformers.py`. Hugging Face names appear
  only in unused llama.cpp / Haystack adapters.

## Phase B — schema compatibility

`JsonSchemaParser` 0.11.3 accepts `configs/second_reader_schema.json` without
weakening it.

The reported raw-file vs validation/generation hash gap is serialization, not
content. There is no separate generation schema file.
`generation_schema()` returns `validation_schema()` from the same on-disk JSON.

| Hash | SHA-256 |
| --- | --- |
| schema_raw_bytes_sha256 | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| schema_canonical_json_sha256 | `c39e6236e509c466850e501969e7e8525bd40a189ab05ed3d591e99bd30d3b05` |
| generation_schema_raw_bytes_sha256 | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| generation_schema_canonical_json_sha256 | `c39e6236e509c466850e501969e7e8525bd40a189ab05ed3d591e99bd30d3b05` |

The on-disk file is pretty-printed (`indent=2`, trailing newline). Decoder
hashes use `json.dumps(..., sort_keys=True, separators=(",", ":"))` of the
same parsed object. Parsed validation, generation, and on-disk JSON compare
equal.

Enforced:

- root object
- required `targets` object
- exactly 12 required target keys, each once
- required fields within each target
- exact status / confidence / evidence / uncertainty enums
- `additionalProperties=false` at root, `targets`, and every target object
- `schema_version` `const`

A generation-only schema is **not** required. The generation schema is the
validation schema. Automated equivalence compares required paths, property
names, enum sets, `additionalProperties`, target order and target set, and
hashes both (identical).

## Phase C — tokenizer compatibility

Pinned Phi-3.5 tokenizer files
(`tokenizer.model` `9e556afd…`, `tokenizer_config.json` `92a6e3f8…`) were
hash-checked. Prefix-tree audit walks the compact JSON skeleton through
LM Format Enforcer's tokenizer tree, confirms EOS IDs
`[32000, 32007, 32001]` after a complete object, and round-trips quotes,
underscores, braces, brackets, commas, colons, escaped characters, every
closed enum string, and the apostrophe in `Baker's`.

Retained: JSON-object stopper, generated-token-only decode, greedy
`do_sample=false`.

## Phase D — constrained three-example smoke

Software fixture: 3/3 schema-valid, 12 targets, zero invalid enums, zero
missing targets, zero unbalanced JSON, zero extra fields, zero max-token
termination, zero gold, zero report-only.

GPU kernel `mahdimzafari/rsna-knee-second-reader-constrained-canary`
version 3, Tesla T4, internet off, no competition dataset:
`CONSTRAINED_SMOKE_PASS`. Backend `local_transformers`. 3/3 schema-valid,
3/3 `targets` objects, 12 targets, zero invalid enums, zero extra
properties, zero unbalanced JSON, zero extra text, zero max-token
termination, `n_gold_studies_processed: 0`,
`n_report_only_studies_processed: 0`. Smoke wall time 175.3 s. Peak GPU
memory 5,177,142,272 bytes.

Kernel versions 1–2 of that constrained notebook are immutable
(`ImportError` on Transformers 5, then unconstrained capture
`n_repeats` guard). They are not gold and do not overwrite v1.3.

## Phase E — constrained 14-example canary

Software fixture (14 examples × 2): `CONSTRAINED_CANARY_PASS`,
`malformed_output_rate = 0`, repeatable, synthetic status accuracy 1.0 versus
the predeclared fixture truth, and no code path that rewrites `uncertain` /
`not_mentioned` / `unknown` to `negative`. Fixture `intervention_rate=0` is
**not** a model result.

GPU kernel version 3, same private T4 notebook:

| Gate | Result |
| --- | --- |
| status | `CONSTRAINED_CANARY_PASS` |
| backend | `local_transformers` |
| unconstrained v1.3 capture | 11/14 schema-valid (examples 07, 12, 13 invalid) |
| constrained schema-valid | 28/28 |
| malformed_output_rate | 0.0 |
| repeatable | true |
| invalid enums / extra properties / missing targets | 0 |
| JSON close / max-token | 14/14 first-run close, 0 max-token |
| n_gold_studies_processed | 0 |
| n_report_only_studies_processed | 0 |
| gold_rerun_authorized | false |
| constrained wall seconds | 926.6 |
| peak GPU memory bytes | 5,236,375,040 |

Paired clinical comparison against unconstrained-valid fields only
(11 examples × 12 targets = 132 fields per attribute):

| Field | n_compared | n_agree | rate |
| --- | --- | --- | --- |
| status | 132 | 132 | 1.0 |
| confidence | 132 | 132 | 1.0 |
| evidence_category | 132 | 132 | 1.0 |
| uncertainty_or_negation_category | 132 | 132 | 1.0 |

`n_outputs_clinical_fields_changed: 0`.
`n_target_level_clinical_decisions_changed: 0`.

Synthetic-truth status accuracy (constrained, 14 × 12 = 168): 0.815
(137/168). Predeclared semantic gate is accuracy < 0.50. No
uncertain/unknown/not_mentioned values were rewritten to negative by
application code (`STATUS_TO_LABEL` maps those to `unknown`).

Per-target status accuracy versus synthetic truth:

| Target | accuracy | n |
| --- | --- | --- |
| acl_injury | 0.786 | 14 |
| baker_cyst | 0.857 | 14 |
| bone_contusion | 0.786 | 14 |
| fracture | 0.857 | 14 |
| joint_effusion | 0.786 | 14 |
| lateral_compartment_oa | 0.857 | 14 |
| lateral_meniscus_injury | 0.786 | 14 |
| mcl_injury | 0.786 | 14 |
| medial_compartment_oa | 0.857 | 14 |
| medial_meniscus_injury | 0.786 | 14 |
| patellofemoral_oa | 0.786 | 14 |
| synovitis | 0.857 | 14 |

Disagreements versus synthetic truth are mostly model under-calling
(`negative`/`uncertain`/`positive` → `not_mentioned`). Constraint
enforcement did not change any unconstrained-valid field.

Constraint-pressure diagnostics (real `transformers` logits processor, not
fixture): 16,304 generated tokens, 16,304 unconstrained top-1 allowed, 0
blocked, `intervention_rate=0.0`, 0 structural interventions, 0 clinical
enum interventions, empty unconstrained→constrained enum map. Mean selected
rank ≈ 0.00012 (unconstrained top-1 almost always already allowed). Full
vocabulary logits and completions were not retained.

Kaggle imported versions: Python 3.12.13, PyTorch 2.10.0+cu128,
Transformers 5.0.0, tokenizers 0.22.2, Pydantic 2.12.3, packaging 26.1,
PyYAML 6.0.3, lm-format-enforcer 0.11.3, interegular 0.3.3. `pip check`
reported pre-existing Kaggle environment conflicts (google-colab,
bigframes); hashed local wheels imported successfully. LMFE 0.11.3 required
an in-process `PreTrainedTokenizerBase` attribute restore for Transformers
5; schema enforcement was not changed.

## Phase F — freeze

v1.4 protocol/config files remain frozen. GPU status is
`CONSTRAINED_CANARY_PASS`. Notebook 07 is **not** executed in this task.
`gold_rerun_authorized: false`. `gold_rerun_executed: false`.

- gold_rerun_authorized: false
- gold_rerun_executed: false

## Coordinator

- Do not rerun the 58-study gold pilot.
- Do not process the 4,349 report-only studies.
- Do not overwrite v7 / v1 / v2 / v1.3 artefacts.
- Do not execute `notebooks/07_kaggle_second_reader_gold_constrained.ipynb`.

