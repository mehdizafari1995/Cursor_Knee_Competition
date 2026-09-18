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

GPU constrained smoke is prepared in
`notebooks/06_kaggle_second_reader_constrained_canary.ipynb` and was **not**
executed in this environment (no attached Phi GPU).

## Phase E — constrained 14-example canary

Software fixture (14 examples × 2): `CONSTRAINED_CANARY_PASS`,
`malformed_output_rate = 0`, repeatable, synthetic status accuracy 1.0 versus
the predeclared fixture truth, and no code path that rewrites `uncertain` /
`not_mentioned` / `unknown` to `negative`.

GPU 14×2 canary, unconstrained-v1.3 field comparison for the 11 previously
valid examples, and constraint-intervention statistics require the Kaggle
notebook and are **not** recorded as a GPU pass here.

## Phase F — freeze

v1.4 protocol/config files are frozen. `CONSTRAINED_CANARY_PASS` is the
fixture-level software status only. GPU `CONSTRAINED_CANARY_PASS` is **not**
authorized. The gold-rerun notebook
`notebooks/07_kaggle_second_reader_gold_constrained.ipynb` is prepared and
must not be executed until a GPU constrained canary passes.

- gold_rerun_authorized: false
- gold_rerun_executed: false

## Coordinator

- Do not rerun the 58-study gold pilot.
- Do not process the 4,349 report-only studies.
- Do not overwrite v7 / v1 / v2 / v1.3 artefacts.
