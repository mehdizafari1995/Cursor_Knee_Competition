# Phase 4C.2 termination inspection

Sanitized tokenizer / generation-config values for the pinned
`microsoft/Phi-3.5-mini-instruct` revision
`2fe192450127e6a83f7441aef6e3ca586c338b77`
(`tokenizer_config.json` SHA-256
`92a6e3f894075fda19f47d8755c6ce65b2a974716307d13ae5d79196011ebb44`).
Prompts, reports, and generated completions are not included.

Kernel version 7 remains `PILOT_EXECUTION_FAIL`. Synthetic canary kernel
version 1 remains `CANARY_FAIL`. v1.0 and v1.1 configs are unchanged.

## Pinned Hugging Face / snapshot mapping

| Field | Value |
| --- | --- |
| `tokenizer.eos_token` | `<\|endoftext\|>` |
| `tokenizer.eos_token_id` | `32000` |
| `pad_token_id` | `32000` (`<\|endoftext\|>`) |
| `generation_config.eos_token_id` | `[32007, 32001, 32000]` |
| `<\|end\|>` | `32007` |
| `<\|endoftext\|>` | `32000` |
| `<\|assistant\|>` | `32001` |

Additional special tokens from `added_tokens_decoder` (IDs 32000–32010):

| ID | Token |
| --- | --- |
| 32000 | `<\|endoftext\|>` |
| 32001 | `<\|assistant\|>` |
| 32002 | `<\|placeholder1\|>` |
| 32003 | `<\|placeholder2\|>` |
| 32004 | `<\|placeholder3\|>` |
| 32005 | `<\|placeholder4\|>` |
| 32006 | `<\|system\|>` |
| 32007 | `<\|end\|>` |
| 32008 | `<\|placeholder5\|>` |
| 32009 | `<\|placeholder6\|>` |
| 32010 | `<\|user\|>` |

`special_tokens_map.json` sets `eos_token` and `pad_token` to
`<|endoftext|>`. It does not list `<|end|>` as `eos_token`.
`additional_special_tokens` is absent from that file; the Phi specials live
in `added_tokens_decoder`.

## v1.1 `model.generate()` configuration (defect)

| Field | Value |
| --- | --- |
| `eos_token_id` passed to `generate()` | `tokenizer.eos_token_id` only (`32000`) |
| Stopping criteria passed to `generate()` | none |
| `do_sample` | `false` |
| `max_new_tokens` | `2048` |
| Decode slice | `generated_ids[:, input_ids.shape[1]:]` |
| `add_generation_prompt` | `true` |

`tokenizer.eos_token_id` is **not** the Phi chat end-of-turn marker. The
pinned `generation_config.json` already lists `[32007, 32001, 32000]`.
Passing only `32000` overrode that list. `<|end|>` (`32007`) therefore did
not stop decoding.

## Exact termination defect

The v1.1 synthetic canary (kernel version 1, `CANARY_FAIL`) showed:

- 14/14 generations began with `{`
- JSON continued after an apparent top-level object
- 10/14 reached `max_new_tokens=2048`
- 13 `forbidden_extra_text`, 1 `prompt_included_in_decoded_output`
- 0 balanced JSON objects under the v1.1 remainder rule
- failure was deterministic (`repeatable: true`)

Prompt slicing and `add_generation_prompt=True` were already correct. The
remaining defect is termination: missing `<|end|>` / generation-config EOS
IDs, and no JSON-aware stop at the first closed top-level object.

## v1.2 repair (software)

- Pass the union of tokenizer EOS, `generation_config.eos_token_id`, and
  defined `<|end|>` / `<|endoftext|>` / `<|assistant|>` IDs to `generate()`.
- Pass `JsonObjectStoppingCriteria` (batch size 1, generated tokens only).
- Production parser protocol `second-reader-transport-v1.2`: exactly one
  JSON object, whitespace only after it, `targets` must be a dictionary
  (never convert a list), exactly the 12 frozen keys.

Do not launch the 58-study gold rerun from this inspection alone.
