# LM Format Enforcer 0.11.3 — licence and network scan

Package: `lm-format-enforcer==0.11.3`
Licence: MIT (Copyright (c) 2023 Noam Gat)
Upstream tag: `v0.11.3`
Tag object SHA: `29514a40ccc64bdb7fc07b970cd31e5f53d1564e`
Commit SHA: `05c7dc3a46305e39c1a67c8622c433d8deda48a2`
Repository: https://github.com/noamgat/lm-format-enforcer
PyPI: https://pypi.org/project/lm-format-enforcer/0.11.3/

Direct dependency `interegular==0.3.3` is also MIT
(https://github.com/MegaIng/regex_intersections).

## Network / telemetry scan (release 0.11.3 wheel)

Scanned every `.py` file in `lm_format_enforcer-0.11.3-py3-none-any.whl` and
`interegular-0.3.3-py37-none-any.whl` for `urllib`, `requests`, `httpx`,
`aiohttp`, `socket`, `httplib`, `urlopen`, `telemetry`, `analytics`, `sentry`,
`opentelemetry`, `posthog`, `huggingface`, `download`, `pypi.org`, and
`websocket`.

Findings:

- `lmformatenforcer/jsonschemaparser.py`, `tokenenforcer.py`,
  `tokenizerprefixtree.py`, and `integrations/transformers.py` contain **no**
  network, telemetry, or runtime-download code.
- `integrations/llamacpp.py`, `integrations/haystackv1.py`, and
  `integrations/haystackv2.py` mention Hugging Face class names. Those
  adapters are **not imported** by the Transformers `prefix_allowed_tokens_fn`
  path used here.
- `interegular` contains **no** network or telemetry code.

Kaggle runs must install from the hashed local wheels with `--no-index`.
Do not `pip install` from PyPI during the notebook.
