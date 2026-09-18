# LM Format Enforcer wheels (Kaggle dataset only)

Wheel files in this directory are **not** committed. They are copied into the
private Kaggle source zip so the internet-off notebook can install from hashed
local files.

Download with:

```bash
python scripts/download_lm_format_enforcer_wheels.py
```

Hashes and licence metadata live in `configs/lm_format_enforcer_wheels.yaml`.
Installation logic lives in `knee/lm_format_enforcer_wheels.py` and
`scripts/install_lm_format_enforcer.py`.
