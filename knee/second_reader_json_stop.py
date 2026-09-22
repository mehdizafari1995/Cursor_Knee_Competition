"""JSON-aware generation stopping for the Phase 4C.2 second reader.

Inspects tokenizer / generation-config EOS IDs and stops decoding when the
first generated top-level JSON object closes. Prompt tokens are never scanned.
Batch size must be 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

PINNED_MODEL_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
PHI_END_TOKEN = "<|end|>"
PHI_END_OF_TEXT_TOKEN = "<|endoftext|>"
PHI_ASSISTANT_TOKEN = "<|assistant|>"
PHI_EOS_TOKEN_STRINGS = (PHI_END_TOKEN, PHI_END_OF_TEXT_TOKEN, PHI_ASSISTANT_TOKEN)

# Pinned microsoft/Phi-3.5-mini-instruct revision 2fe19245… (HF + tokenizer_config
# sha256 92a6e3f8…). tokenizer.eos_token_id is NOT the chat end-of-turn marker.
PINNED_PHI35_TERMINATION = {
    "revision": PINNED_MODEL_REVISION,
    "tokenizer_eos_token": PHI_END_OF_TEXT_TOKEN,
    "tokenizer_eos_token_id": 32000,
    "pad_token_id": 32000,
    "generation_config_eos_token_id": [32007, 32001, 32000],
    "end_token": PHI_END_TOKEN,
    "end_token_id": 32007,
    "endoftext_token": PHI_END_OF_TEXT_TOKEN,
    "endoftext_token_id": 32000,
    "assistant_token": PHI_ASSISTANT_TOKEN,
    "assistant_token_id": 32001,
    "added_special_tokens": {
        32000: PHI_END_OF_TEXT_TOKEN,
        32001: PHI_ASSISTANT_TOKEN,
        32002: "<|placeholder1|>",
        32003: "<|placeholder2|>",
        32004: "<|placeholder3|>",
        32005: "<|placeholder4|>",
        32006: "<|system|>",
        32007: PHI_END_TOKEN,
        32008: "<|placeholder5|>",
        32009: "<|placeholder6|>",
        32010: "<|user|>",
    },
    "v1_1_eos_token_id_passed_to_generate": 32000,
    "v1_1_stopping_criteria": [],
    "defect": (
        "generate() passed only tokenizer.eos_token_id=32000 (<|endoftext|>) and "
        "no StoppingCriteria. generation_config.eos_token_id is "
        "[32007, 32001, 32000] (<|end|>, <|assistant|>, <|endoftext|>). "
        "Phi end-of-turn is <|end|>=32007, so generation continued after `{` "
        "and after an apparent top-level object, often to max_new_tokens=2048."
    ),
}

try:
    from transformers.generation.stopping_criteria import (  # type: ignore
        StoppingCriteria as _TransformersStoppingCriteria,
    )
except Exception:  # pragma: no cover - Cursor fixtures do not load transformers
    try:
        from transformers import StoppingCriteria as _TransformersStoppingCriteria  # type: ignore
    except Exception:  # pragma: no cover
        class _TransformersStoppingCriteria:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass


@dataclass
class JsonScanState:
    """Incremental scan of generated text for one top-level JSON object."""

    closed: bool = False
    started: bool = False
    first_non_ws: str | None = None
    object_depth: int = 0
    array_depth: int = 0
    in_string: bool = False
    escape: bool = False
    unbalanced: bool = False
    close_index: int | None = None


def scan_json_object_progress(text: str) -> JsonScanState:
    """Scan ``text`` and report whether the first top-level object has closed.

    Leading whitespace is ignored. The first non-whitespace character must be
    ``{`` or the scan never reports closure (top-level arrays do not stop).
    Braces and brackets inside JSON strings are ignored. Escaped quotes and
    escaped backslashes are handled. Closure requires object depth 0, array
    depth 0, and not inside a string.
    """
    state = JsonScanState()
    if not text:
        return state
    for index, char in enumerate(text):
        if not state.started:
            if char.isspace():
                continue
            state.first_non_ws = char
            if char != "{":
                return state
            state.started = True
            state.object_depth = 1
            continue
        if state.in_string:
            if state.escape:
                state.escape = False
                continue
            if char == "\\":
                state.escape = True
                continue
            if char == '"':
                state.in_string = False
            continue
        if char == '"':
            state.in_string = True
            continue
        if char == "{":
            state.object_depth += 1
            continue
        if char == "}":
            state.object_depth -= 1
            if state.object_depth < 0:
                state.unbalanced = True
                state.object_depth = 0
                return state
            if state.object_depth == 0:
                if state.array_depth != 0:
                    state.unbalanced = True
                    return state
                state.closed = True
                state.close_index = index
                return state
            continue
        if char == "[":
            state.array_depth += 1
            continue
        if char == "]":
            state.array_depth -= 1
            if state.array_depth < 0:
                state.unbalanced = True
                state.array_depth = 0
                return state
    if state.in_string or state.escape:
        return state
    if state.started and (state.object_depth != 0 or state.array_depth != 0):
        return state
    return state


def json_object_should_stop(text: str) -> bool:
    """True when generated text contains a closed initial top-level object."""
    state = scan_json_object_progress(text)
    return bool(state.closed and not state.unbalanced)


def _sequence_ids(input_ids: Any) -> list[int]:
    if hasattr(input_ids, "shape"):
        shape = input_ids.shape
        if len(shape) == 1:
            raise ValueError("JsonObjectStoppingCriteria supports batch size 1 only")
        if int(shape[0]) != 1:
            raise ValueError("JsonObjectStoppingCriteria supports batch size 1 only")
        row = input_ids[0]
        if hasattr(row, "tolist"):
            return [int(x) for x in row.tolist()]
        return [int(x) for x in row]
    if input_ids is None:
        raise ValueError("input_ids is required")
    rows = list(input_ids)
    if len(rows) != 1:
        raise ValueError("JsonObjectStoppingCriteria supports batch size 1 only")
    return [int(x) for x in rows[0]]


class JsonObjectStoppingCriteria(_TransformersStoppingCriteria):
    """Stop ``model.generate`` when the first generated JSON object closes."""

    def __init__(self, tokenizer: Any, *, prompt_length: int) -> None:
        super().__init__()
        if int(prompt_length) < 0:
            raise ValueError("prompt_length must be >= 0")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.n_generated_tokens = 0
        self.stopped_on_json_close = False
        self.last_scan: JsonScanState | None = None

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        seq = _sequence_ids(input_ids)
        generated = seq[self.prompt_length :]
        self.n_generated_tokens = len(generated)
        if not generated:
            return False
        decode = getattr(self.tokenizer, "decode", None)
        if decode is None:
            return False
        try:
            text = decode(generated, skip_special_tokens=True)
        except TypeError:
            text = decode(generated)
        state = scan_json_object_progress(text or "")
        self.last_scan = state
        if state.closed and not state.unbalanced:
            self.stopped_on_json_close = True
            return True
        return False


def _add_token_id(bucket: dict[int, str], token_id: Any, source: str) -> None:
    if token_id is None:
        return
    if isinstance(token_id, (list, tuple)):
        for item in token_id:
            _add_token_id(bucket, item, source)
        return
    try:
        tid = int(token_id)
    except (TypeError, ValueError):
        return
    if tid < 0:
        return
    bucket.setdefault(tid, source)


def _token_content(tokenizer: Any, token_id: int) -> str | None:
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    raw = decoder.get(token_id)
    if raw is None and isinstance(decoder, dict):
        raw = decoder.get(str(token_id))
    if raw is None:
        convert = getattr(tokenizer, "convert_ids_to_tokens", None)
        if convert is None:
            return None
        try:
            return str(convert(token_id))
        except Exception:
            return None
    return str(getattr(raw, "content", raw))


def _id_for_token_string(tokenizer: Any, name: str) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        tid = convert(name)
    except Exception:
        return None
    if tid is None:
        return None
    try:
        token_id = int(tid)
    except (TypeError, ValueError):
        return None
    if token_id < 0:
        return None
    content = _token_content(tokenizer, token_id)
    if content == name:
        return token_id
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == int(unk) and name != getattr(tokenizer, "unk_token", None):
        return None
    if getattr(tokenizer, "eos_token", None) == name:
        return token_id
    return None


def collect_eos_token_ids(tokenizer: Any, model: Any = None) -> list[int]:
    """Union of tokenizer EOS, generation_config EOS, and Phi end-of-turn IDs."""
    found: dict[int, str] = {}
    _add_token_id(found, getattr(tokenizer, "eos_token_id", None), "tokenizer.eos_token_id")
    gen_cfg = getattr(model, "generation_config", None) if model is not None else None
    if gen_cfg is not None:
        _add_token_id(
            found, getattr(gen_cfg, "eos_token_id", None), "generation_config.eos_token_id"
        )
    for name in PHI_EOS_TOKEN_STRINGS:
        tid = _id_for_token_string(tokenizer, name)
        if tid is not None:
            _add_token_id(found, tid, name)
    return list(found)


def additional_special_token_map(tokenizer: Any) -> dict[str, int]:
    """Sanitized special-token string → id map. No prompts or chat templates."""
    mapping: dict[str, int] = {}
    extra = getattr(tokenizer, "additional_special_tokens", None) or []
    extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None) or []
    for name, tid in zip(extra, extra_ids):
        try:
            mapping[str(name)] = int(tid)
        except (TypeError, ValueError):
            continue
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for tid, spec in decoder.items():
        try:
            token_id = int(tid)
        except (TypeError, ValueError):
            continue
        content = str(getattr(spec, "content", spec))
        if content.startswith("<|") and content.endswith("|>"):
            mapping.setdefault(content, token_id)
    return dict(sorted(mapping.items(), key=lambda item: item[1]))


def inspect_termination_config(tokenizer: Any = None, model: Any = None) -> dict[str, Any]:
    """Sanitized EOS / special-token values. Never includes prompts or completions."""
    payload: dict[str, Any] = {
        "pinned": dict(PINNED_PHI35_TERMINATION),
        "tokenizer_eos_token": getattr(tokenizer, "eos_token", None) if tokenizer else None,
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None) if tokenizer else None,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) if tokenizer else None,
        "additional_special_tokens": additional_special_token_map(tokenizer) if tokenizer else {},
        "generation_config_eos_token_id": None,
        "end_token_id": None,
        "endoftext_token_id": None,
        "eos_token_ids_for_generate": [],
        "stopping_criteria": ["JsonObjectStoppingCriteria"],
        "stopping_criteria_batch_size": 1,
        "json_stop_examines": "generated_tokens_only",
        "chat_template_recorded": False,
        "prompts_printed": False,
        "reports_printed": False,
        "completions_printed": False,
    }
    if tokenizer is not None:
        payload["end_token_id"] = _id_for_token_string(tokenizer, PHI_END_TOKEN)
        payload["endoftext_token_id"] = _id_for_token_string(tokenizer, PHI_END_OF_TEXT_TOKEN)
        payload["eos_token_ids_for_generate"] = collect_eos_token_ids(tokenizer, model)
    if model is not None:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            payload["generation_config_eos_token_id"] = getattr(gen_cfg, "eos_token_id", None)
    return payload


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def inspect_termination_from_snapshot(
    weights_dir: str | Path,
    tokenizer: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    """Read generation_config / tokenizer_config from a local snapshot.

    Drops ``chat_template`` and any prompt-like fields. Does not decode
    generated text.
    """
    root = Path(weights_dir)
    tok_cfg = _load_json_if_present(root / "tokenizer_config.json")
    gen_cfg = _load_json_if_present(root / "generation_config.json")
    special = _load_json_if_present(root / "special_tokens_map.json")
    decoder = tok_cfg.get("added_tokens_decoder") or {}
    added: dict[int, str] = {}
    for tid, spec in decoder.items():
        try:
            token_id = int(tid)
        except (TypeError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        content = str(spec.get("content") or "")
        if content.startswith("<|") and content.endswith("|>"):
            added[token_id] = content
    runtime = inspect_termination_config(tokenizer, model)
    end_id = next((tid for tid, name in added.items() if name == PHI_END_TOKEN), None)
    eot_id = next((tid for tid, name in added.items() if name == PHI_END_OF_TEXT_TOKEN), None)
    file_eos = gen_cfg.get("eos_token_id")
    snapshot = {
        "snapshot_dir_name": root.name,
        "tokenizer_eos_token": tok_cfg.get("eos_token")
        or (special.get("eos_token") or {}).get("content"),
        "tokenizer_pad_token": tok_cfg.get("pad_token")
        or (special.get("pad_token") or {}).get("content"),
        "generation_config_eos_token_id": file_eos,
        "generation_config_pad_token_id": gen_cfg.get("pad_token_id"),
        "added_special_tokens": {str(tid): name for tid, name in sorted(added.items())},
        "end_token_id": end_id,
        "endoftext_token_id": eot_id,
        "chat_template_recorded": False,
    }
    runtime["snapshot"] = snapshot
    if tokenizer is None:
        runtime["tokenizer_eos_token"] = snapshot["tokenizer_eos_token"]
        runtime["pad_token_id"] = gen_cfg.get("pad_token_id")
        runtime["generation_config_eos_token_id"] = file_eos
        runtime["end_token_id"] = end_id
        runtime["endoftext_token_id"] = eot_id
        ids: dict[int, str] = {}
        _add_token_id(ids, eot_id, PHI_END_OF_TEXT_TOKEN)
        _add_token_id(ids, file_eos, "generation_config.eos_token_id")
        _add_token_id(ids, end_id, PHI_END_TOKEN)
        runtime["eos_token_ids_for_generate"] = list(ids)
        runtime["additional_special_tokens"] = {
            name: tid for tid, name in sorted(added.items(), key=lambda item: item[0])
        }
    return runtime


def stopping_cause(
    *,
    stopped_on_json_close: bool,
    last_token_id: int | None,
    eos_token_ids: list[int],
    n_generated_tokens: int,
    max_new_tokens: int,
) -> dict[str, bool]:
    """Classify why generation ended. JSON closure wins over EOS and max tokens."""
    eos_hit = last_token_id is not None and last_token_id in set(eos_token_ids)
    stopped_on_eos = bool(eos_hit) and not stopped_on_json_close
    reached_max = (
        int(n_generated_tokens) >= int(max_new_tokens) > 0
        and not stopped_on_json_close
        and not stopped_on_eos
    )
    return {
        "stopped_on_json_close": bool(stopped_on_json_close),
        "stopped_on_eos": stopped_on_eos,
        "ended_with_eos": bool(eos_hit),
        "reached_max_new_tokens": reached_max,
    }
