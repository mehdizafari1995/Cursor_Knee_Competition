"""Schema-constrained second-reader decoding with LM Format Enforcer 0.11.3.

No parser aliases. No post-generation clinical repair. Gold and report-only
studies are never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, TARGET_KEYS
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .lm_format_enforcer_wheels import (
    LMFE_COMMIT,
    LMFE_LICENSE,
    LMFE_PACKAGE,
    LMFE_TAG,
    LMFE_VERSION,
    LMFE_WHEEL_NAME,
    LMFE_WHEEL_SHA256,
    load_wheel_manifest,
)
from .second_reader_canary import (
    SYNTHETIC_CANARY_EXAMPLES,
    fixture_canary_generate_fn,
    run_synthetic_canary,
    synthetic_truth_fields,
)
from .second_reader_json_stop import PINNED_PHI35_TERMINATION
from .second_reader_parser import json_strict_transport, parse_second_reader_output
from .second_reader_pilot import freeze_manifest, load_pilot_config
from .second_reader_schema import (
    DEFAULT_PILOT_CONFIG_PATH_V1_4,
    DEFAULT_SCHEMA_PATH,
    SCHEMA_VERSION,
    STATUS_TO_LABEL,
    TARGET_FIELD_KEYS,
    allowed_enum_values,
    compact_output_skeleton,
    json_schema,
)

CANARY_VERSION_V1_4 = "second-reader-synthetic-canary-v1.4"
KAGGLE_CONSTRAINED_OUTPUT_DIRNAME = "rsna_second_reader_constrained_canary"
PHI_EOS_IDS = [32000, 32007, 32001]
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "vendor" / "tokenizers" / "phi-3.5-mini-instruct"
PINNED_TOKENIZER_SHA256 = {
    "tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "tokenizer_config.json": "92a6e3f894075fda19f47d8755c6ce65b2a974716307d13ae5d79196011ebb44",
}

# Last constraint-intervention summary from a transformers generate() call.
LAST_CONSTRAINT_STATS: dict[str, Any] | None = None
CONSTRAINT_STAT_LOG: list[dict[str, Any]] = []


def reset_constraint_stats() -> None:
    global LAST_CONSTRAINT_STATS
    CONSTRAINT_STAT_LOG.clear()
    LAST_CONSTRAINT_STATS = None


def empty_intervention_payload(*, source: str, note: str) -> dict[str, Any]:
    return {
        "source": source,
        "note": note,
        "n_calls": 0,
        "n_generated_tokens": 0,
        "n_top1_allowed": 0,
        "n_top1_rejected": 0,
        "intervention_rate": 0.0,
        "n_structural_interventions": 0,
        "n_clinical_enum_interventions": 0,
        "n_target_key_interventions": 0,
        "n_property_name_interventions": 0,
        "n_json_structure_interventions": 0,
        "n_other_interventions": 0,
        "interventions_by_field": {},
        "unconstrained_to_constrained_enum": {},
        "n_rank_measurable": 0,
        "mean_selected_rank": None,
        "mean_prob_gap": None,
        "raw_completions_stored": False,
        "logits_retained": False,
        "fixture_not_a_model_result": source != "transformers_logits_processor",
    }


def aggregate_constraint_stats(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    items = rows if rows is not None else list(CONSTRAINT_STAT_LOG)
    if not items:
        return empty_intervention_payload(
            source="no_model_logits",
            note="No transformers logits were observed. Do not report intervention_rate=0 as a model result.",
        )
    n_tokens = sum(int(item.get("n_generated_tokens") or 0) for item in items)
    n_allowed = sum(int(item.get("n_top1_allowed") or 0) for item in items)
    n_rejected = sum(int(item.get("n_top1_rejected") or 0) for item in items)
    n_structural = sum(int(item.get("n_structural_interventions") or 0) for item in items)
    n_clinical = sum(int(item.get("n_clinical_enum_interventions") or 0) for item in items)
    n_target_key = sum(int(item.get("n_target_key_interventions") or 0) for item in items)
    n_property_name = sum(int(item.get("n_property_name_interventions") or 0) for item in items)
    n_json_structure = sum(int(item.get("n_json_structure_interventions") or 0) for item in items)
    n_other = sum(int(item.get("n_other_interventions") or 0) for item in items)
    by_field: dict[str, int] = {}
    enum_pairs: dict[str, int] = {}
    rank_sum = 0.0
    n_rank = 0
    gap_sum = 0.0
    n_gap = 0
    for item in items:
        for key, count in (item.get("interventions_by_field") or {}).items():
            by_field[key] = by_field.get(key, 0) + int(count)
        for key, count in (item.get("unconstrained_to_constrained_enum") or {}).items():
            enum_pairs[key] = enum_pairs.get(key, 0) + int(count)
        n_item_rank = int(item.get("n_rank_measurable") or 0)
        if n_item_rank and item.get("mean_selected_rank") is not None:
            rank_sum += float(item["mean_selected_rank"]) * n_item_rank
            n_rank += n_item_rank
        n_item_gap = int(item.get("n_prob_gap_measurable") or n_item_rank)
        if n_item_gap and item.get("mean_prob_gap") is not None:
            gap_sum += float(item["mean_prob_gap"]) * n_item_gap
            n_gap += n_item_gap
    return {
        "source": "transformers_logits_processor",
        "note": "Aggregated immediately from per-token top-1 vs allowed-set comparisons. Full-vocabulary logits are not retained.",
        "n_calls": sum(int(item.get("n_calls") or 0) for item in items),
        "n_generated_tokens": n_tokens,
        "n_top1_allowed": n_allowed,
        "n_top1_rejected": n_rejected,
        "intervention_rate": (float(n_rejected) / float(n_tokens)) if n_tokens else 0.0,
        "n_structural_interventions": n_structural,
        "n_clinical_enum_interventions": n_clinical,
        "n_target_key_interventions": n_target_key,
        "n_property_name_interventions": n_property_name,
        "n_json_structure_interventions": n_json_structure,
        "n_other_interventions": n_other,
        "interventions_by_field": dict(sorted(by_field.items())),
        "unconstrained_to_constrained_enum": dict(sorted(enum_pairs.items())),
        "n_rank_measurable": n_rank,
        "mean_selected_rank": (rank_sum / n_rank) if n_rank else None,
        "n_prob_gap_measurable": n_gap,
        "mean_prob_gap": (gap_sum / n_gap) if n_gap else None,
        "raw_completions_stored": False,
        "logits_retained": False,
        "fixture_not_a_model_result": False,
    }

LMFE_SUPPORTED_FEATURES = {
    "root_object": True,
    "required_targets_object": True,
    "exactly_twelve_required_target_keys": True,
    "each_target_once": True,
    "required_fields_within_each_target": True,
    "status_enum": True,
    "confidence_enum": True,
    "evidence_category_enum": True,
    "uncertainty_or_negation_category_enum": True,
    "additionalProperties_false_every_object": True,
    "const": True,
    "generation_schema_required": False,
}


class SchemaCompatibilityError(RuntimeError):
    """The pinned LM Format Enforcer cannot enforce the existing schema exactly."""


# Pretty-printed on-disk file (indent=2, trailing newline).
FROZEN_SCHEMA_RAW_BYTES_SHA256 = (
    "639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20"
)
# Same parsed object, serialized with sort_keys and compact separators.
FROZEN_SCHEMA_CANONICAL_JSON_SHA256 = (
    "c39e6236e509c466850e501969e7e8525bd40a189ab05ed3d591e99bd30d3b05"
)


def hash_schema_payload(schema: dict[str, Any]) -> str:
    blob = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def schema_raw_bytes_sha256(path: str | Path | None = None) -> str:
    return sha256_file(Path(path or DEFAULT_SCHEMA_PATH))


def schema_canonical_json_sha256(schema: dict[str, Any] | None = None) -> str:
    payload = schema if schema is not None else validation_schema()
    return hash_schema_payload(payload)


def schema_hash_provenance(path: str | Path | None = None) -> dict[str, Any]:
    """Reconcile raw-file vs canonical JSON hashes for validation and generation.

    There is no separate generation schema file. ``generation_schema()`` returns
    ``validation_schema()``, which reads ``configs/second_reader_schema.json``.
    The two SHA-256 values differ only because the on-disk file is pretty-printed
    while decoder hashes use canonical ``sort_keys`` compact JSON.
    """
    schema_path = Path(path or DEFAULT_SCHEMA_PATH)
    raw = sha256_file(schema_path)
    parsed = json.loads(schema_path.read_text(encoding="utf-8"))
    val = validation_schema()
    gen = generation_schema()
    canonical = hash_schema_payload(parsed)
    val_canonical = hash_schema_payload(val)
    gen_canonical = hash_schema_payload(gen)
    same_parsed = parsed == val == gen
    if not same_parsed:
        raise SchemaCompatibilityError(
            "validation and generation schemas differ semantically"
        )
    if canonical != val_canonical or canonical != gen_canonical:
        raise SchemaCompatibilityError("canonical schema hashes drifted")
    return {
        "schema_raw_bytes_sha256": raw,
        "schema_canonical_json_sha256": canonical,
        "generation_schema_raw_bytes_sha256": raw,
        "generation_schema_canonical_json_sha256": gen_canonical,
        "same_parsed_object": True,
        "raw_differs_from_canonical_due_to_serialization": raw != canonical,
        "generation_schema_is_validation_schema": True,
        "generation_schema_file": "configs/second_reader_schema.json",
        "frozen_schema_raw_bytes_sha256": FROZEN_SCHEMA_RAW_BYTES_SHA256,
        "frozen_schema_canonical_json_sha256": FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
        "matches_frozen_raw": raw == FROZEN_SCHEMA_RAW_BYTES_SHA256,
        "matches_frozen_canonical": canonical == FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
        "note": (
            "The on-disk schema is pretty-printed JSON (indent=2, trailing newline). "
            "Validation and generation hashes used by the constrained decoder are "
            "SHA-256 of the same parsed object serialized with sort_keys and compact "
            "separators. Contents are identical; only serialization differs."
        ),
    }


def validation_schema() -> dict[str, Any]:
    path = DEFAULT_SCHEMA_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated = json_schema()
    if hash_schema_payload(payload) != hash_schema_payload(generated):
        raise SchemaCompatibilityError("on-disk schema drifted from json_schema()")
    return payload


def generation_schema() -> dict[str, Any]:
    """Generation schema. Identical to the validation schema for LMFE 0.11.3."""
    return validation_schema()


def _object_nodes(schema: dict[str, Any], prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(schema, dict):
        return nodes
    if schema.get("type") == "object":
        nodes.append((prefix or "$", schema))
    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for name, child in properties.items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            nodes.extend(_object_nodes(child, child_prefix))
    return nodes


def schema_semantic_fingerprint(schema: dict[str, Any]) -> dict[str, Any]:
    nodes = _object_nodes(schema)
    additional = {path: bool(node.get("additionalProperties")) for path, node in nodes}
    required = {path: list(node.get("required") or []) for path, node in nodes}
    properties = {
        path: list((node.get("properties") or {}).keys()) for path, node in nodes
    }
    enums: dict[str, list[str]] = {}
    const_paths: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if "enum" in node:
            enums[prefix or "$"] = list(node.get("enum") or [])
        if "const" in node:
            const_paths[prefix or "$"] = node.get("const")
        for name, child in (node.get("properties") or {}).items():
            child_prefix = f"{prefix}.{name}" if prefix else name
            walk(child, child_prefix)

    walk(schema, "")
    targets = ((schema.get("properties") or {}).get("targets") or {}) if isinstance(schema, dict) else {}
    target_required = list(targets.get("required") or [])
    target_properties = list((targets.get("properties") or {}).keys())
    return {
        "required_paths": required,
        "property_names": properties,
        "enum_sets": {key: sorted(values) for key, values in enums.items()},
        "const_paths": const_paths,
        "additionalProperties": additional,
        "target_order": target_required,
        "target_set": sorted(target_properties),
        "schema_sha256": hash_schema_payload(schema),
    }


def assert_generation_schema_equivalent(
    validation: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    val = validation or validation_schema()
    gen = generation or generation_schema()
    left = schema_semantic_fingerprint(val)
    right = schema_semantic_fingerprint(gen)
    mismatches: list[str] = []
    for key in ("required_paths", "property_names", "enum_sets", "additionalProperties", "target_order", "target_set"):
        if left[key] != right[key]:
            mismatches.append(key)
    if left["const_paths"] != right["const_paths"]:
        mismatches.append("const_paths")
    if mismatches:
        raise SchemaCompatibilityError(f"generation schema is not equivalent: {mismatches}")
    return {
        "equivalent": True,
        "validation_schema_sha256": left["schema_sha256"],
        "generation_schema_sha256": right["schema_sha256"],
        "fingerprint": left,
    }


def _lmfe_object_additional_properties(obj: Any) -> Any:
    return getattr(obj, "additionalProperties", None)


def audit_schema_against_lmfe(schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Confirm JsonSchemaParser enforces the existing schema exactly."""
    try:
        from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
        from lmformatenforcer.jsonschemaparser import JsonSchemaParser
    except ImportError as exc:  # pragma: no cover - optional in CPU CI
        return {"ok": False, "skipped": True, "reason": f"lmformatenforcer_unavailable:{exc}"}

    payload = schema or validation_schema()
    parser = JsonSchemaParser(payload, config=CharacterLevelParserConfig())
    root = parser.context.model_class
    failures: list[str] = []
    if getattr(root, "type", None) != "object":
        failures.append("root_not_object")
    if _lmfe_object_additional_properties(root) is not False:
        failures.append("root_additionalProperties")
    if list(getattr(root, "required", None) or []) != ["schema_version", "targets"]:
        failures.append("root_required")
    properties = getattr(root, "properties", None) or {}
    if list(properties.keys()) != ["schema_version", "targets"]:
        failures.append("root_properties")
    schema_version = properties.get("schema_version")
    extras = getattr(schema_version, "extras", None) or {}
    if extras.get("const") != SCHEMA_VERSION:
        failures.append("schema_version_const")
    targets = properties.get("targets")
    if getattr(targets, "type", None) != "object":
        failures.append("targets_not_object")
    if _lmfe_object_additional_properties(targets) is not False:
        failures.append("targets_additionalProperties")
    required_targets = list(getattr(targets, "required", None) or [])
    target_props = list((getattr(targets, "properties", None) or {}).keys())
    if required_targets != list(TARGET_KEYS) or target_props != list(TARGET_KEYS):
        failures.append("target_keys")
    allowed = allowed_enum_values()
    for key in TARGET_KEYS:
        record = (getattr(targets, "properties", None) or {}).get(key)
        if record is None:
            failures.append(f"missing_target:{key}")
            continue
        if _lmfe_object_additional_properties(record) is not False:
            failures.append(f"additionalProperties:{key}")
        if list(getattr(record, "required", None) or []) != list(TARGET_FIELD_KEYS):
            failures.append(f"required_fields:{key}")
        rec_props = getattr(record, "properties", None) or {}
        for field, values in allowed.items():
            prop = rec_props.get(field)
            got = list(getattr(prop, "enum", None) or [])
            if got != list(values):
                failures.append(f"enum:{key}.{field}")

    skeleton = compact_output_skeleton()
    walker = JsonSchemaParser(payload, config=CharacterLevelParserConfig())
    for index, char in enumerate(skeleton):
        if char not in walker.get_allowed_characters():
            failures.append(f"skeleton_rejected:{index}:{char}")
            break
        walker = walker.add_character(char)
    else:
        if not walker.can_end():
            failures.append("skeleton_cannot_end")
        allowed_after = walker.get_allowed_characters()
        if '"' in allowed_after or "," in allowed_after:
            failures.append("extra_fields_allowed_after_complete")

    if failures:
        raise SchemaCompatibilityError("; ".join(failures))
    equivalence = assert_generation_schema_equivalent(payload, payload)
    return {
        "ok": True,
        "skipped": False,
        "library": LMFE_PACKAGE,
        "version": LMFE_VERSION,
        "license": LMFE_LICENSE,
        "features": dict(LMFE_SUPPORTED_FEATURES),
        "generation_schema_required": False,
        **equivalence,
    }


def json_path_from_prefix(text: str) -> str:
    """Best-effort JSON path for synthetic intervention attribution. No report text."""
    path: list[str] = []
    in_string = False
    escape = False
    key_buf: list[str] = []
    reading_key = False
    expecting_key = True
    for char in text:
        if in_string:
            if escape:
                escape = False
                if reading_key:
                    key_buf.append(char)
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
                if reading_key:
                    path.append("".join(key_buf))
                    reading_key = False
                    key_buf = []
                continue
            if reading_key:
                key_buf.append(char)
            continue
        if char == '"':
            in_string = True
            if expecting_key:
                reading_key = True
                key_buf = []
            continue
        if char in "{[":
            expecting_key = True
            continue
        if char == ":":
            expecting_key = False
            continue
        if char == ",":
            if path:
                path.pop()
            expecting_key = True
            continue
        if char in "}]":
            if path:
                path.pop()
            expecting_key = True
    return ".".join(path)


def clinical_fields_from_parsed(parsed) -> dict[str, dict[str, str]]:
    return {
        key: {
            "status": pred.status,
            "confidence": pred.confidence_category,
            "evidence_category": pred.evidence_category,
            "uncertainty_or_negation_category": pred.uncertainty_or_negation_category,
        }
        for key, pred in parsed.predictions.items()
    }


def never_coerces_unknown_to_negative() -> bool:
    """Code must not rewrite unknown / uncertain / not_mentioned into negative."""
    if STATUS_TO_LABEL.get("uncertain") != "unknown":
        return False
    if STATUS_TO_LABEL.get("not_mentioned") != "unknown":
        return False
    if STATUS_TO_LABEL.get("negative") != "negative":
        return False
    return True


def locate_phi_tokenizer_dir(explicit: str | Path | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(DEFAULT_TOKENIZER_DIR)
    for path in candidates:
        if (path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file():
            return path
    return None


def verify_phi_tokenizer_files(directory: Path) -> dict[str, str]:
    hashes = {}
    for name, expected in PINNED_TOKENIZER_SHA256.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned tokenizer file missing: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"tokenizer hash mismatch for {name}: {digest}")
        hashes[name] = digest
    return hashes


def build_token_enforcer_from_tokenizers_json(tokenizer_json: Path, *, eos_token_ids: list[int] | None = None):
    """Prefix-tree tokenizer data without importing transformers."""
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    vocab_size = int(tok.get_vocab_size())
    token_0 = tok.encode("0").ids[-1]
    special = set(eos_token_ids or PHI_EOS_IDS)
    regular_tokens: list[tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        if token_idx in special:
            continue
        decoded_after_0 = tok.decode([token_0, token_idx], skip_special_tokens=False)
        decoded_regular = tok.decode([token_idx], skip_special_tokens=False)
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        piece = decoded_after_0[1:] if decoded_after_0.startswith("0") else decoded_after_0
        if not piece:
            piece = decoded_regular
        regular_tokens.append((token_idx, piece, is_word_start))

    def decode_fn(tokens: list[int]) -> str:
        return tok.decode(tokens, skip_special_tokens=False)

    data = TokenEnforcerTokenizerData(
        regular_tokens,
        decode_fn,
        list(eos_token_ids or PHI_EOS_IDS),
        False,
        vocab_size,
    )
    from lmformatenforcer.jsonschemaparser import JsonSchemaParser

    parser = JsonSchemaParser(generation_schema())
    return tok, TokenEnforcer(data, parser), data


def _token_string(enforcer, token_id: int) -> str:
    tree = enforcer.tokenizer_tree
    mapping = getattr(tree, "tokens_to_strs", None) or {}
    return str(mapping.get(token_id) or "")


def string_is_generable_through_prefix_tree(
    enforcer,
    tokenizer,
    text: str,
    *,
    prompt_ids: list[int] | None = None,
) -> dict[str, Any]:
    """True when the tokenizer encoding of ``text`` is allowed token-by-token."""
    seq = list(prompt_ids or [1])
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return {"ok": False, "reason": "no_encode"}
    encoded = encode(text)
    ids = list(getattr(encoded, "ids", encoded))
    enforcer.get_allowed_tokens(seq)
    for token_id in ids:
        allowed = set(int(x) for x in enforcer.get_allowed_tokens(seq).allowed_tokens)
        if int(token_id) not in allowed:
            return {
                "ok": False,
                "remaining_token_id": int(token_id),
                "n_tokens": len(seq) - len(prompt_ids or [1]),
                "eos_allowed": False,
            }
        seq.append(int(token_id))
    final_allowed = set(int(x) for x in enforcer.get_allowed_tokens(seq).allowed_tokens)
    eos_ok = all(tid in final_allowed for tid in PHI_EOS_IDS)
    return {
        "ok": True,
        "n_tokens": len(ids),
        "eos_allowed": eos_ok,
        "remaining": "",
    }


def tokenizer_prefix_tree_audit(tokenizer_dir: str | Path | None = None) -> dict[str, Any]:
    directory = locate_phi_tokenizer_dir(tokenizer_dir)
    if directory is None:
        return {"ok": False, "skipped": True, "reason": "phi_tokenizer_files_missing"}
    try:
        hashes = verify_phi_tokenizer_files(directory)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "skipped": True, "reason": str(exc)}
    tokenizer_json = directory / "tokenizer.json"
    if not tokenizer_json.is_file():
        return {"ok": False, "skipped": True, "reason": "tokenizer.json_missing"}
    try:
        tok, enforcer, _data = build_token_enforcer_from_tokenizers_json(tokenizer_json)
    except ImportError as exc:
        return {"ok": False, "skipped": True, "reason": f"prefix_tree_deps_unavailable:{exc}"}

    skeleton = compact_output_skeleton()
    skeleton_result = string_is_generable_through_prefix_tree(enforcer, tok, skeleton)
    baker = tok.encode("Baker's")
    quotes = tok.encode('"')
    apostrophe = tok.encode("'")
    escaped = tok.encode("\\n")
    structural = {char: tok.encode(char).ids for char in '{}[],:_'}
    enum_failed = []
    n_enum_ok = 0
    for values in allowed_enum_values().values():
        for value in values:
            encoded = tok.encode(value)
            if not encoded.ids:
                enum_failed.append(value)
            else:
                n_enum_ok += 1
    failed: list[str] = []
    if not skeleton_result.get("ok"):
        failed.append("compact_output_skeleton")
    if not skeleton_result.get("eos_allowed"):
        failed.append("eos_after_complete_object")
    decoded_baker = tok.decode(baker.ids)
    if "Baker's" not in decoded_baker:
        failed.append("bakers_apostrophe")
    if not quotes.ids:
        failed.append("quotes")
    if not apostrophe.ids:
        failed.append("apostrophe")
    return {
        "ok": not failed and not enum_failed,
        "skipped": False,
        "tokenizer_sha256": hashes,
        "vocab_size": tok.get_vocab_size(),
        "phi_eos_ids": PHI_EOS_IDS,
        "bakers_apostrophe_tokens": baker.tokens,
        "quote_token_ids": quotes.ids,
        "apostrophe_token_ids": apostrophe.ids,
        "escaped_token_ids": escaped.ids,
        "structural_token_ids": structural,
        "failed": failed,
        "enum_failed": enum_failed,
        "n_enum_ok": n_enum_ok,
        "skeleton": skeleton_result,
    }


class ConstraintInterventionRecorder:
    """Records unconstrained top-1 vs schema-allowed tokens. Synthetic use only."""

    def __init__(self, prefix_fn: Any, tokenizer: Any, *, prompt_length: int) -> None:
        self.prefix_fn = prefix_fn
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.events: list[dict[str, Any]] = []
        self._LogitsProcessor = None
        try:
            from transformers.generation.logits_process import LogitsProcessor

            self._LogitsProcessor = LogitsProcessor
        except Exception:
            self._LogitsProcessor = object

    def as_logits_processor(self):
        recorder = self

        class _Proc(self._LogitsProcessor):  # type: ignore[misc]
            schema_constraint_mask = True

            def __call__(self, input_ids, scores):
                return recorder._observe(input_ids, scores)

        return _Proc()

    def _observe(self, input_ids: Any, scores: Any) -> Any:
        row = input_ids[0]
        prompt_len = self.prompt_length
        generated = row[prompt_len:] if hasattr(row, "__getitem__") else []
        generated_ids = generated.tolist() if hasattr(generated, "tolist") else list(generated)
        decode = getattr(self.tokenizer, "decode", None)
        prefix_text = ""
        if decode is not None and generated_ids:
            try:
                prefix_text = decode(generated_ids, skip_special_tokens=True)
            except TypeError:
                prefix_text = decode(generated_ids)
        scores_row = scores[0]
        unconstrained_id = int(scores_row.argmax())
        allowed = list(self.prefix_fn(0, row))
        allowed_set = set(int(x) for x in allowed)
        top1_allowed = unconstrained_id in allowed_set
        selected_rank = None
        prob_gap = None
        try:
            import torch

            probs = torch.softmax(scores_row.float(), dim=-1)
            unconstrained_prob = float(probs[unconstrained_id])
            if allowed:
                allowed_idx = [tid for tid in allowed if 0 <= int(tid) < probs.numel()]
                best_allowed = max(allowed_idx, key=lambda tid: float(scores_row[int(tid)]))
                constrained_prob = float(probs[int(best_allowed)])
                order = torch.argsort(scores_row, descending=True)
                rank_map = {int(tok): int(rank) for rank, tok in enumerate(order.tolist()[:512])}
                selected_rank = rank_map.get(int(best_allowed))
                prob_gap = unconstrained_prob - constrained_prob
            else:
                constrained_prob = 0.0
                best_allowed = None
        except Exception:
            best_allowed = None
            unconstrained_prob = None
            constrained_prob = None
        path = json_path_from_prefix(prefix_text or "")
        unconstrained_piece = ""
        convert = getattr(self.tokenizer, "decode", None)
        if convert is not None:
            try:
                unconstrained_piece = convert([unconstrained_id], skip_special_tokens=True)
            except TypeError:
                unconstrained_piece = convert([unconstrained_id])
        self.events.append(
            {
                "json_path": path,
                "top1_allowed": bool(top1_allowed),
                "unconstrained_token_id": unconstrained_id,
                "unconstrained_piece": unconstrained_piece,
                "n_allowed": len(allowed_set),
                "best_allowed_token_id": None if best_allowed is None else int(best_allowed),
                "unconstrained_prob": unconstrained_prob,
                "constrained_prob": constrained_prob,
                "prob_gap": prob_gap,
                "selected_rank": selected_rank,
                "field": _field_from_path(path),
            }
        )
        return mask_scores_to_allowed(scores, allowed_set)

    def summarize(self, generated_ids: list[int] | None = None) -> dict[str, Any]:
        events = list(self.events)
        if generated_ids is not None:
            for index, token_id in enumerate(generated_ids):
                if index < len(events):
                    events[index]["selected_token_id"] = int(token_id)
        n_total = len(events)
        n_rejected = sum(1 for item in events if not item.get("top1_allowed"))
        by_field: dict[str, int] = {}
        enum_pairs: dict[str, int] = {}
        n_structural = 0
        n_clinical = 0
        n_target_key = 0
        n_property_name = 0
        n_json_structure = 0
        n_other = 0
        ranks: list[float] = []
        gaps: list[float] = []
        for item in events:
            if item.get("selected_rank") is not None:
                ranks.append(float(item["selected_rank"]))
            if item.get("prob_gap") is not None:
                gaps.append(float(item["prob_gap"]))
            if item.get("top1_allowed"):
                continue
            field = str(item.get("field") or "other")
            by_field[field] = by_field.get(field, 0) + 1
            if field in TARGET_FIELD_KEYS:
                n_clinical += 1
                proposed = str(item.get("unconstrained_piece") or "").strip().strip('"')
                selected = ""
                convert = getattr(self.tokenizer, "decode", None)
                sid = item.get("best_allowed_token_id")
                if convert is not None and sid is not None:
                    try:
                        selected = str(convert([int(sid)], skip_special_tokens=True)).strip().strip('"')
                    except TypeError:
                        selected = str(convert([int(sid)])).strip().strip('"')
                if proposed and selected and proposed != selected:
                    key = f"{field}:{proposed}->{selected}"
                    enum_pairs[key] = enum_pairs.get(key, 0) + 1
            else:
                n_structural += 1
                if field == "target_key":
                    n_target_key += 1
                elif field in {"schema_version", "targets"}:
                    n_property_name += 1
                elif field == "structure":
                    n_json_structure += 1
                else:
                    n_other += 1
        return {
            "n_generated_tokens": n_total,
            "n_calls": n_total,
            "n_top1_allowed": n_total - n_rejected,
            "n_top1_rejected": n_rejected,
            "intervention_rate": (float(n_rejected) / float(n_total)) if n_total else 0.0,
            "n_structural_interventions": n_structural,
            "n_clinical_enum_interventions": n_clinical,
            "n_target_key_interventions": n_target_key,
            "n_property_name_interventions": n_property_name,
            "n_json_structure_interventions": n_json_structure,
            "n_other_interventions": n_other,
            "interventions_by_field": dict(sorted(by_field.items())),
            "unconstrained_to_constrained_enum": dict(sorted(enum_pairs.items())),
            "n_rank_measurable": len(ranks),
            "mean_selected_rank": (sum(ranks) / len(ranks)) if ranks else None,
            "n_prob_gap_measurable": len(gaps),
            "mean_prob_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "raw_completions_stored": False,
            "logits_retained": False,
            "source": "transformers_logits_processor",
            "fixture_not_a_model_result": False,
        }


def mask_scores_to_allowed(scores: Any, allowed_ids: Any) -> Any:
    """Mask logits after unconstrained top-1 was observed. Full logits are not kept."""
    try:
        import torch
    except Exception:
        return scores
    if scores is None:
        return scores
    allowed = {int(tid) for tid in allowed_ids}
    masked = scores.clone() if hasattr(scores, "clone") else scores
    vocab = int(masked.shape[-1])
    allow = torch.zeros(vocab, dtype=torch.bool, device=masked.device)
    for tid in allowed:
        if 0 <= tid < vocab:
            allow[tid] = True
    if not bool(allow.any()):
        return masked
    fill = torch.finfo(masked.dtype).min
    if masked.dim() == 1:
        return masked.masked_fill(~allow, fill)
    return masked.masked_fill(~allow.unsqueeze(0), fill)


def _field_from_path(path: str) -> str:
    parts = [part for part in (path or "").split(".") if part]
    for part in reversed(parts):
        if part in TARGET_FIELD_KEYS:
            return part
        if part in TARGET_KEYS:
            return "target_key"
        if part in {"schema_version", "targets"}:
            return part
    return "structure"


def ensure_lmfe_transformers_compat() -> dict[str, Any]:
    """Make LM Format Enforcer 0.11.3 importable on Transformers 5.x.

    LMFE imports ``PreTrainedTokenizerBase`` from
    ``transformers.tokenization_utils``. Transformers 5 moved that symbol.
    This restores the attribute in-process. It does not change schema
    enforcement, download packages, or alias clinical enums.
    """
    import transformers
    import transformers.tokenization_utils as tokenization_utils

    patched = False
    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        cls = getattr(transformers, "PreTrainedTokenizerBase", None)
        if cls is None:
            try:
                from transformers.tokenization_utils_base import PreTrainedTokenizerBase as cls
            except Exception as exc:  # pragma: no cover - environment-specific
                raise ImportError(
                    "PreTrainedTokenizerBase is unavailable for LM Format Enforcer 0.11.3"
                ) from exc
        tokenization_utils.PreTrainedTokenizerBase = cls
        patched = True
    return {
        "transformers_version": str(getattr(transformers, "__version__", "unknown")),
        "patched_pretrained_tokenizer_base": patched,
        "internet_used": False,
    }


def attach_schema_constraints(
    tokenizer: Any,
    *,
    prompt_length: int,
    force_json_field_order: bool = False,
) -> dict[str, Any]:
    """Build a masking logits processor that records unconstrained top-1 first.

    Do not also pass ``prefix_allowed_tokens_fn`` into ``model.generate``.
    Transformers wraps that callback as ``PrefixConstrainedLogitsProcessor``,
    which can run before a recorder and make intervention_rate look like 0.
    """
    ensure_lmfe_transformers_compat()
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from lmformatenforcer.jsonschemaparser import JsonSchemaParser

    parser = JsonSchemaParser(
        generation_schema(),
        config=CharacterLevelParserConfig(force_json_field_order=force_json_field_order),
    )
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    tokenizer_data.eos_token_id = list(PHI_EOS_IDS)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
    prefix_fn.token_enforcer.root_parser.config.force_json_field_order = bool(force_json_field_order)
    recorder = ConstraintInterventionRecorder(prefix_fn, tokenizer, prompt_length=prompt_length)
    return {
        "prefix_allowed_tokens_fn": prefix_fn,
        "recorder": recorder,
        "logits_processor": recorder.as_logits_processor(),
        "eos_token_ids": list(PHI_EOS_IDS),
        "library": LMFE_PACKAGE,
        "version": LMFE_VERSION,
    }


def status_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = list(allowed_enum_values()["status"])
    per_target: dict[str, dict[str, Any]] = {}
    for key in TARGET_KEYS:
        matrix = {truth: {pred: 0 for pred in allowed} for truth in allowed}
        n_ok = 0
        n = 0
        for row in rows:
            truth = ((row.get("synthetic_truth") or {}).get(key) or {}).get("status")
            pred = ((row.get("clinical_fields") or {}).get(key) or {}).get("status")
            if truth not in allowed or pred not in allowed:
                continue
            matrix[truth][pred] += 1
            n += 1
            if truth == pred:
                n_ok += 1
        per_target[key] = {
            "accuracy": (float(n_ok) / float(n)) if n else 0.0,
            "n": n,
            "confusion": matrix,
        }
    overall_n = sum(item["n"] for item in per_target.values())
    overall_ok = sum(int(round(item["accuracy"] * item["n"])) for item in per_target.values())
    return {
        "overall_status_accuracy": (float(overall_ok) / float(overall_n)) if overall_n else 0.0,
        "per_target": per_target,
    }


def compare_clinical_fields(left: dict, right: dict) -> list[dict[str, str]]:
    diffs = []
    for key in TARGET_KEYS:
        for field in TARGET_FIELD_KEYS:
            a = str(((left or {}).get(key) or {}).get(field) or "")
            b = str(((right or {}).get(key) or {}).get(field) or "")
            if a != b:
                diffs.append({"target": key, "field": field, "unconstrained": a, "constrained": b})
    return diffs


def evaluate_semantic_gates(
    *,
    constrained_rows: list[dict[str, Any]],
    unconstrained_rows: list[dict[str, Any]] | None,
    intervention: dict[str, Any],
) -> dict[str, Any]:
    confusion = status_confusion(constrained_rows)
    unconstrained_valid_ids = [
        row["example_id"] for row in (unconstrained_rows or []) if row.get("ok")
    ]
    field_mismatches = []
    n_changed = 0
    n_target_any_changed = 0
    n_target_status_changed = 0
    agreement = {
        field: {"n_compared": 0, "n_agree": 0, "rate": 0.0} for field in TARGET_FIELD_KEYS
    }
    if unconstrained_rows:
        unconstrained_by_id = {row["example_id"]: row for row in unconstrained_rows}
        constrained_by_id = {row["example_id"]: row for row in constrained_rows}
        for example_id in unconstrained_valid_ids:
            left = unconstrained_by_id[example_id].get("clinical_fields") or {}
            right = (constrained_by_id.get(example_id) or {}).get("clinical_fields") or {}
            diffs = compare_clinical_fields(left, right)
            if diffs:
                n_changed += 1
                field_mismatches.extend({"example_id": example_id, **item} for item in diffs)
            for key in TARGET_KEYS:
                left_rec = left.get(key) or {}
                right_rec = right.get(key) or {}
                target_changed = False
                for field in TARGET_FIELD_KEYS:
                    a = str(left_rec.get(field) or "")
                    b = str(right_rec.get(field) or "")
                    if not a:
                        continue
                    agreement[field]["n_compared"] += 1
                    if a == b:
                        agreement[field]["n_agree"] += 1
                    else:
                        target_changed = True
                        if field == "status":
                            n_target_status_changed += 1
                if target_changed:
                    n_target_any_changed += 1
        for field, payload in agreement.items():
            n_cmp = payload["n_compared"]
            payload["rate"] = (float(payload["n_agree"]) / float(n_cmp)) if n_cmp else None
    code_ok = never_coerces_unknown_to_negative()
    accuracy = float(confusion["overall_status_accuracy"])
    model_intervention = not bool(intervention.get("fixture_not_a_model_result"))
    intervention_rate = float(intervention.get("intervention_rate") or 0.0)
    material_degrade = False
    if accuracy < 0.50:
        material_degrade = True
    if model_intervention and intervention_rate > 0.25 and accuracy < 0.70:
        material_degrade = True
    if n_changed >= 6 and accuracy < 0.70:
        material_degrade = True
    status = "ok"
    if not code_ok or material_degrade:
        status = "CANARY_SEMANTIC_FAIL"
    semantic_gate_status = "NOT_PREDECLARED"
    sanitized_intervention = {
        "source": intervention.get("source"),
        "note": intervention.get("note"),
        "fixture_not_a_model_result": bool(intervention.get("fixture_not_a_model_result")),
        "n_generated_tokens": intervention.get("n_generated_tokens"),
        "n_top1_allowed": intervention.get("n_top1_allowed"),
        "n_top1_rejected": intervention.get("n_top1_rejected"),
        "intervention_rate": intervention_rate if model_intervention else None,
        "n_structural_interventions": intervention.get("n_structural_interventions"),
        "n_clinical_enum_interventions": intervention.get("n_clinical_enum_interventions"),
        "n_target_key_interventions": intervention.get("n_target_key_interventions"),
        "n_property_name_interventions": intervention.get("n_property_name_interventions"),
        "n_json_structure_interventions": intervention.get("n_json_structure_interventions"),
        "n_other_interventions": intervention.get("n_other_interventions"),
        "interventions_by_field": intervention.get("interventions_by_field") or {},
        "unconstrained_to_constrained_enum": intervention.get("unconstrained_to_constrained_enum") or {},
        "n_rank_measurable": intervention.get("n_rank_measurable"),
        "mean_selected_rank": intervention.get("mean_selected_rank"),
        "mean_prob_gap": intervention.get("mean_prob_gap"),
        "raw_completions_stored": False,
        "logits_retained": False,
    }
    return {
        "status": status,
        "ok": status == "ok",
        "semantic_gate_status": semantic_gate_status,
        "code_never_coerces_unknown_to_negative": code_ok,
        "unknown_uncertain_not_mentioned_coerced_to_negative_by_code": False,
        "n_unconstrained_schema_valid": len(unconstrained_valid_ids),
        "n_outputs_clinical_fields_changed": n_changed,
        "n_target_level_clinical_decisions_changed": n_target_status_changed,
        "n_target_any_field_changed": n_target_any_changed,
        "field_agreement": agreement,
        "field_mismatches_n": len(field_mismatches),
        "field_mismatches": field_mismatches[:80],
        "synthetic_status": confusion,
        "intervention": sanitized_intervention,
        "raw_completions_stored": False,
        "report_text_stored": False,
    }


def freeze_v14_payload(config: dict, prompt: dict, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    freeze = freeze_manifest(config, prompt=prompt)
    freeze["canary_version"] = CANARY_VERSION_V1_4
    freeze["lm_format_enforcer"] = {
        "package": LMFE_PACKAGE,
        "version": LMFE_VERSION,
        "license": LMFE_LICENSE,
        "upstream_tag": LMFE_TAG,
        "upstream_commit_sha": LMFE_COMMIT,
        "wheel_filename": LMFE_WHEEL_NAME,
        "wheel_sha256": LMFE_WHEEL_SHA256,
        "manifest": load_wheel_manifest(),
    }
    freeze["tokenizer_sha256"] = dict(PINNED_TOKENIZER_SHA256)
    freeze["model_revision"] = (config.get("selected_model") or {}).get("revision")
    freeze["phi_eos_ids"] = list(PHI_EOS_IDS)
    provenance = schema_hash_provenance()
    freeze["schema_sha256"] = provenance["schema_canonical_json_sha256"]
    freeze["generation_schema_sha256"] = provenance["generation_schema_canonical_json_sha256"]
    freeze["schema_raw_bytes_sha256"] = provenance["schema_raw_bytes_sha256"]
    freeze["schema_canonical_json_sha256"] = provenance["schema_canonical_json_sha256"]
    freeze["generation_schema_raw_bytes_sha256"] = provenance["generation_schema_raw_bytes_sha256"]
    freeze["generation_schema_canonical_json_sha256"] = provenance["generation_schema_canonical_json_sha256"]
    freeze["schema_hash_note"] = provenance["note"]
    freeze["n_gold_studies_processed"] = 0
    freeze["n_report_only_studies_processed"] = 0
    if extra:
        freeze.update(extra)
    return freeze


def run_constrained_canary(
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    generate_fn=None,
    unconstrained_generate_fn=None,
    weights_path: str | Path | None = None,
    n_repeats: int = 2,
    smoke: bool = False,
    unconstrained_rows: list[dict[str, Any]] | None = None,
    intervention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Constrained synthetic canary. Never processes gold or report-only studies."""
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / KAGGLE_CONSTRAINED_OUTPUT_DIRNAME))
    cfg_path = config_path or DEFAULT_PILOT_CONFIG_PATH_V1_4
    result = run_synthetic_canary(
        output_dir=out_dir,
        config_path=cfg_path,
        generate_fn=generate_fn,
        weights_path=weights_path,
        n_repeats=n_repeats,
        smoke=smoke,
        examples=SYNTHETIC_CANARY_EXAMPLES[:3] if smoke else SYNTHETIC_CANARY_EXAMPLES,
        include_clinical_fields=True,
        include_synthetic_truth=True,
    )
    result["canary_version"] = CANARY_VERSION_V1_4
    result["constrained"] = True
    result["lm_format_enforcer_version"] = LMFE_VERSION
    result["gold_rerun_authorized"] = False
    result["gold_rerun_executed"] = False
    n_repeats_effective = int(n_repeats)
    n_examples = int(result.get("n_examples") or 0)
    n_schema_valid = int(result.get("n_schema_valid") or 0)
    n_schema_valid_all = int(result.get("n_schema_valid_all_repeats") or 0)
    if n_schema_valid_all <= 0:
        if result.get("repeatable") or n_repeats_effective == 1:
            n_schema_valid_all = n_schema_valid * n_repeats_effective
        else:
            n_schema_valid_all = n_schema_valid
    result["n_repeats"] = n_repeats_effective
    result["n_completions"] = n_examples * n_repeats_effective
    result["n_schema_valid_all_repeats"] = n_schema_valid_all
    if intervention is not None:
        intervention_payload = dict(intervention)
        if "source" not in intervention_payload:
            intervention_payload["source"] = "caller"
            intervention_payload["fixture_not_a_model_result"] = True
    elif CONSTRAINT_STAT_LOG:
        intervention_payload = aggregate_constraint_stats()
    else:
        intervention_payload = empty_intervention_payload(
            source="no_model_logits",
            note="Fixture generate_fn or missing logits processor. Do not report intervention_rate=0 as a model result.",
        )
    first_rows = result.get("example_rows") or []
    if not first_rows:
        first_rows = []
    semantic = None
    expected_valid = n_examples * n_repeats_effective
    format_ok = bool(result.get("ok")) and n_schema_valid_all == expected_valid
    if not smoke:
        semantic = evaluate_semantic_gates(
            constrained_rows=result.get("scored_rows") or first_rows,
            unconstrained_rows=unconstrained_rows,
            intervention=intervention_payload,
        )
        result["semantic"] = semantic
        result["intervention"] = semantic["intervention"]
        result["semantic_gate_status"] = semantic.get("semantic_gate_status")
        n_unconstrained_valid = int(semantic.get("n_unconstrained_schema_valid") or 0)
        n_unconstrained = len(unconstrained_rows or [])
        intervention_rate_model = semantic["intervention"].get("intervention_rate")
        model_intervention = not bool(semantic["intervention"].get("fixture_not_a_model_result"))
        diagnostic_fail = (
            model_intervention
            and n_unconstrained > 0
            and n_unconstrained_valid < n_unconstrained
            and format_ok
            and (intervention_rate_model == 0.0 or intervention_rate_model is None)
        )
        if diagnostic_fail:
            result["status"] = "INTERVENTION_DIAGNOSTIC_FAIL"
            result["ok"] = False
            result["stop_gold_rerun"] = True
            result["gold_rerun_authorized"] = False
        elif not format_ok:
            result["status"] = "CANARY_FORMAT_FAIL"
            result["ok"] = False
            result["stop_gold_rerun"] = True
        elif not semantic.get("ok"):
            result["ok"] = False
            result["status"] = "CANARY_SEMANTIC_FAIL"
            result["stop_gold_rerun"] = True
        else:
            result["status"] = "CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED"
            result["ok"] = True
            result["stop_gold_rerun"] = True
            result["gold_rerun_authorized"] = False
    else:
        result["intervention"] = intervention_payload
        if format_ok:
            result["status"] = "CONSTRAINED_SMOKE_PASS"
            result["ok"] = True
        else:
            result["status"] = "CONSTRAINED_SMOKE_FAIL"
            result["ok"] = False
            result["stop_gold_rerun"] = True
    result["n_gold_studies_processed"] = 0
    result["n_report_only_studies_processed"] = 0
    sanitized_dir = Path(result.get("output_dir") or out_dir) / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in result.items() if key not in {"freeze", "scored_rows", "example_rows"}}
    (sanitized_dir / "second_reader_constrained_canary.json").write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
