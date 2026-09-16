"""Offline multilingual instruction-model second reader.

Weights are never downloaded. Loading requires a documented compliance pass
and checksummed files attached locally. Gold labels are not parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT, TARGET_KEYS
from .second_reader_parser import (
    GenerationDiagnostics,
    ParsedSecondReaderOutput,
    diagnostics_from_text,
    parse_second_reader_output,
    transport_protocol_alias,
)
from .second_reader_schema import (
    CONFIDENCE_TO_FLOAT,
    DEFAULT_PROMPT_PATH,
    FORBIDDEN_OUTPUT_KEYS,
    STATUS_TO_LABEL,
    compact_output_skeleton,
)
from .second_reader_json_stop import (
    JsonObjectStoppingCriteria,
    collect_eos_token_ids,
    inspect_termination_from_snapshot,
    stopping_cause,
)

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_UNKNOWN = "unknown"
SECOND_READER_FORBIDDEN = {
    "gold",
    "labels",
    "y_true",
    "expert",
    "expert_labels",
    "target_labels",
    "train_labels",
    "gold_labels",
}

# Candidate public instruction models. None are downloaded here.
CANDIDATE_MODELS = (
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "source": "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "license": "MIT",
        "parameter_count": 3821079552,
        "quantization": "none",
        "expected_gpu_memory_gb": 8.0,
        "public": True,
        "cost": "free",
        "gated": False,
        "structured_json": True,
        "selected_for_gold_pilot": True,
        "notes": "Primary gold-pilot candidate. Attach the snapshot locally.",
    },
    {
        "id": "google/gemma-2-2b-it",
        "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "source": "https://huggingface.co/google/gemma-2-2b-it",
        "license": "Gemma Terms of Use",
        "parameter_count": 2614341888,
        "quantization": "none",
        "expected_gpu_memory_gb": 6.0,
        "public": True,
        "cost": "free",
        "gated": True,
        "structured_json": True,
        "selected_for_gold_pilot": False,
        "notes": "Kaggle Models first-party snapshot. HF is gated; record shard SHA-256 at attach time.",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "source": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "license": "Apache-2.0",
        "parameter_count": 7615616512,
        "quantization": "int8_or_int4_on_16gb",
        "expected_gpu_memory_gb": 8.0,
        "public": True,
        "cost": "free",
        "gated": False,
        "structured_json": True,
        "selected_for_gold_pilot": False,
        "notes": "Stronger multilingual JSON model; BF16 is tight on 16GB. Prefer 8-bit.",
    },
)


@dataclass(frozen=True)
class SecondReaderTarget:
    target_key: str
    label: str
    uncertainty: float
    evidence_class: str  # closed vocabulary; never report text
    supported_language: bool
    status: str = "not_mentioned"
    confidence_category: str = "low"
    evidence_category: str = "not_mentioned"
    uncertainty_or_negation_category: str = "not_mentioned"
    malformed: bool = False


@dataclass
class SecondReaderResult:
    enabled: bool
    skipped_reason: str
    language_supported: bool
    predictions: dict[str, SecondReaderTarget]
    model_id: str | None
    weights_sha256: str | None
    malformed: bool = False
    malformed_reason: str = ""
    dropped_forbidden_fields: tuple[str, ...] = ()
    inference_backend: str = "skipped"
    prompt_version: str | None = None
    schema_version: str | None = None
    generation: GenerationDiagnostics = field(default_factory=GenerationDiagnostics)
    malformed_stage: str = ""
    parsed_root_object: bool = False
    targets_field_present: bool = False
    targets_is_object: bool = False
    targets_not_object: bool = False
    enum_violations: tuple[dict[str, Any], ...] = ()
    extra_properties: tuple[str, ...] = ()


GenerateFn = Callable[[str], str]


def _unknown_predictions(*, reason: str, language_supported: bool) -> dict[str, SecondReaderTarget]:
    return {
        key: SecondReaderTarget(
            target_key=key,
            label=LABEL_UNKNOWN,
            uncertainty=1.0,
            evidence_class=reason,
            supported_language=language_supported,
            status="not_mentioned",
            confidence_category="low",
            evidence_category=reason if reason in {"unsupported_language", "second_reader_disabled", "weights_not_attached", "malformed_output"} else "malformed_output",
            uncertainty_or_negation_category="not_mentioned",
            malformed=reason == "malformed_output",
        )
        for key in TARGET_KEYS
    }


def assert_second_reader_has_no_gold_parameter() -> None:
    params = set(inspect.signature(run_second_reader).parameters)
    leaked = params & SECOND_READER_FORBIDDEN
    if leaked:
        raise TypeError(f"second reader must not accept gold labels: {sorted(leaked)}")


def expected_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def digest_map_from_config(cfg: dict) -> dict[str, str]:
    """Merge weight-shard and config/tokenizer checksums from a frozen config."""
    selected = cfg.get("selected_model") or {}
    merged: dict[str, str] = {}
    for key in ("shard_sha256", "file_sha256"):
        payload = selected.get(key) or cfg.get(key) or {}
        if isinstance(payload, dict):
            merged.update({str(name): str(digest) for name, digest in payload.items() if digest})
    if cfg.get("weights_sha256") and "weights" not in merged:
        merged["weights"] = str(cfg.get("weights_sha256"))
    return merged


_SNAPSHOT_CACHE: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, str]] = {}


def verify_attached_snapshot(directory: Path, digest_map: dict[str, str]) -> dict[str, str]:
    """Hash every listed snapshot file. Raise on a missing file or mismatch."""
    if not digest_map:
        raise RuntimeError("second-reader snapshot checksums are not recorded")
    cache_key = (str(Path(directory).resolve()), tuple(sorted(digest_map.items())))
    cached = _SNAPSHOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    observed: dict[str, str] = {}
    for name, expected in sorted(digest_map.items()):
        path = directory / name if directory.is_dir() else directory
        if not path.is_file():
            raise RuntimeError(f"second-reader shard missing: {name}")
        digest = expected_sha256(path)
        if expected and digest != expected:
            raise RuntimeError(f"second-reader weights sha256 mismatch for {name}")
        observed[name] = digest
    _SNAPSHOT_CACHE[cache_key] = observed
    return observed


def _verify_shards(directory: Path, shard_sha256: dict[str, str]) -> str:
    observed = verify_attached_snapshot(directory, shard_sha256)
    joined = "\n".join(f"{name}:{digest}" for name, digest in sorted(observed.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _second_reader_cfg(config: dict | None) -> dict:
    if not config:
        return {}
    if "second_reader" in config:
        return dict(config.get("second_reader") or {})
    return dict(config)


def assert_no_weight_download(cfg: dict) -> None:
    if cfg.get("download_weights"):
        raise RuntimeError("second reader must not download weights")
    if cfg.get("internet_required"):
        raise RuntimeError("second reader must not require internet")


def assert_compliance_allows_load(config: dict | None) -> None:
    cfg = dict(config or {})
    compliance = cfg.get("compliance") or {}
    if not compliance.get("passed"):
        raise RuntimeError("do not load or download weights until the compliance check passes")
    assert_no_weight_download(_second_reader_cfg(cfg))


def load_prompt_bundle(path: str | Path | None = None) -> dict:
    import yaml

    prompt_path = Path(path or DEFAULT_PROMPT_PATH)
    if not prompt_path.is_file() and not prompt_path.is_absolute():
        candidate = PROJECT_ROOT / prompt_path
        if candidate.is_file():
            prompt_path = candidate
    payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prompt file must be a mapping")
    if not payload.get("forbid_chain_of_thought") or not payload.get("forbid_report_quotations"):
        raise ValueError("prompt must forbid chain-of-thought and quotations")
    return payload


def render_user_prompt(report: str, *, prompt: dict | None = None) -> str:
    bundle = prompt if prompt is not None else load_prompt_bundle()
    template = str(bundle.get("user_template") or "")
    filled = template.replace("{compact_json_skeleton}", compact_output_skeleton())
    return filled.replace("{report}", report or "")


def _predictions_from_parsed(
    parsed: ParsedSecondReaderOutput, *, language_supported: bool
) -> dict[str, SecondReaderTarget]:
    out: dict[str, SecondReaderTarget] = {}
    for key, item in parsed.targets.items():
        out[key] = SecondReaderTarget(
            target_key=key,
            label=item.label,
            uncertainty=1.0 - float(item.confidence_value),
            evidence_class=item.evidence_category,
            supported_language=language_supported,
            status=item.status,
            confidence_category=item.confidence_category if hasattr(item, "confidence_category") else item.confidence,
            evidence_category=item.evidence_category,
            uncertainty_or_negation_category=item.uncertainty_or_negation_category,
            malformed=item.malformed,
        )
    return out


def locate_weights_directory(cfg: dict, weights_path: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if weights_path:
        candidates.append(Path(weights_path))
    if cfg.get("weights_path"):
        candidates.append(Path(str(cfg.get("weights_path"))))
    env_path = os.environ.get("SECOND_READER_WEIGHTS")
    if env_path:
        candidates.append(Path(env_path))
    kaggle_input = Path("/kaggle/input")
    skip_dirs = {"train_series", "test_series", ".git", "__pycache__"}
    if kaggle_input.is_dir():
        for top in sorted(kaggle_input.iterdir()):
            if not top.is_dir() or top.name in {"competitions", "rsna-knee-abnormality-detection"}:
                continue
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = [name for name in dirnames if name not in skip_dirs]
                if "config.json" in filenames:
                    parent = Path(dirpath)
                    if (parent / "tokenizer.json").is_file() or (
                        parent / "tokenizer_config.json"
                    ).is_file():
                        candidates.append(parent)
    for path in candidates:
        if path.is_file():
            path = path.parent
        if path.is_dir() and (path / "config.json").is_file():
            return path
    return None


_TRANSFORMERS_CACHE: dict[str, tuple[Any, Any]] = {}
_TERMINATION_INSPECTED: set[str] = set()


def _cuda_usable() -> bool:
    """True when the attached GPU can run the current PyTorch CUDA kernels."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 7


def _load_causal_lm(weights_dir: Path, load_kwargs: dict[str, Any], selected: dict) -> Any:
    """Load with the in-library Phi-3 class when possible.

    The 2024 snapshot ships ``modeling_phi3.py``. Mixing that file with a newer
    transformers cache API raises ``DynamicCache.from_legacy_cache``. Native
    weights still load locally; remote custom code is the fallback.
    """
    from transformers import AutoModelForCausalLM

    native = dict(load_kwargs)
    native["trust_remote_code"] = False
    try:
        return AutoModelForCausalLM.from_pretrained(str(weights_dir), **native)
    except Exception:
        fallback = dict(load_kwargs)
        fallback["trust_remote_code"] = bool(selected.get("trust_remote_code"))
        return AutoModelForCausalLM.from_pretrained(str(weights_dir), **fallback)


def _generate_with_transformers(
    report: str,
    *,
    weights_dir: Path,
    cfg: dict,
    prompt: dict,
) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - Cursor CPU fixtures do not load weights
        raise RuntimeError("transformers_unavailable") from exc

    selected = cfg.get("selected_model") or cfg
    decoding = cfg.get("decoding") or {}
    cache_key = str(Path(weights_dir).resolve())
    packed = _TRANSFORMERS_CACHE.get(cache_key)
    if packed is None:
        tokenizer = AutoTokenizer.from_pretrained(
            str(weights_dir),
            local_files_only=True,
            trust_remote_code=bool(selected.get("trust_remote_code")),
        )
        use_cuda = _cuda_usable()
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": torch.float16 if use_cuda else torch.float32,
            "device_map": "auto" if use_cuda else "cpu",
        }
        try:
            load_kwargs["attn_implementation"] = "eager"
            model = _load_causal_lm(weights_dir, load_kwargs, selected)
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            model = _load_causal_lm(weights_dir, load_kwargs, selected)
        model.eval()
        _TRANSFORMERS_CACHE[cache_key] = (tokenizer, model)
    else:
        tokenizer, model = packed
    messages = [
        {"role": "system", "content": str(prompt.get("system") or "")},
        {"role": "user", "content": render_user_prompt(report, prompt=prompt)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    else:
        encoded = tokenizer(render_user_prompt(report, prompt=prompt), return_tensors="pt")
    inputs: dict[str, Any] = {}
    if hasattr(encoded, "items") and not hasattr(encoded, "shape"):
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in encoded:
                inputs[key] = encoded[key]
    if "input_ids" not in inputs:
        inputs = {"input_ids": encoded}
    if hasattr(inputs["input_ids"], "dim") and inputs["input_ids"].dim() == 1:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
        if "attention_mask" in inputs and hasattr(inputs["attention_mask"], "dim"):
            if inputs["attention_mask"].dim() == 1:
                inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = None
    if model_device is not None:
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    do_sample = bool(decoding.get("do_sample") or False)
    protocol = transport_protocol_alias(cfg.get("transport_protocol_version"))
    use_json_stop = protocol == "v1.2" or bool(decoding.get("json_object_stop"))
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(decoding.get("max_new_tokens") or 1024),
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(decoding.get("temperature") or 0.0) or 1e-5
        generate_kwargs["top_p"] = float(decoding.get("top_p") or 1.0)
    eos_ids: list[int] = []
    stopper: JsonObjectStoppingCriteria | None = None
    stopping_names: list[str] = []
    if use_json_stop:
        eos_ids = collect_eos_token_ids(tokenizer, model)
        if not eos_ids and tokenizer.eos_token_id is not None:
            eos_ids = [int(tokenizer.eos_token_id)]
        generate_kwargs["eos_token_id"] = eos_ids if len(eos_ids) != 1 else eos_ids[0]
        prompt_length = int(inputs["input_ids"].shape[1])
        stopper = JsonObjectStoppingCriteria(tokenizer, prompt_length=prompt_length)
        try:
            from transformers import StoppingCriteriaList
        except Exception:
            StoppingCriteriaList = list  # type: ignore[misc, assignment]
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([stopper])
        stopping_names = ["JsonObjectStoppingCriteria"]
        try:
            cache_key = str(Path(weights_dir).resolve())
            if cache_key not in _TERMINATION_INSPECTED:
                inspect_payload = inspect_termination_from_snapshot(
                    weights_dir, tokenizer, model
                )
                inspect_payload["eos_token_ids_for_generate"] = eos_ids
                inspect_payload["stopping_criteria"] = stopping_names
                print(
                    "termination_inspection="
                    + json.dumps(inspect_payload, sort_keys=True, default=str)
                )
                _TERMINATION_INSPECTED.add(cache_key)
        except Exception:
            pass
    else:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None:
            eos_ids = [int(tokenizer.eos_token_id)]
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    input_ids = inputs["input_ids"]
    generated_ids = output_ids[:, input_ids.shape[1]:]
    generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    n_new = int(generated_ids.shape[-1])
    if stopper is not None and stopper.n_generated_tokens:
        n_new = int(stopper.n_generated_tokens)
    max_new = int(generate_kwargs["max_new_tokens"])
    last_id = int(generated_ids[0, -1]) if n_new else None
    causes = stopping_cause(
        stopped_on_json_close=bool(stopper and stopper.stopped_on_json_close),
        last_token_id=last_id,
        eos_token_ids=eos_ids,
        n_generated_tokens=n_new,
        max_new_tokens=max_new,
    )
    if not use_json_stop:
        eos_ids_diag = set(eos_ids)
        eos_ids_diag.add(pad_token_id)
        try:
            end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        except Exception:
            end_id = None
        if end_id is not None:
            eos_ids_diag.add(end_id)
        causes["ended_with_eos"] = last_id in eos_ids_diag if last_id is not None else False
        causes["stopped_on_eos"] = bool(causes["ended_with_eos"]) and not causes["stopped_on_json_close"]
        causes["reached_max_new_tokens"] = n_new >= max_new and not causes["stopped_on_json_close"] and not causes["stopped_on_eos"]
    diag = diagnostics_from_text(
        generated,
        max_new_tokens=max_new,
        n_generated_tokens=n_new,
        reached_max_new_tokens=causes["reached_max_new_tokens"],
        ended_with_eos=causes["ended_with_eos"],
        stopped_on_json_close=causes["stopped_on_json_close"],
        stopped_on_eos=causes["stopped_on_eos"],
        protocol=protocol,
        eos_token_ids=tuple(eos_ids),
        stopping_criteria=tuple(stopping_names),
    )
    return generated, diag


def run_second_reader(
    report: str,
    *,
    config: dict | None = None,
    language_supported: bool = True,
    weights_path: str | Path | None = None,
    generate_fn: GenerateFn | None = None,
    prompt: dict | None = None,
) -> SecondReaderResult:
    """Return structured labels or a documented skip. Never fetches weights.

    Gold / expert labels are not parameters and must never be supplied.
    ``generate_fn`` is a local callable used by fixture tests; it is not a
    network client.
    """
    assert_second_reader_has_no_gold_parameter()
    root_cfg = dict(config or {})
    cfg = _second_reader_cfg(root_cfg)
    if "compliance" in root_cfg and "compliance" not in cfg:
        cfg["compliance"] = root_cfg.get("compliance")
    if "selected_model" in root_cfg:
        cfg["selected_model"] = root_cfg.get("selected_model")
    if "decoding" in root_cfg:
        cfg["decoding"] = root_cfg.get("decoding")
    if "transport_protocol_version" in root_cfg:
        cfg["transport_protocol_version"] = root_cfg.get("transport_protocol_version")
    enabled = bool(cfg.get("enabled"))
    model_id = str((cfg.get("selected_model") or {}).get("id") or cfg.get("model_id") or "") or None
    prompt_bundle = prompt
    prompt_version = None
    schema_version = None
    if prompt_bundle is None and enabled:
        try:
            prompt_bundle = load_prompt_bundle(root_cfg.get("prompt_path") or cfg.get("prompt_path"))
            prompt_version = str(prompt_bundle.get("prompt_version") or "")
            schema_version = str(prompt_bundle.get("schema_version") or "")
        except FileNotFoundError:
            prompt_bundle = None
    assert_no_weight_download(cfg)
    if not enabled:
        return SecondReaderResult(
            enabled=False,
            skipped_reason="disabled_until_checksummed_weights_attached",
            language_supported=language_supported,
            predictions=_unknown_predictions(
                reason="second_reader_disabled", language_supported=language_supported
            ),
            model_id=None,
            weights_sha256=None,
            inference_backend="skipped",
        )
    if not language_supported:
        return SecondReaderResult(
            enabled=True,
            skipped_reason="unsupported_language",
            language_supported=False,
            predictions=_unknown_predictions(
                reason="unsupported_language", language_supported=False
            ),
            model_id=model_id,
            weights_sha256=None,
            inference_backend="skipped",
            prompt_version=prompt_version,
            schema_version=schema_version,
        )

    parsed: ParsedSecondReaderOutput | None = None
    weights_digest = None
    backend = "fixture_generate_fn" if generate_fn is not None else "local_transformers"
    protocol = transport_protocol_alias(
        cfg.get("transport_protocol_version") or root_cfg.get("transport_protocol_version")
    )
    if generate_fn is not None:
        text = generate_fn(report)
        decoding_cfg = cfg.get("decoding") or root_cfg.get("decoding") or {}
        parsed = parse_second_reader_output(
            text,
            protocol=protocol,
            generation=diagnostics_from_text(
                text,
                max_new_tokens=int(decoding_cfg.get("max_new_tokens") or 0),
                protocol=protocol,
            ),
        )
    else:
        assert_compliance_allows_load(root_cfg if root_cfg.get("compliance") else {"compliance": cfg.get("compliance"), **cfg})
        directory = locate_weights_directory(cfg, weights_path)
        shard_map = digest_map_from_config(root_cfg if root_cfg.get("selected_model") else cfg)
        if directory is None:
            return SecondReaderResult(
                enabled=True,
                skipped_reason="weights_not_attached",
                language_supported=language_supported,
                predictions=_unknown_predictions(
                    reason="weights_not_attached", language_supported=language_supported
                ),
                model_id=model_id,
                weights_sha256=None,
                inference_backend="skipped",
                prompt_version=prompt_version,
                schema_version=schema_version,
            )
        if shard_map:
            if len(shard_map) == 1 and "weights" in shard_map:
                digest = expected_sha256(directory if directory.is_file() else next(directory.glob("*.safetensors")))
                expected = shard_map["weights"]
                if expected and digest != expected:
                    raise RuntimeError("second-reader weights sha256 mismatch; refusing to load")
                weights_digest = digest
            else:
                weights_digest = _verify_shards(directory, shard_map)
        if prompt_bundle is None:
            raise RuntimeError("frozen prompt is required before loading weights")
        try:
            text, gen_diag = _generate_with_transformers(
                report, weights_dir=directory, cfg=root_cfg if "decoding" in root_cfg else cfg, prompt=prompt_bundle
            )
        except RuntimeError as exc:
            if str(exc) == "transformers_unavailable":
                return SecondReaderResult(
                    enabled=True,
                    skipped_reason="transformers_unavailable",
                    language_supported=language_supported,
                    predictions=_unknown_predictions(
                        reason="malformed_output", language_supported=language_supported
                    ),
                    model_id=model_id,
                    weights_sha256=weights_digest,
                    malformed=True,
                    malformed_reason="transformers_unavailable",
                    malformed_stage="other",
                    inference_backend="skipped",
                    prompt_version=prompt_version,
                    schema_version=schema_version,
                )
            raise
        parsed = parse_second_reader_output(text, protocol=protocol, generation=gen_diag)

    assert parsed is not None
    return SecondReaderResult(
        enabled=True,
        skipped_reason="" if parsed.ok else parsed.malformed_reason,
        language_supported=language_supported,
        predictions=_predictions_from_parsed(parsed, language_supported=language_supported),
        model_id=model_id,
        weights_sha256=weights_digest,
        malformed=parsed.malformed,
        malformed_reason=parsed.malformed_reason,
        dropped_forbidden_fields=parsed.dropped_forbidden_fields,
        inference_backend=backend,
        prompt_version=prompt_version,
        schema_version=schema_version,
        generation=parsed.generation,
        malformed_stage="" if parsed.ok else parsed.malformed_stage,
        parsed_root_object=bool(parsed.parsed_root_object),
        targets_field_present=bool(parsed.targets_field_present),
        targets_is_object=bool(parsed.targets_is_object),
        targets_not_object=bool(parsed.targets_not_object),
        enum_violations=tuple(parsed.enum_violations),
        extra_properties=tuple(parsed.extra_properties),
    )


def combine_conservative(
    deterministic_label: str,
    deterministic_evidence: str,
    reader: SecondReaderTarget | None,
    *,
    language_supported: bool,
    global_normal: bool,
) -> tuple[str, str, str]:
    """Combine deterministic extractor and second reader.

    Returns (label, mask_reason, confidence_band).
    agreement -> eligible for higher-confidence
    disagreement -> unknown
    unsupported language -> unknown
    no finding mentioned -> unknown unless verified global-normal supports a negative
    """
    if not language_supported:
        return LABEL_UNKNOWN, "unsupported_language", "unknown"
    reader_label = reader.label if reader is not None else LABEL_UNKNOWN
    reader_enabled = reader is not None and reader.evidence_class not in {
        "second_reader_disabled",
        "weights_not_attached",
    }
    if deterministic_label == LABEL_UNKNOWN:
        if deterministic_evidence == "global_normal" or (
            global_normal and deterministic_evidence in {"global_normal", "insufficient_evidence", "no_mention"}
        ):
            if not reader_enabled or reader_label in {LABEL_NEGATIVE, LABEL_UNKNOWN}:
                return LABEL_NEGATIVE, "global_normal", "high_confidence_negative"
        return LABEL_UNKNOWN, "no_finding_mentioned", "unknown"
    if not reader_enabled:
        band = (
            "high_confidence_positive"
            if deterministic_label == LABEL_POSITIVE
            else "high_confidence_negative"
        )
        return deterministic_label, "deterministic_only", band
    if reader_label == LABEL_UNKNOWN:
        return LABEL_UNKNOWN, "second_reader_abstain", "unknown"
    if reader_label != deterministic_label:
        return LABEL_UNKNOWN, "disagreement", "unknown"
    band = (
        "high_confidence_positive"
        if deterministic_label == LABEL_POSITIVE
        else "high_confidence_negative"
    )
    return deterministic_label, "agreement", band
