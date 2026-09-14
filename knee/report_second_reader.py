"""Optional offline multilingual second reader.

Weights are never downloaded by this module. The reader stays disabled until
competition-rule compliance is documented *and* versioned checksummed weights
are attached as a local Kaggle input. Gold labels are not parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any

from .config import TARGET_KEYS

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

# Candidate public models. None are downloaded. Eligibility depends on
# reports/competition_pretrained_model_rules.md.
CANDIDATE_MODELS = (
    {
        "id": "xlm-roberta-base",
        "source": "https://huggingface.co/FacebookAI/xlm-roberta-base",
        "license": "MIT",
        "public": True,
        "cost": "free",
        "notes": (
            "Multilingual encoder. Would be attached as a Kaggle dataset and "
            "loaded with local_files_only=True. Not used in this revision."
        ),
    },
    {
        "id": "bert-base-multilingual-cased",
        "source": "https://huggingface.co/google-bert/bert-base-multilingual-cased",
        "license": "Apache-2.0",
        "public": True,
        "cost": "free",
        "notes": "Same attachment/offline constraints. Not used in this revision.",
    },
)

@dataclass(frozen=True)
class SecondReaderTarget:
    target_key: str
    label: str
    uncertainty: float
    evidence_class: str  # closed vocabulary; never report text
    supported_language: bool


@dataclass
class SecondReaderResult:
    enabled: bool
    skipped_reason: str
    language_supported: bool
    predictions: dict[str, SecondReaderTarget]
    model_id: str | None
    weights_sha256: str | None


def _unknown_predictions(*, reason: str, language_supported: bool) -> dict[str, SecondReaderTarget]:
    return {
        key: SecondReaderTarget(
            target_key=key,
            label=LABEL_UNKNOWN,
            uncertainty=1.0,
            evidence_class=reason,
            supported_language=language_supported,
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


def run_second_reader(
    report: str,
    *,
    config: dict | None = None,
    language_supported: bool = True,
    weights_path: str | Path | None = None,
) -> SecondReaderResult:
    """Return structured labels or a documented skip. Never fetches weights.

    Gold / expert labels are not parameters and must never be supplied.
    """
    assert_second_reader_has_no_gold_parameter()
    cfg = ((config or {}).get("second_reader") or {})
    enabled = bool(cfg.get("enabled"))
    if cfg.get("download_weights"):
        raise RuntimeError("second reader must not download weights")
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
        )
    if not language_supported:
        return SecondReaderResult(
            enabled=True,
            skipped_reason="unsupported_language",
            language_supported=False,
            predictions=_unknown_predictions(
                reason="unsupported_language", language_supported=False
            ),
            model_id=str(cfg.get("model_id") or ""),
            weights_sha256=None,
        )
    path = Path(weights_path or cfg.get("weights_path") or "")
    expected = str(cfg.get("weights_sha256") or "")
    if not path.is_file():
        return SecondReaderResult(
            enabled=True,
            skipped_reason="weights_not_attached",
            language_supported=language_supported,
            predictions=_unknown_predictions(
                reason="weights_not_attached", language_supported=language_supported
            ),
            model_id=str(cfg.get("model_id") or ""),
            weights_sha256=None,
        )
    digest = expected_sha256(path)
    if expected and digest != expected:
        raise RuntimeError("second-reader weights sha256 mismatch; refusing to load")
    # Explicitly do not load a model in this revision. The interface is ready
    # for an offline forward pass once weights are attached and reviewed.
    raise RuntimeError(
        "second reader weights are present but loading is not implemented in "
        "this revision; refuse rather than run an unreviewed model"
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
