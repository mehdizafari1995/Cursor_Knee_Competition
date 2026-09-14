"""Local language and formatting audit for radiology reports.

Detection is a deterministic script + function-word + medical-cue heuristic.
It never calls a network service or a pretrained language model, never claims
certainty, and never returns report text in aggregate outputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import re
import unicodedata

from .language import SCRIPT_RANGES, scripts_in_text

# Extended function-word and report-section cues. Scores are counts, not probabilities.
LANGUAGE_CUES: dict[str, tuple[str, ...]] = {
    "en": (
        "the", "and", "with", "without", "there", "is", "of", "findings",
        "impression", "intact", "tear", "joint", "knee", "no",
    ),
    "es": (
        "los", "las", "del", "una", "con", "sin", "hallazgos", "rotura",
        "ligamento", "menisco", "rodilla", "impresion", "impresión",
    ),
    "fr": (
        "les", "des", "une", "avec", "sans", "dans", "constatations",
        "rupture", "genou", "ligament", "menisque", "ménisque",
    ),
    "de": (
        "und", "der", "die", "das", "ohne", "mit", "befunde", "knie",
        "meniskus", "band", "keine", "fraktur",
    ),
    "it": (
        "della", "delle", "una", "con", "senza", "reperti", "ginocchio",
        "menisco", "legamento", "rotula",
    ),
    "pt": (
        "uma", "com", "sem", "dos", "das", "achados", "joelho", "menisco",
        "ligamento", "rotura",
    ),
    "nl": (
        "een", "van", "niet", "met", "zonder", "bevindingen", "knie",
        "meniscus", "band",
    ),
    "pl": (
        "oraz", "bez", "jest", "nie", "kolana", "więzadło", "łąkotka",
        "wynik", "opis",
    ),
    "tr": (
        "ve", "ile", "bir", "yok", "bulgular", "diz", "menisküs", "bağ",
        "kırık", "öykü",
    ),
            "el": (
                "και", "του", "την", "στο", "της", "των", "από", "δεν", "χωρίς",
                "ευρήματα", "γόνατος", "μηνίσκος", "σύνδεσμος", "ρήξη", "κάταγμα",
            ),
    "bg": (
        "на", "за", "при", "без", "находка", "коляно", "менискус",
        "връзка", "фрактура", "излив",
    ),
    "ru": (
        "при", "без", "находка", "колена", "мениск", "связка", "разрыв",
        "перелом", "выпот",
    ),
    "uk": (
        "при", "без", "коліна", "меніск", "зв'язка", "розрив", "перелом",
    ),
    "ro": (
        "din", "fără", "cu", "glezna", "genunchi", "menisc", "ligament",
        "constatari",
    ),
    "cs": (
        "bez", "pro", "kolene", "meniskus", "vaz", "nález", "zlomenina",
    ),
    "hu": (
        "és", "nem", "térd", "meniszkusz", "szalag", "lelet",
    ),
    "sv": (
        "och", "utan", "med", "knä", "menisk", "ligament", "fynd",
    ),
    "da": (
        "og", "uden", "med", "knæ", "menisk", "fund",
    ),
    "fi": (
        "ja", "ilman", "polven", "nivel", "meniski", "löydös",
    ),
    "ca": (
        "els", "les", "amb", "sense", "genoll", "menisc", "lligament",
    ),
}

SCRIPT_LANGUAGE_HINTS: dict[str, str] = {
    "greek": "el",
    "cyrillic": "ru",  # may be bg/uk; marked uncertain unless cues break the tie
    "arabic": "ar",
    "hebrew": "he",
    "cjk": "zh",
    "hiragana": "ja",
    "katakana": "ja",
    "hangul": "ko",
    "thai": "th",
}

CYRILLIC_SPECIFIC: dict[str, tuple[str, ...]] = {
    "bg": ("коляно", "находка", "връзка", "излив", "фрактура", "менискус"),
    "ru": ("колена", "перелом", "выпот", "связка", "разрыв", "мениск"),
    "uk": ("коліна", "зв'язка", "розрив", "меніск"),
}

SECTION_HEADER_RE = re.compile(
    r"(?i)(?:^|\n)\s*("
    r"findings?|impression|conclusion|history|indication|clinical(?:\s+history)?"
    r"|technique|protocol|comparison|opinion|summary|"
    r"hallazgos|impresi[oó]n|antecedentes|indicaci[oó]n|t[eé]cnica|"
    r"constatations|conclusion|technique|"
    r"befunde|beurteilung|technik|anamnese|"
    r"reperti|conclusione|tecnica|"
    r"achados|impress[aã]o|"
    r"ευρήματα|συμπέρασμα|ιστορικό|τεχνική|"
    r"находк[аи]|заключение|анамнез|методика|"
    r"bulgular|yorum|teknik|öykü|"
    r"wynik|opis|wnioski"
    r")\s*:",
)

BOILERPLATE_RE = re.compile(
    r"(?i)electronically signed|this report is|confidential|page\s+\d+\s+of|"
    r"i have personally reviewed|computer generated|please correlate clinically|"
    r"dicom|hl7|pacs",
)

HTML_ENTITY_RE = re.compile(r"&(?:[a-z]+|#\d+|#x[0-9a-f]+);", re.I)
REPLACEMENT_CHAR = "\ufffd"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageResult:
    """Per-report language classification. Contains no report text."""

    language: str
    score: float
    runner_up: str | None
    runner_up_score: float
    uncertain: bool
    reason: str
    scripts: tuple[str, ...]
    mixed_scripts: bool
    supported: bool
    heuristic_language: str | None = None
    heuristic_uncertain: bool = True
    ngram_language: str | None = None
    ngram_score: float = 0.0
    ngram_uncertain: bool = True
    detectors_agree: bool = False


@dataclass
class FormatAudit:
    """Encoding / structure flags. Contains no report text."""

    n_chars: int
    n_lines: int
    empty: bool
    has_replacement_char: bool
    has_control_chars: bool
    has_html_entities: bool
    non_ascii: bool
    utf8_roundtrip_ok: bool
    n_section_headers: int
    n_empty_sections: int
    boilerplate_hits: int
    length_bucket: str
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _cue_scores(tokens: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    token_set_counts = Counter(tokens)
    for lang, cues in LANGUAGE_CUES.items():
        score = float(sum(token_set_counts[cue] for cue in cues))
        if score:
            scores[lang] = score
    return scores


def _script_bonus(scripts: list[str]) -> dict[str, float]:
    bonus: dict[str, float] = {}
    for script in scripts:
        lang = SCRIPT_LANGUAGE_HINTS.get(script)
        if lang:
            bonus[lang] = bonus.get(lang, 0.0) + 4.0
    return bonus


def _disambiguate_cyrillic(tokens: list[str], scores: dict[str, float]) -> None:
    blob = " ".join(tokens)
    for lang, cues in CYRILLIC_SPECIFIC.items():
        hit = sum(1 for cue in cues if cue in blob)
        if hit:
            scores[lang] = scores.get(lang, 0.0) + 3.0 * hit


def _latin_plus_single_body_script(scripts: tuple[str, ...]) -> bool:
    """Latin MRI abbreviations plus one body script (Greek/Cyrillic) is expected."""
    body = [s for s in scripts if s in {"greek", "cyrillic"}]
    extra = [
        s for s in scripts
        if s not in {"latin", "greek", "cyrillic", "other_non_ascii"}
    ]
    return "latin" in scripts and len(body) == 1 and not extra


def detect_language_heuristic(
    text: str,
    *,
    supported: tuple[str, ...] | None = None,
    uncertain_margin: float = 0.15,
    min_score: float = 2.0,
) -> LanguageResult:
    """Cue + script heuristic only. Empty text is ``und`` (undetermined)."""
    supported_set = set(supported or LANGUAGE_CUES)
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return LanguageResult(
            language="und",
            score=0.0,
            runner_up=None,
            runner_up_score=0.0,
            uncertain=True,
            reason="empty_report",
            scripts=(),
            mixed_scripts=False,
            supported=False,
            heuristic_language="und",
            heuristic_uncertain=True,
        )

    scripts = tuple(scripts_in_text(stripped))
    tokens = _tokens(stripped)
    scores = _cue_scores(tokens)
    if "cyrillic" in scripts:
        _disambiguate_cyrillic(tokens, scores)
    for lang, bonus in _script_bonus(list(scripts)).items():
        scores[lang] = scores.get(lang, 0.0) + bonus

    if not scores:
        language = "und"
        score = 0.0
        runner_up = None
        runner_up_score = 0.0
        reason = "no_cues"
        uncertain = True
    else:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        language, score = ranked[0]
        if len(ranked) > 1:
            runner_up, runner_up_score = ranked[1]
        else:
            runner_up, runner_up_score = None, 0.0
        uncertain = False
        reason = "top_cue_score"
        if score < min_score:
            uncertain = True
            reason = "low_score"
        elif runner_up is not None and score > 0:
            relative = (score - runner_up_score) / score
            if relative < uncertain_margin:
                uncertain = True
                reason = "close_runner_up"
        if language == "ru" and "cyrillic" in scripts and runner_up in {"bg", "uk"}:
            uncertain = True
            if reason == "top_cue_score":
                reason = "cyrillic_family"

    mixed = len(scripts) > 1
    latin_plus_body = _latin_plus_single_body_script(scripts)
    if mixed and reason in {"top_cue_score", "low_score"} and not latin_plus_body:
        extra_scripts = [s for s in scripts if s not in {"latin", "other_non_ascii"}]
        if len(extra_scripts) > 1:
            uncertain = True
            reason = "mixed_scripts"

    supported_flag = language in supported_set
    return LanguageResult(
        language=language,
        score=float(score),
        runner_up=runner_up,
        runner_up_score=float(runner_up_score),
        uncertain=uncertain,
        reason=reason,
        scripts=scripts,
        mixed_scripts=mixed,
        supported=supported_flag,
        heuristic_language=language,
        heuristic_uncertain=uncertain,
    )


def detect_language(
    text: str,
    *,
    supported: tuple[str, ...] | None = None,
    uncertain_margin: float = 0.15,
    min_score: float = 2.0,
) -> LanguageResult:
    """Ensemble: cue heuristic plus independent character-trigram detector.

    Uncertain assignments stay uncertain when the two detectors disagree.
    No network, translation API, or pretrained neural model is used.
    """
    from .report_ngram_language import detect_language_ngram

    heuristic = detect_language_heuristic(
        text,
        supported=supported,
        uncertain_margin=uncertain_margin,
        min_score=min_score,
    )
    ngram = detect_language_ngram(text)
    supported_set = set(supported or LANGUAGE_CUES)

    language = heuristic.language
    score = heuristic.score
    uncertain = heuristic.uncertain
    reason = heuristic.reason
    agree = (
        ngram.language == heuristic.language
        and heuristic.language not in {"und", ""}
        and ngram.language not in {"und", ""}
    )
    if heuristic.language == "und" and ngram.language not in {"und", ""}:
        language = ngram.language
        score = ngram.score
        uncertain = True
        reason = "ngram_only_kept_uncertain"
        agree = False
    elif agree:
        if ngram.uncertain and heuristic.uncertain:
            uncertain = True
            reason = "detectors_agree_both_uncertain"
        else:
            uncertain = False
            reason = "detectors_agree"
    else:
        # Keep the heuristic code but do not promote a disagreement to certain.
        uncertain = True
        reason = "detector_disagreement"
        if heuristic.language == "und":
            language = ngram.language
            score = ngram.score

    return LanguageResult(
        language=language,
        score=float(score),
        runner_up=heuristic.runner_up,
        runner_up_score=heuristic.runner_up_score,
        uncertain=uncertain,
        reason=reason,
        scripts=heuristic.scripts,
        mixed_scripts=heuristic.mixed_scripts,
        supported=language in supported_set,
        heuristic_language=heuristic.language,
        heuristic_uncertain=heuristic.uncertain,
        ngram_language=ngram.language,
        ngram_score=float(ngram.score),
        ngram_uncertain=ngram.uncertain,
        detectors_agree=agree,
    )


def _length_bucket(n_chars: int) -> str:
    if n_chars == 0:
        return "empty"
    if n_chars < 200:
        return "short"
    if n_chars < 1500:
        return "medium"
    if n_chars < 6000:
        return "long"
    return "very_long"


def _has_control_chars(text: str) -> bool:
    for char in text:
        if char in "\t\n\r":
            continue
        if unicodedata.category(char).startswith("C"):
            return True
    return False


def audit_report_format(text: str) -> FormatAudit:
    """Encoding, empty-section, boilerplate and structure flags. No text stored."""
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    empty = not stripped
    flags: list[str] = []
    utf8_ok = True
    try:
        raw.encode("utf-8").decode("utf-8")
    except UnicodeError:
        utf8_ok = False
        flags.append("utf8_roundtrip_failed")

    replacement = REPLACEMENT_CHAR in raw
    if replacement:
        flags.append("replacement_char")
    control = _has_control_chars(raw)
    if control:
        flags.append("control_chars")
    html = bool(HTML_ENTITY_RE.search(raw))
    if html:
        flags.append("html_entities")
    non_ascii = any(ord(ch) > 127 for ch in raw)
    if non_ascii:
        flags.append("non_ascii")

    headers = list(SECTION_HEADER_RE.finditer(raw))
    n_empty_sections = 0
    for i, match in enumerate(headers):
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[start:end].strip()
        if not body:
            n_empty_sections += 1
    if n_empty_sections:
        flags.append("empty_section")
    if not headers:
        flags.append("no_section_headers")

    boilerplate = len(BOILERPLATE_RE.findall(raw))
    if boilerplate:
        flags.append("boilerplate")
    if empty:
        flags.append("empty")

    n_chars = len(raw)
    bucket = _length_bucket(n_chars)
    if bucket in {"short", "very_long"} and not empty:
        flags.append(f"length_{bucket}")

    return FormatAudit(
        n_chars=n_chars,
        n_lines=raw.count("\n") + (1 if raw else 0),
        empty=empty,
        has_replacement_char=replacement,
        has_control_chars=control,
        has_html_entities=html,
        non_ascii=non_ascii,
        utf8_roundtrip_ok=utf8_ok,
        n_section_headers=len(headers),
        n_empty_sections=n_empty_sections,
        boilerplate_hits=boilerplate,
        length_bucket=bucket,
        quality_flags=tuple(sorted(set(flags))),
    )


def aggregate_language_audit(
    reports: list[str],
    *,
    supported: tuple[str, ...] | None = None,
    uncertain_margin: float = 0.15,
    min_score: float = 2.0,
) -> dict:
    """Counts-only language and formatting audit. Never includes report text."""
    language_counts: Counter[str] = Counter()
    uncertain_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    n_uncertain = 0
    n_unsupported = 0
    n_empty = 0
    n_mixed = 0
    n_non_ascii = 0
    n_ngram_agree = 0
    n_ngram_disagree = 0
    ngram_counts: Counter[str] = Counter()
    heuristic_counts: Counter[str] = Counter()
    n_uncertain_kept = 0
    for raw in reports:
        lang = detect_language(
            raw,
            supported=supported,
            uncertain_margin=uncertain_margin,
            min_score=min_score,
        )
        fmt = audit_report_format(raw)
        language_counts[lang.language] += 1
        heuristic_counts[str(lang.heuristic_language or lang.language)] += 1
        ngram_counts[str(lang.ngram_language or "und")] += 1
        if lang.detectors_agree:
            n_ngram_agree += 1
        elif lang.ngram_language:
            n_ngram_disagree += 1
        if lang.uncertain:
            n_uncertain += 1
            uncertain_counts[lang.language] += 1
            n_uncertain_kept += 1
        reason_counts[lang.reason] += 1
        if not lang.supported:
            n_unsupported += 1
        if lang.mixed_scripts:
            n_mixed += 1
        for script in lang.scripts:
            script_counts[script] += 1
        if fmt.empty:
            n_empty += 1
        if fmt.non_ascii:
            n_non_ascii += 1
        length_counts[fmt.length_bucket] += 1
        for flag in fmt.quality_flags:
            flag_counts[flag] += 1

    return {
        "n_reports": len(reports),
        "n_empty": n_empty,
        "n_non_empty": len(reports) - n_empty,
        "n_non_ascii": n_non_ascii,
        "n_uncertain_language": n_uncertain,
        "n_unsupported_language": n_unsupported,
        "n_mixed_scripts": n_mixed,
        "n_independent_detector_agree": n_ngram_agree,
        "n_independent_detector_disagree": n_ngram_disagree,
        "n_uncertain_kept_separate": n_uncertain_kept,
        "language_counts": dict(language_counts),
        "heuristic_language_counts": dict(heuristic_counts),
        "ngram_language_counts": dict(ngram_counts),
        "uncertain_language_counts": dict(uncertain_counts),
        "reason_counts": dict(reason_counts),
        "scripts_observed": dict(script_counts),
        "quality_flag_counts": dict(flag_counts),
        "length_bucket_counts": dict(length_counts),
        "note": (
            "Language codes combine a Unicode-script + function-word heuristic "
            "with an independent offline character-trigram detector. "
            "Uncertain classifications are counted separately and are never "
            "promoted when the two detectors disagree. Report text is not exported."
        ),
    }


def language_result_to_dict(result: LanguageResult) -> dict:
    return asdict(result)
