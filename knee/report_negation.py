"""Clause/sentence splitting and clinical modifiers for report-label extraction.

Deterministic ConText-style windows: negation before and after a finding,
uncertainty/hedging, history versus current, postoperative language, and
normal-structure statements. No network calls. Does not store report text
outside of in-memory spans used during extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

SECTION_TYPES = ("history", "findings", "impression", "technique", "comparison", "other")

_SECTION_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "history",
        (
            "history", "indication", "clinical", "clinical history", "clinical information",
            "reason for exam", "reason for examination", "antecedentes", "indicacion",
            "indicación", "anamnese", "anamnesis", "klinische angaben", "frage",
            "ιστορικό", "ένδειξη", "анамнез", "история", "klinik", "öykü",
        ),
    ),
    (
        "findings",
        (
            "findings", "finding", "hallazgos", "constatations", "befunde", "reperti",
            "achados", "bevindingen", "ευρήματα", "находка", "находки", "bulgular",
            "opis", "wynik",
        ),
    ),
    (
        "impression",
        (
            "impression", "conclusion", "opinion", "summary", "impresión", "impresion",
            "zusammenfassung", "beurteilung", "conclusione", "impressão",
            "συμπέρασμα", "заключение", "yorum", "wnioski",
        ),
    ),
    (
        "technique",
        (
            "technique", "protocol", "procedure", "técnica", "tecnica", "technik",
            "τεχνική", "методика", "teknik",
        ),
    ),
    (
        "comparison",
        (
            "comparison", "comparado", "comparaison", "vergleich", "σύγκριση",
            "сравнение",
        ),
    ),
)

_HEADER_RE = re.compile(
    r"(?im)^[ \t]*([A-Za-zÀ-ÖØ-öø-ÿΑ-ωА-яЁёİıŞşĞğÇç][A-Za-zÀ-ÖØ-öø-ÿΑ-ωА-яЁёİıŞşĞğÇç /-]{0,48})\s*:\s*"
)

_SENTENCE_RE = re.compile(
    r"(?<=[\.\!\?;։。])\s+|(?:\n[ \t]*){2,}|\n(?=[A-ZΑ-ΩА-ЯЁ])"
)

_CLAUSE_RE = re.compile(
    r"\s*(?:,|/|;| - | – | — |(?<!\w)(?:but|however|although|though|mientras|pero|mais|aber|ma|mas|jedoch)\s+)\s*",
    re.IGNORECASE,
)

PRE_NEGATION = (
    "no", "not", "without", "absent", "absence of", "negative for", "denies",
    "denied", "ruled out", "rule out", "neither", "nor", "free of",
    "sin", "ningun", "ningún", "no hay", "sin evidencia", "se descarta",
    "sans", "aucune", "pas de", "ne montre pas",
    "kein", "keine", "ohne", "nicht",
    "senza", "assente", "nessun",
    "sem", "nenhum", "nao", "não",
    "zonder", "geen",
    "bez", "nie",
    "yok", "yoktur", "görülmedi", "izlenmedi",
    "δεν", "χωρίς", "αρνητικό",
    "нет", "без", "отсутствует", "нема", "няма",
)

POST_NEGATION = (
    "intact", "normal", "unremarkable", "preserved", "continuous",
    "not identified", "not seen", "not demonstrated", "is absent",
    "is negative", "excluded", "is intact", "remains intact",
    "integro", "íntegro", "intacto", "conservado",
    "intacte", "normale",
    "unauffällig", "erhalten",
    "integro", "nella norma",
    "sağlam", "intakt", "normaldir",
    "ακέραιος", "φυσιολογικός",
    "интактен", "сохранен", "цялий",
)

PSEUDO_NEGATION = (
    "not only", "not just", "gram negative", "no change in", "not limited to",
    "cannot be excluded is not",
)

UNCERTAINTY = (
    "possible", "possibly", "probable", "probably", "likely", "unlikely",
    "cannot exclude", "cannot be excluded", "could represent", "may represent",
    "may be", "might be", "suggestive of", "suspected", "suspicious for",
    "questionable", "equivocal", "indeterminate", "versus", " vs ",
    "differential includes", "concern for", "appears to", "seems",
    "compatible with", "consistent with",  # compatible/consistent kept as uncertain unless morphology is explicit
    "posible", "probablemente", "no se puede excluir", "dudoso", "sugerente",
    "possible", "evocateur", "ne peut exclure",
    "möglich", "v.a.", "verdacht", "nicht auszuschließen",
    "possibile", "dubbio",
    "possível", "não se pode excluir",
    "olası", "şüpheli", "dışlanamaz",
    "πιθανή", "δεν μπορεί να αποκλειστεί",
    "вероятно", "нельзя исключить", "възможна",
)

HISTORY = (
    "history of", "hx of", "prior", "previous", "previously", "remote",
    "old ", "status post", "s/p", "s / p", "known", "chronic history",
    "antecedente", "antecedentes", "previo", "previa", "historia de",
    "antecedent", "antecedents", "ancien",
    "anamnese", "zustand nach", "z.n.", "z. n.", "vorbeschrieben",
    "pregresso", "pregressa",
    "história", "historia de",
    "öykü", "geçirilmiş", "eski",
    "ιστορικό", "προηγούμενη",
    "анамнез", "ранее", "предишна",
)

POSTOPERATIVE = (
    "reconstruction", "reconstructed", "graft", "repair", "repaired",
    "meniscectomy", "arthroplasty", "arthroscopic", "tka", "uka",
    "acl reconstruction", "aclr", "postoperative", "post-operative",
    "postsurgical", "post-surgical", "status post", "s/p",
    "reconstruccion", "reconstrucción", "injerto", "protesis", "prótesis",
    "reconstruction", "greffe",
    "rekonstruktion", "transplantat", "prothese", "prothese",
    "ricostruzione", "protesi",
    "reconstrução", "enxerto",
    "rekonstrüksiyon", "greft", "ameliyat",
    "αποκατάσταση", "μόσχευμα",
    "реконструкция", "графт", "ендопротеза",
)

NORMAL_STRUCTURE = (
    "intact", "normal", "unremarkable", "preserved", "within normal limits",
    "no tear", "no rupture", "no injury", "continuous fibres", "continuous fibers",
    "integro", "intacto", "sin rotura",
    "intacte", "sans rupture",
    "unauffällig", "intakt",
    "nella norma",
    "sağlam", "normal",
    "ακέραιος", "φυσιολογικός",
    "интактен", "без разкъсване",
)


def normalize_text(text: str) -> str:
    """NFKC normalize and collapse whitespace. Does not strip clinical punctuation."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ")
    value = value.replace("\u2019", "'").replace("\u2018", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = re.sub(r"[ \t]+", " ", value)
    return value


def _classify_header(header: str) -> str:
    key = unicodedata.normalize("NFKC", header).strip().lower().rstrip(":")
    key = re.sub(r"\s+", " ", key)
    for section, names in _SECTION_MAP:
        if key in names or any(key.startswith(name) for name in names):
            return section
    return "other"


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int
    section: str
    index: int


@dataclass(frozen=True)
class Clause:
    text: str
    start: int
    end: int
    section: str
    sentence_index: int


@dataclass(frozen=True)
class Modifiers:
    negated: bool
    negation_position: str | None  # before / after / None
    uncertain: bool
    historical: bool
    postoperative: bool
    normal_structure: bool
    section: str
    pseudo_negation: bool


def split_sections(text: str) -> list[tuple[str, int, int]]:
    """Return (section_type, start, end) spans covering the whole document."""
    normalized = normalize_text(text)
    headers = list(_HEADER_RE.finditer(normalized))
    if not headers:
        return [("other", 0, len(normalized))]
    spans: list[tuple[str, int, int]] = []
    if headers[0].start() > 0:
        spans.append(("other", 0, headers[0].start()))
    for i, match in enumerate(headers):
        section = _classify_header(match.group(1))
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(normalized)
        spans.append((section, start, end))
    return spans


def _section_at(spans: list[tuple[str, int, int]], offset: int) -> str:
    for section, start, end in spans:
        if start <= offset < end:
            return section
    return spans[-1][0] if spans else "other"


def split_sentences(text: str) -> list[Sentence]:
    normalized = normalize_text(text)
    if not normalized.strip():
        return []
    spans = split_sections(normalized)
    parts = _SENTENCE_RE.split(normalized)
    sentences: list[Sentence] = []
    cursor = 0
    index = 0
    for part in parts:
        if part is None:
            continue
        loc = normalized.find(part, cursor)
        if loc < 0:
            loc = cursor
        end = loc + len(part)
        stripped = part.strip()
        if stripped:
            sentences.append(
                Sentence(
                    text=stripped,
                    start=loc,
                    end=end,
                    section=_section_at(spans, loc),
                    index=index,
                )
            )
            index += 1
        cursor = max(end, cursor + 1)
    if not sentences and normalized.strip():
        sentences.append(
            Sentence(
                text=normalized.strip(),
                start=0,
                end=len(normalized),
                section=_section_at(spans, 0),
                index=0,
            )
        )
    return sentences


def split_clauses(sentence: Sentence) -> list[Clause]:
    pieces = _CLAUSE_RE.split(sentence.text)
    clauses: list[Clause] = []
    cursor = 0
    body = sentence.text
    for piece in pieces:
        if piece is None:
            continue
        loc = body.find(piece, cursor)
        if loc < 0:
            loc = cursor
        end = loc + len(piece)
        stripped = piece.strip()
        if stripped:
            clauses.append(
                Clause(
                    text=stripped,
                    start=sentence.start + loc,
                    end=sentence.start + end,
                    section=sentence.section,
                    sentence_index=sentence.index,
                )
            )
        cursor = max(end, cursor + 1)
    if not clauses:
        clauses.append(
            Clause(
                text=sentence.text,
                start=sentence.start,
                end=sentence.end,
                section=sentence.section,
                sentence_index=sentence.index,
            )
        )
    return clauses


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = f" {text.lower()} "
    for phrase in phrases:
        needle = phrase.lower()
        if needle.startswith(" ") or needle.endswith(" "):
            if needle in lowered:
                return True
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", lowered):
            return True
    return False


def classify_modifiers(
    sentence_text: str,
    *,
    match_start: int,
    match_end: int,
    section: str,
) -> Modifiers:
    """Classify modifiers around a finding span inside a sentence.

    ``match_start`` / ``match_end`` are offsets into ``sentence_text``.
    """
    text = sentence_text
    before = text[:match_start]
    after = text[match_end:]
    window_before = before[-80:]
    window_after = after[:80]
    full = text

    pseudo = _contains_phrase(full, PSEUDO_NEGATION)
    hist_section = section == "history"
    historical = hist_section or _contains_phrase(full, HISTORY)
    postop = _contains_phrase(full, POSTOPERATIVE)
    uncertain = _contains_phrase(full, UNCERTAINTY)
    # "consistent with X tear" is reasonably certain; keep uncertainty for hedges.
    if re.search(r"(?i)\b(?:complete|full[- ]thickness|displaced|definite)\b", full):
        if not _contains_phrase(full, ("cannot exclude", "possible", "equivocal", "questionable")):
            uncertain = False

    negated_before = (not pseudo) and _contains_phrase(window_before, PRE_NEGATION)
    negated_after = (not pseudo) and _contains_phrase(window_after, POST_NEGATION)
    normal = _contains_phrase(full, NORMAL_STRUCTURE)
    # "intact ACL" / "ACL is intact" is a normal-structure negative, not a positive.
    if normal and not _contains_phrase(full, ("tear", "torn", "rupture", "rotura", "ρήξη", "yırtık")):
        negated_after = True

    if negated_before:
        position = "before"
    elif negated_after:
        position = "after"
    else:
        position = None

    return Modifiers(
        negated=bool(negated_before or negated_after),
        negation_position=position,
        uncertain=uncertain,
        historical=historical,
        postoperative=postop,
        normal_structure=normal,
        section=section,
        pseudo_negation=pseudo,
    )


def sentence_containing(sentences: list[Sentence], start: int, end: int) -> Sentence | None:
    for sentence in sentences:
        if sentence.start <= start < sentence.end or sentence.start < end <= sentence.end:
            return sentence
    return sentences[0] if sentences else None
