"""Independent offline character-trigram language detector.

No network, no pretrained neural weights, no translation API, no LLM.
Profiles are built from short public-domain function-word / section-header
reference strings shipped in this module — never from competition reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import unicodedata

# Public-domain function words, numerals, and radiology *section headers*
# (not patient reports). Used only to seed compact trigram profiles.
_REFERENCE: dict[str, str] = {
    "en": (
        "the and with without there is of findings impression intact tear joint "
        "knee no history indication technique comparison conclusion summary "
        "medial lateral meniscus ligament fracture effusion"
    ),
    "es": (
        "los las del una con sin hallazgos impresion impresión rotura ligamento "
        "menisco rodilla antecedentes indicación técnica comparación conclusión "
        "medial lateral fractura derrame"
    ),
    "fr": (
        "les des une avec sans dans constatations conclusion rupture genou "
        "ligament menisque ménisque technique comparaison impression "
        "médial latéral fracture épanchement"
    ),
    "de": (
        "und der die das ohne mit befunde knie meniskus band keine fraktur "
        "technik vergleich beurteilung anamnese medial lateral"
    ),
    "it": (
        "della delle una con senza reperti ginocchio menisco legamento "
        "conclusione tecnica comparazione mediale laterale frattura"
    ),
    "pt": (
        "uma com sem dos das achados joelho menisco ligamento rotura "
        "impressão técnica comparação medial lateral fratura"
    ),
    "nl": (
        "een van niet met zonder bevindingen knie meniscus band techniek "
        "vergelijking mediaal lateraal fractuur"
    ),
    "pl": (
        "oraz bez jest nie kolana więzadło łąkotka wynik opis wnioski "
        "przyśrodkowa boczna złamanie"
    ),
    "tr": (
        "ve ile bir yok bulgular diz menisküs bağ kırık öykü teknik yorum "
        "karşılaştırma iç dış yırtık efüzyon kireçlenme"
    ),
    "el": (
        "και του την στο ευρήματα γόνατος μηνίσκος σύνδεσμος ρήξη κάταγμα "
        "ιστορικό ένδειξη τεχνική σύγκριση συμπέρασμα έσω έξω υπαρθρισμός"
    ),
    "bg": (
        "на за при без находка коляно менискус връзка фрактура излив "
        "анамнез методика заключение медиална латерална"
    ),
    "ru": (
        "при без находка колена мениск связка разрыв перелом выпот "
        "анамнез методика заключение медиальный латеральный"
    ),
    "uk": (
        "при без коліна меніск зв'язка розрив перелом анамнез "
        "методика висновок"
    ),
    "ro": (
        "din fără cu genunchi menisc ligament constatari tehnica "
        "comparatie concluzie"
    ),
    "cs": (
        "bez pro kolene meniskus vaz nález zlomenina technika "
        "srovnání závěr"
    ),
    "hu": (
        "és nem térd meniszkusz szalag lelet törés technika "
        "összehasonlítás"
    ),
    "sv": (
        "och utan med knä menisk ligament fynd teknik jämförelse "
        "sammanfattning"
    ),
    "da": (
        "og uden med knæ menisk fund teknik sammenligning "
        "konklusion"
    ),
    "fi": (
        "ja ilman polven nivel meniski löydös tekniikka "
        "vertailu yhteenveto"
    ),
    "ca": (
        "els les amb sense genoll menisc lligament tècnica "
        "comparació conclusió"
    ),
}


def _letters(text: str) -> str:
    chars: list[str] = []
    for char in unicodedata.normalize("NFKC", text).lower():
        if char.isalpha():
            chars.append(char)
        elif chars and chars[-1] != " ":
            chars.append(" ")
    return "".join(chars).strip()


def trigram_counts(text: str) -> Counter[str]:
    blob = f"  {_letters(text)}  "
    if len(blob) < 3:
        return Counter()
    return Counter(blob[i : i + 3] for i in range(len(blob) - 2))


def _unit(counter: Counter[str]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in counter.values())) or 1.0
    return {k: v / norm for k, v in counter.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = a.keys() & b.keys()
    return float(sum(a[k] * b[k] for k in keys))


_PROFILES: dict[str, dict[str, float]] = {
    lang: _unit(trigram_counts(text)) for lang, text in _REFERENCE.items()
}


@dataclass(frozen=True)
class NgramLanguageResult:
    language: str
    score: float
    runner_up: str | None
    runner_up_score: float
    uncertain: bool
    reason: str
    n_trigrams: int


def detect_language_ngram(
    text: str,
    *,
    min_score: float = 0.08,
    uncertain_margin: float = 0.12,
) -> NgramLanguageResult:
    """Independent detector. Empty text is undetermined and uncertain."""
    raw = "" if text is None else str(text)
    letters = _letters(raw)
    if not letters:
        return NgramLanguageResult(
            language="und",
            score=0.0,
            runner_up=None,
            runner_up_score=0.0,
            uncertain=True,
            reason="empty_or_no_letters",
            n_trigrams=0,
        )
    counts = trigram_counts(raw)
    n_trigrams = int(sum(counts.values()))
    vector = _unit(counts)
    ranked = sorted(
        ((lang, _cosine(vector, profile)) for lang, profile in _PROFILES.items()),
        key=lambda item: (-item[1], item[0]),
    )
    language, score = ranked[0]
    runner_up, runner_up_score = (ranked[1] if len(ranked) > 1 else (None, 0.0))
    uncertain = False
    reason = "trigram_cosine"
    if score < min_score:
        uncertain = True
        reason = "low_score"
    elif runner_up is not None and score > 0:
        relative = (score - runner_up_score) / score
        if relative < uncertain_margin:
            uncertain = True
            reason = "close_runner_up"
    # Script veto: Greek letters are decisive for el even if Latin MRI tokens exist.
    n_greek = sum(1 for ch in raw if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff")
    n_cyr = sum(1 for ch in raw if "\u0400" <= ch <= "\u04ff")
    n_latin = sum(1 for ch in raw if "A" <= ch <= "Z" or "a" <= ch <= "z")
    if n_greek >= 8 and n_greek >= n_cyr:
        language = "el"
        score = max(score, 0.5)
        uncertain = False
        reason = "greek_script_veto"
    elif n_cyr >= 8 and n_cyr > n_greek and language not in {"bg", "ru", "uk"}:
        # Keep family-level guess uncertain; do not invent a Slavic split.
        uncertain = True
        reason = "cyrillic_family"
    return NgramLanguageResult(
        language=language,
        score=float(score),
        runner_up=runner_up,
        runner_up_score=float(runner_up_score),
        uncertain=uncertain,
        reason=reason,
        n_trigrams=n_trigrams,
    )


def ngram_result_to_dict(result: NgramLanguageResult) -> dict:
    return {
        "language": result.language,
        "score": result.score,
        "runner_up": result.runner_up,
        "runner_up_score": result.runner_up_score,
        "uncertain": result.uncertain,
        "reason": result.reason,
        "n_trigrams": result.n_trigrams,
        "detector": "char_trigram",
        "pretrained_weights": False,
        "network": False,
    }
