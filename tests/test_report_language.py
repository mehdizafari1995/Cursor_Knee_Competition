"""Language detection and formatting audit: counts only, no report text."""

from __future__ import annotations

from knee.report_language import (
    aggregate_language_audit,
    audit_report_format,
    detect_language,
)


def test_detects_english_spanish_french_and_marks_uncertain():
    en = detect_language("FINDINGS: The ACL is intact. There is no tear of the knee.")
    es = detect_language("Hallazgos: rotura del menisco medial de la rodilla sin fractura.")
    fr = detect_language("Constatations: rupture du ligament croise anterieur du genou.")
    assert en.language == "en"
    assert es.language == "es"
    assert fr.language == "fr"
    empty = detect_language("   ")
    assert empty.language == "und"
    assert empty.uncertain is True


def test_greek_and_cyrillic_scripts_are_recorded():
    el = detect_language("Ευρήματα: ρήξη του προσθίου χιαστού συνδέσμου του γόνατος.")
    assert "greek" in el.scripts
    assert el.language == "el"
    ru = detect_language("Находка: разрыв передней крестообразной связки колена, перелом не выявлен.")
    assert "cyrillic" in ru.scripts
    assert ru.language in {"ru", "bg", "uk"}


def test_format_audit_flags_empty_sections_and_boilerplate_without_storing_text():
    text = "TECHNIQUE:\n\nFINDINGS: intact ACL.\nElectronically signed by Dr X.\n"
    audit = audit_report_format(text)
    payload = str(audit)
    assert "intact ACL" not in payload
    assert audit.n_section_headers >= 2
    assert audit.n_empty_sections >= 1
    assert audit.boilerplate_hits >= 1
    assert "empty_section" in audit.quality_flags
    assert "boilerplate" in audit.quality_flags


def test_aggregate_language_audit_is_counts_only():
    reports = [
        "The knee findings are normal.",
        "Hallazgos de la rodilla con rotura.",
        "",
    ]
    summary = aggregate_language_audit(reports)
    blob = str(summary)
    assert "The knee findings" not in blob
    assert "Hallazgos de la rodilla" not in blob
    assert summary["n_reports"] == 3
    assert summary["n_empty"] == 1
    assert "language_counts" in summary
    assert "n_uncertain_language" in summary
