"""Contract tests for the bounded first literature pilot."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import yaml

from knee.config import TARGET_KEYS

ROOT = Path(__file__).resolve().parents[1]
LITERATURE = ROOT / "literature"
PAPERS = LITERATURE / "papers.csv"
CLAIMS = LITERATURE / "claims.jsonl"

REQUIRED_COLUMNS = [
    "pmid",
    "pmcid",
    "doi",
    "title",
    "authors",
    "year",
    "journal",
    "abstract",
    "publication_type",
    "target_abnormalities",
    "source_url",
    "open_access",
    "license",
    "retrieval_date",
]
PMID_RE = re.compile(r"^[1-9]\d{0,8}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ALLOWED_LICENSES = {
    "",
    "CC0",
    "CC BY",
    "CC BY 4.0",
    "CC BY-SA",
    "CC BY-SA 4.0",
    "CC BY-NC",
    "CC BY-NC 4.0",
    "CC BY-NC-SA",
    "CC BY-NC-SA 4.0",
    "CC BY-NC-ND",
    "CC BY-NC-ND 4.0",
    "CC BY-ND",
    "CC BY-ND 4.0",
}
FORBIDDEN_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


def read_papers() -> list[dict[str, str]]:
    with PAPERS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_title(title: str) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_title.lower()).strip()


def test_required_pilot_files_exist():
    for relative_path in (
        "search_queries.json",
        "papers.csv",
        "papers.ris",
        "ontology.yaml",
        "claims.jsonl",
    ):
        assert (LITERATURE / relative_path).is_file()


def test_papers_have_exact_required_columns_and_bounded_size():
    with PAPERS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == REQUIRED_COLUMNS
        rows = list(reader)
    assert 1 <= len(rows) <= 100


def test_populated_pmids_and_dois_have_valid_formats():
    for row in read_papers():
        if row["pmid"]:
            assert PMID_RE.fullmatch(row["pmid"]), row["pmid"]
        if row["doi"]:
            assert DOI_RE.fullmatch(row["doi"]), row["doi"]


def test_papers_are_deduplicated_in_documented_priority():
    rows = read_papers()
    queries = json.loads((LITERATURE / "search_queries.json").read_text(encoding="utf-8"))
    assert queries["deduplication_priority"] == ["pmid", "doi", "normalized_title"]

    pmids = [row["pmid"] for row in rows if row["pmid"]]
    dois = [row["doi"].lower() for row in rows if row["doi"]]
    titles = [normalized_title(row["title"]) for row in rows]
    assert len(pmids) == len(set(pmids))
    assert len(dois) == len(set(dois))
    assert len(titles) == len(set(titles))


def test_all_twelve_targets_have_paper_and_ontology_coverage():
    paper_targets = {
        target.strip()
        for row in read_papers()
        for target in row["target_abnormalities"].split(";")
        if target.strip()
    }
    ontology = yaml.safe_load((LITERATURE / "ontology.yaml").read_text(encoding="utf-8"))
    ontology_targets = [record["target_id"] for record in ontology["targets"]]
    assert set(TARGET_KEYS) <= paper_targets
    assert ontology_targets == list(TARGET_KEYS)


def test_every_claim_has_a_resolvable_identifier_present_in_papers():
    papers = read_papers()
    known_pmids = {row["pmid"] for row in papers if row["pmid"]}
    known_pmcids = {row["pmcid"] for row in papers if row["pmcid"]}
    known_dois = {row["doi"].lower() for row in papers if row["doi"]}

    with CLAIMS.open(encoding="utf-8") as handle:
        claims = [json.loads(line) for line in handle if line.strip()]
    assert claims
    for claim in claims:
        assert claim.get("claim_id")
        assert claim.get("citations"), claim["claim_id"]
        for citation in claim["citations"]:
            identifiers = {
                "pmid": citation.get("pmid"),
                "pmcid": citation.get("pmcid"),
                "doi": citation.get("doi"),
            }
            assert any(identifiers.values()), claim["claim_id"]
            assert (
                identifiers["pmid"] in known_pmids
                or identifiers["pmcid"] in known_pmcids
                or (identifiers["doi"] or "").lower() in known_dois
            ), claim["claim_id"]


def test_licenses_use_allowed_verbatim_values():
    observed = {row["license"] for row in read_papers()}
    assert observed <= ALLOWED_LICENSES, observed - ALLOWED_LICENSES


def test_no_pdfs_or_figures_were_downloaded():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in LITERATURE.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    assert not offenders
