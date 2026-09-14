"""Contract tests for the bounded first literature pilot."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import yaml

from knee.config import TARGET_KEYS

ROOT = Path(__file__).resolve().parents[1]
LITERATURE = ROOT / "literature"
PAPERS = LITERATURE / "papers.csv"
CLAIMS = LITERATURE / "claims.jsonl"
REJECTED_CLAIMS = LITERATURE / "rejected_claims.jsonl"
REPORTS = ROOT / "reports"

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


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        "rejected_claims.jsonl",
    ):
        assert (LITERATURE / relative_path).is_file()
    for report in (
        "literature_pilot.md",
        "evidence_verification.md",
        "claim_remediation.md",
    ):
        assert (REPORTS / report).is_file()


def test_reviewable_markdown_reports_are_tracked():
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.splitlines()
    )
    assert {
        "reports/literature_pilot.md",
        "reports/evidence_verification.md",
        "reports/claim_remediation.md",
    } <= tracked


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

    claims = read_jsonl(CLAIMS)
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


def test_rejected_claims_are_preserved_but_removed_from_active_evidence():
    active = read_jsonl(CLAIMS)
    rejected = read_jsonl(REJECTED_CLAIMS)
    active_ids = {claim["claim_id"] for claim in active}
    rejected_ids = {claim["claim_id"] for claim in rejected}

    assert len(active) == 83
    assert len(rejected) == 75
    assert active_ids.isdisjoint(rejected_ids)
    assert {claim["prior_verdict"] for claim in rejected} == {
        "partially_supported",
        "unsupported",
    }
    assert sum(claim["prior_verdict"] == "partially_supported" for claim in rejected) == 72
    assert sum(claim["prior_verdict"] == "unsupported" for claim in rejected) == 3

    required = {
        "claim_id",
        "original_claim",
        "cited_identifiers",
        "prior_verdict",
        "rejection_reason",
        "remediation_action",
        "remediated_at_utc",
    }
    for claim in rejected:
        assert required <= claim.keys()
        assert claim["original_claim"]
        assert claim["cited_identifiers"]
        assert claim["rejection_reason"]
        assert claim["remediation_action"] == "removed_from_active_evidence"
        assert any(
            identifier.get(key)
            for identifier in claim["cited_identifiers"]
            for key in ("pmid", "pmcid", "doi")
        )


def test_rejected_original_claims_match_the_pre_remediation_ledger():
    baseline_text = subprocess.run(
        ["git", "show", "237372c:literature/claims.jsonl"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    baseline = {
        claim["claim_id"]: claim["claim"]
        for claim in (json.loads(line) for line in baseline_text.splitlines())
    }
    for rejected in read_jsonl(REJECTED_CLAIMS):
        assert rejected["original_claim"] == baseline[rejected["claim_id"]]


def test_each_target_retains_at_least_three_active_clinical_claims():
    counts = {target: 0 for target in TARGET_KEYS}
    claims = read_jsonl(CLAIMS)
    for claim in claims:
        target = claim.get("target_id")
        if target in counts:
            counts[target] += 1
    assert all(count >= 3 for count in counts.values()), counts
    assert counts["bone_contusion"] == 5
    assert {claim["pilot_status"] for claim in claims} == {"verified"}

    ontology = yaml.safe_load((LITERATURE / "ontology.yaml").read_text(encoding="utf-8"))
    assert {
        target["verification"]["status"] for target in ontology["targets"]
    } == {"verified"}
    assert "**PASS**" in (REPORTS / "evidence_verification.md").read_text(encoding="utf-8")


def test_active_supporting_quotes_are_at_most_25_words():
    for claim in read_jsonl(CLAIMS):
        for citation in claim["citations"]:
            quote = citation.get("quote")
            if quote:
                assert len(quote.split()) <= 25, claim["claim_id"]


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
