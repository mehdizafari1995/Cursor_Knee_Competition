"""Validates the custom-agent system: agent definitions, policy docs and scaffolding.

Run with `python -m pytest tests/ -q` after editing anything under `.cursor/agents/`
or `docs/`.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / ".cursor" / "agents"
DOCS_DIR = REPO_ROOT / "docs"

# Fields documented by Cursor for `.cursor/agents/*.md` frontmatter.
ALLOWED_FRONTMATTER_KEYS = {"name", "description", "model", "readonly", "is_background"}
REQUIRED_FRONTMATTER_KEYS = ALLOWED_FRONTMATTER_KEYS

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
FRONTMATTER_PATTERN = re.compile(r"\A---\n(?P<frontmatter>.*?)\n---\n(?P<body>.*)\Z", re.DOTALL)

# agent name -> (readonly, is_background, description keywords that make it specific)
EXPECTED_AGENTS: dict[str, tuple[bool, bool, tuple[str, ...]]] = {
    "literature-searcher": (False, True, ("pubmed", "pmc", "pmid", "pmcid", "doi")),
    "clinical-ontology": (
        False,
        True,
        ("twelve", "synonym", "anatomy", "sequence", "negation", "uncertainty",
         "postoperative", "citation"),
    ),
    "evidence-verifier": (True, False, ("citation", "claim", "licence", "verdict")),
    "competition-data-auditor": (True, False, ("csv", "dicom", "missing", "leakage")),
    "repository-verifier": (True, False, ("test", "exist")),
}

REQUIRED_BODY_SECTIONS = (
    "when invoked",
    "hard rules",
    "return to the coordinator",
)

REQUIRED_RETURN_ELEMENTS = ("evidence", "file paths", "tests", "unresolved questions")

FORBIDDEN_TRACKED_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp",
    ".dcm", ".dicom", ".nii", ".mha", ".mhd", ".nrrd",
    ".ckpt", ".pt", ".pth", ".h5", ".hdf5", ".onnx", ".safetensors", ".pkl", ".joblib",
    ".pem", ".key",
}


def flat(text: str) -> str:
    """Collapse whitespace so phrase assertions survive line wrapping."""
    return re.sub(r"\s+", " ", text).lower()


def agent_path(name: str) -> Path:
    return AGENTS_DIR / f"{name}.md"


def read_agent(name: str) -> tuple[dict, str]:
    """Return (frontmatter mapping, body) for an agent definition file."""
    text = agent_path(name).read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(text)
    assert match is not None, f"{name}.md must start with a '---' delimited YAML frontmatter block"
    frontmatter = yaml.safe_load(match.group("frontmatter"))
    assert isinstance(frontmatter, dict), f"{name}.md frontmatter must parse to a mapping"
    return frontmatter, match.group("body")


AGENT_NAMES = sorted(EXPECTED_AGENTS)


class TestAgentDirectory:
    def test_agents_directory_exists(self):
        assert AGENTS_DIR.is_dir(), "expected project subagents in .cursor/agents/"

    def test_exact_set_of_agent_files(self):
        found = sorted(p.stem for p in AGENTS_DIR.glob("*.md"))
        assert found == AGENT_NAMES


@pytest.mark.parametrize("name", AGENT_NAMES)
class TestAgentFrontmatter:
    def test_file_exists(self, name):
        assert agent_path(name).is_file()

    def test_frontmatter_is_valid_yaml_mapping(self, name):
        frontmatter, _ = read_agent(name)
        assert frontmatter

    def test_required_keys_present(self, name):
        frontmatter, _ = read_agent(name)
        missing = REQUIRED_FRONTMATTER_KEYS - set(frontmatter)
        assert not missing, f"{name}.md is missing frontmatter keys: {sorted(missing)}"

    def test_no_undocumented_keys(self, name):
        frontmatter, _ = read_agent(name)
        unknown = set(frontmatter) - ALLOWED_FRONTMATTER_KEYS
        assert not unknown, f"{name}.md has undocumented frontmatter keys: {sorted(unknown)}"

    def test_name_matches_filename_and_naming_convention(self, name):
        frontmatter, _ = read_agent(name)
        assert frontmatter["name"] == name
        assert NAME_PATTERN.match(frontmatter["name"]), "use lowercase letters and hyphens"

    def test_model_is_inherit(self, name):
        frontmatter, _ = read_agent(name)
        assert frontmatter["model"] == "inherit"

    def test_readonly_and_is_background_are_booleans(self, name):
        frontmatter, _ = read_agent(name)
        assert isinstance(frontmatter["readonly"], bool), "quote-free YAML boolean required"
        assert isinstance(frontmatter["is_background"], bool), "quote-free YAML boolean required"

    def test_readonly_and_is_background_match_the_agent_contract(self, name):
        frontmatter, _ = read_agent(name)
        expected_readonly, expected_background, _ = EXPECTED_AGENTS[name]
        assert frontmatter["readonly"] is expected_readonly
        assert frontmatter["is_background"] is expected_background

    def test_description_is_specific(self, name):
        frontmatter, _ = read_agent(name)
        description = frontmatter["description"]
        assert isinstance(description, str)
        assert len(description) >= 120, "description must be specific, not a one-liner"
        assert "TODO" not in description and "TBD" not in description
        lowered = description.lower()
        _, _, keywords = EXPECTED_AGENTS[name]
        missing = [kw for kw in keywords if kw not in lowered]
        assert not missing, f"{name} description omits distinguishing terms: {missing}"


@pytest.mark.parametrize("name", AGENT_NAMES)
class TestAgentBody:
    def test_body_is_substantive(self, name):
        _, body = read_agent(name)
        assert len(body.strip()) >= 500

    def test_body_has_required_sections(self, name):
        _, body = read_agent(name)
        lowered = body.lower()
        for section in REQUIRED_BODY_SECTIONS:
            assert section in lowered, f"{name}.md body is missing a '{section}' section"

    def test_body_defines_the_return_contract(self, name):
        _, body = read_agent(name)
        lowered = body.split("return to the coordinator", 1)[-1].lower()
        for element in REQUIRED_RETURN_ELEMENTS:
            assert element in lowered, f"{name}.md must return '{element}'"

    def test_body_forbids_committing_restricted_artefacts(self, name):
        _, body = read_agent(name)
        assert re.search(r"never commit|do not commit", body, re.IGNORECASE), (
            f"{name}.md must state the no-commit policy"
        )

    def test_readonly_agents_state_their_restriction(self, name):
        frontmatter, body = read_agent(name)
        if frontmatter["readonly"]:
            assert "read-only" in body.lower()

    def test_writing_agents_require_isolated_worktrees(self, name):
        frontmatter, body = read_agent(name)
        if not frontmatter["readonly"]:
            assert re.search(r"worktree", body, re.IGNORECASE), (
                f"{name}.md must require an isolated worktree or branch for parallel writes"
            )


class TestAgentDescriptionsAreDistinct:
    def test_descriptions_are_unique(self):
        descriptions = [read_agent(name)[0]["description"] for name in AGENT_NAMES]
        assert len(set(descriptions)) == len(descriptions)


class TestScaffolding:
    @pytest.mark.parametrize(
        "relative_path",
        [
            "docs/literature_protocol.md",
            "docs/citation_and_figure_policy.md",
            "docs/clinical_ontology_schema.md",
            "reports/.gitkeep",
            "literature/.gitkeep",
            "tests/test_agent_configuration.py",
            "AGENTS.md",
            ".gitignore",
        ],
    )
    def test_required_path_exists(self, relative_path):
        assert (REPO_ROOT / relative_path).is_file()


@pytest.fixture(scope="module")
def protocol_text():
    return (DOCS_DIR / "literature_protocol.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def citation_policy_text():
    return (DOCS_DIR / "citation_and_figure_policy.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ontology_schema_text():
    return (DOCS_DIR / "clinical_ontology_schema.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gitignore_patterns():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


class TestLiteratureProtocol:
    def test_requires_reproducible_api_searches(self, protocol_text):
        text = flat(protocol_text)
        assert "e-utilities" in text
        assert "esearch" in text and "efetch" in text
        assert "pmc" in text

    def test_forbids_html_scraping(self, protocol_text):
        assert re.search(r"scrap\w+ .*prohibited", flat(protocol_text))

    def test_defines_the_all_papers_rule(self, protocol_text):
        text = flat(protocol_text)
        assert "all records returned by the documented search strategies" in text
        assert "never means that every publication" in text

    def test_requires_query_and_date_logging(self, protocol_text):
        for field in ("query", "executed_at_utc", "hit_count", "pmid", "pmcid", "doi"):
            assert field in protocol_text

    def test_forbids_paywalled_downloads(self, protocol_text):
        assert "paywalled" in flat(protocol_text)

    def test_requires_isolated_worktrees_for_parallel_work(self, protocol_text):
        assert "worktree" in flat(protocol_text)

    def test_defines_the_agent_return_contract(self, protocol_text):
        text = flat(protocol_text)
        for element in REQUIRED_RETURN_ELEMENTS:
            assert element in text


class TestCitationAndFigurePolicy:
    def test_forbids_invented_citations(self, citation_policy_text):
        assert "do not invent citations" in flat(citation_policy_text)

    def test_requires_legal_figure_access(self, citation_policy_text):
        text = flat(citation_policy_text)
        assert "open-access" in text
        assert "paywalled figures" in text

    def test_requires_licence_and_source_url_per_figure(self, citation_policy_text):
        assert "licence" in flat(citation_policy_text)
        assert "source_url" in citation_policy_text
        assert "manifest.csv" in citation_policy_text

    def test_separates_literature_figures_from_kaggle_images(self, citation_policy_text):
        text = flat(citation_policy_text)
        assert "literature/figures/" in text
        assert "never mix" in text

    def test_gates_training_use_of_figures(self, citation_policy_text):
        text = flat(citation_policy_text)
        assert "licence review" in text
        assert "patient-level label verification" in text
        assert "coordinator approval" in text

    def test_lists_never_committed_artefacts(self, citation_policy_text):
        text = flat(citation_policy_text)
        for artefact in ("pdf", "figure", "dicom", "report", "credential", "checkpoint"):
            assert artefact in text


class TestClinicalOntologySchema:
    def test_registry_has_exactly_twelve_target_slots(self, ontology_schema_text):
        text = ontology_schema_text
        slots = [int(m) for m in re.findall(r"^\|\s*(\d+)\s*\|", text, re.MULTILINE)]
        assert slots == list(range(1, 13))

    def test_registry_targets_are_not_invented(self, ontology_schema_text):
        assert "competition data dictionary" in flat(ontology_schema_text)

    def test_schema_covers_every_required_field(self, ontology_schema_text):
        for field in (
            "definition",
            "synonyms",
            "anatomy",
            "mri_findings",
            "sequences",
            "negation_cues",
            "uncertainty_terms",
            "postoperative_terms",
        ):
            assert field in ontology_schema_text, f"schema is missing the '{field}' field"

    def test_requires_a_citation_per_claim(self, ontology_schema_text):
        text = flat(ontology_schema_text)
        assert "citations are per claim" in text
        assert "no invented citations" in text

    def test_separates_negation_from_uncertainty(self, ontology_schema_text):
        assert "never appears in both" in flat(ontology_schema_text)


class TestGitignorePolicy:
    @pytest.mark.parametrize(
        "pattern",
        ["*.pdf", "*.png", "*.dcm", "*.nii", "*.ckpt", "*.pt", "*.safetensors",
         "reports/*", "data/", "kaggle.json", ".env"],
    )
    def test_restricted_artefact_is_ignored(self, gitignore_patterns, pattern):
        assert pattern in gitignore_patterns

    def test_report_placeholder_stays_tracked(self, gitignore_patterns):
        assert "!reports/.gitkeep" in gitignore_patterns


class TestNoRestrictedArtefactsTracked:
    def test_git_tracks_no_restricted_artefacts(self):
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=REPO_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
            pytest.skip(f"git is unavailable: {exc}")
        offenders = [
            path
            for path in result.stdout.split("\0")
            if path and Path(path).suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
        ]
        assert not offenders, f"restricted artefacts are tracked in git: {offenders}"
