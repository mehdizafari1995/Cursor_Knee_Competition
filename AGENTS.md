# Agent Instructions

Knee-MRI competition repository. The main Cursor Agent acts as **coordinator**: it
plans work, delegates to the project subagents in `.cursor/agents/`, merges their
output, and is the only party that grants approvals.

## Subagents

| Agent | Role | Read-only | Background |
| --- | --- | --- | --- |
| `literature-searcher` | PubMed/PMC retrieval with reproducible, logged queries | no | yes |
| `clinical-ontology` | Structured, fully cited ontology for the twelve targets | no | yes |
| `evidence-verifier` | Independent audit of citations, claims and licences | yes | no |
| `competition-data-auditor` | CSV/DICOM schema, missingness and leakage audit | yes | no |
| `repository-verifier` | Runs tests and confirms claimed code exists and works | yes | no |

Delegate discovery and writing work to the background agents; block on the
verifiers before accepting any result.

## Policies

These are binding on the coordinator and every subagent. Full text in
`docs/literature_protocol.md` and `docs/citation_and_figure_policy.md`.

1. Literature retrieval uses the PubMed E-utilities and PMC APIs, with every query,
   date and hit count logged so the search is reproducible.
2. "All papers" means all records returned by the documented search strategies on
   the documented dates. It is never a claim that every publication was found.
3. Article figures are retrieved only when full-text and figure access is legal.
4. Every retrieved figure records its licence and canonical source URL.
5. Literature figures live in `literature/figures/` and never mix with Kaggle
   training images under `data/`.
6. Literature figures are not used for model training without licence review,
   patient-level label verification and explicit coordinator approval.
7. PDFs, figures, DICOM files, reports, credentials and model checkpoints are never
   committed. `.gitignore` enforces this.
8. Parallel writing work uses isolated git worktrees or separate branches.
9. Every agent returns: evidence, file paths, tests run with results, and unresolved
   questions.

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

`tests/test_agent_configuration.py` validates the agent definitions and the policy
docs. Run it after editing anything under `.cursor/agents/` or `docs/`.
