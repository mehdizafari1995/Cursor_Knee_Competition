# Claim Lineage Audit

Baseline commit: `237372c`
Baseline ledger: `literature/claims.jsonl`
Machine-readable audit: `literature/claim_lineage.json`

## Conservation arithmetic

The immutable baseline contains 154 unique original claim IDs,
`KO-PILOT-0001` through `KO-PILOT-0154`.

- Original dispositions: 154 = 79 active supported originals + 75 rejected
  removed originals.
- Final active ledger: 83 = 79 active originals + 4 replacement claims.
- Raw final rows across the active and rejected ledgers: 158 = 83 active rows
  + 75 rejected original rows.
- The 158 raw rows span 154 original claims plus 4 new replacement claims.
  Historical versions are not counted as distinct claims.
- Conservation across origins and final ledgers:
  154 originals + 4 replacements = 83 active + 75 rejected originals = 158.

The complete disposition table for all 154 original IDs, including each final
disposition and current location, is
`literature/claim_lineage.json#/original_claim_dispositions`.

## Replacement claims

These are replacements, not originals. Each is active supported, targets
`bone_contusion`, and was added after the second audit to restore directly
supported bone-contusion coverage:

- `KO-PILOT-0155`
- `KO-PILOT-0156`
- `KO-PILOT-0157`
- `KO-PILOT-0158`

## Reconciliation exceptions

- Duplicate or versioned rejection-history records: 0; IDs: none (`[]`).
- Active/rejected overlap: 0; IDs: none (`[]`).
- Originals without a final disposition: 0; IDs: none (`[]`).

## Final counts

| Measure | Count |
| --- | ---: |
| Unique original baseline IDs | 154 |
| Active supported originals | 79 |
| Rejected removed originals | 75 |
| Replacement claims | 4 |
| Final active claims | 83 |
| Raw active plus rejected rows | 158 |
| Duplicate/versioned rejection-history records | 0 |
| Active/rejected overlap IDs | 0 |
| Original IDs without final disposition | 0 |
