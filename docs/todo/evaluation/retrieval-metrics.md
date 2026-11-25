Add the retrieval metrics readers know: MRR, nDCG@5/10, precision@5, hit rate@5/10, recall@k. ^desc

## Why
Today only document recall is reported.

## How
- Strict (gold only) and with alternate gold as relevant ([[ambiguity-labels-alternate-gold]]).
- No model calls.

**Done when:** both variants in the report.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Card: [[Run the final evaluation]].
