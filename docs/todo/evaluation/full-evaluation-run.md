The 500 questions plus the off-topic set on `field-agent-clarify`, one run (budget, 23 Nov). ^desc

## Why
The final numbers. Agent noise is 6–10 points, so the report states it is a single run.

## How
- `uv run train-rag evaluate --system field-agent-clarify`.
- ≈ $20 (see Cost). Ask first.

## Cost (estimate, 23 Nov)
**≈ $20** for one `field-agent-clarify` run on 575 questions (500 + ~75 off-topic): agent,
answer and strict judge ≈ $16 (≈ $0.027 per question, measured in smoke runs 32–34), simulated
person ≈ $1, lenient judge ≈ $2.3, RAGAS own judge on the 100-question sample ≈ $0.90 (the
sample: [[ragas-judge-build-eval]]). Search and rerank are local.
Scope cut on 23 Nov for budget: no `field-agent` run, no repeats (they would add ≈ $20 and
≈ $24 per run).

**Done when:** the run is stored and judged both ways.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 7. Card: [[Run the final evaluation]].
