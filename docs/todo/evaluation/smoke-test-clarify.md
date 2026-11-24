Check everything end to end on a small sample before the expensive runs. ^desc

## Why
A bug found on 20 questions costs cents, on 500 it costs a run.

## How
- Mention-cap check (`answer_check.py`, cap 2, ≈ $0.80).
- Memory notes for every question the clarifying run asks (`train-rag write-memories`).
- 20 questions on `field-agent-clarify`; read the traces.

## Cost (estimate, 23 Nov)
**≈ $1.7.** Mention-cap check $0.80 (measured $0.81–0.85); 20 questions × ≈ $0.03 (agent, person, answer, judge; smoke runs 32–34 measured $0.027) ≈ $0.6; both judges and RAGAS on those 20 answers ≈ $0.25. Memory notes are in [[memory-notes-v3-tiers]].

**Done when:** `field-agent-clarify` runs clean; every metric appears in the report.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 7. Card: [[Run the final evaluation]].
