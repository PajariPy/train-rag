Check everything end to end on a small sample before the expensive runs.

## Why
A bug found on 20 questions costs cents, on 500 it costs a run.

## How
- Mention-cap check (`answer_check.py`, cap 2, ≈ $0.80).
- Memory notes for every question the clarifying run asks (`train-rag write-memories`).
- 20 questions on `field-agent` and `field-agent-clarify`; read the traces.

**Done when:** both systems run clean; every metric appears in the report.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 7. Board: [[Run the final evaluation]].
