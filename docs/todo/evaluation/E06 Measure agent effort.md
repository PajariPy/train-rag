Count what the agent spends per question, from the stored traces.

## Why
Shows the cost of asking and searching, and where the agent wanders.

## How
- From `TraceStep`s: model turns, tool calls by kind (general search, field search, ask, check), searches until the right document first appears, asks, tokens (input / cached / output / thinking).
- Mean, p50, p95. No model calls.

**Done when:** the report shows effort per system.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Board: [[Run the final evaluation]].
