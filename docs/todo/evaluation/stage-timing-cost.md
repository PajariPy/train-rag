Record wall time per stage per question and the cost per question. ^desc

## Why
Only a run's total time and tokens are stored today; production choices need per-stage numbers.

## How
- Stages: rewrites, first search, rerank, each agent turn and tool, check, ask, answer (judge excluded).
- Latency p50 / p95, overall and per stage; $ per question and per 1,000; split by asked vs not.

**Done when:** timings stored with the run's results and in the report.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Card: [[Run the final evaluation]].
