Give the agent's check a fourth verdict, "no document answers", which leads to the exact `{not_found}` reply. ^desc

## Why
Today the agent always answers from its best documents.

## How
- In the `judge_cases` check turn on the final top 5; runs on the cached prefix, almost free.

**Done when:** off-topic questions get the not-found reply in the smoke run.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 6. Card: [[Run the final evaluation]].
