A command that rechecks the off-topic set after new documents arrive.

## Why
The corpus will grow; a question off-topic today can become answerable.

## How
- Reruns search + rating on the stored set; moves answerable questions out, reported, not silently dropped.
- Extends the set from the stored generator instructions.

**Done when:** the recheck runs and reports moved questions.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 5. Board: [[Run the final evaluation]].
