Rewrite the simulated person's memory notes with tiers, so asking is tested on realistic memories. ^desc

## Why
Today every note is clean and vague; a real person remembers a concrete detail, a wrong one, or nothing.

## How
- `NOTE_PROMPT_VERSION = "v3"`; tiers by seed like `pick_wrong`: vague 40%, concrete 25%, wrong 25%, none 10%.
- Store the tier on `QuestionMemoryRecord`; questions without gold get `none`.
- Keep `leaks_title` and `has_number`; hand-review a sample.

## Cost (estimate, 23 Nov)
**≈ $2.** One Gemini call per question (the gold document, ~3k tokens in, a short note out) ≈ $0.004 × 500. The 10% `none` tier needs no call.

**Done when:** all 500 notes written in v3 with a stored tier.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 2. Card: [[Run the final evaluation]].
