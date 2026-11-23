Run the `ragas` library on a fixed ~100-question sample and compare it with our judge.

## Why
So the article can state how well our RAGAS-style numbers agree with the real library.

## How
- `ragas` with Gemini through `langchain-google-genai`; seeded sample.
- Mean difference and correlation per metric.

**Done when:** an agreement table per metric.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4. Board: [[Run the final evaluation]].
