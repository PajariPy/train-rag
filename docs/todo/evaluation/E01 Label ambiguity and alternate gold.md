Label all 500 questions consistent / specific / ambiguous and store the twin documents that answer fully (alternate gold).

## Why
The answers now name a main case plus similar cases; without alternate gold a right twin counts as a miss.

## How
- Gemini rates each document of a cheap union-rerank top 5 as full / partial / none (method: `exploration/slack_extraction/ambiguity.py`).
- Store per question: the label and the alternate gold ids.
- Hand-check ~30 labels.

**Done when:** every question has a label; the hand-check agrees.

Spec: `docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 1. Board: [[Run the final evaluation]].
