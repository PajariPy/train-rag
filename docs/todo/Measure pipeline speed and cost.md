Measure latency and price per question of the full pipeline, per stage, for the production
version.

## Why
Each stage adds time and money: rewrites, gte + BM25, reranking 100 documents on their first
1,024 tokens, agent rounds, asking, answer. Known pieces: union rerank about 6 s per
question; agent v3 about $0.002 per question on DeepSeek.

## How
1. Time and token counts per stage on the test set; p50 / p95 latency, $ per 1,000 questions.
2. Where it's slow or expensive, list the options (fewer rerank documents, shorter chunks,
   cheaper model, caching).

## Design (23 Nov)
Moved to Next, right after [[Detect out-of-scope questions]]. Part of the evaluation
update (`docs/superpowers/specs/2026-11-23-evaluation-update-design.md`, part 4):
- wall time per stage per question: rewrites, first search, rerank, each agent turn and
  tool, check, ask, answer (judge excluded); stored with the run's results;
- latency p50 / p95, overall and per stage; cost per question and per 1,000 questions;
- split by `field-agent` vs `field-agent-clarify` and by asked vs not asked.
Today only a run's total time (`seconds_elapsed`) and total tokens are stored.
