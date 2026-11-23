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
