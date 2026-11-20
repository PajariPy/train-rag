Rerank the whole union of gte's top 50 and BM25's top 50 (up to ~100 documents) instead of the fused top 50.

## Why
Fusion keeps only 50 of the up to ~100 candidates the two searches return. On the 125 semantic questions it cut 8 right documents that the fine-tuned gte had found (at its ranks 35–48). The reranker puts about 90% of the right documents it sees into the top 5, so this could add about 7 semantic questions (semantic check, 13 Nov).

## How
- Full text, nothing cut: we want the representative result first. If it works, the question becomes speed.
- Base and fine-tuned gte, both candidate lists, 125 semantic questions. Local, free.
- Script: `exploration/embedder_finetune/union_rerank_check.py`, resumable; output in `exploration/embedder_finetune/data/union_rerank_check.json`.

## Result (15 Nov)
Right document in the final 5, 125 semantic questions:

| | fused 50 (today) | union (~97 docs) |
|---|---|---|
| base gte | 69 | 73 |
| fine-tuned gte | 72 | **80** |

- Fine-tuned + union vs today's production (base, fused): 69 → 80, 55.2% → 64.0%.
- Fine-tuned: union gained 10 questions and lost 2 against its own fused 50; right documents not among the candidates at all: 45 → 31.
- Rerank time doubles: median 3.3 s → 6.3 s per question.
- Next: a full benchmark run with the union (paid), then make it faster.
