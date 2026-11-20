Fine-tune other base embedders (e.g. jina-embeddings-v3, Qwen3-Embedding-0.6B) and compare with gte, using the same training/dev sets and dev check (`exploration/embedder_finetune`).

## Parked (13 Nov)
The fine-tuned gte found 38 → 59 of 125 semantic right documents on its own, but BM25 already had most of them, so the fused list gained only 5. Another embedder trained the same way likely hits the same wall.

## If reopened
Qwen first (better than gte as-is, no custom code). The 12 Nov Qwen run crashed at 60% (disk full); a rerun needs ~$1, a 60 GB disk and `save_total_limit=1`, then ~$5 more to re-embed the corpus.
