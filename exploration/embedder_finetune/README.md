# Embedder fine-tune

Fine-tune gte-modernbert-base on our own questions with hard negatives, re-embed the corpus
with it, and compare it with the base model. Results and decisions: article 006
("Fine-tuning the embedder", 12 November). Data: brainstorm 005 and
`exploration/training_questions/`. Renting GPUs: `docs/vast_ai.md`.

## Steps

| # | Script | Where | What |
|---|---|---|---|
| 1 | `export.py` | laptop, free | Copies the training and dev sets and writes every document they mention to `data/documents.jsonl`, as production embedded it. |
| 2 | `train.py` | rented GPU (`vast/run.sh`) | Fine-tunes and runs the dev check before and after. `--model`, `--qwen-prompt` and `--grad-checkpointing` for other base models. |
| 3 | `export_corpus.py` | laptop, free | Writes all 512,052 corpus chunks to `data/corpus.jsonl.gz` (830 MB). |
| 4 | `embed_corpus.py` | rented multi-GPU (`vast/embed.sh`) | Embeds the corpus with the chosen model, one process per GPU, in resumable parts. |
| 5 | `import_vectors.py` | laptop, free | Loads the vectors into the `<source>_embeddings_768_ft` tables, then builds their index. |
| 6 | `dev_search.py` | laptop, free | Runs the dev questions through real search with the base and the fine-tuned embedder. |

The chosen model (`lr5e-6-e2`) is copied to `models/gte-modernbert-ft`, where
`FineTunedGteEmbedder` loads it (override with `FINETUNED_EMBEDDER`). Evaluation retrievers:
`gte-ft-chunk`, `hybrid-ft`, `hybrid-ft-rerank`.

`data/`, `runs/` and `models/` are gitignored.
