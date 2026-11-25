---
date: 2026-11-25
todo: "[[Investigate the reranker cost and how to reduce it]]"
status: done (25 Nov), decided: 100 per search
---

# Plan (25 Nov): does LateOn on deeper candidates (top 200, full documents) raise accuracy?

## Context
- **Today:** the first search hands the reranker gte top 50 + BM25 top 50 (~97 candidates). The right document is among them for only 51.6% of the 380 dev questions. The cross-encoder can't afford more; LateOn (14 ms per question) can.
- **Question:** with gte + BM25 top 200 each (~383 candidates, coverage 68.7%) and LateOn reading **whole documents**, how much higher is end-to-end total@5 / ans@5 than today (cross-encoder on ~97)?
- **Already done (24 Nov):** `deep_candidates.py` collected the deep candidates: `data/deep_found.json` (with gte_rank / bm25_rank / is_expected per candidate) and `data/deep_documents.parquet` (71,683 texts beyond the 31,972 already in `documents.parquet`). Only the scoring is missing.
- **Finding while checking:** the 1,024-token LateOn run stopped at 936 tokens max (ColBERT drops punctuation), but the texts average ~5,600 chars (~1,300 tokens), max ~24,500 (~6,000 tokens). So today ~half the documents are cut. "Full context" = `--doc-length 8192` (ModernBERT's limit), and nothing will be cut.

## Dataset: extend, don't make a new version
It's the same 380 questions with the same right documents. Only the candidate list gets longer, so:
- The deep layer sits beside the original as extra files (`deep_found.json`, `deep_documents.parquet`, `deep_scores_*.json`). The original set with its cross-encoder scores stays untouched as the baseline.
- `labels.json` is keyed by (question, document), so Gemini labels carry over to the deep layer. Only new top-5 documents need rating.
- Depth 50 of the deep layer reproduces today's candidates (coverage 51.6%), so one scoring run gives every depth 50/100/150/200 by filtering on `gte_rank` / `bm25_rank`. No re-collection per depth.

## Steps
1. **Make `colbert.py` safe for long runs** (the 24 Nov run was lost when it was stopped at the timing step):
   - **Token-budget batches:** documents are already sorted by length. Batch size is the token budget divided by the batch's longest document, so 8,192-token documents don't OOM the 8 GB GPU. `--batch-tokens` defaults to about 32k, tuned in the smoke test.
   - **Checkpoints:** write the scores file every ~2,000 documents and resume from it, skipping documents already scored.
   - **Save first:** write the scores before the timing step, then time.
   - Follow the python-refine rules (spacing, docstrings).
2. **Full-length run (main experiment):**
   - `colbert.py --model lightonai/LateOn --name lateon-full --deep --doc-length 8192` on the ~103k unique deep documents, free on the local GPU.
   - Estimated 1.5–2.5 h, since documents are ~1.4× longer on average and attention grows faster. The smoke test (`--limit 10`) measures documents/s first and gives the real ETA.
   - Shown on the progress page as a workflow.
3. **1,024-token deep run (optional, ~1 h):** `--name lateon --deep`. It separates the gain from going deeper from the gain from reading whole documents. Run it after step 2, overnight-style. Recommended.
4. **New `deep_score.py`** (reuses `orderings.load_questions`, `load_labels`, `score.hit`, `deep_candidates.within`):
   - Rows (all end-to-end, over all 380 questions, plus per split):
     - **baselines:** cross-encoder 1,024 on today's ~97 (`candidates.jsonl`) and LateOn 1,024 on today's ~97 (`scores_lateon.json`);
     - **deep:** each `deep_scores_*.json` at depths 50 / 100 / 150 / 200 per search.
   - Columns: coverage, **total@1 / total@5** (strict), rerank@5 (given coverage), ans@1 / ans@5, unlabelled in top 5.
   - Writes `data/deep_answered_summary.json`, re-included in `.gitignore` like `answered_summary.json`.
   - Free strict numbers first: report them to the user before any paid step.
5. **Gemini labels (paid, ask first):**
   - Teach `label.py` the deep orderings (`--ordering "deep lateon-full @200"`, etc.) through a shared `deep_orderings()` in `orderings.py`.
   - Run `--estimate` and report the dollars. Rough guess: ≤ 1,900 pairs per ordering minus the ones already labelled, so a few dollars at ~$0.002 each.
   - Label only after the user says yes, with `--max-dollars`.
6. **Record:**
   - Research map, rp3: the "? Next: let ColBERT look wider" card becomes a result card. Its table has depth × (coverage / total@5 / ans@5) against today's cross-encoder, and it gets a decision or next step.
   - The reranker-cost todo note.
   - Article 011, ColBERT section: also catch up on the earlier pending request (the ColBERT results).

## Verification
- **Smoke test:** `--deep --doc-length 8192 --limit 10`.
  - Every candidate has a score.
  - The logged max tokens are well above 936, which confirms the full length took effect.
  - No OOM on the longest batch.
  - Documents/s gives the ETA.
- **Sanity:** at depth 50, the deep LateOn 1,024 run (step 3) should match `scores_lateon.json`'s total@5 within noise.
- **Checks:** `uv run ruff check exploration/reranker_dev && uv run mypy exploration/reranker_dev` (if mypy is configured for exploration).
- **Final table:** total@5 / ans@5 per depth against cross-encoder today. The decision is whether to move to 200 (or 100/150) per search with LateOn.
