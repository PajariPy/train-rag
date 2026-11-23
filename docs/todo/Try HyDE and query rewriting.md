Search with LLM-written text instead of (or next to) the raw question.

## Why
Recall is the bottleneck: for 45 of 125 semantic questions the right document never reaches the reranker's 50 candidates, base or fine-tuned gte (`exploration/embedder_finetune/benchmark_semantic_check.py`).

## Ideas
- **HyDE:** an LLM writes a hypothetical answer document, and we search with it.
- **Query rewriting:** an LLM rewrites the question into several search queries.

## How
Test on the semantic questions first; compare against the [[Union rerank check]].

## Dev run (started 15 Nov)
- Script: `exploration/query_expansion/expansion_check.py`, resumable; Gemini outputs cached in `data/expansions.json`.
- Fine-tuned gte. All 422 dev questions: fused 50 vs union. The 118 semantic dev questions also get a HyDE passage and 4 rewritten queries (Gemini 3.8 Flash, ~$0.25); each is searched with gte and BM25.
- Every candidate is scored once by the reranker against the original question, so fused 50, union, union + HyDE and union + rewrites are compared on the same scores.

## Dev result (15 Nov)
118 semantic dev questions, fine-tuned gte, right document in the reranked top 5 (reranker ~61 ms per document):

| Candidates | Docs reranked | Top 5 | Rerank time |
|---|---:|---:|---:|
| fused 50 (production) | 50 | 56.8% | 3.0 s |
| union of q lists | 97 | 61.0% | 5.9 s |
| union + HyDE (gte + BM25) | 180 | 65.3% | 11.0 s |
| union + 4 rewrites | 392 | **69.5%** | 23.8 s |
| union + HyDE + rewrites | 460 | 68.6% | 28.0 s |
| q + HyDE + rewrites, RRF top 100 | 100 | **67.8%** | 6.1 s |

- Rewrites bring the right document into the candidates for 84.7% (union: 67.8%); HyDE adds nothing on top of them.
- Capping all 12 lists with RRF to 100 keeps most of the gain at the union's speed.
- Gemini: $0.21 for 118 questions (both methods).
- All 422 dev, union vs fused 50: top 5 82.2% → 83.9%; basic 91.9% → 92.7%.

## Merged before the search (15 Nov)
Still one gte + one BM25 search, so the reranker reads ~96 documents (union), 118 semantic dev questions, top 5:

| gte searches with | BM25 searches with | Top 5 |
|---|---|---:|
| question (today's union) | question | 61.0% |
| question | keyword rewrite | 63.6% |
| mean of question + 4 rewrite vectors | question + rewrites appended | 65.3% |
| question weighted half + rewrites | keyword rewrite | 66.1% |
| **mean of question + HyDE vectors** | **keyword rewrite** | **66.9%** |
| question + rewrites appended (one text) | keyword rewrite | 65.3% |

- Best: gte with the mean of the question's and the HyDE passage's vectors, BM25 with the keyword rewrite: 61.0% → 66.9% (+7 questions) at the same ~96 documents and ~6 s rerank.
- Close to the 12-search results (69.5% at 392 docs; 67.8% for RRF top 100) with only 2 searches.
- Replacing the question instead of merging hurts gte (HyDE alone 50.0%): the fine-tuned gte matches questions.
- One question is 0.85 points; differences of 1–2 points are noise.
- Scripts: `exploration/query_expansion/expansion_merge.py` (local, $0).

## Test set: benchmark's 125 semantic questions (15 Nov)
Fine-tuned gte, one gte + one BM25 search (~95 docs), right document in the top 5 (dev result in brackets):

| gte searches with | BM25 searches with | Test | Dev |
|---|---|---:|---:|
| question (today's union) | question | 64.0% | 61.0% |
| question | question + rewrites appended | **67.2%** | 63.6% |
| mean(question, HyDE) | question | 65.6% | 63.6% |
| mean(question, HyDE) | question + rewrites appended | 65.6% | 66.1% |
| mean(question, 4 rewrites) | question + rewrites appended | 65.6% | 65.3% |
| mean(question, HyDE) | keyword rewrite | ✗ 60.8% | **66.9%** |
| question | keyword rewrite | ✗ 60.8% | 63.6% |

- The dev winner doesn't hold: BM25 with the keyword rewrite *instead of* the question loses 4 questions on the benchmark.
- What holds on both: keep the question and **append** the rewrites for BM25 (test +4 questions, dev +3).
- Gains are small: one question = 0.8 points. More right documents reach the candidates (up to 82.4% vs 75.2%), but the reranker loses them among near-duplicates.
- Gemini $0.22. Script: `exploration/query_expansion/benchmark_merge_check.py`.

## Decision (16 Nov)
- Rewrites go to **BM25 only, appended to the question**; gte keeps the plain question. HyDE and vector means are dropped: the LLM doesn't know the answer or our data, so its guesses only add noise to the vector.
- BM25 alone, right document in its top 50 (test / dev): question 62.4% / 42.4%, keyword rewrite alone 52.0% / 49.2%, question + 4 rewrites **72.0% / 59.3%**. One right "golden keyword" is enough for BM25.
- The reranker loses most of that gain among near-duplicates (+10–17 points found, +3–4 questions in the top 5). Next: brainstorm before any paid run.
- Recorded in article 006 ("The merged query on the benchmark") and the research map (step 12d).
