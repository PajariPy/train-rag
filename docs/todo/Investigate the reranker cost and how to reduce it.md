Find out what reranking really costs on AWS, and how to bring it down without losing the quality it adds. ^desc

## Why
GPU benchmarks on 24 Nov, projected to SageMaker L4 instances (article 011, `docs/articles/011_model_costs_on_aws.md` in the AWS-architecture-research workspace):
- **Today's field-agent setting** (100 documents × 1,024 tokens): ~1.1 s of GPU per question. At 25 questions/s that is ~25–30 L4s, ~$21–25k a month on SageMaker.
- **The lighter setting** (50 × 512): ~0.25–0.3 s per question, still ~5–8 L4s (~$4–6.5k a month).
- **Everything else is small by comparison:** the embedder fits on a fraction of one GPU, and ingestion costs ~$0.15–0.25 per 1,000 documents.

## How
1. **Measure quality against cost:** recall and answer quality at 100 × 1,024, 50 × 1,024, 100 × 512 and 50 × 512 (fits into [[Run the final evaluation]], or a small separate run).
2. **Measure on a real instance:** one hour on an L4 (`ml.g6.xlarge` or plain EC2 `g6.xlarge`) to replace the 2–3× projection from the laptop GPU. Paid: ask first.
3. **Price the options to cut it:**
   - one GPU shared by the embedder and the reranker;
   - working hours only (scheduled scaling);
   - plain EC2 instead of SageMaker, and Savings Plans;
   - cheaper GPUs (T4);
   - a smaller or distilled reranker, or an ONNX / TensorRT build;
   - reranking only when the first ranking is unsure;
   - caching repeated questions.
4. **Get the expected traffic** (questions per day and the peak), which decides how many GPUs are needed at all.
5. **Write the result into article 011**, then merge the cost section into article 010.

## Progress (24 Nov)
**What the reranker reads:** 96.8 documents per question (median 99, range 71–100). gte and BM25 share only ~3 documents, so the union is nearly both top-50 lists. Each document is read up to 1,024 tokens, and 61% get cut.

**What it buys** (125 benchmark semantic questions, the same 50 candidates with and without the reranker):

| | R@1 | R@5 | R@10 |
|---|--:|--:|--:|
| no reranker | 20.8% | 44.0% | 51.2% |
| reranker | 46.4% | 57.6% | 59.2% |

That is about +13 points of semantic accuracy.

**Dev reranking set** (`exploration/reranker_dev/`, 380 semantic questions from dev and dev_slack, 36,798 pairs):

| | R@1 | R@5 | R@10 | s/question |
|---|--:|--:|--:|--:|
| fused order | 10.0% | 22.4% | 29.2% | – |
| reranker, 1,024 tokens | 26.3% | 40.8% | 46.1% | 2.44 |
| reranker, 8,192 tokens | 27.9% | 40.5% | 45.3% | 6.01 |

- On the 195 questions whose right document is among the candidates, R@5 is **79.5%** at 1,024 tokens. That is the bar for any alternative.
- 1,024 tokens loses nothing against 8,192 at rank 5, at 2.5× less time.
- Only 51.3% of right documents reach the reranker (44.0% for Slack). That is a first-search limit.

**Next:** score alternatives on the dev set: ColBERT, a smaller cross-encoder, 512 tokens, 50 documents.

## Details to remember (24 Nov)
**Metrics on the dev reranking set:**
- **strict@k:** the source document (the one the question was written from) is in the top k.
- **answered@k:** the top k holds the source or a document Gemini rated "full", read alone. answered+partial@k also accepts "partial".
- answered@k measures the reranker, not answer correctness. A near-twin with other facts (another customer, another number) counts as answering. Choosing which case the person means is the agent's job (asking, the answer rule).

**The bar for any alternative:** answered@5 ≈ 84% and answered@1 ≈ 63% (1,024 tokens). Strict R@5 is 79.5% on the 195 questions with the source among the candidates.

**Scoring a new reranker:**
1. Score `data/candidates.jsonl` with the texts in `data/documents.parquet`. These are the same inputs the current reranker read: `title + text`, the question normalized.
2. Rate its unlabelled top-5 documents first, with `label.py` and the shared cache (~$0.002 each; only new pairs cost money).
3. Then run `score.py`, and report top-5 agreement with today's reranker as well.

**Keep in mind:**
- **dev_slack is the ambiguity dev set, ambiguous by design.** Hypotheses are tested there, not on the benchmark, to avoid overfitting the test set.
- **About half the right documents never reach the reranker** (51.3% are among the candidates). That is a first-search limit, so don't blame or credit the reranker for it.
- **The fused order's answered@k is a lower bound:** its top 5 is only partly labelled.
- **1,024 tokens is the setting to beat.** It matches 8,192 at rank 5 at 2.5× less time.
- **Cost so far:** labels $3.29 (Gemini). The dev set build and the benchmark check were free (local).
- **Files:**
  - `exploration/reranker_dev/`: build.py, existing_labels.py, estimate_labels.py, label.py, score.py;
  - committable: data/summary.json, data/answered_summary.json;
  - data files are gitignored.
- **Excalidraw format:** keep the maps as plain `.excalidraw` (compatibility mode) until the scripts support `.excalidraw.md` (Backlog card [[Support the excalidraw.md format in the map scripts]]).

## Status at end of 24 Nov
- **Decided:** LateOn alone replaces the cross-encoder. Reranker-only (rerank@5 on ~97 candidates): 71.8% vs the cross-encoder's 79.5%. End to end ans@5: 80.3% vs 83.7%. No reranker GPU cost.
- **Running overnight:** ColBERT-Zero (Apache 2.0) and Reason-ModernColBERT (CC-BY-NC, comparison only), the same rerank@5 test. Results go to `data/scores_<name>.json`; run `score.py` to compare.
- **Next, the deep test:** can LateOn find the right document when it has ~380 candidates?
  - The candidates are collected: `deep_found.json` and `deep_documents.parquet`. Deeper search raises coverage from 51.6% to 68.7%.
  - The LateOn scoring was paused partway. It isn't resumable, so the rerun takes ~1 h: `colbert.py --model lightonai/LateOn --name lateon --deep`.
  - Then report rerank@5 at each depth (top 50 / 100 / 150 / 200 per search) against 79.5%.
- **Later:** fine-tune LateOn on our training set (hard negatives ready); try smaller models (mxbai-edge-colbert 32M).

## 25 Nov: deep test with whole documents
- **Plan:** [[2026-11-25-lateon-deep-full-context]]. LateOn scores gte + BM25 top 200 each (~383 candidates) at full document length (8,192 tokens), compared with today's cross-encoder on ~97.
- **Dataset:** the existing dev set is extended with deep files, not versioned. Same questions, labels shared by (question, document).
- **Result:** LateOn at 1,024 tokens, ans@5 by depth per search: 50 → 80.8%, 100 → 82.1%, 150 → 81.6%, 200 → 80.5% (cross-encoder today 83.7%). Whole documents are worse at every depth. Gemini labels cost $1.46.
- **Decided (25 Nov): 100 per search** for LateOn, applied when LateOn goes into the pipeline. The remaining gap is in ranking; the next lever is fine-tuning LateOn. On the research map, see the "Reranker price on AWS" problem, step 3.
