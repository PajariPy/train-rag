# Solution plan (20 Nov 2026)

What goes into production, why each part is there, how it is measured and what it costs.
Every part links back to the experiment that put it here. Status: **✓ accepted** (measured,
in the pipeline), **▶ in progress**, **○ planned**.

## 1. Problem

A company assistant has to answer employees' questions from internal documents.
EnterpriseRAG-Bench simulates one company (Redwood Inference) with 511,962 documents from nine
tools; Slack is 55.8% of them, and the least structured. It comes with 500 questions with
gold answers.

What makes it hard, as measured:

- **Plain search misses semantic questions** (the question's words aren't the document's
  words): base gte added only ~3% over BM25, and semantic questions failed most.
- **Near-twin documents.** Search reaches the right topic but loses to threads on the same
  topic about another case (another customer, incident, release). Topic modelling doesn't
  help: the topic is already right, what differs is the detail.
- **Ambiguity.** On the 200-question Slack semantic dev set, for 71% of the questions whose
  right thread answers, another shown document answers too with conflicting facts or about
  a different case. The question alone doesn't say which one the person means.

## 2. Solution: the pipeline

In the order a question goes through it.

| # | Step | What | Status | Why it's here (evidence) |
|---|---|---|---|---|
| 1 | Ingestion | Parse and normalize every source without an LLM, chunk, store in Postgres | ✓ | Article 001 |
| 2 | Embeddings | gte-modernbert fine-tuned with hard negatives (picked by reranker score) on 2,067 synthetic questions | ✓ | Dev top 10 80% → 84%, semantic 49% → 61%; benchmark 75.4% → 77.6% correct (run 31). Article 006 |
| 3 | Mined fields | Per Slack thread: kind, area, topic, bug, cause, fix, status, names, channel. DeepSeek labels now; a fine-tuned SLM (Qwen3.5-2B LoRA) to run on the whole corpus | ✓ labels / ○ SLM | Field search reaches 47% of the questions plain search misses in top 50 (gte: 1%). Article 007 |
| 4 | Query from chat history | Rewrite the question with what the chat already said, before the first search | ○ | Needed once there is a conversation; not measured yet |
| 5 | First search | Hybrid: gte on the plain question + BM25 on the question with 4 LLM rewrites appended | ✓ | Benchmark semantic top 5: 64.0% → 67.2%. HyDE and rewritten vectors hurt (dropped) |
| 6 | Reranker | gte-modernbert reranker on the union (~97 docs), first 1,024 tokens | ✓ | Union rerank 55.2% → 64.0% semantic top 5; 3.3 s → 6.3 s |
| 7 | Search agent | Gemini Flash agent (DeepSeek until 20 Nov) with field search (areas, sources, channels filters), up to 4 searches, then its top 100 to the reranker | ✓ (exploration code) | Slack dev top 5: 39.0% → 55.5% |
| 8 | Disagreement check | The agent judges whether its top 5 describe different cases; if so it asks the person (or shows the cases side by side) | ▶ | Classifier on dev-contradictions: ask precision 0.85, recall 0.76 (below) |
| 9 | Ask the person | `ask_user`, at most 2 questions; the reply becomes a filter | ✓ (exploration code) | Top 10 on asked questions 27% → 39% (noise 6–10 pts) |
| 10 | Answer | Gemini Flash answers from the final top 5 | ✓ | Judge-scored benchmark runs |
| 11 | Out-of-scope | Say "not in our documents" instead of answering from near misses | ○ | |
| 12 | Guardrails | Input and output checks | ○ | |
| 13 | Observability | Per-query traces, cost, latency, RAG quality metrics | ○ | |

### Step 8 in detail (decided 20 Nov)

The agent itself (Gemini Flash, the same model as the rest of the pipeline) classifies its
final top 5, with no separate classifier call:

- **same answer** (consistent): answer;
- **one case fits** (specific): the question's details single out one document; answer from it;
- **several cases fit** (ambiguous): ask what separates the cases, or show them side by side.

Whatever the verdict, the final answer never picks one of several similar documents
silently: when more than one answers fully or in part, it gives the best-fitting one as the
main answer and mentions the others in a sentence each (their case and how they differ),
so the person can look at them too. This rule is in every answering prompt in
`src/train_rag/prompts.yaml` (20 Nov).

For this step the agent sees the 5 documents in full (cut at 12,000 characters, like the
answer step). They are already in its context, so the check is one more turn on a cached
prefix. The 300-character search snippets are not enough (below); full text scores the same
as the first 3,000 characters, within noise (decided 20 Nov).

### Not going to production (tried and closed)

- Topic modelling and clustering: search already finds the topic (16 Nov).
- Zero-shot GLiNER / NuExtract for fields: micro F1 0.41 (16 Nov).
- LLM summaries / fact extraction as the index (stopped 9 Nov).
- HyDE and rewrites as extra searches (more candidates than the reranker can sort).
- Mined fields as reranker features: noise. They work as search, not as a score.
- A separate rewritten "v2" disagreement prompt: worse than the plain one (20 Nov).

## 3. Evaluation

| Set | Size | Measures | Latest |
|---|---|---|---|
| Synthetic dev | 422 questions | Embedder top 10 | 84% (fine-tuned gte) |
| EnterpriseRAG-Bench | 500 questions | Judged answer correctness, end to end | 77.6% (run 31) |
| Benchmark semantic subset | 125 questions | Retrieval top 5 | 67.2% |
| Slack semantic dev set | 200 questions | Right thread in top 5/10/50/100 | top 5 55.5% with agent + reranker |
| Same, lenient | 200 questions | Any fully answering document in top 5 | 88.0% (agent), 84.5% (today) |
| Dev-contradictions | 83 questions (27 consistent, 27 specific, 29 ambiguous; hand-reviewed) | Ask or not, and which document | baseline Gemini prompt: 3-class 72%, ask P 0.85 / R 0.76; right document picked 27/27 consistent and specific |

Disagreement check by input (baseline prompt, Gemini):

| Input per document | Ask precision / recall | Missed asks (of 29) |
|---|---|---|
| 300-char snippet | 0.89 / 0.59 | 12 |
| first 3,000 chars | 0.85 / 0.76 | 7 |
| up to 12,000 chars | 0.82 / 0.79 | 6 |

Still to measure: the agent's own check on this set; the Gemini agent's search scores (all
agent numbers above are DeepSeek runs); the full pipeline with
clarifying questions on the test set (simulated person, memory notes); repeated runs for noise.

## 4. Price

Prices: Gemini 3.8 Flash $0.75 / M input, $3.75 / M output (thinking billed as output). For
comparison, DeepSeek flash $0.15 / M input ($0.003 cached), $0.60 / M output off-peak,
doubled at peak. Embedder and reranker run locally.

### Per query

| Step | Model | Cost | Basis |
|---|---|---|---|
| Rewrites for BM25 | Gemini Flash | ~$0.002 | $0.25 for 125 questions (with HyDE, now dropped) |
| Search + rerank | local gte, BM25, reranker | GPU time only | 6.3 s rerank on ~97 docs |
| Search agent (4 searches) | Gemini Flash | ~$0.01 (estimate) | DeepSeek did it for $0.395 / 200 questions; Gemini is 5–6× the price per token |
| Disagreement check, inside the agent | Gemini Flash | ~$0.003 (estimate) | 5 full documents (~10–15k tokens) already in context, read from the cache at 10% of the price; ~0.5k output tokens |
| Asking the person | Gemini Flash | < $0.001 | a short turn |
| Answer | Gemini Flash | ~$0.005 (estimate) | 5 documents in |
| **Total** | | **~$0.02 per question** | ≈ $20 per 1,000 questions (with DeepSeek as the agent: ~$0.01) |

### One-off

| Item | Cost |
|---|---|
| Fine-tuning gte (Vast.ai GPU) | a few dollars |
| DeepSeek labels, 25,633 Slack threads | $26.54 |
| SLM fine-tune + mining the corpus | ~$15–35 (estimate) |
| Benchmark run (500 questions, judged) | $4.53 |

Estimates are marked; they come from token counts and price tables, not billed runs.
