# First retrieval baselines: summaries, chunks, whole documents and BM25

Status: draft.

In the [first post](001_data_exploration_and_parsing.md) I read and parsed the 511,962
documents of EnterpriseRAG-Bench without building any search. This post is the first search:
how I measure it, the baseline I started from, a small, cheap comparison of four ways to
find the right documents, then the two best of them on the full corpus, alone, combined and
reranked, and what it would cost to run.

## How I measure

The benchmark has 500 questions. Each comes with a gold answer, a list of **answer facts**
(the individual facts a full answer must contain), and the **expected documents** that hold
the evidence. I run two evaluations.

**The full evaluation** asks the RAG system each question. The system sees only the question
text and returns an answer plus the documents it used. Then:

1. **Document recall** is the share of expected documents the system returned.
2. **An LLM judge** (Gemini) compares the answer with the gold answer. It decides whether the
   answer is *correct* and which answer facts it covers. *Completeness* is the share of facts
   covered.
3. **The score** is correctness × completeness, as on the benchmark's leaderboard, so a wrong
   answer scores 0 however many facts it lists.

Every answer, grade and judge's reasoning is stored, so I can read any run later. Both the
answers and the judge use Gemini 3.8 Flash. A full run over 500 questions, answering from 5
retrieved documents, costs about $4; every run now prints its exact token count and cost.

**What the numbers are made of.** The judge only gives yes/no verdicts: one for "is the
answer correct", and one per answer fact for "does the answer state it". The metrics are
built from those:

| Metric | Per question | From |
|---|---|---|
| Correct | 0 or 1 | the judge, one verdict |
| Complete | facts covered ÷ all facts, e.g. 2 of 3 = 0.67 | the judge, one verdict per fact |
| Score | correct × complete: 0 when wrong, otherwise the completeness | both |
| Document recall | expected documents retrieved ÷ expected documents | document ids only, no LLM |

Every reported number is the average over the questions. A few things follow:

- **Correctness has no partial credit;** completeness is there to show how much of an answer
  is right.
- **Small groups are noisy.** With 175 questions one answer is 0.6 points; with 20 it's 5
  points. A few points between runs on a small question type is chance.
- **The judge is one LLM call per answer,** with no second opinion, so a borderline verdict
  could flip between runs. It also shares a model with the answers, and could favour answers
  that read like its own.
- **Document recall is exact and costs nothing,** which is why retrieval comparisons use
  only that kind of measure.

**The retrieval evaluation** skips the answer and the judge, so it's free and takes minutes.
It only asks: *are the expected documents among the top k results?* That's **Recall@k**,
averaged over the 470 questions that expect documents. Next to it I report **In index**: the
share of expected documents the search can find at all. Search can't find a document that
isn't indexed.

When I'm comparing ways to index, the retrieval evaluation is the one that matters. The full
evaluation comes later, on the winner.

## The question types

The 500 questions come in ten types, and they test different things:

| Type | Questions | Docs per question | What it tests |
|---|---:|---:|---|
| basic | 175 | 1.0 | one fact in one document, asked in the document's own words |
| semantic | 125 | 1.0 | one document, but the question is reworded to avoid its words |
| intra_document_reasoning | 40 | 1.0 | one document, combining several places in it |
| constrained | 30 | 1.4 | very specific conditions (a region, a protocol, an error code) |
| miscellaneous | 20 | 1.0 | off-topic documents, like the office fridge schedule |
| project_related | 40 | 4.2 | one topic followed across tools, linked by ticket and PR ids |
| completeness | 20 | 6.5 | counting or listing across many near-identical documents |
| conflicting_info | 20 | 2.0 | two documents disagree; the answer must take the latest |
| high_level | 10 | 0 | the company as a whole; no single document |
| info_not_found | 20 | 0 | the answer isn't in the corpus; the right answer says so |

The last two expect no documents, so they're left out of the retrieval scores.

## The baseline: free summaries

Three sources come with a summary I didn't have to generate (see the first post): the
Fireflies summary, the Jira issue summary and the GitHub description. I embedded those 23,843
texts with `all-MiniLM-L6-v2`, a small and fast embedding model. The vectors live in Postgres,
in one table per source, with a pgvectorscale StreamingDiskANN index on cosine distance.

| | Recall@1 | Recall@5 | Recall@10 | Recall@20 | In index |
|---|---:|---:|---:|---:|---:|
| MiniLM on free summaries | 8.3% | 12.5% | 14.5% | 17.1% | 28.9% |

Only 4.7% of the corpus has a free summary, so at best 28.9% of the expected documents can be
found at all. The search finds about half of those.

I also ran a first full evaluation on top of it. The system retrieves the 10 closest summaries
and Gemini answers from them. On the first 100 questions, which all happen to be basic ones, it
scored **4.1%** (6% correct, 5% complete, 21% document recall). Low, as expected: most answers
simply aren't in the index.

## Why not summarize everything?

The obvious fix is to summarize every document with an LLM and embed the summaries. Before
doing it, I priced it.

MiniLM reads at most 256 word pieces, about 1,000 characters. Fireflies and Jira summaries fit
(a median of 77 and 64 word pieces). GitHub descriptions don't: their median is 735, and 7,808
of 8,051 are too long. They aren't really summaries. Together with the six sources that have no
summary at all, **495,930 documents (97%) would need one**, about 779 million input tokens.

With Gemini 3.8 Flash on the Batch API, that's about **$480**, or about $850 if the model is
allowed to think first. Input alone is $292, so shorter summaries barely help. And Slack and
Gmail are 82% of the cost.

That's too much to spend before knowing whether summaries beat simply embedding the documents
themselves. So I compared the free options first.

## A small test set

The full corpus is expensive to embed too, in time rather than money. On my laptop GPU (an RTX
2000 Ada with 8 GB), I measured:

- **MiniLM on small chunks:** about 57 documents per second, so about 2.5 hours for the corpus.
- **gte-modernbert-base on whole documents:** about 17,600 tokens per second. The corpus is
  about 660 million tokens, so about **10 to 11 hours**.

Instead of starting a 13-hour run, I built a test set: **every document a question cites (722)
plus 20,000 random other documents as distractors**, 20,691 in all. A fixed seed picks the
distractors, so every method indexes exactly the same documents. Running again without the
sample later adds the rest of the corpus, so nothing is wasted.

The catch: with 20,000 distractors instead of 511,000, search is much easier. The numbers below
are inflated. They're good for ranking the methods against each other, not as final results.

## Three ways to search the whole text

**1. MiniLM on small chunks.** MiniLM reads about 1,000 characters, so I cut each document with
recursive chunking: at paragraph breaks first, then line breaks, then sentence ends, then
between words, joining pieces back up to 600 characters. Each chunk also gets the document
title and about 100 characters from its neighbours: the end of the chunk before it on top, the
start of the next one at the bottom. That way a chunk that starts mid-topic still says what the
topic is. 95% of chunks fit in MiniLM's window; the rest only lose the end of their bottom
context. The sample became 218,183 chunks, about 10 per document. A document is ranked by its
best chunk.

**2. gte-modernbert-base on whole documents.** This model reads 8,192 tokens, about 30,000
characters. A typical document here is 2 to 6 pages (about 4,300 characters), so almost every
document fits in one piece. Only documents over 24,000 characters, a few long meeting
transcripts, are split. The sample became 20,699 vectors, one per document.

### Why gte-modernbert-base

I wanted a model that reads a whole document at once, is good at English retrieval, and is
small enough to embed half a million long documents on an 8 GB laptop GPU overnight.
`Alibaba-NLP/gte-modernbert-base` fits all three:

- **It reads 8,192 tokens**, so 99.98% of documents are embedded whole, without chunking.
- **It's small**: 149M parameters and 768-dimensional vectors. It's built on ModernBERT, which
  skips padding and handles long inputs efficiently. That matters when most inputs are
  thousands of tokens long.
- **It's strong on English for its size.** On its model card it averages 64.4 on MTEB English
  (56 tasks), 55.3 nDCG@10 on BEIR retrieval and 88.9 on LoCo, a long-document retrieval
  benchmark.

The obvious alternative was **BGE-M3** (BAAI), also 8,192 tokens, but multilingual and built
on XLM-RoBERTa-large:

| | gte-modernbert-base | BGE-M3 |
|---|---:|---:|
| Parameters | 149M | 568M |
| Vector size | 768 | 1,024 |
| Languages | English | 100+ |
| MTEB English, 56 tasks | **64.4** | 59.8 |
| BEIR (nDCG@10) | **55.3** | 48.7 (dense), 49.4 (dense + sparse) |
| LoCo (nDCG@10) | **88.9** | 84.9 (dense), 87.4 (dense + sparse) |

The gte numbers are from its [model card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base);
the BGE-M3 numbers are from the [mGTE paper](https://arxiv.org/abs/2407.19669). Published
BEIR averages don't always cover the same datasets, so small gaps between papers mean little,
but a 6-point gap on BEIR and 4.5 on MTEB likely does.

On English, gte-modernbert is ahead on every row with a quarter of the parameters, so it
should also embed about 3 to 4 times faster. BGE-M3's strengths are other languages and its
two extra outputs: sparse keyword weights (a learned BM25) and one vector per token. In the
[BGE-M3 paper](https://arxiv.org/abs/2402.03216) the sparse weights beat its own dense vector
by about 10 points on long documents (MLDR, NarrativeQA). The corpus is English and I already
pair the dense model with BM25, so BGE-M3 would have to earn its place with those sparse
weights. That's still worth testing on the sample.

**3. BM25 on whole documents.** No model at all: the classic keyword ranking, with words
lowercased, stemmed and English stopwords removed (using the `bm25s` library). Indexing the
sample took 18 seconds on the CPU.

## Results

All three searched the same 20,691 documents; scores are over the 470 questions that expect
documents.

| | Vectors | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|---:|
| MiniLM, small chunks | 218,183 | 40.9% | 58.7% | 66.1% | 71.8% |
| gte-modernbert, whole document | 20,699 | 48.1% | 66.7% | 72.0% | 77.9% |
| **BM25, whole document** | — | **64.0%** | **84.0%** | **87.7%** | **90.3%** |

Recall@10 by question type:

| Question type | MiniLM chunks | gte whole document | BM25 |
|---|---:|---:|---:|
| basic (175) | 70.9% | 84.0% | **94.3%** |
| semantic (125) | 40.8% | 43.2% | **70.4%** |
| intra_document_reasoning (40) | 80.0% | 92.5% | **100%** |
| project_related (40) | 74.2% | 71.1% | **92.8%** |
| constrained (30) | 98.3% | 93.3% | **100%** |
| completeness (20) | 39.6% | 45.3% | **61.9%** |
| conflicting_info (20) | 82.5% | 75.0% | **97.5%** |
| miscellaneous (20) | 100% | 100% | 100% |

### What I take from it

- **BM25 wins almost everywhere, by a lot.** The questions name specific customers, projects,
  services and tickets ("ClearEdge", "po_fingerprint", "PR-19422"). An exact word match finds
  those directly; small embedding models blur rare names into general topics.
- **Even the semantic questions favour BM25** (70% against 43%). They're reworded, but they're
  also long, and a few rare words survive the rewording.
- **Whole documents beat small chunks for the embeddings.** gte gains most on basic and
  intra-document questions, where the answer depends on what the whole document is about. It
  does this with one vector per document instead of ten.
- **Small chunks win on narrow details.** MiniLM beats gte on constrained and conflicting-info
  questions, where one specific incident or value decides. A whole-document vector dilutes it.
- **Completeness is the open problem for everything.** Six near-identical status pages compete
  for ten places. That needs more results or a filter (project, date), not a better ranking.

### Caveats

- **The sample makes search easy.** On the full corpus there are 25 times more distractors, and
  many documents reuse the same names. That could hurt BM25 more than the embeddings. Only the
  full corpus can tell; see below.
- **The models are small.** MiniLM is from 2021, and gte-modernbert is a base-size model. A
  larger embedding model could close part of the gap with BM25.

## The full corpus

The sample said BM25 first, gte second, so I ran both on all 511,962 documents.

**gte on everything.** Embedding the corpus took about 7.5 hours of GPU time, closer to 18,000
tokens per second than my estimate. That was spread over three runs, because the pipeline can
stop and resume: it crashed once on a stray NUL character in an email, and the laptop
restarted once. Almost every document fit in one piece: 512,054 vectors for 511,962
documents.

**BM25 on everything** took 7.6 minutes on the CPU and 8 GB of memory, and is 1.2 GB on disk.

**A hybrid of the two.** The retrieval pipeline runs both searches at once, takes the top 50
documents from each, and merges them with **reciprocal rank fusion (RRF)**. A document scores
`1 / (60 + rank)` in each search that found it, and the scores are summed. A document missing
from one search gets nothing from it. **For now both searches have equal weight**, and `k = 60`
is the value from the original paper (Cormack et al., 2009).

| Full corpus (511,962 documents) | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| MiniLM on free summaries (baseline) | 8.3% | 12.5% | 14.5% | 17.1% |
| gte-modernbert, whole document | 30.9% | 45.4% | 51.1% | 56.6% |
| BM25, whole document | **45.7%** | 62.0% | 68.3% | **75.8%** |
| **Hybrid: BM25 + gte, equal-weight RRF** | 41.7% | **62.9%** | **71.4%** | 75.7% |

Recall@10 by question type:

| Question type | gte | BM25 | Hybrid |
|---|---:|---:|---:|
| basic (175) | 60.0% | 76.0% | **82.3%** |
| semantic (125) | 19.2% | 45.6% | **46.4%** |
| intra_document_reasoning (40) | 67.5% | **95.0%** | 90.0% |
| project_related (40) | 50.8% | 58.6% | **59.3%** |
| constrained (30) | 80.0% | 85.0% | **90.0%** |
| completeness (20) | 32.9% | 39.6% | **43.8%** |
| conflicting_info (20) | 67.5% | **90.0%** | **90.0%** |
| miscellaneous (20) | **100%** | 90.0% | **100%** |

### What I take from it

- **The sample was about 20 points too optimistic for both.** BM25 fell from 87.7% to 68.3% at
  Recall@10, and gte from 72.0% to 51.1%. More distractors didn't hurt BM25 more than the
  embeddings; the order stayed the same.
- **The hybrid helps the top 10 but hurts the top 1.** It finds 3 points more at Recall@10 than
  BM25 alone, mostly on basic, constrained and completeness questions. But it puts the right
  document first 4 points less often.
- **The cause is the equal weights.** With `k = 60` the scores are close (rank 1 gives 0.0164,
  rank 10 gives 0.0143), so a document both searches rank around 30th (about 0.022) beats
  either search's top hit. gte is much weaker than BM25 here, yet it gets the same say, and
  when it disagrees it can push BM25's correct first hit down a few places.
- **Semantic questions are still the weak spot** at 46%. The embedding adds almost nothing to
  them, even though rewording is exactly what embeddings are supposed to handle.

## Adding a reranker

Both searches compare the question with each document indirectly: BM25 counts shared words,
and gte compares two vectors computed separately. A **cross-encoder reranker** reads the
question and a document together, as one input, so every word of the question can attend to
every word of the document. That's far more precise, but nothing can be computed ahead of
time: it costs one full model pass per document, per question. So it only reorders the top
of the fused list.

I used `Alibaba-NLP/gte-reranker-modernbert-base`. It's from the same family as the embedder:
149M parameters, 8,192 tokens, so whole documents fit, and Apache 2.0. On its model card it
scores 56.7 on BEIR and 90.7 on LoCo. Larger rerankers score higher in published comparisons
(BGE-Reranker-v2-M3, Qwen3-Reranker), but this one is a quarter of their size, which matters
on an 8 GB GPU.

The pipeline fuses BM25 and gte as before, then the reranker scores the top 50 fused
documents (title plus text) and sorts them by its score. I saved all 50 reranked results per
question, so I could also score reranking only the top 10, 20 or 30 from the same run.

| Full corpus (511,962 documents) | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| gte alone | 30.9% | 45.4% | 51.1% | 56.6% |
| BM25 alone | 45.7% | 62.0% | 68.3% | 75.8% |
| Hybrid, no reranker | 41.7% | 62.9% | 71.4% | 75.7% |
| Hybrid + rerank top 10 | 59.3% | 69.7% | 71.4% | 76.1% |
| Hybrid + rerank top 20 | 61.4% | 72.5% | 74.9% | 76.1% |
| Hybrid + rerank top 30 | 63.1% | 75.2% | 77.9% | 79.6% |
| **Hybrid + rerank top 50** | **64.0%** | **76.2%** | **78.8%** | **81.0%** |

The fused top 50 holds 82.4% of the expected documents, so that's the most any reranking of
it can reach. The top-10, 20 and 30 rows are recomputed from the top-50 run. Where two
documents tie on fusion score, the reranker's order decides which one makes the cut, which can
move those rows by about a point.

Recall@10 by question type:

| Question type | Hybrid | + rerank top 30 | + rerank top 50 |
|---|---:|---:|---:|
| basic (175) | 82.3% | 88.6% | **89.7%** |
| semantic (125) | 46.4% | 55.2% | **56.8%** |
| intra_document_reasoning (40) | 90.0% | **100%** | **100%** |
| project_related (40) | 59.3% | 67.7% | **69.4%** |
| constrained (30) | 90.0% | 90.0% | 90.0% |
| completeness (20) | 43.8% | **50.4%** | **50.4%** |
| conflicting_info (20) | **90.0%** | **90.0%** | 87.5% |
| miscellaneous (20) | 100% | 100% | 100% |

### What I take from it

- **The reranker is the biggest gain so far.** The right document comes first 64% of the time
  instead of 42%, and on basic questions 82%.
- **It finally helps the semantic questions.** Recall@1 goes from 17% to 44%, and Recall@10 from
  46% to 57%. Reading the question and document together is what reworded questions needed.
- **Five reranked documents beat ten unreranked ones** (76.2% against 71.4%). That matters for
  cost: the LLM could read 5 documents instead of 10 and still see more of the right ones.
- **Depth has diminishing returns.** Reranking only the top 10 can't raise Recall@10, it only
  reorders. The top 30 gets almost all of the top 50's gain (75.2% against 76.2% at Recall@5)
  with 60% of the work.
- **It's slow.** On my laptop GPU the top 50 took about 3.7 seconds per question on real
  documents, twice my estimate from a synthetic test: real documents are longer, and attention
  cost grows faster than length.
- **Still unsolved:** completeness (about 50%; these questions need 6 or more near-identical
  documents) and project_related at rank 1 (22%; they span about 4 documents each).

## Answers, not just documents

Recall says whether the right document was found. What matters in the end is the answer. So
I ran the full evaluation on the best retrieval: BM25 and gte, fused, the top 50 reranked,
and **the top 5 documents** given to Gemini 3.8 Flash (each cut at 12,000 characters, thinking
set to low). The same model judges the answers.

| | Basic (175) | Semantic (125) |
|---|---:|---:|
| **Score** | **84.4%** | **52.9%** |
| Correct | 86.3% | 54.4% |
| Complete | 84.6% | 54.6% |
| Document recall (expected document in the top 5) | 88.6% | 55.2% |
| Gemini cost, answers and judge | $1.38 | $1.06 |

For comparison, the summary baseline scored 4.1% on the first 100 basic questions (6% correct,
21% document recall).

Why the wrong answers were wrong:

| | Basic | Semantic |
|---|---:|---:|
| Wrong answers | 24 | 57 |
| The expected document wasn't in the top 5 | 20 | 55 |
| …and Gemini said it couldn't find the information | 15 | 42 |
| The expected document was there, but the answer was wrong | 4 | 2 |

Where the expected document was ranked, against the answer. Ranks 1–5 come from the answer
runs. Ranks 6–50 come from the reranker run that kept all 50 results. The two runs put the
same documents in the top 5 for every question, so they can be combined.

| Rank of the expected document | Basic: answers | correct | Semantic: answers | correct |
|---|---:|---:|---:|---:|
| 1 | 143 | 141 | 55 | 54 |
| 2–5 | 12 | 10 | 14 | 13 |
| 6–10 | 2 | 0 | 2 | 0 |
| 11–20 | 3 | 0 | 3 | 0 |
| 21–50 | 0 | 0 | 1 | 0 |
| not in the top 50 | 15 | 0 | 50 | 1 |

- **When the document is first, the answer is right 98% of the time.** At ranks 2–5 it's
  still about 88%.
- **The misses aren't just below the cut.** Only 5 basic and 6 semantic documents sat at
  ranks 6–50, so giving Gemini 10 or 20 documents instead of 5 would rescue very few answers.
- **Most misses are a search failure:** 15 basic and 50 semantic questions had no expected
  document anywhere in the top 50. Reranking can't fix that; the question has to be searched
  differently, for example with other words.

### What I take from it

- **Retrieval is the limit, not the model.** Given the expected document, Gemini answered
  correctly about 97% of the time on both types (4 failures out of about 155 on basic, 2 out of
  about 69 on semantic).
- **So semantic questions fail at search, not at answering.** Almost every wrong semantic
  answer is a missing document: 55 of 57.
- **It rarely makes things up.** When the document was missing, it said so in 15 of 20 basic
  and 42 of 55 semantic cases instead of guessing. The other 18 deserve a read: they may be
  answers from a different document that says the same thing.
- **The retrieval evaluation predicts the answers.** Recall@5 was 88.6% for basic and 55.2%
  for semantic; document recall here is the same, and the score follows it within a few
  points. So the cheap retrieval evaluation is a reliable guide, and the full evaluation is
  only needed to check the answer step.
- **An answer costs about $0.006** (5 documents, about 7,500 input tokens), as the cost
  section below estimates for 5 documents. Answer and judge together are about $0.008 per
  question, so a full run of 500 questions is about $4.

## What it would cost to run

Before going further I wanted a rough price for running this in production, at two loads:
**50 requests per second** (about 131 million queries a month) and **50 requests per minute**
(about 2.2 million a month). These are estimates from November 2026, not measurements on
server hardware, so read them as ±50%. I'll update them as the system changes.

### Assumptions

- **Hardware:** AWS on-demand prices in us-east-1, 730 hours a month. An L4 GPU (g6.xlarge)
  is $0.805 an hour and an L40S (g6e.xlarge) $1.86 an hour. Each GPU is planned at 70% load,
  to leave room for peaks.
- **Search** (BM25, gte query embedding, Postgres vector search) runs on CPU. Embedding a
  question with gte takes 74 ms on one CPU core in float32, measured on my laptop. Postgres
  needs about 32 GB of RAM. The CPU servers also include the BM25 index and the API.
- **Reranker:** `gte-reranker-modernbert-base` on top of the fused results. On my laptop GPU
  the top 50 took about 3.7 seconds per question on real documents. I scaled that to server
  GPUs by their fp16 compute: an L4 does about 2.7 times my laptop's work, an L40S about 8
  times. Reranking the top 30 or top 10 costs proportionally less.
- **LLM answer:** Gemini Flash at $0.75 per million input tokens and $3.75 per million output
  tokens (aggregator prices, not Google's page). An answer reads 10 documents (about 13,000
  input tokens) and writes about 300 tokens, so **about $0.011 per answer**.
  Measured since: an answer from 5 documents uses about 7,500 input tokens and about 200
  output and thinking tokens, so **about $0.006 per answer**, in line with the "5 documents"
  rows below.
- **Five documents instead of ten:** with the reranker, the top 5 holds more of the expected
  documents than the unreranked top 10 (76.2% against 71.4%). Whether answers are as good from
  5 documents is not measured yet. Only the input halves: the output and the fixed prompt
  don't, so the LLM bill drops by about 45%, not 50%.

### Monthly cost

| Part | 50 requests/second | 50 requests/minute |
|---|---:|---:|
| Search only, no reranker (CPU servers) | ~$0.9k | ~$0.3k |
| Reranker, top 10 documents | ~$11k (8 × L40S) | ~$0.6k (1 × L4) |
| Reranker, top 30 documents | ~$27k (20 × L40S) | ~$0.6–1.2k (1–2 × L4) |
| Reranker, top 50 documents | ~$44k (32 × L40S) | ~$1.2k (2 × L4) |
| LLM answer, 10 documents | **~$1.43M** | **~$24k** |
| LLM answer, 5 documents | ~$0.79M | ~$13k |

| Whole system | 50 requests/second | 50 requests/minute |
|---|---:|---:|
| Search + LLM on 10 documents (no reranker) | ~$1.43M | ~$24k |
| Search + reranker on top 10 + LLM on 10 documents | ~$1.44M | ~$25k |
| Search + reranker on top 30 + LLM on 5 documents | ~$0.82M | ~$14k |
| Search + reranker on top 50 + LLM on 5 documents | ~$0.83M | ~$15k |

At 50 requests per minute an L4 reranks the top 50 in about 1.4 seconds per query, so two of
them are needed for the load; the top 30 just fits on one.

### What I take from it

- **The LLM is almost all of the bill**: about 90–99% of it, at either load. Search is a rounding
  error, and the reranker is small next to the answer step.
- **So the reranker pays for itself through the LLM.** It costs about $27–44k a month at 50
  requests per second. If it lets the LLM read 5 documents instead of 10, it saves about $0.6M.
  The retrieval numbers say it can; the answer evaluation still has to confirm it.
- **Top 30 is the sweet spot**: nearly the top 50's recall for about 60% of its GPU cost.
- **At low traffic, fixed costs dominate search.** Search costs about $0.007 per 1,000
  queries at 50 requests per second, but about $0.14 at 50 per minute, because the servers run
  around the clock either way.
- **Search alone is cheap enough to never worry about.** Even at 50 requests per second it's
  under $1k a month on CPU only. Query embedding doesn't need a GPU: indexing does, once.
- **Things these numbers leave out:**
  - re-embedding new or changed documents, a background GPU job;
  - savings plans or spot instances, about 30–70% off the hardware;
  - caching repeated questions, which cuts the LLM bill directly;
  - SageMaker's markup over plain EC2, typically 20–40% from memory.

What others do to cut these costs, and which of it fits here, is in
[the companion note on RAG cost savings](002-1_rag_cost_savings.md).

## What's next

- **Weight the fusion.** Give BM25 more weight than gte, for example
  `2 / (60 + rank_bm25) + 1 / (60 + rank_gte)`. Tune the ratio on the sample, then confirm it
  once on the full corpus so it isn't fitted to the reported numbers.
- **Make "rerank top 30, answer from 5 documents" the default candidate**, and confirm with the
  answer evaluation that 5 reranked documents answer as well as 10.
- **Make the reranker cheaper**: cut documents at 1,024 to 2,048 tokens, or install
  flash-attention, and measure what it costs in recall.
- **Compare LLMs on price and quality**, including cheaper models on Amazon Bedrock, with the
  same retrieved documents.
- **Try BGE-M3's sparse weights** against BM25 on the sample.
- Run the full answer evaluation, with a sample balanced across question types, on the best
  retrieval.
