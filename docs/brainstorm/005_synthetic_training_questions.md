# Our own training questions, made the benchmark's way

Status: design, approved in chat 11 November 2026. Follows brainstorms 003 (negative-sampled
training set) and 004 (the reranker's biggest drop). Decisions are recorded in article 006.

## Why

We fine-tune the embedder with hard negatives. Training on the benchmark's own 500
questions would turn the test into a memory test, so we make our own questions and keep the
benchmark for evaluation only. The benchmark's generation code and prompts are public (MIT,
[onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench),
`src/prompts/`, flows in the paper's Appendix C), so our questions can have the same style
and difficulty.

## Train, dev and test (the user, 11 November)

| Set | Made from | Used for |
|---|---|---|
| **Train** | our synthetic questions, about 2,000 kept | fine-tuning the embedder |
| **Dev** | our synthetic questions, about 400 kept, from different documents | every hypothesis and setting: the drop rule, how many negatives, training settings, comparing variants |
| **Test** | the 500 benchmark questions | checked **once**, when the approach is final |

So we don't tune on the test set. The knee check on the benchmark's 100-question sample
(brainstorm 004) was a first look; from now on such checks run on dev.

- Dev documents are picked first. They and their 20 nearest documents by gte are excluded from
  train, like the benchmark's documents, so train never holds a dev answer or its twin.
- Both sets use the same type and source mix.

## Goals and non-goals

- **Goal:** about 2,000 one-document training questions, each with its source document as the
  positive and 4–5 hard negatives picked by the biggest-drop rule.
- **Goal:** the same style and mix as the benchmark's one-document questions.
- **Non-goal (for now):** the multi-document types (project related, constrained,
  conflicting info, completeness). They need corpus-exploring agents or planted structures.
- **Non-goal:** high level and info not found. They have no right document.
- **Non-goal:** gold answers and fact lists. The embedder needs only question + document.
- **Anti-goal:** any training question generated from, or any negative that is, a document
  used in evaluation.

## Decisions

| | Decision | Whose |
|---|---|---|
| Types | basic, semantic, intra-document reasoning, miscellaneous | user |
| Mix | the benchmark's 175 : 125 : 40 : 20 (about 49 / 35 / 11 / 6%) | user ("same distribution") |
| Prompts | the benchmark's own prompts for these four types, unchanged where possible | user ("same ideas") |
| Sources | sample documents to match the benchmark's per-source mix for each type (paper Table 10) | user |
| Evaluation documents | exclude the right documents of all 500 questions as sources and as negatives | user |
| Near-duplicates of those | also excluded as sources: each right document's 20 nearest documents by gte | our call |
| Size | about 2,000 train + 400 dev kept; generate about 3,100 (30% extra, the drop rule skips some) | user (dev set), our call (sizes) |
| Generator | DeepSeek V4.1 Flash (`deepseek-flash`), off-peak where possible; about $1.50 | our call |
| Intra-document check | keep the benchmark's check that the question can't be answered from one passage | our call |
| Negatives | biggest-drop rule (brainstorm 004): skip the question if its source sits below the drop; drop everything above the drop; take negatives just below it | agreed earlier |

## Where the documents come from

- **basic, semantic:** random documents, stratified by source as in the benchmark.
- **intra-document reasoning:** long documents only (the paper uses a minimum length; we take
  the benchmark's choice if the code states it, otherwise about the top third by length).
- **miscellaneous:** the benchmark's off-topic folders. Our database has no file paths, but
  the repo's `generated_data/uuid_index.json` maps every doc_id to its path. The benchmark's
  20 miscellaneous questions come from `slack/sports`, `slack/memes`, `confluence/misc`,
  `google_drive/.../misc-assets`, `jira/misc-requests`, `fireflies/interviews`,
  `google_drive/.../_temp_uploads` and `github/hackathons`.

## The pipeline

1. **Pick source documents** per type and source, excluding evaluation documents and their
   near-duplicates.
2. **Generate** one question per document with the type's prompt (DeepSeek). For
   intra-document questions, run the conformance check and drop failures.
3. **Search** each question as production does: gte top 50 + BM25 top 50.
4. **Rerank** the source document and every candidate (gte-reranker, local).
5. **Biggest drop** in the top 30: skip the question if its source is below; otherwise keep
   4–5 negatives just below the drop, never an evaluation document.
6. **Save** the training set (question, type, source document, negatives, reranker scores)
   for the fine-tuning step.

Every long step reports to the progress page. Everything is resumable and cached.

## What it should tell us on the way

- The skip rate per type: how often the source document isn't above the drop.
- Grey-zone size per question.
- Cost per question, against the estimate.

## Left to the builder

- Exact sampling seed and per-source counts (rounded from Table 10).
- File layout under `exploration/` and the output format.
- Whether to use the benchmark's few-shot examples verbatim (yes, unless they leak benchmark
  content; check them).

## First run (11 November)

Generation: 3,120 source documents, 3,011 valid questions, **$0.96** on DeepSeek (+$0.01 for
pilots). Intra-document questions passed the benchmark's check in 238 of 347 cases, even with
up to 3 tries. Mining (search + rerank + biggest drop) is local and free, about 3.2 hours.

| Split | Type | Kept | Of | Skipped |
|---|---|---:|---:|---:|
| train | basic | 1,258 | 1,373 | 8% |
| train | semantic | 585 | 903 | 35% |
| train | intra-document | 191 | 199 | 4% |
| train | miscellaneous | 33 | 35 | 6% |
| **train** | **all** | **2,067** | **2,510** | **18%** |
| dev | basic | 259 | 275 | 6% |
| dev | semantic | 118 | 180 | 34% |
| dev | intra-document | 38 | 39 | 3% |
| dev | miscellaneous | 7 | 7 | 0% |
| **dev** | **all** | **422** | **501** | **16%** |

Grey zone (documents above the drop): median 1, p75 3–4, p90 16–19, largest 30 (the whole
window). Negatives per kept question: median 5.

What the numbers say:

- **Semantic questions are where search fails, as on the benchmark.** For 656 of 1,083
  semantic questions (61%) the source document was not in the fused top 50 at all; for basic
  it was 234 of 1,648 (14%). The reranker still puts the source above the drop for 703 of the
  1,083, so most of these misses become training examples: exactly the cases the embedder
  needs to learn.
- **Skips are mostly semantic** (380 of 522 skipped questions). A skipped question is one the
  reranker doesn't rank its own source above the drop for: either the question is too vague
  or the reranker is wrong. Both are worth dropping.
- **Most questions have a sharp drop right after the source** (median 1 above the drop), so
  the negatives sit just below a clear gap, as in the pilot.

Negatives never include an evaluation document or the other split's answers (the review
found 16 + 13 such cross-split negatives in the first write; fixed, counts unchanged).

Outputs: `exploration/training_questions/data/training_set.jsonl` (2,067) and `dev_set.jsonl`
(422); each row has the question, type, source, up to 5 negatives and every reranker score.
