# A negative-sampled training set for the embedding model: reranker + judge

Status: proposal, 11 November 2026. Not yet planned. Follows the decision in article 006
(10–11 November): fine-tune the embedding model with negative sampling.

## 1. Why

- gte adds only about 3% on top of BM25 (article 006, "What does the embedding model add?").
  On semantic questions it works poorly: 42 of 125 right documents are found by neither
  search, and BM25 alone finds 45 that gte misses.
- gte separates *topics* well. A question scores 0.74 against its own top 50 and 0.56 against
  another question's top 50. It does *not* separate near-duplicates: its whole top 50 spans
  about 0.05 of cosine, and a missed right document sits 0.04–0.09 below rank 1.
- A bigger general model (Qwen3-Embedding-0.6B) doesn't help inside the hybrid search.
- So the training signal must be **hard negatives**: near-duplicates of the right document,
  not random documents.

## 2. Which teacher (data, 10–11 November)

On the 46 semantic failures, about 53 candidates each, where the right document lands:

| Ranked by | Top 1 | Top 5 | Top 10 | Median rank |
|---|---:|---:|---:|---:|
| gte (its own search) | 0 | 0 | 0 | outside its top 50 for 44 |
| **gte-reranker** (full text, local, free) | **16** | **25** | **33** | **4** |
| gemini-embedding-2 | 9 | 17 | 25 | 8 |

- The reranker is the best teacher we have, and it's free.
- Gemini embeddings spread the near-duplicates about 2.7× wider than gte, but they rank
  worse than the reranker (better in 15 failures, worse in 26).
- Across the full evaluation, the reranker moves 481 of the 577 right documents in the
  fused top 50 (83%) into the final 5.

**The reranker can't spot false negatives on its own.** Its scores are squashed (half of all
candidates fall between 0.77 and 0.86). The judge had already rated 229 hard candidates.
Those it said fully answer the question (false negatives) scored about the same as those it
said don't answer (true hard negatives). The rule "drop if the score is ≥ the right
document's score − 0.02" removes all 6 false negatives, but also 145 of the 200 true hard
negatives.

The false-negative rate itself is **not known**. The 6 of 229 (2.6%) come from wrong answers
only, a set that by construction holds few documents that answer. See open points.

## 3. The proposal

The reranker decides **what** gets judged; the judge decides **what it is**.

1. **Questions.** Synthetic questions written from sampled corpus documents; the source
   document is the positive.
2. **Candidates.** gte top 50 + BM25 top 50 per question, as in production.
3. **Reranker** scores the positive and every candidate.
4. **Judge the grey zone only:** candidates scoring above the positive or within about 0.02
   below it. Cap at about 5 per question.
   - full → false negative: drop it (or keep it as a second positive);
   - partial → drop;
   - none → confirmed hard negative.
5. **Below the grey zone,** candidates become negatives without judging.
6. **Train** with MultipleNegativesRanking (1 positive + a few hard negatives); optionally add
   the reranker's score gaps as soft labels (MarginMSE, the GPL idea).

The same pass also filters the questions: if the judge says the source document doesn't
fully answer its own question, or the reranker scores the source low, the question is too
generic or broken. Drop it.

## 4. Decisions (the user, 11 November)

1. **Training questions must not touch documents used in evaluation.** Exclude the right
   documents of all 500 benchmark questions (and probably their nearest near-duplicates)
   from the documents the questions are generated from.
2. **Question style:** generated questions paraphrase and avoid the document's own wording,
   like the benchmark's semantic questions. Otherwise the embedder learns what BM25 already
   does.
3. **No model grades itself:** if DeepSeek writes the questions, Gemini judges (or the other
   way round).
4. **Check the judge first:** measure the false-negative rate (about $1.50, see below) and
   have the user spot-check about 50 of its verdicts.
5. **Keep the judging cost down** by capping the grey zone per question.
6. **Accepted limit:** the embedder learns the reranker's view of relevance. That targets
   our bottleneck (getting the right document into the reranker's list), not the cases
   where the reranker itself ranks it low (about half of the hard failures).

## 5. Rough cost for 2,000 questions

| Step | Cost |
|---|---:|
| Generate the questions (DeepSeek, off-peak) | ~$1 |
| Mine candidates, reranker scores | $0 (laptop) |
| Judge the grey zone (Gemini 3.8 Flash, ~$0.0024 per rating, ≤5 per question) | ~$15–25 |
| Train (Vast.ai RTX 4090, ~$0.36–0.48/h) | ~$0.50 |
| Quick evaluation without a re-index (100-question sample + 5k distractors) | $0 (laptop) |
| **Total** | **~$20–30** |

The judging is most of it. With DeepSeek as the judge it would be about $5–10.

## 6. Open points

- **The real false-negative rate.** Proposed test: about 30 benchmark questions, half
  answered right and half wrong, about 20 candidates each spread over gte's and BM25's
  top 50, all rated by the judge (about 600 ratings, about $1.50). It also shows whether the
  grey zone catches the false negatives.
- **Judge vs human:** never checked.
- **How big the grey zone is** for synthetic questions (it sets the judging cost).
- **Base model:** gte-modernbert-base (cheap re-index, about 7 h on the laptop) or
  Qwen3-Embedding-0.6B (better alone, about 23 h).
- **Which judge:** the document judge from notebook 006 (`document_answer_instructions`), or
  the ambiguity judge, or a new prompt for "does this document also answer?".
- **The user's own research** on frameworks and methods comes first; this note is the
  starting proposal for the brainstorm.

## 7. Related work (from 11 November)

- NVIDIA Nemotron embedding recipe (March 2026): SDG → mine top-5 negatives below 95% of the
  positive → fine-tune. https://huggingface.co/blog/nvidia/domain-specific-embedding-finetune
- GPL (UKPLab, NAACL 2022): synthetic questions + cross-encoder margins (MarginMSE).
  https://github.com/UKPLab/gpl
- NV-Retriever: positive-aware mining cuts false negatives by 50–57%.
  https://www.arxiv.org/pdf/2407.15831
- sentence-transformers `mine_hard_negatives(relative_margin=...)`.
  https://sbert.net/docs/package_reference/util/hard_negatives.html
- "Don't Retrieve, Generate": LLM-generated negatives alone are worse than mined ones.
  https://arxiv.org/html/2504.21015

## 8. Data and scripts

- `exploration/retriever_attribution/`: replayed gte / BM25 / fused top 50 for all 500
  questions.
- `exploration/embedder_comparison/`: the 100-question sample with distractors (a quick
  evaluation without a re-index).
- `exploration/gemini_embedding/`: Gemini vs gte vs reranker on the 18 cached questions and the
  46 failures.
- Judge ratings: `exploration/question_ambiguity/rating_cache.json`.
