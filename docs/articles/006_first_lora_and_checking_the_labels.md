# The first fact-extraction LoRA, and checking its labels

Status: notes and decisions, 7–8 November 2026. Follows
[Cheap labels for a small fact-extraction model](005_cheap_labels_for_fact_extraction.md).
Code: `exploration/fact_extraction/` (`train.py`, `vast_train.sh`, `judge_labels.py`).

## What the facts are for

The facts and entities a small model extracts from each document are **search metadata**.
They help the search find the right document among near-duplicates that differ in one
detail. They don't replace the document: once a document is found, the answering model reads
its full text. That purpose sets the bar for the labels:

- **They must not be made up.** An invented or wrong number or name can send a search to the
  wrong near-duplicate, which is the failure we are trying to fix.
- **They don't need to be complete.** A fact only has to point to the right document; the
  explanation, the steps and the "why" stay in the document.

## The first LoRA run (7 November)

Qwen3.5-2B with LoRA (rank 16, 16.8M trainable parameters, text layers only) on 2,695 DeepSeek
labels, 3 epochs, effective batch 16, learning rate 1e-4. Run on a rented RTX 4090 (Vast.ai,
$0.42/h), logged to Weights & Biases (`rag-train/train-rag-facts`, run `qwen35-2b-r16-e3`).

| | |
|---|---|
| Training time | 68 min (median example 2,020 tokens, longest cut to 4,096) |
| Cost of the run | $1.01, plus $0.74 in five failed attempts before it |
| JSON that parses, 50 held-out documents | 100% |
| Word F1 against DeepSeek's facts / entities | 0.75 / 0.85 |
| Generation speed (plain `generate`) | about 25 s per document |

Word F1 compares the sets of words in the model's facts with DeepSeek's. It shows the model
learned the format and the vocabulary, but not whether a fact is true: a wrong number in an
otherwise matching fact costs one word. Hence the checks below.

What the five failed attempts taught (each fixed in `vast_train.sh` or `train.py`):

1. `str.splitlines()` splits JSONL lines on U+2028 inside document text; split on `"\n"`.
2. The fast DeltaNet kernels (flash-linear-attention) compile with Triton, which needs a C
   compiler; the slim image has none.
3. A host can be "running" without accepting SSH; check SSH for real and move on to the next
   offer.
4. The plain `pytorch/pytorch` image left hosts "loading" for 15+ minutes; Vast's own "PyTorch
   (Vast)" template boots in about a minute.
5. A 248k-token vocabulary makes the output scores huge: about 4 GB per 4k-token example in
   float32. Batch 4 ran out of memory on 24 GB; batch 1 with 16 accumulation steps fits and
   trains the same.

## Checking the labels (8 November)

The fine-tuned model can only be as good as DeepSeek's labels, so the labels were checked
first, at three levels of cost.

### A free screen: numbers, ids and URLs

Every token with a digit, and every URL, in a fact must appear in the document, written as is
or with number and unit set apart ("30-minute" matches "30 minutes"). Run on all 64,617 facts
of the 5,370 labelled documents:

| Source | Facts | Flagged |
|---|---:|---:|
| google_drive | 6,916 | 4.0% |
| fireflies | 2,735 | 3.0% |
| gmail | 13,832 | 2.3% |
| confluence | 2,454 | 1.3% |
| slack | 23,981 | 1.0% |
| hubspot | 5,081 | 0.8% |
| jira | 2,455 | 0.7% |
| linear | 4,935 | 0.5% |
| github | 2,228 | 0.2% |
| **all** | **64,617** | **1.6%** |

A flag is a lead, not a verdict. Some flags are rewritten values ("~13:30 UTC" when the
document gives the time in another zone, "$12600/month" for "$12.6k per month"). Others look
like real errors, e.g. "partner-demo window 07:30–6:30 PT", or table values that may have been
mixed up between rows. Table-heavy sources (Drive, meeting transcripts) are flagged most.

### Are the facts made up? (LLM judge)

Gemini 3.8 Flash got the document and its label, and gave every fact a verdict: *supported*,
*unsupported* (the document doesn't say it) or *contradicted* (the document says something
else), with the passage behind it. The sample was 301 labels of training documents, spread
over the sources in proportion to their size. The judge is strict about specifics: one wrong
number, date or name fails the fact.

| Source | Docs | Facts | Unsupported | Contradicted | Docs with ≥ 1 bad fact |
|---|---:|---:|---:|---:|---:|
| slack | 108 | 1,301 | 0.6% | 0.5% | 13.0% |
| gmail | 54 | 650 | 0.3% | 0.3% | 7.4% |
| linear | 26 | 313 | 0.3% | 0.3% | 7.7% |
| google_drive | 23 | 279 | 0.4% | 0.4% | 4.3% |
| hubspot | 20 | 240 | 0.0% | 0.0% | 0.0% |
| fireflies | 18 | 216 | 0.9% | 0.9% | 16.7% |
| github | 18 | 216 | 1.4% | 0.0% | 11.1% |
| confluence | 17 | 204 | 0.0% | 0.5% | 5.9% |
| jira | 17 | 204 | 0.5% | 0.0% | 5.9% |
| **all** | **301** | **3,623** | **0.5%** | **0.4%** | **9.3%** |

**31 of 3,623 facts (0.9%) fail.** Every "about" sentence and every entity is backed by its
document. The failures are not invented content but small slips in busy threads:

- **mixed-up attribution:** a requirement said by Anne-Marie given to Samir; a 30% rate that
  Maya *suggested* turned into one she *approved*;
- **two statements merged wrongly:** a deadline that applies to one action item extended to
  another; a sampler that gates one process attached to a different one;
- **an off-by-one specific:** a CloudTrail timestamp one second off; "next Tuesday" from a
  Wednesday call computed as 2026-09-14 (a Monday) instead of 09-15;
- **cause added:** "the escalation failure caused the error spikes" when the document reports
  both but doesn't link them.

Chat and meeting sources (Slack, Fireflies) have the most slips, as expected for many-speaker
threads. **Verdict: the labels pass the acceptance rule.** About 1 fact in 100 has a wrong
or added detail, and no fact is made up from nothing. That is good enough to train on and to index.
Checking DeepSeek's labels cost $3.85 in total (the 13-document trial $0.14, 301 labels $3.35, the
gold-fact check $0.36).

### Do the facts keep what the questions ask? (LLM judge)

For the 58 benchmark questions whose expected documents are labelled, Gemini 3.8 Flash got
the question, the reference answer, its gold facts, the full documents and the extracted
facts. It decided for each gold fact whether the document states it and whether the
extraction keeps it, and whether the question could be answered from the extraction alone.

| Question type | Questions | Gold facts | In document | Kept | Partly kept | Answerable from facts | Partly |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic | 5 | 11 | 100% | 100% | 0% | 100% | 0% |
| conflicting_info | 1 | 8 | 100% | 75% | 0% | 0% | 0% |
| intra_document_reasoning | 1 | 2 | 100% | 100% | 0% | 100% | 0% |
| semantic | 51 | 210 | 100% | 46% | 31% | 45% | 43% |
| **all** | **58** | **231** | **100%** | **50%** | **28%** | **50%** | **38%** |

Every gold fact is in its document. The earlier word-matching check found only 71% because
the benchmark words its facts differently from the documents. The extraction keeps half of
them whole. What it drops:

- a procedure reduced to its name: "Ten-Step Restoration Sprint", without the steps;
- the mechanism or the "why", with the "what" kept: the cooldown is 15 minutes, but not that
  spikes within it count as one billable event;
- a PR's title when the question asks where a proposal is.

Twelve facts of at most 20 words can't hold a step-by-step checklist, and they aren't meant
to. **For retrieval this is enough**: the kept facts are the distinguishing ones (names,
numbers, ids) that point to the document, and the answer is read from the document itself.
It would not be enough to answer from the facts alone, which we don't plan to do.

## The fine-tuned model against the document and DeepSeek (8 November)

The checks above are of DeepSeek's labels, the training data. The fine-tuned model's own
outputs were then generated for the 305 held-out documents on a laptop GPU (RTX 2000 Ada,
8 GB, about 12 s per document, free). Every output parsed as JSON; 28% had 13–14 facts
instead of at most 12. For 101 of them, one Gemini call per document got the document,
DeepSeek's label and the model's output, and judged each model fact against the document
(not against DeepSeek, whose label can be incomplete) and each DeepSeek fact for whether the
output keeps it. Cost $1.21.

| Source | Docs | Model facts | Unsupported | Contradicted | Docs with ≥ 1 bad fact | DeepSeek facts kept | Partly | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slack | 36 | 433 | 0.7% | 1.8% | 27.8% | 57% | 30% | 13% |
| gmail | 18 | 216 | 0.9% | 1.9% | 33.3% | 65% | 23% | 12% |
| google_drive | 8 | 100 | 0.0% | 2.0% | 25.0% | 57% | 28% | 15% |
| linear | 8 | 99 | 1.0% | 1.0% | 25.0% | 55% | 22% | 23% |
| fireflies | 7 | 86 | 3.5% | 3.5% | 57.1% | 35% | 29% | 37% |
| github | 7 | 86 | 0.0% | 1.2% | 14.3% | 67% | 23% | 11% |
| hubspot | 7 | 81 | 1.2% | 1.2% | 14.3% | 62% | 19% | 19% |
| confluence | 5 | 59 | 1.7% | 0.0% | 20.0% | 61% | 17% | 22% |
| jira | 5 | 57 | 0.0% | 1.8% | 20.0% | 58% | 25% | 17% |
| **all** | **101** | **1,217** | **0.9%** | **1.7%** | **27.7%** | **58%** | **26%** | **16%** |

Against DeepSeek's own rate (0.9% bad facts, 9.3% of documents with one, measured on 301
other documents of the same sample), the fine-tuned model makes about **three times as many
errors: 2.6% of facts, in 28% of documents.** The errors are of the same kind as DeepSeek's,
only more often, and mostly *contradicted* rather than invented:

- **wrong person:** a proposal made by Redwood given to the customer Garnet; an offer by
  Karthik Iyer given to Dylan Brooks; Samir's explanation given to Ava;
- **wrong clause:** a fallback command of the automated rollback attached to the
  human-in-the-loop rollback; a proposed fix ("abort on zero-length part") stated as what
  the uploader did;
- **two slots or owners merged:** an Americas-friendly time slot listed as an APAC one.

Meeting transcripts (Fireflies) are the weak spot: 7% bad facts, more than half the
documents affected, and only 35% of DeepSeek's facts kept. A transcript is long, with many
speakers, and is where a 2B model most often loses track of who said what.

Completeness: 58% of DeepSeek's facts are kept whole and 26% partly. Most of the "missing"
are replaced by other, supported facts of the same document, since the model also writes
about 12.

## Do the facts help the search? A first pilot (8 November)

The 46 search failures of post 004 each have a pool: gte's top 50 for the question plus the
expected document, about 51 documents, all labelled by DeepSeek. Each pool was ranked seven
ways and the expected document's rank recorded (`retrieval_pilot.py`, local, free):

| Variant | Top 1 | Top 5 | Top 10 | Median rank |
|---|---:|---:|---:|---:|
| gte, full text (2,000-character chunks) | 0 | 0 | 0 | 51 |
| BM25, full text | 8 | 20 | 31 | 8 |
| gte, one fact block per document | 0 | 1 | 1 | 50 |
| gte, each fact separately (best counts) | 3 | 7 | 12 | 31 |
| BM25, fact block | 9 | 19 | 26 | 8 |
| fusion: gte text + gte fact block | 0 | 1 | 1 | 51 |
| fusion: all four of gte/BM25 × text/facts | 1 | 2 | 3 | 23 |
| fusion: gte each fact + BM25 facts | 3 | 7 | 20 | 14 |
| fusion: gte each fact + BM25 facts + BM25 text | 3 | 20 | 27 | 7 |

Read with care: the pools are gte's own top 50 *without* the expected document, so they are
built from exactly the documents gte prefers. gte on the full text scoring 0 is by
construction. BM25 is not biased that way, and its statistics come from the pool only.

- **One embedding of the whole fact block doesn't help.** It is better than the full text in
  24 of 46 questions, a coin flip, and still near the bottom. Averaging twelve facts into
  one vector blurs the single detail that tells near-duplicates apart.
- **One embedding per fact helps.** Better than the full text in 42 of 46, 12 of 46 in the
  top 10 instead of 0. The one fact that matches the question can win on its own.
- **BM25 on the facts is about as good as BM25 on the full text** (26 vs 31 in the top 10)
  from a fraction of the text. The two find partly different documents: 3 only through the
  facts, 8 only through the text, 34 of 46 through either.
- **Plain rank fusion with gte hurts here,** because gte ranks the expected document last
  in every pool; on these failures gte is the noise.
- **Fusing the per-fact vectors with BM25 doesn't beat BM25 alone.** gte each fact + BM25
  facts gets 20 in the top 10, between its two parts (12 and 26), and only 3 at rank 1
  against BM25's 9. Adding BM25 on the full text brings it level with BM25 on the full text
  alone (27 vs 31 in the top 10, median 7 vs 8). Equal-weight fusion averages a strong
  signal with a weaker one; the per-fact vectors would need a lower weight, or a role
  of their own (e.g. adding candidates the text search misses), to add to BM25.

**Weights, and the fusion we run today.** The current pipeline fuses gte on the full text
with BM25 on the full text. In these pools that fusion does badly (0 at rank 1, 2 in the
top 10, median rank 24), because gte puts the expected document last and drags BM25 down;
that is the failure the pools were built from. Adding to it, with gte each fact at weight w
(RRF, the others at weight 1):

| Fusion | w | Top 1 | Top 5 | Top 10 | Median rank |
|---|---:|---:|---:|---:|---:|
| current (gte text + BM25 text) | – | 0 | 0 | 2 | 24 |
| current + BM25 facts | 0 | 0 | 1 | **20** | 14 |
| current + BM25 facts + gte each fact | 0.5 | 0 | 6 | 17 | 13 |
| current + BM25 facts + gte each fact | 1 | 0 | 9 | 16 | 16 |
| BM25 facts + gte each fact | 0.25 | 6 | 19 | 26 | 10 |
| BM25 facts + gte each fact | 1 | 3 | 7 | 20 | 14 |
| BM25 text + BM25 facts | 0 | 7 | 21 | 29 | 6 |
| BM25 text + BM25 facts + gte each fact | 0.25 | 9 | 22 | 30 | 7 |
| BM25 text alone | – | 8 | 20 | 31 | 8 |

- **A BM25 field on the facts rescues the current fusion:** from 2 to 20 of 46 in the top
  10. It is the single most useful addition.
- **Lowering the weight of the per-fact vectors never makes them add to BM25.** Each
  BM25 combination gets a little worse as w rises. Next to the current fusion they move
  documents into the top 5 (1 → 9) but out of the top 10 (20 → 16).
- **In these pools, nothing beats BM25 on its own text by much.** BM25 text + facts
  (+ per-fact vectors at 0.25) gets 9 / 22 / 30 against 8 / 20 / 31. The pools are gte's
  favourites, so they reward anything that ignores gte; the full-corpus test has to say
  whether the BM25 facts field also helps when gte's ranking is right.

**MiniLM instead of gte for the per-fact vectors.** all-MiniLM-L6-v2 has 384 dimensions,
half of gte's, so per-fact vectors would take half the space.

| Variant | Top 1 | Top 5 | Top 10 | Median rank |
|---|---:|---:|---:|---:|
| gte, full text | 0 | 0 | 0 | 51 |
| MiniLM, full text (1,000-character chunks) | 3 | 6 | 13 | 32 |
| gte, each fact | 3 | 7 | 12 | 31 |
| MiniLM, each fact | 4 | 8 | 14 | 27 |
| MiniLM, fact block | 0 | 4 | 8 | 32 |
| current + BM25 facts + gte each fact (w 0.75) | 0 | 6 | 16 | 15 |
| current + BM25 facts + MiniLM each fact (w 0.75) | 0 | 9 | 20 | 12 |
| BM25 facts + MiniLM each fact (w 0.25) | 10 | 20 | 25 | 10 |

MiniLM does as well as gte on single facts here, and pulls the fusions down less. But the
pools are gte's favourites, so any model that is *not* gte has an edge: MiniLM on the full
text scores 13 in the top 10 where gte scores 0, which says more about the pools than about
MiniLM. The fair conclusion is only that MiniLM is not worse than gte for short facts, and
that neither adds to BM25.

So the facts are worth indexing **as a BM25 field first**; per-fact vectors beat whole-text
vectors but don't add to BM25 at any weight tried. One summary vector per document doesn't
help. Whether that holds over the whole corpus, where the expected document
competes with all 512k documents and not just gte's top 50, is the next test.

## A wider reranker list, and reranking on facts (9 November)

Two things helped in the earlier tests: the reranker, when it gets to see the right
document (post 004: top 5 in 29 of 46), and BM25 over the facts. This section tests both
together. From here on the pools are called **candidates**: for each failure, gte's top 50
plus the expected document.

**How deep would the reranker have to read?** (`depth_pilot.py`, free.) Today's hybrid
search was run over the whole corpus: gte chunks and BM25, fused with RRF. This time it
kept 1,000 documents from each ranking instead of 50. The table counts, for each depth,
the failures where the expected document is among the fused results:

| Fused documents the reranker would read | gte alone | BM25 alone | Fused |
|---:|---:|---:|---:|
| 50 (today) | 3 | 10 | 8 |
| 100 | 6 | 13 | 14 |
| 200 | 10 | 15 | 19 |
| 300 | 12 | 19 | 22 |
| 500 | 13 | 24 | 25 |
| 1,000 | 18 | 34 | 30 |

A wider list helps, but slowly. At 300 documents, half the failures reach the reranker. At
1,000, 16 of the 46 still don't, and reading 1,000 full documents takes about a minute per
question. No reranker setting can rescue those 16; they need a way in that doesn't depend
on today's search: routing, keywords or buckets. Even at 50, 8 failures already had the
document in the fused list, and the reranker ranked it out of the top 10.

**Reranker on full text, on facts only, and fused with BM25 facts.**
(`rerank_pilot.py`, free.) Every candidate of the 46 failures was scored by
gte-reranker-modernbert twice: once on its text, up to the reranker's 8,192 tokens, and once
on the DeepSeek fact block only (title, about line, entities, facts). Rank of the expected
document among its ~51 candidates:

Recall@5 and recall@10 are the share of the 46 questions whose expected document is in the
top 5 or top 10. "Missed" counts the questions where it isn't in the top 10.

| Variant | Top 1 | Top 5 | Top 10 | Recall@5 | Recall@10 | Missed | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25, full text | 8 | 20 | 31 | 43% | 67% | 15 | 8 |
| BM25, facts | 9 | 19 | 26 | 41% | 57% | 20 | 8 |
| **Reranker, full text** (59 ms per document) | **16** | **25** | **33** | **54%** | **72%** | **13** | **4** |
| **Reranker, facts only** (11 ms per document) | 12 | **25** | 28 | **54%** | 61% | 18 | 5 |
| Reranker text + BM25 facts | 14 | 25 | 30 | 54% | 65% | 16 | 4 |
| Reranker text + BM25 facts + BM25 text | 9 | 27 | 34 | 59% | 74% | 12 | 5 |
| Reranker text + reranker facts | 16 | 26 | 34 | 57% | 74% | 12 | 5 |
| Reranker text + 0.25 × BM25 facts | 12 | 25 | 34 | 54% | 74% | 12 | 4 |
| Reranker text + 2 × BM25 facts | 12 | 22 | 29 | 48% | 63% | 17 | 6 |
| Cheap fusion → its top 30 → reranker text | 16 | 26 | 33 | 57% | 72% | 13 | 4 |
| Cheap fusion → its top 10 → reranker text | 13 | 24 | 28 | 52% | 61% | 18 | 5 |

("Cheap fusion" is BM25 text + BM25 facts + gte text at weight 0.25.)

- **Once the right document is in the list, the reranker does the work.** Adding BM25 over
  the facts moves the top 5 from 25 to 27 at most and costs places at rank 1, at every weight
  tried. The earlier gain from BM25 facts (2 → 20 of 46 in the top 10) was against today's
  fusion *without* a reranker; with the reranker it mostly disappears.
- **The reranker on facts only is about 5× faster** (11 instead of 59 ms per document)
  with the same top 5 (25 of 46), but fewer at rank 1 (12 instead of 16). This is the one
  real use of the facts found so far: with them, a deeper list is affordable. 300 fact blocks take
  about 3.3 s per question, about what 50 full documents take today (3.2 s); 300 full
  documents take about 18 s.
- **Rough estimate, not measured:** reading 300 documents on facts brings about 22 of 46
  failures into the list, and about half of those reach the top 5. That is +10–12 semantic
  questions, **52.8% → about 61–62%** (the end-to-end test below measured at most 60.0%). The candidates are gte's top 50, and a fused list of
  300 holds different near-duplicates, so the end-to-end test has to confirm it.

Two caveats. Reranking on facts needs facts for every document, and that route is parked
(see Decisions, 9 November). And the 16 failures that never reach even 1,000 deep need
routing, not a better reranker.

**Where the misses land.** For the 21 questions whose right document isn't in the reranker's
top 5 on full text, the median rank is 17 (6 to 43). Eight sit just below, at ranks 6–9;
the other 13 sit at 12–43, among near-duplicates the reranker prefers. On facts, the misses
sink lower (only 3 at 6–10), which matches the lower recall@10.

**An LLM as reranker on the fact blocks? Not financially viable.** A fact block is short:
about 438 tokens per document, against about 1,500 for the full text (counted with the
reranker's tokenizer). For the ~53 candidates of a question that is about 23k tokens, against
80k. Gemini 3.8 Flash reading them and returning the top 10 would cost:

| Setup | Input per question | Per question | Per 125 semantic questions | Per 1,000 queries |
|---|---:|---:|---:|---:|
| LLM, facts, 53 documents | ~24k | ~$0.019 | ~$2.3 | ~$19 |
| LLM, full text, 53 documents | ~80k | ~$0.061 | ~$7.6 | ~$61 |
| LLM, facts, 300 documents | ~132k | ~$0.10 | ~$12.6 | ~$100 |
| gte-reranker on a dedicated L4 GPU (~$580 per month), facts, at full load | – | ~$0.00007 | – | ~$0.07 |

The GPU is about 250× cheaper per question, and it is already paid for: today's pipeline
reranks 50 full documents on it. The LLM would break even with a dedicated GPU only below
about 1,000 questions a day, it is slower (an estimated 2–5 s against 0.3–0.6 s) and its
ranking of near-duplicates is untested. It would also need facts for every document. (GPU
prices are approximate on-demand rates, and the L4 speed is the laptop's scaled by an
assumed 2×; neither was measured.)

### End-to-end test: search 300 deep, then rerank (9 November)

The estimate above assumed the candidates behave like a real deep list. This test drops the
assumption (`e2e_pilot.py`, free). For each failure: today's hybrid search with 300
documents from each ranking, fused with RRF and cut to 300; the full text of all 300 loaded
from Postgres; the reranker run over all of them, once on the full text and once on the fact
block. Only 2,774 of the 13,800 fused documents (20%) have facts; for the rest the "facts"
input is the first 2,000 characters of the text. "@50" reranks only the fused top 50, as
today, but on full text.

| Variant | In the list | Top 5 | Top 10 | Recall@5 | Recall@10 | Missed | User waits per question |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fusion alone, no reranker | 20 | 1 | 1 | 2% | 2% | 45 | 0.5 s |
| Reranker, full text @50 (today's depth) | 7 | 2 | 3 | 4% | 7% | 43 | ~3.8 s |
| Reranker, facts / 2k characters @50 | 7 | 1 | 3 | 2% | 7% | 43 | ~1.2 s |
| **Reranker, full text @300** | 20 | **9** | **10** | **20%** | **22%** | 36 | **~20.4 s** |
| Reranker, facts / 2k characters @300 | 20 | 5 | 9 | 11% | 20% | 37 | ~4.7 s |
| Reranker, full text + facts (RRF) @300 | 20 | 8 | 10 | 17% | 22% | 36 | ~25 s |

The wait is measured on the laptop (RTX 2000 Ada): search 0.16 s, loading 300 texts
0.35 s, then the reranker, 19.8 s on full text or 4.2 s on facts (means). The @50 rows are
the reranker time scaled to 50 documents.

- **The estimate was too optimistic.** Reading 300 documents brings 20 of the 46 into the
  list and 9 into the top 5 on full text, not the 10–12 expected. On facts it's only 5.
  Even if every new top-5 hit gave a correct answer, semantic would go from 52.8% to at most
  **60.0%** on full text (+9 of 125) and **56.8%** on facts (+5). That's no better than the
  snippet agent (60.8%), at 20 s per question.
- **A deeper list is harder to rank.** Among its ~53 candidates the reranker found 25 of 46
  in the top 5. Of the 11 that are in the 300 but not in the top 5, the right document sits
  at 6–139 (median 24), behind near-duplicates that the candidates didn't contain. On facts
  the misses spread to 6–249.
- **The facts variant is mostly not facts.** 80% of the fused documents had no fact block, so
  the reranker compared fact blocks with text openings, scores that aren't on the same
  scale. Facts for all 300 would need about 11k more fact blocks (counting repeats) for this
  test, and that route is stopped.
- **None of the 46 got worse** with depth: no question in today's top 5 dropped out at 300.
  Questions that already pass weren't tested.
- **26 of the 46 aren't in the fused 300 at all.** No reranker setting reaches them;
  that is the case for routing.

### The same speed without facts: rerank on the first N tokens (9 November)

The fact blocks made the reranker about 5× faster. Is that the extraction, or just the
shorter input? `trunc_pilot.py` (free) cuts every document to its first N tokens (the title
comes first) and reranks both settings again.

| On the ~53 candidates | Top 1 | Top 5 | Top 10 | Recall@5 | Recall@10 | ms per document |
|---|---:|---:|---:|---:|---:|---:|
| Full text | 16 | 25 | 33 | 54% | 72% | 58.5 |
| Fact block | 12 | 25 | 28 | 54% | 61% | 11.4 |
| First 256 tokens | 12 | 20 | 23 | 43% | 50% | 5.9 |
| First 512 tokens | 14 | 24 | 26 | 52% | 57% | 11.8 |
| **First 1,024 tokens** | **15** | **27** | **32** | **59%** | **70%** | **26.0** |

| End to end, 300 deep | Top 1 | Top 5 | Top 10 | Recall@5 | Recall@10 | ms per document | Rerank per question |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full text | 5 | 9 | 10 | 20% | 22% | 66.1 | ~19.8 s |
| First 256 tokens | 3 | 5 | 8 | 11% | 17% | 6.2 | ~1.9 s |
| First 512 tokens | 4 | 7 | 10 | 15% | 22% | 12.4 | ~3.7 s |
| **First 1,024 tokens** | 4 | **9** | **11** | **20%** | **24%** | **27.1** | **~8.1 s** |

- **The first 1,024 tokens rank as well as the full text, at about 2.3× the speed,** in both
  settings. Today's 50-document rerank would drop from about 3.1 s to about 1.3 s.
- **The first 512 tokens match the fact blocks** in speed and nearly in quality (top 5: 24
  against 25). The facts' speed came from being short, not from the extraction.
- **No model, no labels and nothing to keep up to date** as documents are added: a one-line
  cut before the reranker. This is the cheap speed-up to use when latency is needed.

## Why our own judge prompts, not Ragas

Our two checks measure what Ragas calls *faithfulness* (are the statements backed by the
context?) and *context recall* (are the reference statements found in the context?). We
considered running Ragas itself, since it is a known, citable reference. Its faithfulness
prompt is one instruction ("verdict 1 if the statement can be directly inferred from the
context, otherwise 0") with one toy example. It was validated against human annotators on
WikiEval (Wikipedia pages, GPT-3.5), a different domain and model from ours. Our prompts are
stricter about specifics: one wrong number or name fails the fact. They also separate
*contradicted* (says something else) from *unsupported* (not said), which matters for search,
and quote the passage behind each verdict, so a verdict can be checked. Neither prompt is
calibrated on our data; that would take a human-labelled sample of facts, which we haven't
done (see the open points).

## Decisions

**7 November**

- **Model:** Qwen3.5-2B, not 0.8B, vision encoder included on purpose (it may help with
  visual documents later). Details in brainstorm 002.
- **Training stack:** Hugging Face `transformers` (main), `peft`, plain `Trainer`, loss on the
  JSON answer only. Weights & Biases for monitoring. Vast.ai RTX 4090, rented by script and
  destroyed when the run ends.
- **Batch 1 × 16 accumulation steps,** because of the 248k vocabulary.

**8 November**

- **No 10,000-label set for now.** About $2 of the $5 DeepSeek credit is left, not enough
  for 7,000 more labels (about $9). We continue with the model trained on 3,000.
- **Judge the labels instead of growing them,** with our own prompts and Gemini 3.8 Flash as
  judge, not DeepSeek, which would grade itself. Not Ragas (see above).
- **Acceptance rule: the facts must not be made up.** If the labels pass the hallucination
  check, the summaries are good to use. They passed: 0.9% of facts have a wrong or added
  detail and none is made up from nothing, so the DeepSeek labels and the model trained on them go ahead.
- **Facts are for retrieval only.** They go into the index with the document's metadata; the
  answer comes from the full document. So 50% coverage of the gold facts is fine and is not
  a reason to change the prompt.

**9 November**

- **No index of one vector per fact.** It would add about 6.7M vectors (+15–50 GB, about
  +$175–500 per month on managed Postgres) and never beat BM25 over the same facts.
- **The LLM-summary route (facts for every document) is stopped for now.** The summaries didn't
  help enough, even in the small pilot. It also doesn't scale: the corpus keeps growing,
  every new document needs a summary, and the accuracy of the old ones can't be guaranteed
  over time. A structured taxonomy lowers that risk but doesn't remove it. The reranker on
  facts (above) is the one result that could reopen it.
- **Document growth is a separate problem** with its own research map. The work on
  semantic failures goes on assuming a fixed corpus.
- **BM25 over facts doesn't add to the reranker.** Getting the right document into the
  reranker's list is the problem. A wider list reaches half the failures at 300 documents.
  The rest need routing.
- **Bucketing idea (in the document-growth map):** a small model writes keywords per
  document, an LLM deduplicates them, and a keyword index used as a routing step points to
  a small group of documents for the reranker. TnT-LLM as published is too token-hungry
  for us: it needs an LLM summary of every sampled item and repeated taxonomy passes.
- **No LLM reranker on the fact blocks: not financially viable.** About $0.019 per question
  on top of a GPU that already runs the reranker, about 250× the GPU's cost per question,
  with no measured gain in quality. Reopen if a test on the 46 failures (about $0.90) shows
  an LLM ranks near-duplicates clearly better than gte-reranker, or if LLM prices fall about
  tenfold.
- **No deeper reranker list (end-to-end test).** Reranking 300 documents on full text gets 9
  of the 46 failures into the top 5 (at most 52.8% → 60.0% semantic) at about 20 s per
  question; on facts, 5 of 46 at about 4.7 s. Neither is worth it next to the snippet
  agent. The next step is routing for the 26 failures the search never reaches.
- **For reranker speed, cut documents to their first 1,024 tokens instead of extracting
  facts.** Same quality as the full text at about 2.3× the speed, with no model to run and
  nothing that goes stale. Facts are no longer needed for speed. Not switched on yet; use it
  when the latency is needed (agents, a deeper list).

## Open points

- **Routing for the 26 failures** whose right document isn't in the fused 300 (see the
  end-to-end test): keywords or buckets, in the document-growth map.
- **Does a deeper list hurt the questions that already pass?** The end-to-end test ran on the
  46 failures only.

- The judge itself isn't checked against human labels.
- The fine-tuned model has about three times DeepSeek's error rate (2.6% of facts), mostly
  wrong attribution; whether that is acceptable for retrieval is open.
- `train.py` keeps only five generated samples; `generate.py` now writes all of them.
- 25 s per document with plain `generate` is too slow for 512k documents; serving needs vLLM
  or batched generation.

## Sources

- [RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)](https://arxiv.org/abs/2309.15217)
- [Ragas source, `metrics/collections/faithfulness` (v0.4.3)](https://github.com/explodinggradients/ragas)
