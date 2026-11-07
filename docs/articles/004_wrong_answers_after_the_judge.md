# Wrong answers after the judge: the problem is finding, not ranking

Status: draft notes, 4 November 2026. Follows [Why search misses](003_why_search_misses.md).
The live version is stage 2 of the notebook `notebooks/006_why_search_misses.py`.

Post 003 looked at 20 search misses one by one. This post does three things:

1. It checks whether two text fixes moved the numbers.
2. It adds a second judge, which asks whether a wrong answer came from a document that really
   answers the question, to separate ambiguous questions from real failures.
3. It measures the real failures over the whole corpus, then tests whether smaller gte
   chunks would rescue them.

## 1. Two fixes that changed nothing

On 3 November, 115,315 documents turned out to be stored as Python list literals
(`['...', '...']`), almost all of them Gmail threads. `normalize_content` now decodes them, so
their chunks were re-embedded. On 4 November the BM25 index was rebuilt from the same
normalized text.

| Semantic questions (125) | Correct | Recall@5 |
|---|---:|---:|
| Run 15, before both fixes | 54.4% | 55.2% |
| Run 17, after the re-embed | 52.8% | 55.2% |
| Search only, after the BM25 rebuild too | — | 55.2% |

- The re-embed changed the top 5 for 38 of the 125 questions, but it never added or dropped an
  expected document.
- 2 answers went from right to wrong with the same documents. That is the LLM varying between
  runs, not search.
- After the BM25 rebuild: Recall@1 44.0%, Recall@10 56.8%, Recall@20 59.2%. Recall barely
  grows past the top 5, so the misses are deep, not just below the cut.

The list literals looked broken, but they didn't cause the misses.

## 2. A second judge: does the cited document answer?

The main judge only says whether the final answer matches the gold answer. A wrong answer can
mean two different things:

- **search failed**: the right document was never found;
- **the question is ambiguous**: another document answers it too, the pipeline found that one,
  and its answer is reasonable but different from the gold one.

Only the first one is a search problem. To tell them apart, a second check runs on every wrong
answer whose expected document wasn't given to the LLM:

1. **Which documents was the answer taken from?** Since run 15, the answering LLM cites the
   numbered documents it used. For older runs Gemini names them afterwards
   (`citation_instructions`).
2. **Does each cited document answer the question on its own?** Gemini reads the question and
   one document, without the gold answer, and rates it `full`, `partial` or `none`, with the
   answer and a word-for-word quote (`document_answer_instructions` in
   `exploration/question_ambiguity/prompts.yaml`).

The rating rules are strict about details. A document about the same topic but a different
customer, date, product or version gets `none`, because the question rules it out. A close
version of the same event gets `partial`, because a reader could build a plausible answer
from it.

Ratings and citations are cached by a hash of the model, instructions and prompt, so rerunning
never pays twice. Rating run 15's cited documents cost about $0.15.

**Checking the judge.** Before trusting it, the same per-document check labelled a sample of
30 questions for ambiguity:

- 10 questions with the highest similarity for gte's #2 document, 10 with the lowest and 10
  in the middle;
- a question counts as ambiguous when a document in gte's top 2 that isn't the expected one
  answers it fully or partly;
- 11 of the questions were also labelled by hand on a side-by-side page (`compare.py`).

The rules agree with the hand labels on **9 of 11**. Both disagreements are questions the
rules call ambiguous through a weak match (`low_similarity`), where by hand they're not.

**Run 15's 57 wrong answers:**

| Outcome | Answers |
|---|---:|
| Expected document given to the LLM, answer still wrong | 1 |
| Cited document answers fully | 8 |
| Cited document answers partly | 2 |
| Cited document doesn't answer | 1 |
| No document cited ("could not find") | 45 |

So 10 wrong answers (8 + 2) are most likely ambiguous questions: the pipeline found a
document that answers, just not the expected one. The other **46 look like real search
failures**. The rest of this post is about them. (Section 6 rates every document the LLM
was given, not only the cited ones, and splits these 46 further: 31 search failures and 15
where a given document did answer.)

## 3. The 46 real failures, measured

For each one: the expected document's exact rank over all 511,962 documents in BM25 and in gte,
and the reranker's score for it next to its score for the pipeline's #1 pick
(`exploration/retrieval_failures/analyze.py`, on the indexes after both fixes).

| Expected document within | BM25 | gte | Better of the two |
|---|---:|---:|---:|
| top 5 | 2 | 0 | 2 |
| top 50 | 10 | 3 | 13 |
| top 500 | 24 | 13 | 29 |
| top 1,000 | 34 | 18 | 38 |
| **median rank** | 446 | 2,334 | 214 |

| Reranker | |
|---|---:|
| Prefers the expected document over the #1 pick | **17 of 46** |
| Median score, expected document | 0.891 |
| Median score, #1 pick | 0.903 |

- **gte is much weaker than BM25 here.** On half the failures the expected document ranks
  below about 2,300 in gte.
- **The reranker recognises the right document.** It scores the expected document almost
  exactly as high as the winner, and higher in 17 cases. It never gets the chance: the
  document doesn't reach the fused top 50 it reorders.
- Giving the reranker more candidates would help, but it raises the system's compute
  requirements, so I ruled it out.

## 4. Smaller gte chunks don't help

One explanation for gte's weakness is dilution. A gte chunk is 24,000 characters, so a long
document is one vector, and the answer passage is averaged away. If so, smaller chunks should
lift the expected document.

The test (`exploration/retrieval_failures/gte_small_chunks.py`):

- the pool for each failure is gte's own top 50, plus the expected document and the pipeline's
  picks (about 51–56 documents);
- every pool document is re-chunked at 4,000, 2,000 and 1,000 characters, embedded with
  gte-modernbert-base, and scored by its best chunk;
- the expected document starts outside gte's top 50, so it ranks last on the stored chunks.

| Expected document in its pool's | 24,000 (stored) | 4,000 | 2,000 | 1,000 |
|---|---:|---:|---:|---:|
| top 10 | 0 | 0 | 0 | 1 |
| median rank | 51 | 51 | 51 | 49 |

Even at 1,000 characters, and in a small pool where nothing else can rise, the expected
document barely moves. **It's not dilution.** gte finds 50 other documents that really are
as close to the question. An earlier test with MiniLM and 600-character chunks (post 003)
pointed the same way.

## 5. A cheaper look before reading: snippet search for the agent

Two things showed up in the agent's traces:

- the agent quotes phrases (`stepdown "pass rate" machine`) in 11 of 24 keyword searches, but
  BM25 strips the quotes and scores each word on its own;
- every keyword result came back as a full document, so each search was expensive and the
  agent had only 2 searches after the automatic one.

So a new system, `agent-snippets` (`evaluation/agent_snippets.py`), changes the tools:

- **keyword search shows snippets**: 10 results, each with its id and the lines that hold the
  query's words (as BM25 sees them: stemmed, no stopwords), with 2 lines of context. On a real
  query, 10 results took 9,860 characters against 57,090 as full text;
- **a new tool, `read_document(id)`**, returns one document in full;
- **budget**: 5 searches and 3 full reads per question, against 3 searches.

| Semantic questions (125) | Correct | Doc recall | Cost |
|---|---:|---:|---:|
| Run 17, pipeline | 52.8% | 55.2% | $1.05 |
| Run 12, agent | 58.4% | 58.4% | about $3.2 |
| **Run 18, snippet agent** | **60.8%** | **60.8%** | **$7.17** |

- Against the pipeline: 10 answers newly right, none newly wrong. Against the old agent: 6
  newly right, 3 newly wrong.
- The agent made 142 general searches, 100 keyword searches and 52 full reads. It read a
  document in 30 questions, but the expected one in only 4, and the snippets showed the
  expected document in only 5 questions.
- So snippets don't uncover the hidden documents. The gain most likely comes from the larger
  search budget.
- The cost more than doubled. The median question used about 17k input tokens, the worst
  329k: every turn resends the whole conversation, and some questions took 10 turns.

## 6. The judge on every document, not only the cited ones

Section 2 rated only documents an answer cited, so the 45 "could not find" answers had no
ratings at all. Here the same judge rates all 5 documents given to the LLM and the expected
document, for all 57 wrong answers: 324 new ratings, about $0.67
(`exploration/retrieval_failures/rate_failures.py`).

| The 46 "not found" cases: best rating among the 5 given documents | Cases |
|---|---:|
| None of them answers | **31** |
| One answers partly | 9 |
| One answers **fully** | **6** |

- **31 are clean search failures:** nothing the LLM got answers the question.
- **15 are not pure search failures.** A given document answered partly or fully, and the
  LLM still replied "could not find". The 6 "fully" cases point at the LLM not using what
  it had, or being too strict.
- The expected document answers fully in 43 of 46, partly in 2, and not at all in one
  (qst_0249), which may be a faulty question or gold answer.

The judge's reasoning makes the near-duplicates concrete. For qst_0185 (how to split a
year-long commitment for about eight accelerators across three regions) it rejects all
five given documents on one detail each: a fleet of 48 GPUs, of 24 + 12 + 4, of 18 nodes,
of hundreds of GPUs, or a 3-year term. All five are about the same topic.

## 7. Would the reranker pick the right document? Mostly yes

The reranker only sees the fused top 50. Scoring the expected document as if it were there
(`analyze.py`, title and first chunk):

| If the expected document were among the candidates | Ranked #1 | In the top 5 |
|---|---:|---:|
| Against the pipeline's 5 picks, all 46 problem cases | 17 | **29** |
| Against the pipeline's 5 picks, the 31 clean failures | 12 | **20** |
| Against all 50 candidates, all 46 problem cases (section 9 setup) | 12 | 25 |

qst_0185 shows how close it can be: the expected document scores 0.935, above all five
picks (0.914–0.922), and sits at BM25 #66 and gte #71, just outside both top-50 lists.
**The reranker is not the weak link; getting the document into its candidates is.**

## 8. What a reranked document costs in time

Measured on the laptop's RTX 2000 Ada (8 GB, half precision, batches of 4), on 10 problem
questions' real candidate lists:

| Documents reranked | Time per question (mean, range) | Per document |
|---:|---|---:|
| 10 | 0.71 s (0.26–1.24) | 71 ms |
| **50 (today)** | **3.18 s (1.18–4.68)** | 64 ms |
| 100 | 6.19 s (2.48–9.10) | 63 ms |

Time grows linearly with the number of documents, about 65 ms each, and is driven by input
length: the reranker reads the title plus the best gte chunk, median about 5,900 characters
and up to 24,000. Reranking 100 instead of 50 would cost about 3 more seconds per question,
and would still miss qst_0185 unless each search also returned more than its top 50.

## 9. Shorter reranker inputs: faster, but less precise

All 125 semantic questions, candidates = today's fused top 50 plus the expected document
when missing, reranked with three inputs (`rerank_window.py`):

| Reranker input | Per document | Per question | Average input |
|---|---:|---:|---:|
| Title + best chunk (today) | 65.9 ms | 3.32 s | 6,176 characters |
| Title + 2,000-character window with the most query words | **14.4 ms** | **0.73 s** | 2,044 characters |
| Title + first 2,000 characters | 14.4 ms | 0.73 s | 2,045 characters |

| Expected document in the top 5 | Today's chunk | Best window | First 2,000 |
|---|---:|---:|---:|
| Already in the top 50 (75 questions) | **68** | 59 | 60 |
| Added from outside (50 questions) | 30 | **32** | 26 |

- The window is 4.6× faster, but loses 10 of today's top-5 hits (gains 1): the answer is
  sometimes outside the densest 2,000 characters. That is about 7 points of recall.
- On the misses it is slightly better, since it cuts noise around the matching words.
- So a short input doesn't come free; it buys depth. At the same time budget the reranker
  could read about 230 windows instead of 50 chunks. Whether that depth makes up for the
  lost precision is still to be measured.

## 10. Embedding experiments: smaller chunks, another model, late chunking

All on the same pools as section 4 (gte's own top 50 plus the expected document, 51
documents), scored by each document's best chunk. Rank of the expected document, 46 problem
cases (`pools.py`, `jina_late_chunking.py`):

| | gte 24k (stored) | gte 1k | jina-v3 1k | jina-v3 late chunking 1k | jina-v3 whole document |
|---|---:|---:|---:|---:|---:|
| top 1 | 0 | 1 | **3** | 0 | 0 |
| top 5 | 0 | 1 | **4** | 0 | 0 |
| top 10 | 0 | 1 | **9** | 3 | 0 |
| top 20 | 1 | 4 | **18** | 10 | 8 |
| median rank | 51 | 49 | **26.5** | 32.5 | 33 |

- **Late chunking hurts here:** against plain 1k chunks with the same model it is better in
  15 cases and worse in 27. Giving each chunk its whole document's context blurs the one
  detail that tells it apart from its near-duplicates.
- **A different model with small chunks helps a little** (jina-embeddings-v3, 1k chunks),
  but the pools are gte's own favourites, the hardest distractors for gte, not for Jina, so
  part of the gain is the pool. It reaches the top 5 in only 4 of 46.
- Practical notes: Jina's model code needs transformers 4, so it ran in a separate
  environment. Without flash-attn, attention memory grows with the square of the length, so
  late chunking used 12,000-character windows instead of the full 8k tokens. Jina v3 is
  licensed for non-commercial use.

## 11. An agent that leans on keyword search, with caching

A third agent, `agent-keywords` (`evaluation/agent_keywords.py`): 2 general searches and 10
keyword searches (snippets with ids), 3 full reads, and instructions to paraphrase when a
keyword search finds nothing, to check every detail of the question against the document,
and to collect details from several documents.

To keep the cost down it caches the start of each conversation with Gemini's explicit
context caching (`evaluation/agent_cache.py`): the instructions, tools and first 5
documents, and again after each general search. Measured:

| | Result |
|---|---|
| Implicit (automatic) caching on our model | none in a bare 3-call test with the same 12,900-token prefix, but it does happen in real agent conversations (section 13) |
| Explicit cache on the same prefix | 12,916 of 12,926 tokens read from the cache |
| Latency, uncached vs cached call (3 each) | 2.06 s vs 2.24 s mean: no speed-up; creating a cache takes about 1.3 s |
| Live, qst_0185, first instructions | 5 calls, 62k tokens cached, about $0.05 (uncached about $0.065); answered from a wrong document after 3 of 10 keyword searches |
| Live, qst_0185, stricter instructions | 9 calls, 7 keyword searches, 155k tokens cached, about $0.13 (uncached about $0.19); replied "could not find" |

Caching saves money, not time. Gemini needs at least one new message after a cache and
can't switch tools off on a cached call, so the question goes after the cached documents,
and the last turn after the budget is spent goes uncached. The agent kept putting quotes in
its keyword queries despite the instructions; they are ignored by BM25 and are better
stripped in code. Cached input is priced as an assumption (10% of input), not yet checked
for this model.

## 12. Ideas costed, not yet tried

| Idea | LLM cost | Local load |
|---|---|---|
| Gemini writes 10 questions per document, index them | about $1,250 (about $600 in batch mode), thinking off | 5.1M questions to embed, about 8–16 GB |
| Gemini labels 10k documents with key facts, LoRA on a 1–2B model, local extraction | about $27 | 5–10 h training (fits in 8 GB; 3–4B needs QLoRA), then about 2 days of GPU |
| Same, but one vector of all facts per document | same | 512k vectors, about 1.5 GB |

Before any of these, a pilot on the 2,370 pool documents (about $6) would compare: all facts
as one vector, one vector per fact, groups of facts, and BM25 over fact text.

## 13. The pipeline on the whole benchmark

The pipeline (hybrid-rerank) on all 500 questions: basic (run 7) and semantic (run 17)
from before, and the 8 other types run now (runs 19–26, about $2.09).

| Question type | Questions | Correct | Complete | Doc recall |
|---|---:|---:|---:|---:|
| high_level | 10 | 40.0% | 43.2% | — |
| semantic | 125 | 52.8% | 51.7% | 55.2% |
| completeness | 20 | 55.0% | 47.8% | 40.6% |
| project_related | 40 | 65.0% | 53.7% | 54.5% |
| constrained | 30 | 80.0% | 91.2% | 90.0% |
| basic | 175 | 86.3% | — | 88.6% |
| conflicting_info | 20 | 90.0% | 86.3% | 87.5% |
| intra_document_reasoning | 40 | 92.5% | 91.7% | 100.0% |
| info_not_found | 20 | 100.0% | 100.0% | — |
| miscellaneous | 20 | 100.0% | 99.0% | 100.0% |
| **All** | **500** | **≈ 75.4%** | | **≈ 76.3%** |

(— : these questions have no expected documents. Totals are weighted by questions.)

- Where the right document is easy to find, the pipeline is strong: 90–100%.
- The four weak types are all about finding documents: completeness and project_related
  need several documents and the top 5 holds only some; semantic is the near-duplicate
  problem; high_level asks for an overview across many documents.
- These four are 185 of the 500 questions, so a better first stage, or an agent that keeps
  searching until it has every piece, matters well beyond the semantic questions.

**The snippet agent on completeness** (run 27, about $0.52), the type that needs several
documents:

| Completeness, 20 questions | Pipeline (run 23) | Snippet agent (run 27) |
|---|---:|---:|
| Correct | 55.0% | **70.0%** |
| Complete | 47.8% | **72.7%** |
| Document recall | 40.6% | **64.1%** |

4 answers newly right, 1 newly wrong; recall up 24 points, as the extra searches find the
second and third documents. One question (qst_0439) first failed in the judge with a
`503 UNAVAILABLE` and was rerun on its own (wrong, completeness 0.4, about $0.06).

The run also reported 628,252 cached input tokens with no cache written: Gemini's implicit
caching does work in real agent conversations, unlike the bare 3-call test in section 11.
So run 18's $7.17 was likely an overestimate (the meter then priced all input in full), and
the explicit cache saves less than estimated there. The cost above prices implicit hits at
the assumed cached rate.

## What I take from it

- **Of 57 wrong answers, 10 are ambiguous questions, 15 are the LLM not using a document
  that answered, and 31 are documents search can't find.**
- **Text fixes, chunk size, late chunking and a different embedding model won't fix them.**
  The corpus is full of same-topic, near-duplicate documents. Wording alone can't tell
  them apart: the right document and 50 others look equally relevant to every text-based
  scorer.
- **The reranker can tell them apart** (it would put the expected document in the top 5 in
  29 of 46 cases), but it only sees the top 50, and reranking more costs about 65 ms per
  document.
- **What differs is structure:** the project, the customer, the people, the channel, the
  time. The next step is an index of those, so the agent can narrow the search down before
  ranking by text. An LLM over all 512k documents costs too much, so it has to come mostly from
  the parsed fields, with an LLM only where a source needs it.

### Caveats

- The small-chunk test is optimistic (a small pool), so a clear "no" is meaningful, but a
  "yes" wouldn't have been.
- The per-question ranks are measured on today's indexes, while run 15's answers came from the
  indexes before both fixes. The fixes didn't change recall, so the picture holds.
- The judge itself was checked on only 11 hand labels.

- Run 18 is a single run; LLM answers vary by a couple of questions between runs (section 1).
- The embedding tests (sections 4 and 10) use pools picked by gte, which favours any other
  scorer; the reranker and window tests use today's indexes.
- Latency numbers are from one laptop GPU and a handful of calls.

Spend for this part: run 17 about $1.05, the judge ratings about $0.15 and $0.67, run 18
about $7.17, the keyword agent's live checks and caching probes about $0.25.
