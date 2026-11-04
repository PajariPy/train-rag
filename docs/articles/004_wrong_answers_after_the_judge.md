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
document that answers, just not the expected one. The other **46 are real search failures**.
The rest of this post is about them.

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

## What I take from it

- **Ambiguity explains 10 of 57 wrong answers.** The other 46 are documents search can't
  find.
- **Text fixes, chunk size and a better reranker won't fix them.** The corpus is full of
  same-topic, near-duplicate documents. Wording alone can't tell them apart: the right
  document and 50 others look equally relevant to every text-based scorer.
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

Spend for this part: run 17 cost about $1.05, the judge ratings about $0.15, run 18 about
$7.17.
