Check whether mined fields improve search on the Slack semantic dev set, before training any
extractor. DeepSeek's extraction stands in for the SLM: it is the quality ceiling, so if its
fields don't help, a trained small model's won't either.

## Why
The plan is to mine fields once per document and let the agent narrow down by them
([[Understand how data could be bucketed by content]]). The agent doesn't know what values
the database holds, so matching works by words (names) and by embeddings (phrases). Training
the SLM is only worth it if this narrowing helps. Baseline to beat
([[Build the Slack semantic dev set]], best setup): top 1 25.5%, **top 5 34.0%**, top 10
39.5%; right thread among the candidates for 46.5% of the 200 questions.

## How
Before the test:
1. **Fix the extraction prompt** on the 13 trial threads (short values, clear boundaries
   between feature, component and product, a few strong relations); hand check. ≈ $0.01.
2. **Extract fields from the 200 questions** with the same schema, so a question can be
   compared with a document. ≈ $0.05.

**Step 1 done (17 Nov): prompt frozen at `fc8ae8878988`** (after the field review below; first
frozen at `268d2b46a58e`) (`exploration/slack_extraction/`
`schema.py` + `label.py`). The extended schema: 24 fields (name / phrase / sentence),
2 labels (`thread_kind`, `status`), 15 relations (`answers` dropped: DeepSeek forced it onto
unrelated decisions). Names are copied word for word; phrases and sentences may be
rewritten. Invalid relations (unknown ends, wrong field types, head = tail) are removed in
code before saving, so training never sees them. Four rounds on the 13 trial threads,
$0.04 in total:

| Problem | Round 1 | Round 3 (JSON mode) | Frozen (strict schema) |
|---|---:|---:|---:|
| Same value in two fields | 22 | 13 | 7 |
| Durations as date_time | 16 | 4 | 4 |
| Bots or roles as person or team | 10 | 3 | 2 |
| Invalid relations | kept | 17 removed | 4 removed |

The frozen version uses DeepSeek's strict tool call (beta endpoint): the server enforces the
JSON schema built from `schema.py` (all 24 fields present, labels from their lists, known
relation names), and pydantic models built from the same schema check it again. JSON mode
only guaranteed valid JSON, and the old answer model accepted any field name or label.

≈ 860 output tokens and ≈ $0.0007 a thread off-peak, below the 1,400 tokens assumed.
Left as is: borderline duplicates (a date also as identifier); harmless for search, since a
value matches in whichever field it sits.

**Field review after the smoke test (17 Nov, 30 mixed items, $0.035).** Added for the two
routes: `area` (multi-label, 19 fixed business areas: the bucket key for filtering before
search), `concept` (up to 6 generic terms such as "session affinity", "expense
reimbursement": what paraphrased questions share with documents) and `doc_date` (the
corpus stores no document date). Also: Redwood is never a client, `status` has "not stated"
for questions, Slack texts start with their channel. Smoke check: both questions got the
right area, concepts read like search terms, no Redwood clients. Inputs:
`build_inputs.py` → `data/inputs.jsonl` (25,633 items); labels append to
`data/labels.jsonl`.

The test, for each question:
```
1. Reuse today's search: ~96 candidates, each with a reranker score
   (exploration/query_expansion/data/slack_dev_baseline.json). No new search.
2. Extract fields from those candidates (DeepSeek).
3. Compare each candidate's fields with the question's → a field-match score:
     names (client, region, version, ...): same word after normalising?
     phrases (bug, fix, topic, decision, ...): similar meaning? (gte embedding)
4. Re-order the candidates by reranker score + field-match score.
5. Is the right thread in the top 5 now?
```
Count over the 200 questions: top 1 / 5 / 10 before and after.

Not now: dropping candidates with a clear mismatch (a hard filter). Unclear how to set
"clear mismatch" safely; revisit only if re-ordering shows the fields carry signal.

**Where this leads: a two-stage reranker.** If field matching works, it is cheap (word
compares and dot products on fields mined once), so it can score a much larger pool than the
cross-encoder can afford:
```
stage 1: field-match score over a large pool (e.g. gte + BM25 top 300-1,000) → keep top ~100
stage 2: today's deep-learning reranker on those ~100 → final order
```
Stage 1 replaces "gte top 50 + BM25 top 50" as the gate into the reranker, which is where
53.5% of the dev questions are lost today (see Limit below). This test only checks the
signal on the existing ~96 candidates; the two-stage setup is the follow-up if it shows a
gain.

- **Top 5 goes up:** the fields separate near-twins → train the SLM
  ([[Train the Slack field extractor SLM]]).
- **No change:** the fields add nothing the reranker doesn't already see → stop, save the
  training.

**Limit:** re-ordering can only help the questions whose right thread is already among the
candidates (105 of 200 when all three lists are pooled). The rest need the fields as a search
of their own (match question fields against a deeper pool, e.g. gte + BM25 top 300 per
question, ≈ $40–70); only worth testing if this one shows a gain. Semantic questions avoid
names on purpose, so expect phrase fields to matter more than names.

**Scope (17 Nov): label the candidates of all 200 questions, not only the 105.** The labels
feed two routes after this test, and both need the other 95:
1. **A second reranker stage** (above): more near-twin distractors to rank against.
2. **Agentic search and filtering before embedding and rerank** (the bucketing idea, for a
   corpus that keeps growing): the 95 questions whose right thread never reaches the
   reranker are exactly the cases it has to fix.
Route 2 also needs the right thread itself labelled, so the 94 source threads that are in no
candidate list are labelled too (dev data, never trained on). Labels are stored per document,
so each one serves every later step.

## Cost (DeepSeek, off-peak, ask first)
Measured 17 Nov: the 200 questions have 25,798 candidate slots (≈ 129 each, the three lists
pooled) but only 2,025 documents shared between questions, so **23,148 distinct documents**.
Half are not Slack and run longer (median tokens: Slack 898, Linear 1,352, Drive 1,883,
Gmail 1,905, Confluence 2,492, Fireflies 2,881); 31.7M input tokens in all. A Slack thread
cost $0.00083 on the trial.

| Part | Size | Cost |
|---|---|---:|
| Prompt fix (done) | 13 threads, four rounds | $0.04 |
| Candidate fields | 23,148 documents | ≈ $21–23 |
| Source threads in no list | 94 | ≈ $0.10 |
| Question fields | 200 dev + ~1,000 train questions | ≈ $0.10 |
| Random Slack threads (variety for training) | 1,000 | ≈ $0.80 |
| **Total** | | **≈ $22–24** (top-up $30 with margin) |

## Labels reused as SLM training data (17 Nov)
Every DeepSeek label from this test is also training data for
[[Train the Slack field extractor SLM]], so the test pays for most of the labelling.
1. **Freeze the prompt before the big run.** All candidate labels come from one prompt
   version (the prompt hash in the label key), so they form one consistent training set. The
   raw JSON output is the SLM's target as is.
2. **Split the 200 questions in half** (fixed seed):
   - **half A**: its candidates' labels go to SLM training;
   - **half B**: its candidates stay held out, for re-running this search test with the
     SLM's own fields later. Trained on them, the SLM would just repeat DeepSeek's labels
     and the check would flatter it.
   - A document in both halves' lists counts as held out.
   The field test itself still runs on all 200 questions with DeepSeek.
3. **Never train on:** the 200 source threads, the 200 questions, or the 200 extraction-test
   threads in `exploration/slack_extraction/` (the SLM's extraction score).
4. **Drop broken outputs** (cut off, invalid JSON, `not_in_text` values) from training; log
   how many.
5. **Variety:** half A's candidates cluster around 100 topics; about half are not Slack
   (Linear, Drive, Gmail, Confluence, ...), which is good for later sources. Add ~1,000 random Slack threads from the training pool (≈ $0.80).
6. **Question-side examples:** at query time the SLM also extracts fields from questions.
   Label ~1,000 **train-split** questions (≈ $0.05); never dev questions.

Expected: ≈ 10,000 usable training documents (half A) + 1,000 random threads, far more than
the 2,000 the SLM card planned, with no separate labelling run. All costs are in the table
above.

## Result (17 Nov)
Labelling: 25,632 of 25,633 items, $26.54 in all. `fc8ae8878988` stays the version; repairs added
after the run: strict mode does not enforce the label lists (off-list `area` / `thread_kind`
values are mapped or reset in code) and some answers had an extra closing brace.

Test: `exploration/slack_extraction/field_match.py` → `data/field_match_result.json`. Reranker
scores from `slack_dev_baseline.json` (three lists pooled, so the baseline here is top 5 39.0%,
not 34.0%). Four features, z-scored: name overlap, gte similarity of "topic; concepts", of
"bug; cause; fix; question", area overlap. Weights picked on half A, checked on half B.

| | Top 1 | Top 5 | Top 10 |
|---|---:|---:|---:|
| Half B, reranker only | 31.0 | 43.0 | 46.0 |
| Half B, + fields (names 0.05, topic 0.1, problem 0.1) | 30.0 | **45.0** | 48.0 |
| All 200, reranker only | 27.5 | 39.0 | 44.5 |
| All 200, + fields | 26.5 | 43.0 | 47.0 |

- Held out: +2 questions in top 5, +2 in top 10, −1 in top 1. Within noise for 100 questions.
- Any feature at weight ≥ 0.25 or alone hurts (top 5 23.5–33.5%): the fields are much weaker
  than the cross-encoder and only help as a light tie-breaker.
- Area (the bucket key) adds nothing to re-ordering (weight 0 in both grids).
- Read: **as a re-ordering signal the fields carry little**. Not tested yet: the fields as a
  gate over a deeper pool (stage 1 of the two-stage reranker) and as a filter for the 95
  questions whose right thread never reaches the reranker. Flagged labels (24%) still to check.

**Names by embedding (17 Nov):** adding gte similarity of the joined names gives no gain
(half B top 5 44.0%, top 1 27.0%). Fields stay a weak re-ordering signal.

**Field search over a deeper pool (17 Nov)** (`field_pool.py` → `data/field_pool_result.json`):
the tool an agent would call. Rank all 24,432 labelled documents by field profiles only
(subject + problem + names, gte), compared with today's dense search (fine-tuned gte, best
stored chunk) on the same pool. Right thread within the top N:

| | Reached (105): top 10 / 50 / 100 | Missed (95): top 10 / 50 / 100 / 300 |
|---|---|---|
| Raw gte | 35.2 / 58.1 / 82.9 | 1.1 / 1.1 / 36.8 / 70.5 |
| **Field search** | **54.3 / 83.8 / 90.5** | **23.2 / 47.4 / 64.2 / 83.2** |
| Raw gte + fields | 55.2 / 81.9 / 93.3 | 10.5 / 29.5 / 58.9 / 86.3 |

- **As a search, the fields are much stronger than as a re-ordering signal.** Field search
  finds 47% of the missed right threads in its top 50, where gte finds 1%.
- Caveats: the pool is 24k documents, not the full corpus; the "missed" group was chosen
  because gte+BM25 missed it (favours fields there), the "reached" group because they found
  it (favours gte), and fields still win clearly in both groups.
- Next: corpus-wide check (needs fields for the whole Slack corpus: SLM, or ≈ $0.001 a
  document with DeepSeek) and then the agent loop with field search as its tool.

**Agent field search (17 Nov)** (`agent_search.py` → `data/agent_search_result.json`, $0.07):
DeepSeek with a `search_fields(query, profile, areas)` tool over the same 24k pool, 4
searches per question (it always used all 4; area filter in 16% of questions), results fused
by reciprocal rank. Right thread within top 10 / 50 / 100:

| | Reached (105) | Missed (95) |
|---|---|---|
| Raw gte | 35.2 / 58.1 / 82.9 | 1.1 / 1.1 / 36.8 |
| **One-shot field search** | **54.3 / 83.8 / 90.5** | **23.2 / 47.4 / 64.2** |
| Agent, 4 searches | 42.9 / 71.4 / 81.9 | 20.0 / 38.9 / 52.6 |

The agent beats raw gte but loses to the one-shot field search. Likely reason: the one-shot
search compares the question's *extracted fields* with the documents' fields (same extractor,
same style on both sides), while the agent writes free-text queries in its own style.
Next try: seed the agent with the one-shot result and let it only add searches.

**Agent v2 and v3 (17 Nov).** v2: each result shows id, topic, areas, key facts (bug, fix,
client, component, person) and the first 300 characters of the text; the agent ends with
`finish(ids)` and its picks go first ($0.38). v3: same, plus a documented prompt: what each
of the 19 areas holds and how many documents it has, how the filter works (any area
matches), when to use each profile, and a worked example ($0.38). Max 4 searches (v2 found
the agent asking for several per turn; now capped).

| Top 10 / 50 / 100 | Reached (105) | Missed (95) |
|---|---|---|
| Raw gte | 35 / 58 / 83 | 1 / 1 / 37 |
| One-shot field search | 54 / 84 / 91 | 23 / 47 / 64 |
| Agent v1 (one line per result) | 43 / 71 / 82 | 20 / 39 / 53 |
| Agent v2 (+ snippets, picks) | 55 / 73 / 82 | 22 / 35 / 52 |
| **Agent v3 (+ documented areas and profiles)** | **70 / 86 / 92** | **34 / 53 / 62** |

- Documentation is what made the agent work: area filter used in 99% of questions (11% in
  v2), right thread among its picks in 65% (reached) and 30% (missed) of questions.
- v3 beats the one-shot field search at top 10 (+15 and +11 points) and top 50; equal at top
  100. Total DeepSeek for the agent tests: ≈ $0.85.
- Same caveat: 24k labelled pool, not the full corpus.

**Rerank of the agent's candidates (17 Nov)** (`agent_rerank.py` → `data/agent_rerank_result.json`):
new candidates shown to the reranker as title + chunk closest to the question; known ones keep
their baseline score. All 200 questions:

| To the reranker | Reaches it | Top 1 | Top 5 | Top 10 |
|---|---:|---:|---:|---:|
| Today's candidates (~128) | 52.5 | 27.0 | **39.0** | 44.5 |
| Agent v3 top 50 | 70.0 | 32.0 | 53.5 | 60.5 |
| **Agent v3 top 100** | 78.0 | 33.5 | **55.5** | 63.0 |
| Today's + agent top 100 (~187) | 82.0 | 34.0 | 51.0 | 60.0 |

**Agent v3 + clarifying questions (17 Nov)** (`clarify_agent.py` → `data/clarify_agent_result.json`,
notes in `data/memory_notes.json`, DeepSeek $0.36 + 66 Gemini replies): the 9 Nov
clarification setup (memory notes, 25% with one wrong detail, simulated person on Gemini)
with `ask_user` up to 2 times. The agent asked in 33% of questions. No gain: top 10 / 50 / 100
reached 66 / 83 / 90 (v3: 70 / 86 / 92), missed 28 / 45 / 60 (v3: 34 / 53 / 62). Questions it
didn't ask about dropped too (top 10 60 → 55), so about 5 points is run-to-run noise.
Why: the replies say what the person remembers (a chat thread in the release channel, the
team, roughly when), and the tool can't use any of it: the field index has no source type,
channel, team or date filter. Next: add those filters (channel and source are in the data;
`doc_date` and `team` are mined), then rerun.

**Agent v4: source and channel filters, with and without asking (17 Nov)**
(`agent_search_v4_result.json`, `clarify_agent_v4_result.json`, $0.81). `search_fields` gets
`sources` (chat, email, Linear/Jira ticket, PR, wiki, shared doc, meeting, CRM; counts in the
prompt) and `channels` (the 35 Slack channels, with counts); result cards show both. Date and
team filters left out: `doc_date` is mined for only 16% of documents, team is searchable as a
name. With asking, the agent asked in 43% of questions and used the filters from the replies
(sources in 79 questions, channels in 39).

| Top 10 / 50 / 100 | Reached (105) | Missed (95) |
|---|---|---|
| v3 | 70 / 86 / 92 | 34 / 53 / 62 |
| v4, no asking | 64 / 85 / 91 | 24 / 47 / 62 |
| **v4 + asking** | 68 / 83 / 87 | **35 / 56 / 64** |

- On the 85 questions where it asked: top 10 27 → 39, top 50 58 → 64 (vs v4 without asking).
  On the 115 where it didn't: 58 → 62 top 10, i.e. noise of a few points.
- Clean notes gain (top 10 46 → 55); notes with a wrong detail barely (42 → 44, top 100 72 → 66):
  a wrong channel or source filter hides the thread.
- Run-to-run noise is large: v3 and v4 without asking differ by 6-10 points at top 10 with the
  same information available. Conclusions need repeated runs or more questions.

**Ambiguity of the 200 dev questions (17 Nov)** (`ambiguity.py` → `data/ambiguity_result.json`,
1,498 Gemini ratings, method and cache of `exploration/question_ambiguity/answers.py`): each
document read alone with the question → full / partial / none. Rated: the right thread, and
the final top 5 of today's setup and of agent v3 top 100 after the reranker.

- **The right thread answers its own question**: full 189, partial 10, none 1. The questions
  are answerable from their thread without extra details.
- **But 76.5% of questions have another shown document that also answers fully** (91.5% fully
  or partly). Spot check of 4: all real near-twins, the same kind of problem in another
  thread with *different* facts (another fix, deadline, owner). E.g. "the partially completed
  EU deployment with latency spikes on warmup": the right thread pins the old model on three
  hosts; a twin thread raises prefill parallelism. Both fit the question's words.

| Final top 5 after the reranker | Right thread top 1 / 5 | A fully answering document top 1 / 5 |
|---|---|---|
| Today | 27.5 / 39.0 | 63.0 / 84.5 |
| Agent v3 top 100 | 33.5 / 55.5 | 67.0 / 88.0 |

Read: the dev set is mostly an ambiguity test. From the question text alone, a system usually
finds *a* thread that answers, but often the wrong one, and the answer then has wrong facts.
Only the right-thread score counts as correct; "answering document" is the ceiling a system
could reach without knowing which case the person means. Closing that gap needs information
from the person (asking), which is why the asking + filters result matters.
