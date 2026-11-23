# Mined fields, a search agent, and questions that fit two threads

Status: notes and decisions, 16–17 November 2026. Follows
[The first fact-extraction LoRA, and checking its labels](006_first_lora_and_checking_the_labels.md).
Code: `exploration/slack_extraction/` (`label.py`, `field_match.py`, `field_pool.py`,
`agent_search.py`, `agent_rerank.py`, `clarify_agent.py`, `ambiguity.py`). Notes:
`docs/todo/Test mined fields as search filters.md`. Research map: semantic search failures, data preprocessing problem, step 6.

## The problem

On our Slack semantic dev set (200 questions written from Slack threads, names stripped on
purpose), today's best search reaches **39.0% top 5**: gte top 50 + BM25 top 50 + BM25 on
the question with rewrites, pooled (~128 documents) and reranked. The reranker isn't the main
loss. For **95 of the 200 questions the right thread never reaches it**: gte and BM25 rank
look-alike threads higher (another thread on the same kind of problem, a Confluence page or
an email on the topic).

The plan from 16 November: mine short fields from every document once (topic, concepts,
bug, cause, fix, names, business areas) and let search narrow down by them. A small model
(SLM) would do the mining at scale, but first we checked that the fields help at all, with
DeepSeek as the extractor. DeepSeek is the quality ceiling: if its fields don't help, a
small model's won't either.

## Labelling (17 November)

One schema for documents and questions (`schema.py`): 26 fields (names copied word for
word, phrases and sentences may be rewritten), two labels (`thread_kind`, `status`), the
multi-label `area` (1–3 of 19 business areas: the bucket key) and 15 relations. DeepSeek
answers through a strict tool call, so the server enforces the JSON schema, and pydantic
models built from the same schema check it again.

Labelled: every candidate of the 200 questions, all 200 right threads, 1,000 random Slack
threads, the 200 dev questions and 1,000 train-split questions: **25,632 items, $26.54**.
The labels are also the SLM's training data: the 200 questions are split in half, half A's
candidates train, half B's stay held out for a later check of the SLM's own fields.

What went wrong: DeepSeek's strict mode does **not** enforce enum values. It wrote areas
from the wrong list ("HR or people") and thread kinds from the area list ("sales and
contracts"), and some answers closed the JSON with an extra brace. 568 items failed until
the code repaired these (near-miss areas mapped by their first word, off-list labels reset,
the first JSON object parsed).

## Fields as a re-ordering signal: no

The first test kept today's candidates and added a field-match score to the reranker's:
shared names, gte similarity of "topic; concepts", of "bug; cause; fix; question", and area
overlap. Weights were picked on half A and checked on half B.

| Half B (held out) | Top 1 | Top 5 | Top 10 |
|---|---:|---:|---:|
| Reranker only | 31.0 | 43.0 | 46.0 |
| Reranker + fields | 30.0 | 45.0 | 48.0 |

Two questions out of 100, noise. Matching names by embedding instead of word for word
didn't change it, and any weight of 0.25 or more made things worse. The cross-encoder reads
the question and the document together; a few mined words tell it nothing new.

## Fields as a search: yes

The second test used the fields to *find* candidates. Every labelled document (24,432) was
ranked for each question by its fields alone, compared with today's dense search (fine-tuned
gte, best stored chunk) on the same pool. Right thread in the first 10 / 50 / 100:

| | Today reaches it (105) | Today misses it (95) |
|---|---|---|
| gte on the raw text | 35 / 58 / 83 | 1 / 1 / 37 |
| Field search, one shot | 54 / 84 / 91 | 23 / 47 / 64 |

A question is short and paraphrased; a document is long and noisy. Comparing the question's
extracted fields with the document's puts both sides in the same short form, written by
the same extractor, and the comparison gets much cleaner.

## A search agent on the fields

Next, an agent drives that search. DeepSeek gets one tool, `search_fields(query, profile,
areas)`: the query is embedded with gte and compared with one profile of every document
(subject = topic + concepts, problem = bug + cause + fix + question, names, or all), limited
to some business areas if asked. It sees the top 10, searches up to 4 times, and its
searches are merged by reciprocal rank.

| Right thread in first 10 / 50 / 100 | Reached (105) | Missed (95) |
|---|---|---|
| One-shot field search | 54 / 84 / 91 | 23 / 47 / 64 |
| Agent v1: one line per result (topic, areas, concepts) | 43 / 71 / 82 | 20 / 39 / 53 |
| Agent v2: + key facts and a 300-character snippet; ends with `finish(ids)`, its picks first | 55 / 73 / 82 | 22 / 35 / 52 |
| **Agent v3: + documentation of the tool** | **70 / 86 / 92** | **34 / 53 / 62** |

v1 lost to the one-shot search: it wrote queries in its own words, which match the
extractor's phrases less well. v2 gave it something to judge with (snippets) and a way to
act on it (picks); that lifted the top 10, not the depth. v3 changed only the prompt: what
each of the 19 areas holds and how many documents it has, that a document passes the filter
if any of its areas match, when to use each profile with an example query, and one worked
session. Area filter use went from 11% to 99% of questions, and the agent became the best
method on both groups. **An agent is only as good as what it's told about its tools.**

The cost is small: about $0.38 for the 200 questions, 4 searches each.

## The end-to-end number: through the reranker

The agent's candidates go to the same reranker as today's. Documents already in today's
list keep their score; new ones are shown like the pipeline does (title + the chunk closest
to the question).

| To the reranker | Reaches it | Top 1 | **Top 5** | Top 10 |
|---|---:|---:|---:|---:|
| Today's candidates (~128) | 52.5 | 27.0 | **39.0** | 44.5 |
| Agent v3 top 50 | 70.0 | 32.0 | 53.5 | 60.5 |
| **Agent v3 top 100** | 78.0 | 33.5 | **55.5** | 63.0 |
| Today's + agent top 100 (~187) | 82.0 | 34.0 | 51.0 | 60.0 |

**+16.5 points top 5.** The reranker turns ~71% of the threads that reach it into a top-5
answer, as before; the whole gain is reach. Adding today's candidates back hurts: more
look-alikes, more chances to rank a wrong one first. The caveat for all of this: the agent
searched only the 24k labelled documents, not the whole corpus, so the number will drop
there.

## Letting the agent ask

The clarification setup from 9 November moved to this agent: for every question a memory
note of what the person remembers (kind of document, team, roughly when, what it was about;
no names or numbers; a quarter of the notes hold one slightly wrong detail), and Gemini
playing the person, answering only from the note. The agent may call `ask_user` twice.

The first try didn't help: the agent asked in a third of the questions, but replies like
*"a chat thread in our release channel with engineering and ops, about a week ago"* had
nowhere to go. The tool had no source or channel filter. v4 added `sources` (chat, email,
ticket, PR, wiki, doc, meeting, CRM) and `channels` (the 35 Slack channels), shown on every
result card.

| Right thread in first 10 / 50 / 100 | Reached (105) | Missed (95) |
|---|---|---|
| v4, no asking | 64 / 85 / 91 | 24 / 47 / 62 |
| v4 + asking | 68 / 83 / 87 | 35 / 56 / 64 |

On the 85 questions where it asked, top 10 went from 27% to 39%. Clean notes gain (46% →
55%); notes with a wrong detail barely do, since a wrong channel filter hides the thread.
One warning about all agent numbers: two runs with the same information (v3 and v4 without
asking) differ by 6–10 points at top 10. Firm conclusions need repeated runs.

## Most questions fit two threads

Why does the right thread lose to look-alikes so often? Gemini read each shown document
alone with the question and said whether it answers fully, partly or not at all (the
method of `exploration/question_ambiguity`), for the right thread and both setups' final top
5: 1,498 ratings.

- **The right thread answers its own question**: full for 189, partial for 10, none for 1.
- **For 76.5% of questions another shown document answers fully too**, with different facts.
  "What did the team decide about the partially completed EU deployment with latency spikes
  on warmup?": the right thread pins the old model on three hosts; a twin thread raises
  prefill parallelism. Both fit every word of the question.

| Final top 5 | Right thread top 1 / 5 | Some fully answering document top 1 / 5 |
|---|---|---|
| Today | 27.5 / 39.0 | 63.0 / 84.5 |
| Agent v3 top 100 | 33.5 / 55.5 | 67.0 / 88.0 |

Search already finds *a* thread that answers for most questions. The gap between the two
columns is the choice of *which case the person means*, and a twin's answer reads well but
has the wrong facts. The question text alone can't close that gap; only the person can.

## Decisions

**17 November**

- **Fields are a search, not a reranker signal.** Drop field-match re-ordering.
- **The reranker gets the agent's top 100 alone** (39.0% → 55.5% top 5 on the labelled pool).
- **Document the agent's tools** like an API: what every filter value holds, how it matches,
  when to use what, one worked example. That was the biggest single gain.
- **Train the SLM**: field search is the reason to have fields for the whole corpus.
- **Next: make asking target the twins.** Show the agent where its top results disagree
  (different fixes, owners, dates) so it asks about what separates them.

## Open points

- Corpus-wide check: the agent searched the 24k labelled documents; the full Slack corpus
  needs SLM labels first.
- Run-to-run noise of the agent (6–10 points): repeat runs before trusting small gains.
- The 10 questions whose right thread answers only partly, and the 1 it doesn't answer.
- Wrong memory: a wrong channel or source filter hides the thread; the agent could drop a
  filter when results look off.
- Date and team filters: `doc_date` is mined for only 16% of documents.

## Next: one extractor for every source (plan, 20 November)

The field search worked on DeepSeek's labels for 24k documents. To search the whole corpus
(511,962 documents) we train a small model, Qwen3.5-2B with LoRA, on those labels and let it
mine every document once. Plan, made before any paid run:

- **All sources in one pass.** Training covers Slack and the other eight sources together,
  so nothing is left for later. Training labels: 12,930 (6,289 Slack, 1,487 Linear, 1,282
  Drive, 1,114 Gmail, 1,001 questions, and fewer than 500 each for Fireflies, Confluence,
  Jira, GitHub and HubSpot), plus **~2,000 extra DeepSeek labels** for the thin sources:
  Gmail is 24% of the corpus but 9% of the training labels.
- **Why the split is near 50/50 and stays there.** The halves are halves of the 200 dev
  questions. Half B's 12,531 candidate documents are never trained on, so the field-search
  test can be rerun with the model's own fields without flattering it. A 70/30 split would
  leave 60 questions for that test, too few when one agent run moves 6–10 points from the
  next. More labels are cheaper than a re-split, and they also fill the thin sources.
- **Cross-validation would be the right check, but there's no budget for it.** Five folds
  are five trainings: about $15 and 30+ GPU hours. With 13–15k training examples, and every
  held-out document scored (about ±1–2 points of uncertainty), one run is enough to decide.
  If the score is borderline, a second run with one changed setting tells us more for $3.
- **Every run logs to Weights & Biases** (project `train-rag-fields`), and the training log
  is copied back with the adapter.
- **Score first, mine after.** The model is scored per source on the held-out documents
  against DeepSeek, then on the field-search test (half B). Only then is the corpus mined.

| Fine-tuning | Cost |
|---|---|
| 2,000 extra labels (DeepSeek, off-peak, $0.00104 a document) | ≈ $2.10 |
| Smoke run and machine setup (RTX 4090) | ≈ $1.00 |
| LoRA, 2 epochs, ~14.9k examples, ≈ 6.5 h at $0.42/h | ≈ $2.70 |
| Scoring all 12.5k held-out documents | ≈ $0.20 |
| Download fees, failed boots | ≈ $0.50–1.00 |
| **Total** | **≈ $6.50–7** (+ ≈ $3 if a second run is needed) |

Mining the whole corpus afterwards: about 615M output tokens, ≈ $9–13 on one 4090 (21–30
hours). DeepSeek on the same corpus would cost ≈ $500, and again after every schema change.

## The training run (20–21 November)

The plan held, with one lost attempt and a longer run than estimated.

- **Labels.** DeepSeek labelled 1,998 of the 2,000 extra documents for $1.84, so the model
  trained on 14,919 examples from every source.
- **The first GPU was lost.** The ssh connection dropped while the machine was being set up.
  The script read the flag for the fast attention kernels as empty and started on the slow
  path, 68 seconds a step instead of 16, a 50-hour run. It was stopped and destroyed ($0.46),
  and the scripts now put a time limit on every remote call and run long jobs detached.
- **The run.** Two epochs, 1,866 steps, 9.6 hours on one RTX 4090 at $0.40 an hour. The plan
  said 6.5 hours: the examples are longer than estimated (median 3,729 tokens). The training
  loss fell from 0.46 to 0.14. The loss on 400 held-out documents fell from 0.247 to 0.171
  and was flat for the last ~250 steps: the model isn't overfitting, and a third epoch
  would add little.
- **A first look.** At the end of training the model wrote the fields for 20 held-out
  documents. All 20 were valid JSON, and the fields matched DeepSeek's with a mean word F1
  of 0.78. Names are near perfect (person, client, region, model ≥ 0.99), and so are questions
  0.90, products 0.89, topic and fix 0.84. The weak fields are the ones that ask for
  judgement: decision 0.30, impact 0.40, action item 0.56. DeepSeek is least consistent on
  those itself, and field search leans on topic, problem and names, not on them.
- **Cost.** $1.84 for labels and $5.23 for GPUs, failed attempt included: **$7.07**, against
  the plan's $6.50–7.

Twenty documents only show the direction. The score that decides whether we mine the corpus
comes from all 12,711 held-out documents, per source.

## The held-out score (21 November)

The model wrote the fields for every held-out document, 12,711 from all nine sources plus
the questions, none of them seen in training. vLLM served the model merged into its
weights, and one RTX 4090 took 80 minutes. The outputs were compared with DeepSeek's in two
ways: word F1 (the words of a field's values, joined) and value F1 (each value as written).

| Source | Documents | Valid JSON | Word F1 | Value F1 |
|---|---:|---:|---:|---:|
| Slack | 6,584 | 99.7% | 0.787 | 0.555 |
| Google Drive | 1,706 | 98.9% | 0.767 | 0.574 |
| Linear | 1,618 | 99.6% | 0.780 | 0.608 |
| Gmail | 687 | 99.7% | 0.813 | 0.584 |
| Confluence | 633 | 98.6% | 0.758 | 0.568 |
| Jira | 495 | 99.8% | 0.785 | 0.595 |
| GitHub | 293 | 99.7% | 0.792 | 0.572 |
| Fireflies | 268 | 99.6% | 0.733 | 0.575 |
| HubSpot | 227 | 98.7% | 0.779 | 0.623 |
| **All** | **12,711** | **99.5%** | **0.785** | **0.571** |

- **No source is left behind.** The thin sources, which got the extra labels, score like
  Slack. Fireflies is lowest (long meeting transcripts, the most often cut).
- **Names are copied as DeepSeek copies them:** person 0.98, client 0.90, region 0.88,
  identifier 0.81 exact value F1. These are what field filters and the names profile use.
- **Phrases agree in words, not in wording.** A topic matches DeepSeek's in words (0.75) but
  rarely word for word (0.16); bug, cause and fix look the same. Field search embeds these
  phrases, so the same words in another order still land close.
- **Decision and impact stay weak** (0.55 and 0.59 word F1). They ask what counts as a
  decision, and DeepSeek isn't consistent on that either.
- Zero-shot GLiNER reached 0.41 exact value F1 against DeepSeek on 13 threads. The
  fine-tuned 2B model reaches 0.57 on 12,711 documents of every kind.

The whole fine-tuning cost **$7.84**: $1.84 for DeepSeek labels and $6.00 of GPU time, the
lost first attempt and the scoring included. The plan's go criteria hold: almost every
output parses, and names match. What remains before mining is the question that matters for
search: does field search find the right documents as well with these fields as with
DeepSeek's? Half B's candidates are among the held-out documents, so that test can run on
the outputs we already have.

## Fine-tuning works: field search with the SLM's fields (22 November)

Agreeing with DeepSeek is not the goal; finding the right thread is. So the "Fields as a
search" test ran again, twice on the same ground: half B's 100 dev questions, never trained
on, and a pool of the 12,511 held-out documents. Once every question and document had
DeepSeek's fields, once the SLM's. The same search (gte similarity of the subject, problem
and names profiles), no agent and no reranker, so the extractor is the only difference.

| Right thread in the top … (100 questions) | 10 | 50 | 100 | 300 | Median rank |
|---|---:|---:|---:|---:|---:|
| Field search, DeepSeek's fields | 38 | 68 | 80 | 96 | 19 |
| **Field search, SLM's fields** | **38** | **66** | **80** | 90 | 23 |

The fine-tuned 2B model gives the same search as DeepSeek down to the top 100, which is as
deep as candidates go to the reranker. One question is one point here, so the 2-point gap at
the top 50 is noise. It is a little weaker only deeper down (top 300: 90% against 96%). The
phrases it writes differ from DeepSeek's in wording, but they embed to the same place, which
is what field search uses.

**The small model can stand in for DeepSeek**, at about $10 for the whole corpus instead of
about $500.
