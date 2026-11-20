An extended dev split, `dev_slack`, of semantic questions whose answer is a Slack thread, to
measure whether mined Slack fields help search.

## Why
The first dev split has only 11 semantic questions with a Slack answer (the benchmark has 13),
too few to measure anything. Slack is 55.8% of the corpus, and the field-mining plan starts
there ([[Understand how data could be bucketed by content]]).

## How
The same pipeline as the other splits (brainstorm 005):
1. `exploration/training_questions/slack_dev_sources.py`: 200 Slack threads, leaving out the
   benchmark answers and every train or dev source (each with its 20 nearest documents),
   every training negative, the extraction sample and the off-topic folders. Done: 66,195
   excluded, 267,884 left to draw from.
2. `generate.py`: the benchmark's semantic prompt on DeepSeek. Done: 200 questions, $0.038.
3. **No mining.** `mine.py` makes training data (hard negatives) and drops questions the
   reranker finds ambiguous; an evaluation set needs neither, and the drop would remove the
   near-twin cases we want to test. All 200 questions stay, like the benchmark's. The set is
   the `dev_slack` rows of `data/questions.jsonl` (question + source thread).
4. Baseline: today's best setup on these 200 questions (gte top 50 + BM25 top 50, union
   reranked), so the field test has a number to beat.
5. A section "Extended Slack semantic dev" in brainstorm 005 with the numbers.

## Result (16 Nov)
Baseline, `exploration/query_expansion/slack_dev_baseline.py` (Gemini rewrites $0.08; lists
and reranker scores in `data/slack_dev_baseline.json`):

| Search | Right thread in its top 50 |
|---|---:|
| gte on the question | 29.5% |
| BM25 on the question | 29.5% |
| BM25 on question + rewrites | 35.5% |

| gte + BM25 | In union | Top 1 | Top 5 | Top 10 | Docs |
|---|---:|---:|---:|---:|---:|
| question + question | 44.0% | 25.5% | **35.0%** | 39.5% | 97 |
| question + concat (best setup) | 46.5% | 25.5% | **34.0%** | 39.5% | 96 |

Much harder than the benchmark's semantic questions (67.2% top 5): the right thread never
reaches the reranker for 53.5% of the questions. The losers are near-twins (another thread on
the same kind of problem, a Confluence page or email on the topic), and the semantic prompt
strips names on purpose ("those broken image references", "the isolated customer
environment"), so the questions rarely state a client, id or version to match word for word.
