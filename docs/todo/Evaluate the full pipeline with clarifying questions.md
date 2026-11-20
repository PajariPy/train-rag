Run the whole pipeline (first search, agent, asking, answer, judge) on a full test set, with
questions that cover the cases we now handle: ambiguous near-twins, multi-turn chats,
out-of-scope questions.

## Why
Today's numbers are retrieval only, on the 200 Slack semantic dev questions and a 24k
labelled pool (agent top 100 + reranker: 55.5% top 5). The benchmark's semantic questions
reach about 67% top 5. There is no end-to-end answer accuracy with asking yet, and 76.5% of
dev questions also fit a twin thread.

## How
1. Extend the test set: mark ambiguous questions (from `ambiguity.py`), add multi-turn and
   out-of-scope questions; memory notes for every question.
2. Score: answer correct (judge), right document cited, asks per question, refusals.
3. Repeat runs (agent noise is 6–10 points) and report the spread.

Depends on [[Rewrite the query from the chat history]], [[Detect out-of-scope questions]],
[[Show the agent where its results disagree]]. Paid: ask first.
