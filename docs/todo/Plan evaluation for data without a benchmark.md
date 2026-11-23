Work out how to evaluate retrieval (and mined fields) on a dataset that comes with no
questions and no answers, so we have to build the evaluation set ourselves.

## Why
Everything we tune today leans on EnterpriseRAG-Bench: its 500 questions, its question types
and its prompts, which we reuse to write our own train, dev and `dev_slack` questions
(brainstorm 005). A real company's data has none of that. Without a tactic, we can't tell
whether a change helps there, and the synthetic sets may flatter us: the same kind of LLM
writes the questions and the documents ([[Approach human-style data problems]]).

## How
Questions to think through, before any build:
- **Where questions come from:** LLM-written from sampled documents (our current way), real
  user queries and search logs, help-desk tickets, or people writing a small set by hand.
- **Question types:** which mix (basic, semantic, multi-document, "not found") matches what
  people actually ask, and how to get loose-match questions without copying the document.
- **Right answers:** one source document per question is cheap; questions with several right
  documents or none need a judge or a person.
- **Twins and leakage:** keep evaluation documents and their nearest neighbours out of
  training, as in brainstorm 005.
- **Trust:** how many hand-checked questions are enough, and how to tell a bad question
  from a search failure (the drop rule skips the first).
- **Cost:** what a minimal set costs per source and per question type.

Related: [[Synthetic train and dev sets]], [[Build the Slack semantic dev set]],
`docs/brainstorm/005_synthetic_training_questions.md`.
