Move every accepted experiment from `exploration/` into the pipeline in `src/train_rag/`, as
one system the final evaluation can run end to end.

## Why
The pieces were measured one by one, mostly in exploration scripts on the Slack dev set:
field search and the agent (top 5 39.0% → 55.5%), the disagreement check (ask F1 0.80),
and the answer with the main and similar cases, cited (right answer in 96.4% of the
ambiguity set). None of this is in the system the benchmark runs yet. Plan and prices:
[[solution_plan]] (docs/solution_plan.md).

## How
What goes in, in the order a question passes through it:
1. First search as today: fine-tuned gte on the question + BM25 with 4 rewrites appended,
   union reranked (first 1,024 tokens).
2. Mined fields from the SLM ([[Train the Slack field extractor SLM]]) stored per
   document, with the profile vectors (subject, problem, names) for field search.
3. The search agent on Gemini (`agent_search.py` + `agent_llm.py`): field search with
   area, source and channel filters, at most 4 searches, its top 100 to the reranker.
4. The disagreement check inside the agent, on the final top 5 in full, as a turn on the
   cached documents (baseline prompt in `agreement.py`); several cases → `ask_user` (at
   most 2), or the cases side by side.
5. The answer (`document_answer_instructions`): the main case plus at most 2 similar
   cases, one line each, only when their answer differs; `[n]` citations turned into
   `[doc_id]` by `link_citations`.

Open choices, settle before the run:
- Cap the mentions at 2 and rerun `answer_check.py` (~$0.80) to confirm no right answers
  are lost.
- The judge rule changed on 20 Nov (a matching mentioned case counts as right): keep it or
  not for the final run.
- Field search covers only the documents the SLM has mined: Slack first, so other sources
  rely on the first search until they are mined.

Checks: unit tests for the new parts; a 20-question smoke run before the full benchmark.
Then [[Evaluate the full pipeline with clarifying questions]].
