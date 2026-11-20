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

## Result (20 Nov)
Built on branch `feature-pipeline-agent` as two systems, `field-agent` (no asking) and
`field-agent-clarify` (asks the simulated person at most 2 times). Both run with
`uv run train-rag evaluate --system <name>`.

- **General search as a tool.** The fixed first search became the agent's
  `search_documents` tool: fine-tuned gte on the query, BM25 on the query with Gemini's 4
  rewrites appended (`query_rewrite.py`), and the union of both reranked on the first
  1,024 tokens (`RetrievalPipeline(union=True)`, `GteModernBertReranker(max_tokens=1024)`).
  The agent decides which tool to call first.
- **Mined fields** are stored in the `document_fields` table (raw output plus parsed fields,
  one row per document and extractor) and in `document_field_vectors` (the subject, problem
  and names profiles). `train-rag import-fields` loaded the DeepSeek labels (26,431 rows,
  12,872 of them Slack), and `train-rag embed-fields` embedded them locally. The SLM's output
  is imported with the same command as extractor `slm-<run>`; field search prefers `slm-`
  rows, with no code change. Checked against the exploration index: top-10 overlap 297/300.
- **The agent** (`evaluation/field_agent/`) uses Gemini Flash.
  - Tools: `search_documents` and `search_fields` (areas, sources and channels filters), at
    most 4 searches in total of which at most 2 general, plus `finish`.
  - Its prompt documents the tools like v3 did, with a worked example that uses both tools.
  - Its picks and every search's results are fused, reranked against the question, and the
    top 5 become its final documents.
- **The check** is a `judge_cases` turn on the same conversation, run on the final 5 in full
  (cut at 12,000 characters) with the baseline prompt.
  - When several cases fit and an ask is left, the agent asks and then finishes again (at
    most 2 checks).
  - Otherwise the answer shows the cases side by side (`side_by_side_note`).
- **The answer** keeps the rule as it was: the main case plus similar cases, with `[n]`
  citations turned into `[doc_id]` (`answer_from_hits`).
- **Free retrieval check** (`evaluate-search --retriever hybrid-ft-union`: union rerank
  on 1,024 tokens, no rewrites) on the 125 semantic benchmark questions: recall@5 62.4%,
  recall@10 68.8%. Compare: union on full text, 64.0% of questions with a right document in
  the top 5 (different measure, same order of size).
- **Checks:** 285 local tests pass (Gemini mocked), along with ruff and mypy.

- **Smoke run** (`field-agent-clarify`, 20 semantic benchmark questions with memory notes,
  runs 32–34, $1.51 Gemini). The tool calls, the check turn, asking and the cited answers
  all worked, with 0 errors.
  - First pass: 5/20 correct, right document in the final 5 for 5/20. The agent shortened
    the question for its first general search, and the search lost the right document
    (that query found it in the top 5 once, the plain question 6 times).
  - Fix: the prompt now says to pass the question as asked in the first `search_documents`
    call, and that the search has no quotes, OR or AND. The 15 failed questions were
    rerun: 5 more correct.
  - Combined: 10/20 correct, right document in the final 5 for 10/20. The plain union
    search alone gets 6/20 into the top 5 on these 20; they are mostly hard ones.
  - Tools: `search_documents` was always called first, then usually a second general
    search and two field searches. The agent asked on 7 of 35 attempts (8 asks).
  - Check verdicts: 19 same answer, 16 one case fits, 1 several cases fit.
  - Time: about 4 minutes per 15–17 questions with 6 in parallel.

Next: [[Prepare the evaluation for the final run]].
