Settle how the final run is judged and compared, then run `field-agent` and
`field-agent-clarify` end to end on the benchmark.

## Why
The pipeline is now one system in `src/` ([[Update the pipeline with the accepted
experiments]]), but two things are open before its numbers mean anything:

- **The judge rule changed on 20 Nov.** `judge_instructions` now counts an answer as right
  when its main answer *or one of its mentioned similar cases* matches the reference. The
  answers now mention similar cases on purpose (right answer in 96.4% of the ambiguity set,
  12% of them only as a mentioned case), so the strict rule would mark those wrong.
- **Comparability with run 31** (`hybrid-ft-rerank`, 77.6% correct, judged with the old,
  strict rule). A lenient run can't be compared with it as is.

## How
1. Decide the judge rule: lenient (today's prompt), strict (the rule before 20 Nov), or
   both. Both means two judge prompts in `prompts.yaml` and a `--judge` option per rule
   (`JUDGES` in `evaluation/registry.py`), same answers judged twice.
2. For comparison with run 31: rejudge run 31's stored answers with the chosen rule(s), or
   judge the new runs strictly as well.
3. Memory notes for every question the clarifying run asks about (`train-rag
   write-memories`).
4. Run `uv run train-rag evaluate --system field-agent` and `--system field-agent-clarify`
   on the 500 questions; report correct, right document cited, asks per question,
   side-by-side answers, cost. Repeat 2–3 times (agent noise is 6–10 points).
5. Then [[Evaluate the full pipeline with clarifying questions]] (extended test set).

Paid: about $0.02–0.03 per question per run plus the judge and the simulated person
(≈ $15–25 per 500-question run). Ask first.
