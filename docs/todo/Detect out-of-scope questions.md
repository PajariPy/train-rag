Recognise questions the corpus can't answer (off topic, or about data we don't hold) and
say so instead of answering from a wrong document.

## Why
Nothing in the pipeline checks this yet. The agent always returns its best documents, so an
out-of-scope question gets a confident answer from a look-alike. Related work: the
abstention problem (`docs/research/research_map.canvas`) and the "info not found"
part of [[Approach human-style data problems]].

## How
1. Collect out-of-scope questions: off-topic ones, and in-domain ones whose answer isn't in
   the corpus (e.g. remove the right document from the index for some dev questions).
2. Signals: reranker score of the top document, agent picks empty, an LLM check that the top
   documents answer (the ambiguity rating: full / partial / none).
3. Measure: how many out-of-scope questions are caught, how many answerable ones are wrongly
   refused.

## Design (23 Nov)
Settled in `docs/superpowers/specs/2026-11-23-evaluation-update-design.md` (parts 5 and 6)
and [[Prepare the evaluation for the final run]]:
- **Test set:** 50–100 off-topic questions, built in rounds: Gemini writes candidates →
  search wider than the pipeline (BM25 + gte + field search, top 50) → each document rated
  full / partial / none → accept only when all are none, else dismiss and add a line to the
  generator's instructions. Levels: clearly unrelated, other companies and public facts,
  general tech how-to. General-knowledge questions are refused.
- **Rerunnable**, because the document set will grow: each question stores its check and
  the corpus it was checked against; a recheck moves questions that became answerable out
  of the set; the generator's instructions are kept so a later run extends the set.
- The 20 look-alike `info_not_found` questions already exist; reported as their own line.
- **Detection:** a fourth verdict, "no document answers", in the agent's `judge_cases`
  check turn → the exact `{not_found}` reply. No pre-search gate for now.
- **Measure:** correct refusals (off-topic + `info_not_found`), wrong refusals on
  answerable questions, from the exact `{not_found}` reply.
