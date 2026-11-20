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
