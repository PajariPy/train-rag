Run the fine-tuned field extractor (`fields-r16-e2`) over every document of the corpus once,
store its raw output and fields, and embed the field profiles, so field search covers all
511,962 documents instead of the ~26k DeepSeek labelled.

## Why
Field search is what reaches the threads today's search never finds: with the agent and the
reranker, top 5 39.0% → 55.5% on the Slack dev set (article 007). Today it only sees the
DeepSeek-labelled documents, so the final evaluation would miss it on the rest of the
corpus.

The SLM is ready ([[Train the Slack field extractor SLM]]):
- **Held-out score** (12,711 documents, every source, never trained on): JSON valid 99.5%,
  mean field word F1 vs DeepSeek 0.785, 0.73–0.81 on every source; names near exact
  (person 0.98, client 0.90). Report: `exploration/slm_fields/runs/fields-r16-e2/score.md`.
- **Fine-tuning works for search** (half B's 100 dev questions, pool 12,511 held-out
  documents, same field search): right thread in the top 10 / 50 / 100: DeepSeek's fields
  38 / 68 / 80, the SLM's **38 / 66 / 80**. Report: `runs/fields-r16-e2/field_search.md`.
- DeepSeek on the whole corpus would cost ≈ $500 (and again after every schema change);
  the SLM ≈ $9–13.

## How
1. **Top up Vast.ai** (credit $3.99 on 22 Nov; the run needs ≈ $9–13 plus a margin, say $20).
2. **Inputs** (local, free): every document's text as DeepSeek saw it
   (`exploration/slack_extraction/build_inputs.document_text`), minus the 12,711 documents
   already mined by the scoring run (`runs/fields-r16-e2/outputs.jsonl`, reuse them). Write
   them in shards (~50k documents each) so a lost machine loses one shard, not the run.
3. **Mining** (GPU, paid): generalise `exploration/slm_fields/vast_score.sh` +
   `generate_vllm.py` (already resumable, chunked, with the early stop if outputs don't
   parse): merge the LoRA once, vLLM 0.33.0, greedy, prompt cut at 5,000 tokens, up to 3,000
   output tokens.
   - ≈ 615M output tokens at ≈ 6,000–8,500 tokens/s: **21–30 h on one RTX 4090**, or
     **3 machines in parallel ≈ 8–10 h**, same total ≈ $9–13 at ≈ $0.40/h.
   - Every ssh call with a timeout, jobs detached, copy back per shard, destroy on exit
     (`docs/vast_ai.md`; lesson of 21 Nov: never send a heredoc through `ssh -n`, and never
     edit a script while it runs).
   - Progress page: one workflow ("Mine the corpus") with a step per shard and a bar in
     documents.
4. **Import** (local, free): `uv run train-rag import-fields --file <labels> --extractor
   slm-fields-r16-e2` into `document_fields`. It reads DeepSeek's labels format, so the SLM
   outputs (`{doc_id, source, text}`) need converting first, or a reader for them. The raw
   output is kept, so a schema change re-parses, never re-mines. Field search prefers
   `slm-` rows with no code change.
5. **Embed** the subject, problem and names profiles with the fine-tuned gte:
   `uv run train-rag embed-fields` (≈ 1.5M strings; local GPU, hours). The field index then
   needs a faster load at 512k documents (open point from the pipeline update).
6. **Check:** documents with valid JSON per source (expect ≈ 99.5%); re-run
   `exploration/slm_fields/field_search_compare.py` on the full pool as a smoke check;
   a few outputs per source read by hand.

Paid: ask first. Then [[Prepare the evaluation for the final run]].
