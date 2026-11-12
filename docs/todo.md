---

kanban-plugin: board

---

## Next

- [ ] Benchmark run with the fine-tuned embedder<br>train-rag evaluate with retriever hybrid-ft-rerank (weights in models/gte-modernbert-ft, vectors in the _768_ft tables), compared with the base run. Paid (Gemini judge): ask first.


## Later

- [ ] Fine-tune other base embedders and compare with gte<br>e.g. jina-embeddings-v3, Qwen3-Embedding-0.6B: same training/dev sets and dev check as gte (exploration/embedder_finetune). Only gte has been fine-tuned so far; jina and Qwen were only tested as they are. Qwen first (better than gte as-is, no custom code). 12 Nov: Qwen run crashed at 60% (disk full), dropped for credit; rerun needs ~$1 + a 60 GB disk and save_total_limit=1, ~$5 more to re-embed the corpus.
- [ ] Review the code map (docs/code-map, swyg on localhost:8770)
- [ ] Port to LangGraph / LangChain<br>Evaluation agents, pipeline answering and judging, exploration scripts (label.py, judge_labels.py). Design first. Re-run agent baselines 18, 28–30 afterwards (paid).
- [ ] Training-set minors from review<br>Neighbours of evaluation docs come from the first chunk only; score cache is keyed by doc_id and goes stale if questions are regenerated.


## Done

**Complete**
- [x] Fine-tuned gte (lr5e-6-e2) and re-embedded the corpus: dev search, semantic in top 10 49% → 61% (article 006, 12 Nov)
- [x] Synthetic train/dev sets: 2,067 / 422 questions, 5 hard negatives each (brainstorm 005)




%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[false,false,false]}
```
%%
