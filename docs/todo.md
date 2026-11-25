---

kanban-plugin: board

---

## Backlog

- [ ] [[Support the excalidraw.md format in the map scripts]]: teach the drawing scripts the plugin's new format, then convert #low
- [ ] [[off-topic-set-verified-rounds]]: careful, rerunnable off-topic set with search and ratings #evaluation #low
- [ ] [[fix-cut-long-documents]]: full fields for the 3,151 long documents the SLM read cut #evaluation #low


## Later

- [ ] [[Investigate the reranker cost and how to reduce it]]: reranking dominates the AWS bill; measure quality vs cost and cheaper setups #medium
- [ ] [[Review CLIP as an alternative to LLM video descriptions]]: embed video frames instead of describing videos with an LLM #medium
- [ ] [[Private and personal data questions]]: how to answer about salaries, HR records and other private data (article only) #article-only #medium
- [ ] [[Port the prod path to LangGraph]]: the prod agent as a LangGraph graph, when the prod code is written #medium
- [ ] [[Add observability and RAG quality tracking]]: traces per request and quality metrics from real chats #medium
- [ ] [[Article - problems by metric and how to fix them]]: a guide from metric symptoms to causes and fixes #low
- [ ] [[Article - future map]]: where the project could go beyond the todo #low
- [ ] [[Approach human-style data problems]]: how the RAG copes with the way people really write at work #medium


## Next

- [ ] [[Run the final evaluation]]: measure the built pipeline on 500 + off-topic questions, ≈ $32–36 #high
- [ ] [[Rewrite the query from the chat history]]: an LLM composes the search query from the chat; its own evaluation #high
- [ ] [[Add guardrails]]: what to refuse, what never to leak, unsafe input #medium


## In progress

- [ ] [[AWS infrastructure for the solution]]: where each part would run on AWS and its cost (article only) #article-only #medium


## Done

**Complete**
- [x] [[Mine the corpus with the SLM]] #high
- [x] [[Article - detect data drift for the SLM]] #low
- [x] [[Article - strategy for messier documents]] #low
- [x] [[Train the Slack field extractor SLM]] #high
- [x] [[Update the pipeline with the accepted experiments]] #done
- [x] [[Show the agent where its results disagree]] #done
- [x] [[Plan evaluation for data without a benchmark]] #medium
- [x] [[Test mined fields as search filters]] #high
- [x] [[Understand how data could be bucketed by content]] #high
- [x] [[Build the Slack semantic dev set]]
- [x] [[Try HyDE and query rewriting]]
- [x] [[Union rerank check]]
- [x] [[Benchmark run with the fine-tuned embedder]]
- [x] [[Fine-tune gte and re-embed the corpus]]
- [x] [[Synthetic train and dev sets]]
- [x] [[Review the code map]]




%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[false,false,false,false,false],"move-tags":true,"tag-colors":[{"tagKey":"#high","color":"rgba(250, 249, 251, 1)","backgroundColor":"rgba(167, 56, 56, 1)"},{"tagKey":"#medium","color":"rgba(255, 255, 255, 1)","backgroundColor":"rgba(159, 158, 52, 1)"},{"tagKey":"#low","color":"rgba(255, 255, 255, 1)","backgroundColor":"rgba(50, 61, 175, 1)"},{"tagKey":"#evaluation","color":"rgba(255, 255, 255, 1)","backgroundColor":"rgba(120, 72, 160, 1)"},{"tagKey":"#done","color":"rgba(255, 255, 255, 1)","backgroundColor":"rgba(76, 174, 27, 1)"}],"new-note-folder":"docs/todo"}
```
%%