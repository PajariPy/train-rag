# Why search misses: 20 semantic questions up close

Status: analysis notes, November 2026. Follows
[First retrieval baselines](002_first_retrieval_baselines.md). The live version, with the
breakdown of misses by source, is the notebook `notebooks/006_why_search_misses.py`.

In post 002 the pipeline (BM25 and gte, fused, top 50 reranked, top 5 to Gemini) answered
54% of the semantic questions correctly. Nearly every wrong answer was a search failure: for
50 of the 125 semantic questions, no expected document was anywhere in the fused top 50. An
agent that searches again with its own words rescued 6–9 of them.

This post looks at 20 of those 50 misses, one by one, to see why search fails and what would
fix it.

## How I measured

For each question I took its expected document and measured, over the whole corpus of
511,962 documents:

- **BM25 rank and score**, scoring every document against the question (the top document's
  score is given for scale);
- **gte rank and cosine similarity** of the document's best chunk, against every chunk of
  every document (exact search, not the approximate index);
- **the reranker's score** for the question with the document, computed the same way the
  pipeline would (title and first chunk);
- **what the pipeline picked instead**: its top 5, with each one's rank in BM25 and gte and
  its reranker score;
- **whether the agent found it** in its extra searches (run 14).

The 20 are the first 20 by question id among the 50. The scripts are in
`exploration/retrieval_failures/`: `analyze.py` measures the ranks and scores, and
`case_texts.py` collects the questions, answers and source passages for the case notes. Their
output is in `exploration/retrieval_failures/data/` (not committed).

## The 20 at a glance

Ranks are out of 511,962 documents. "Rerank vs picks" compares the reranker's score for the
expected document with its best score among the five documents actually picked; **beats**
means the reranker would have put the expected document first, had it been a candidate.

| Question | Expected document | BM25 rank (score / top) | gte rank (similarity) | Rerank: expected vs best pick | Agent found | Main reason (hypothesis) |
|---|---|---:|---:|---|:---:|---|
| qst_0178 | GitHub PR: kernel stability thresholds and precision annealing | 703 (8.7 / 13.7) | 20,652 (0.594) | 0.792 vs 0.830 | no | paraphrase + look-alike PR |
| qst_0184 | Gmail: CytoHealth, schedule private hosting review | 4,599 (9.5 / 15.6) | 1,763 (0.691) | 0.879 vs 0.932 | no | near-duplicates, client unnamed |
| qst_0185 | Slack #eng-platform: regional GPU commit split | 66 (12.7 / 18.0) | 40 (0.742) | 0.935 vs 0.922 **beats** | no | just below the cut |
| qst_0186 | Drive: Fractured Context, session anchoring notes | 634 (13.4 / 23.0) | 179 (0.735) | 0.899 vs 0.907 | no | paraphrase + similar notes |
| qst_0187 | Linear: acceptance matrix for quality shocks | 668 (10.8 / 17.7) | 58,623 (0.632) | 0.927 vs 0.906 **beats** | no | detail buried in a long spec |
| qst_0190 | Confluence: Optimize 1.3 release notes | 617 (10.0 / 13.8) | 196,200 (0.592) | 0.904 vs 0.892 **beats** | no | one line in a long document |
| qst_0191 | Jira: Falkon index mismatch and audit-log gap | 57 (13.5 / 16.4) | 1,026 (0.689) | 0.903 vs 0.853 **beats** | yes | paraphrase, just below the cut |
| qst_0196 | Drive: microbench notelet, small quant runs | 937 (10.6 / 18.5) | 214 (0.735) | 0.903 vs 0.899 **beats** | no | paraphrase + similar notes |
| qst_0197 | Confluence: Ten-Step Restoration Sprint handbook | 37 (11.3 / 13.6) | 2,164 (0.663) | 0.894 vs 0.881 **beats** | yes | **cut by fusion** |
| qst_0201 | Slack #eng-oncall: APAC model-cache incident | 594 (15.2 / 20.8) | 292 (0.760) | 0.948 vs 0.912 **beats** | no | paraphrase + many similar incidents |
| qst_0204 | HubSpot: Pillarhaven Insightworks | 683 (12.6 / 19.0) | 23,749 (0.657) | 0.880 vs 0.925 | no | near-duplicates, customer unnamed |
| qst_0207 | GitHub PR: per-route function-call telemetry | 13,209 (5.6 / 12.3) | 41,930 (0.639) | 0.854 vs 0.872 | no | total vocabulary mismatch |
| qst_0212 | Drive: observability slice storyboard | 101 (15.4 / 19.0) | 1,725 (0.712) | 0.902 vs 0.897 **beats** | yes | just below the cut |
| qst_0213 | HubSpot: BrambleBox Content Labs | 104 (9.9 / 13.6) | 1,538 (0.676) | 0.883 vs 0.843 **beats** | no | paraphrase, just below the cut |
| qst_0214 | Linear: capacity triage and rollback plan | 17,509 (9.5 / 18.7) | 5,997 (0.688) | 0.893 vs 0.896 | no | near-duplicate incident alerts |
| qst_0218 | Linear: actionable incident cards | 4,049 (8.1 / 14.4) | 480 (0.705) | 0.822 vs 0.867 | no | UI names paraphrased + look-alikes |
| qst_0219 | Gmail: third-party IP indemnity redlines | 211 (13.3 / 24.6) | 166 (0.668) | 0.918 vs 0.914 **beats** | yes | paraphrase + similar negotiations |
| qst_0220 | Slack #eng-runtime: OOM with KV prefetch | 943 (14.6 / 22.4) | 2,387 (0.715) | 0.924 vs 0.919 **beats** | yes | paraphrase of config names |
| qst_0221 | Confluence: Employee Lifecycle Operating Manual | 3,523 (6.3 / 16.0) | 2,144 (0.722) | 0.889 vs 0.891 | yes | one line in a long manual |
| qst_0224 | Slack #docs: audit-log query examples | 5,161 (7.8 / 17.1) | 343 (0.659) | 0.909 vs 0.886 **beats** | no | paraphrase of field names |

Summary of the ranks:

| | BM25 | gte | Better of the two |
|---|---:|---:|---:|
| Median rank | 676 | 1,744 | 412 |
| Within the top 100 | 3 | 1 | 3 |
| Within the top 200 | 5 | 3 | 7 |
| Within the top 500 | 6 | 7 | 11 |
| Within the top 1,000 | 14 | 7 | 16 |

### Median scores

To help set thresholds later, here are the median scores of the expected documents next to
those of the documents the pipeline picked instead. "#1 pick" is the document the pipeline
put first; "all picks" are its five picks for all 20 questions (100 documents). BM25 scores
are not comparable across questions, so the table also gives each score as a share of that
question's top BM25 score.

| Score | Expected document | #1 pick | All picks |
|---|---:|---:|---:|
| BM25 score | 10.7 | 13.5 | 12.7 |
| BM25 score / question's top score | 0.62 | 0.85 | 0.80 |
| gte cosine similarity | 0.688 | 0.727 | 0.726 |
| Reranker score | **0.900** | 0.896 | 0.882 |

The median top BM25 score per question is 17.4. Ranges and quartiles:

- **BM25 / top score:** expected documents run from 0.40 to 0.83 of the top score.
- **gte similarity:** expected documents have quartiles 0.658 / 0.688 / 0.720, the picks
  0.672 / 0.726 / 0.759. Every expected document and pick lies between 0.57 and 0.80.
- **Reranker score:** expected documents have quartiles 0.881 / 0.900 / 0.915, the picks
  0.862 / 0.882 / 0.905. The fifth pick has a median of 0.874.

What this means for thresholds:

- **A cut-off on BM25 or gte scores would drop the expected documents first.** They score
  below the documents picked instead in both searches. The candidate lists should be cut
  by count, not by a score floor; if a floor is used at all, a relative one (a share of the
  top BM25 score) must sit below 0.4 to keep these documents.
- **gte similarities are squeezed into a narrow band.** Right and wrong documents both sit
  around 0.66–0.76, so an absolute similarity threshold can't separate them.
- **The reranker can't say "the answer isn't here".** Wrong #1 picks have a median score of
  0.896, the same as the missing right documents (0.900). A "not found" threshold on the
  reranker would need a clear gap between questions where the right document is present
  and ones where it isn't, and these 20 show none at the #1 position. Comparing with the
  questions the pipeline answered correctly is the next measurement.

## Case notes

For each question: the query, the expected (gold) answer, the answer the pipeline gave (run
8, long answers shortened with "…"), the pipeline's top 5 documents, the searches the agent
ran on top of its automatic first search (run 14), where the expected document is in the
dataset, the passage of it that holds the answer, and why I think search missed it. The
agent's first search is the question itself, so its results are the pipeline's top 5.

### qst_0178: GitHub PR, kernel stability thresholds and precision annealing

- **Query:** In the recent change that made low bit math safer for inference, what is the
  default pass rate used to decide when a machine is allowed to step down from the safest
  numeric mode?
- **Expected answer:** The default pass-rate threshold is 0.92 (92% of the microbenchmark
  runs must pass) before a node is allowed to step down from the safest precision.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. gmail: Eval sign-off criteria for FP8/INT8 defaults (hosted catalog) (gte #9; 0.830)
  2. slack: eng-runtime (gte #18; 0.815)
  3. github: route-sampler: staged route verification and numeric stepdown for safe alternate-kernel ro (BM25 #2; 0.791)
  4. confluence: Per-model quant prescriptions & runtime guardrails (Model Catalog) (BM25 #25; 0.776)
  5. slack: eng-ml (gte #12; 0.762)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"low bit math" "safest numeric mode" OR "pass rate" step down`:
    1. github: route-sampler: staged route verification and numeric stepdown for safe alternate-kernel rollout
    2. google_drive: Quantization slicebench probe — int4/awq/gptq comparison
    3. slack: eng-ml
  - keyword `stepdown "pass rate" machine`:
    1. github: route-sampler: staged route verification and numeric stepdown for safe alternate-kernel rollout
    2. linear: Surge affinity, soft-stepdown reclaim, and invoice-smoothing policy for temporary boosts
    3. google_drive: Stepdown Orchestra — warmup & canary journal (personal notes)
- **Source document** (6,406 characters; `dsid_22a7e9af2760474dbdf611c2c21cd231`, row 15,841 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Configuration and defaults: default kernel_stability_threshold=0.92 (fraction of
  > microbench runs that must pass), anneal_step_interval=30s, min_stable_window=2m.
- **Why it was missed:** The question replaces every technical term: "low bit math" for
  FP8/INT8, "machine" for node, "numeric mode" for precision, "pass rate" for stability
  threshold. Picked instead: an email on FP8/INT8 sign-off criteria and a different PR,
  "route-sampler: … numeric stepdown for safe rollout" (BM25 #2), which even shares the
  question's words. The reranker also preferred it. *Paraphrase plus a look-alike document.*

### qst_0184: Gmail, CytoHealth private hosting review

- **Query:** When is the 60 to 90 minute technical deep dive scheduled with the healthcare
  client about running model serving inside their own isolated network, and what is the time
  window in Pacific time?
- **Expected answer:** Tuesday, October 24, 2028, from 10:00 to 11:15 AM Pacific Time.
- **Given answer:** Based on the email thread with MedAlign Health… Tuesday, June 2, 2026,
  09:30–11:00 PT. *(Wrong customer.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. gmail: Next steps: technical deep-dive & demo — scheduling (BM25 #3, gte #15; 0.932)
  2. gmail: MedAlign: post-call recap — isolation & network acceptance path (BM25 #8; 0.918)
  3. gmail: Post-discovery: scheduling 60-90m technical deep dive (gte #1; 0.913)
  4. gmail: Post-discovery: schedule 60-90m ML/Infra deep dive — availability & attendees (BM25 #20, gte #46; 0.912)
  5. gmail: Next steps: lock ML/Infra deep-dive (60–90m) and agenda owners (BM25 #4; 0.901)
- **Agent's own searches** (run 14): none; it answered from the first search.
- **Source document** (5,827 characters; `dsid_74cd6ebda84547e2a534f8f6d1b49739`, row 88,739 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Proposed 3 slots (all PT):
  > - Tue Oct 24 — 10:00–11:15
  > …
  > Tue Oct 24 10:00 PT works for our team.
- **Why it was missed:** The corpus has many "schedule a 60–90 minute deep dive" emails to
  healthcare prospects; the expected one is to CytoHealth, about a VPC/on-prem deployment.
  The question never names the client. Picked instead: the same kind of email to Fidelis
  Health, MedAlign and Mediva Health, all scored higher by the reranker. *Near-duplicates;
  the question is underspecified.*

### qst_0185: Slack #eng-platform, regional GPU commit split

- **Query:** What was the recommended way to split a year long commitment for about eight
  high end inference accelerators across North America, Europe, and Southeast Asia while
  keeping a small extra pool for short spikes?
- **Expected answer:** Commit about 40% in us-east, 40% in eu-west and 20% in ap-southeast,
  plus a ~10% global burst pool on low-cost spot capacity for short spikes.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: eng-infra (gte #2; 0.922)
  2. slack: eng-infra (gte #5; 0.917)
  3. slack: eng-infra (gte #7; 0.917)
  4. gmail: Orion Labs: regional SKU split and latency tradeoffs for dedicated commitment (gte #3; 0.915)
  5. slack: eng-infra (BM25 #2, gte #10; 0.914)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"eight" inference accelerators North America Europe Southeast Asia commitment spikes`:
    1. slack: eng-infra
    2. slack: eng-infra
    3. slack: eng-platform
  - search `commitment "Southeast Asia" inference spikes North America Europe`:
    1. gmail: Region pinning + transient replication for analytics/metrics
    2. confluence: Multi-Region GPU Commitment & Egress Guide (US/EU/AP) — Inference Fleet
    3. slack: eng-infra
- **Source document** (4,885 characters; `dsid_be67d91a95e24fe690cba692cb1762c6`, row 226,397 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Karthik: Quick proposal from the numbers + constraints: commit 40% capacity in us-east
  > (cheap + availability), 40% in eu-west (SLO + discounted commit), 20% in ap-southeast
  > (smaller commit, accept mixed SKUs). Keep a 10% global burst pool on low-cost spot-ish
  > capacity for short-lived surges.
- **Why it was missed:** The thread says "8x A100-equivalent", "us-east / eu-west /
  ap-southeast". It ranked 66th in BM25 and 40th in gte: close, but outside the fused top 50,
  where four other region-cost threads sat. The reranker scores it above all of them. *Just
  below the cut.*

### qst_0186: Google Drive, Fractured Context session anchoring notes

- **Query:** In the notes about keeping long, stop-and-go chat sessions cheap without
  replaying the whole history, what storage setup and time-to-live were proposed for keeping
  the compact per-session state for recent sessions versus longer retention?
- **Expected answer:** Hot, recent per-session anchors in a per-customer LRU store in Redis;
  longer-term anchors in S3 with a 30-day TTL.
- **Given answer:** Lists three other designs: "Cobalt Sprint Mini-PRD" (local LRU, Redis,
  S3, 24 hours), "Stowaway Sessions" (Redis with a 24-hour TTL, S3, 7 days) and
  "Conversation State Snapshotting" (local SSD, S3 WAL). *(Look-alike notes, wrong TTL.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. google_drive: Microstate gardening — rough PRD and experiment notes (BM25 #2, gte #22; 0.907)
  2. google_drive: Conversation Memory Compaction - Arch Review Notes (gte #13; 0.886)
  3. google_drive: Conversation state snapshotting - arch jam notes (BM25 #3, gte #4; 0.880)
  4. google_drive: Cobalt sprint: compact-session-orchestrator mini-PRD fragments (BM25 #6; 0.878)
  5. google_drive: Stowaway Sessions — UX & proto notes (BM25 #12, gte #10; 0.874)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `stop and go chat sessions storage setup time to live compact per-session state recent sessions longer retention`:
    1. google_drive: Stowaway Sessions — UX & proto notes
    2. google_drive: Microstate gardening — rough PRD and experiment notes
    3. slack: eng-runtime
  - search `"stop-and-go" chat sessions storage TTL`:
    1. google_drive: Stowaway Sessions — UX & proto notes
    2. google_drive: Conversation Slices — rough PRD & idea pad
    3. slack: incidents
  - search `"stop-and-go"`: refused, search limit reached
- **Source document** (6,088 characters; `dsid_0ae9b752fef446ec86e376e2dea49c28`, row 168,604 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Storage: per-customer LRU store (Redis) for hot sessions; long-term anchors in S3 with
  > TTL 30d.
- **Why it was missed:** The notes call it "fractured context" and "session anchors"; none of
  those words are in the question. Picked instead: the three sets of notes above, on session
  compaction and snapshotting. *Paraphrase plus many similar notes.*

### qst_0187: Linear, acceptance matrix for quality shocks

- **Query:** When a production model experiment shows a sustained increase in harmful output
  warnings for two evaluation windows, what automated actions should the platform take
  regarding traffic shifting and incident/oncall notifications?
- **Expected answer:** Shift traffic back to the last-known-good model for up to 75% of
  traffic, open an incident with a pre-filled template, and notify on-call via PagerDuty and
  the Console alerting channel.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: incidents (gte #1; 0.906)
  2. slack: incidents (gte #14; 0.905)
  3. slack: incidents (gte #24; 0.905)
  4. slack: incidents (gte #8; 0.905)
  5. google_drive: Game Day Plan: Auto-rollback on SLO breach (rollout controller integration) (BM25 #9; 0.902)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `harmful output warnings evaluation windows traffic shifting incident oncall`:
    1. slack: incidents
    2. slack: incidents
    3. slack: incidents
  - search `"harmful output warnings" "two evaluation windows"`:
    1. google_drive: Safety annotation IAA casebook and adjudication notes
    2. google_drive: escape-hatch-exercises-corpus-v1
    3. gmail: Clarify Eval acceptance & A/B escape criteria for NovaCorp promptset
  - keyword `"harmful output"`: refused, search limit reached
- **Source document** (6,190 characters; `dsid_a27715e7309c494d812c75cf03bcf916`, row 199,795 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > AC7: Auto-remediation flows: If a critical regression (safety or composite drop >
  > threshold) is detected and sustained for two consecutive evaluation windows, the platform
  > must: (a) automatically shift traffic back to the last-known-good model for up to 75% of
  > traffic, (b) open an incident in the tracking system with a pre-filled template, and (c)
  > notify on-call via PagerDuty and the Console alerting channel.
- **Why it was missed:** The ticket is a long acceptance matrix for "quality shocks and
  auto-remediation"; the rule asked about is one entry (AC7) in it. gte ranks it 58,623rd: its
  vector describes the whole spec, not this rule. "Harmful output warnings" is also a
  paraphrase of "safety regression". Picked instead: Slack incident-bot alerts about safety
  regressions, which look more like the question. *A detail buried in a long document.*

### qst_0190: Confluence, Optimize 1.3 release notes

- **Query:** In the February 2026 update to our inference tuning product, what change caused
  older route configuration payloads to start getting rejected with a 400 error, and what
  field needs to be removed to fix it?
- **Expected answer:** Optimize 1.3 removed the deprecated field `optimize.legacy_hint` and
  rejects API calls that include it; remove it from saved route config payloads.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: support (BM25 #4; 0.892)
  2. slack: incidents (gte #21; 0.890)
  3. slack: support (BM25 #10; 0.873)
  4. slack: eng-platform (gte #5; 0.861)
  5. slack: eng-oncall (BM25 #12; 0.859)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `February 2026 inference tuning route configuration 400 error`:
    1. slack: eng-sre
    2. gmail: Service degradation observed — SLA claim and postmortem request
    3. gmail: Fwd: Frostline live stream throttle & intermittent chunk timeouts
  - keyword `"inference tuning"`:
    1. gmail: Request: Locked-Down Inference — No Tuning, Response IP, and Cross-Border Transit
    2. gmail: Nonlinear tier path and feature carveout discussion
    3. gmail: Penumbra dedicated pool — context headroom & quantization playbook
  - search `route configuration rejected 400 error field remove February 2026`: refused, search limit reached
  - keyword `"route" "400"`: refused, search limit reached
- **Source document** (8,371 characters; `dsid_b02e6ce0d9814b35af846c309c25e551`, row 1,761 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Deprecated field removed: optimize.legacy_hint (previously used for heuristic-only
  > recommendations). Action: remove any usages in saved route configs. Failure to remove
  > will return 400 in API calls that include the field.
- **Why it was missed:** The answer is one line in long release notes that are mostly about
  Performance Profiles and cost telemetry, so gte ranks them 196,200th. The question also
  says "inference tuning product" instead of "Optimize". Picked instead: Slack threads about
  400 errors and schema validation. *One line in a long document, plus a product name
  paraphrased away.*

### qst_0191: Jira, Falkon index mismatch and audit-log gap

- **Query:** During a short surge in vector writes last March, a customer in the western US
  region saw search answers point to the wrong source records and also couldn't verify
  charges because telemetry for about twenty minutes never arrived; what was the suspected
  technical cause behind both symptoms?
- **Expected answer:** Indexer pods in us-west-2 were OOM-killed during the ingest burst
  because of aggressive in-memory batching (made worse by a recent batch-size increase); on
  restart, unflushed batches never reached the commit log and the audit-log Kafka topic,
  causing wrong vector-to-doc-id mappings and a gap in billing telemetry.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. google_drive: Vector/rerank oddity - quick debugging breadcrumbs (BM25 #12, gte #19; 0.853)
  2. google_drive: Trace stallcheck debug diary (gte #10; 0.845)
  3. gmail: Unexpected invoice uplift — usage telemetry and offset proposal (gte #17; 0.834)
  4. gmail: Unexpected staging traffic causing March overage / request for credit (gte #22; 0.830)
  5. gmail: Observed heartbeat anomalies and elevated response latency — customer advisory (gte #16; 0.809)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `vector writes western US search answers wrong source telemetry twenty minutes`:
    1. gmail: Vectors API latency spike (p95/p99) after catalog patch - Orion Platform
    2. gmail: PilotRetail — vector alignment & auth next steps
    3. jira: Falkon Systems: embedding index reconciliation mismatch & audit-log delivery gap during high ingest  ✅
- **Source document** (7,083 characters; `dsid_ef7f1d5cba204b728e1f4d7cddda7daa`, row 189,041 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Root cause hypothesis: Combination of aggressive in-memory batching in the
  > dense-index-service and a recent change that increased batch size for latency
  > optimization caused larger transient memory usage. When a pod was OOM-killed, in-memory
  > batch in producer was lost before flush to audit log, creating both search inconsistency
  > and missing billing events.
- **Why it was missed:** The ticket says "high ingest", "incorrect document ids", "audit-log
  delivery gap", "us-west-2"; the question rewords all of them. BM25 ranks it 57th. The agent
  found it by rewording. *Paraphrase; just below the cut.*

### qst_0196: Google Drive, microbench notelet

- **Query:** In the quick A100 microbenchmark notes comparing 4-bit weight compression to half
  precision on a 7B model at single-item batching, what speedup and quality hit were observed
  for the simplest run?
- **Expected answer:** NF4 post-training quantization gave about 1.8× token throughput versus
  fp16, with roughly +0.6 perplexity on a small set of story prompts.
- **Given answer:** From other A100 notes (`open-7b-a1`): token latency ~48 ms in fp16 vs
  ~29 ms in 4-bit; "retained ~97% of float16 ROUGE-L for short answers". *(Look-alike notes.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. google_drive: KV + prompt explorations: microbench and eval notes (BM25 #13; 0.899)
  2. google_drive: Runlet sprint pad — latency/cost microbench (gte #25; 0.892)
  3. google_drive: mini ops experiment capture — afternoon runs (gte #3; 0.889)
  4. google_drive: Harper's quant & eval day brief (BM25 #5; 0.887)
  5. slack: eng-ml (gte #23; 0.882)
- **Agent's own searches** (run 14): none; it answered from the first search.
- **Source document** (4,049 characters; `dsid_3413f1905a0740d88b66c196b54f4df5`, row 166,123 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > NF4 PTQ on 7B gives ~1.8x token throughput improvement vs fp16 at batch=1 on A100, loss
  > ~+0.6 ppl for story prompts (small sample).
- **Why it was missed:** The notes say "NF4 PTQ", "fp16", "batch=1", "ppl"; the question says
  "4-bit weight compression", "half precision", "single-item batching", "quality hit".
  Picked instead: four other sets of microbenchmark notes, one of which answered a similar
  question. *Paraphrase plus similar notes.*

### qst_0197: Confluence, Ten-Step Restoration Sprint handbook

- **Query:** During a major production outage, what is the prescribed sequence of actions for
  the first 60 to 90 minutes, from opening the live coordination call through confirming the
  service is healthy again?
- **Expected answer:** The Ten-Step Restoration Sprint: (1) acknowledge and open the incident
  bridge and channel (0–5m); (2) quick impact triage (5–10m); (3) stabilize traffic with rate
  limits, routing or lighter model variants (10–20m); (4) hotfix window with small reversible
  mitigations (20–40m); (5) validate restoration with synthetic probes and p95/p99 checks
  (30–50m).
- **Given answer:** From the "Contingency Coordinator Handbook": triage and confirmation
  (0–10 min), fast containment (10–30), remediation loop (30–90), verification and close
  (90–120). *(A different runbook.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. confluence: Rapid escalation protocols and runbook bootstrap (BM25 #1; 0.881)
  2. confluence: Severity matrix and escalation triggers (Support P0–P4 ↔ Incident Sev0–Sev3) (BM25 #5; 0.875)
  3. confluence: Contingency Coordinator Handbook and Rotation Patterns (BM25 #7; 0.874)
  4. confluence: Containment and Recovery Runbook — Customer Impact Incidents (BM25 #3; 0.870)
  5. confluence: Ten-Minute Stabilization Checklist and Customer Briefs (gte #13; 0.869)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"live coordination call" OR "live coordination" major production outage sequence`:
    1. slack: 2165432100
    2. gmail: Persistent 5xx/429 surge — edge translation bottleneck affecting Lumena Pay production
    3. slack: eng-oncall
  - search `"First-hour runbook" OR "first 60" OR "first 90" "coordination call" OR "bridge"`:
    1. confluence: Contingency Coordinator Handbook and Rotation Patterns
    2. confluence: First-Hour Stabilization SOP and Executive Incident Brief
    3. confluence: Rapid escalation routing playbook and multi-audience status canvas
    4. confluence: Ten-Step Restoration Sprint & Shadow Mentoring Handbook ✅
- **Source document** (9,470 characters; `dsid_25b0de9bfa854d3c9acfa961555c9d45`, row 2,936 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > 2) Quick Impact Triage (5–10m)
  > - Primary Resolver: confirm service(s) affected using dashboards…
  > 3) Stabilize Traffic (10–20m)
  > - If applicable, apply graduated traffic control: rate-limits, scaled back routing
  > policies, or traffic-division to healthy regions.
- **Why it was missed:** BM25 ranks the handbook **37th**, inside its own top 50, yet it
  didn't reach the reranker. Fusion merged BM25's top 50 and gte's top 50 (up to 100
  documents) and kept only the top 50 by fusion score. A document found by one search alone,
  at rank 37, scores 1/(60+37) = 0.0103, below every document both searches found, so it was
  cut. *Cut by the fusion step, not missed by search.*

### qst_0201: Slack #eng-oncall, APAC model-cache incident

- **Query:** During the Feb 14, 2026 APAC region outage where vector generation requests
  intermittently failed with server errors due to a cache layer getting overloaded by a
  scheduled warmup task, what immediate mitigation did the oncall apply to reduce the
  cascading retries across downstream nodes?
- **Expected answer:** The on-call flipped the model-cache circuit breaker to "soft" (via the
  admin API) to reduce fanout to slow downstream nodes and cut the retry storm.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: eng-oncall (gte #5; 0.912)
  2. google_drive: Shift resume, scrub & follow-ups — APAC night -> EU morning (gte #16; 0.910)
  3. slack: eng-oncall (gte #1; 0.910)
  4. slack: eng-oncall (gte #18; 0.908)
  5. slack: 1802006789 (gte #21; 0.895)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - keyword `"Feb 14" warmup cache retries`:
    1. gmail: Concurrency & Latency Profiling — detailed methodology and scaling playbook
    2. gmail: Urgent: MedFlow amplified 429/5xx wave — proposed fallback & mitigation plan
    3. gmail: [Incident] US-WEST1 throughput signal impacting hosted inference - OrchidAI
  - search `"Feb 14, 2026" "APAC" "vector" OR "warmup" OR "cache" OR "cascading retries"`:
    1. gmail: Service degradation: intermittent 5xx/429 timeouts impacting Nimbus production
    2. gmail: MAP: Multi-region benchmarks + close timeline
    3. gmail: APAC zone isolation — immediate traffic reroute, opt-in & follow-up
- **Source document** (5,436 characters; `dsid_96e315a4f8644c8bbafe05c2e0c0b775`, row 259,992 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Sam C: +1. I flipped the circuit-breaker for the model-cache service to 'soft' at 20:35
  > UTC to reduce fanout to downstream p99-slow nodes. … This has cut the retry storm but
  > error_rate still ~3.8%.
- **Why it was missed:** The thread has the answer almost verbatim but says "embeddings
  pipeline", "ap-southeast-1", "INC-2026-98", "5xxs" where the question says "vector
  generation", "APAC", "server errors". There are dozens of on-call threads about 5xx
  incidents. The reranker scores the expected thread highest of all (0.948), had it been a
  candidate. *Paraphrase plus many similar incidents.*

### qst_0204: HubSpot, Pillarhaven Insightworks

- **Query:** What are the immediate follow up actions for the mid market product analytics
  SaaS that wants cheaper and more consistent in app assistant responses and is planning a
  short trial on the shared service before moving to a private network isolated deployment?
- **Expected answer:** Run a 2-week Hosted API POC and prepare a Dedicated pricing proposal:
  Pillarhaven uploads the prompt set and 2k sample chats; the SE delivers the parity-run plan
  by 2026-03-04; Redwood sends the Dedicated term sheet by 2026-03-10; security QA kickoff
  the week of 2026-03-08.
- **Given answer:** Based on the CRM record for Lighthouse Canvas Systems… POC runbook by
  2026-03-14, VPC peering info by 2026-03-16, kickoff the week of 2026-03-22… *(Wrong
  customer.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. hubspot: Cedarglade Product Systems (BM25 #16; 0.925)
  2. hubspot: RollingWave Inferlytics (BM25 #11; 0.913)
  3. hubspot: Lighthouse Canvas Systems (BM25 #6; 0.912)
  4. hubspot: GaleBrook Platforms (BM25 #7; 0.911)
  5. hubspot: Rosebank Signalworks (BM25 #1; 0.911)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"cheaper and more consistent in app assistant" "private network" OR "network isolated"`:
    1. fireflies: 2028-05-12 LumenForge - streaming latency economics discovery
    2. gmail: Private deployment connectivity options for Acme Analytics — decision and pilot request
    3. fireflies: POC: Deployment mode evaluation — Cedarbank Finance scoping
  - search `"trial" "shared" "private network" OR "private network isolated" "product analytics"`:
    1. gmail: Shared vs Private SLA rubric & next steps
    2. gmail: Re: Renewal + isolated analytics cluster for Nova Clinics
    3. gmail: Recap: Trial environment trade-offs, success criteria & owner commitments
- **Source document** (3,869 characters; `dsid_a45c07f3daaf47c18b7bc75d3cbf2112`, row 174,362 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > next_steps_detail: 1) Upload canonical prompt set & 2k sample chats to POC sandbox… 2)
  > SE to deliver parity-run plan + baseline metrics by 2026-03-04. 3) Redwood to send
  > Dedicated term sheet and committed tier options by 2026-03-10. 4) Schedule security QA
  > with Pillarhaven infra week of 2026-03-08.
- **Why it was missed:** At least three other CRM records describe a "mid-market product
  analytics SaaS building an in-app assistant" (Cedarglade, RollingWave, Lighthouse Canvas),
  and the reranker prefers them. *The question fits several documents; only the customer's
  name would tell them apart.*

### qst_0207: GitHub PR, per-route function-call telemetry

- **Query:** In the recent observability change for tool invocation tracking, what is the
  staged rollout schedule and the temporary kill switch name used to gate it?
- **Expected answer:** Gated behind the experiment flag "obs.route_tags_tool_calls"; three
  days in canary, then 25% of traffic, then full rollout if there are no regressions.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. github: Feature flag + staged rollout controls for tiered batching defaults (BM25 #7; 0.872)
  2. linear: Call surface fidelity audit and rollback plan (BM25 #5; 0.869)
  3. linear: Rollout feature flag + staged rollout controls + kill switch for tiered batching defaults (BM25 #21; 0.869)
  4. google_drive: Runtime 1.22 Launch Brief — Workload-aware batching defaults (fast/standard/cost-efficient (BM25 #24; 0.864)
  5. github: tool-result-merge-policy-and-exponential-backoff-for-parallel-calls (BM25 #8; 0.860)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `tool invocation tracking observability rollout schedule kill switch`:
    1. linear: Chained invoker rate safeguards + recursion detector for multi-tool execution
    2. github: tool-result-merge-policy-and-exponential-backoff-for-parallel-calls
    3. linear: Acceptance Criteria: Progressive Schema Tolerance, Tool Backpressure, and Post-Stream Reconciliation
  - keyword `tool invocation tracking observability rollout kill switch`:
    1. linear: Chained invoker rate safeguards + recursion detector for multi-tool execution
    2. linear: Acceptance Criteria: Progressive Schema Tolerance, Tool Backpressure, and Post-Stream Reconciliation
    3. github: Introduce invocation-sherpa-and-token-bias-attenuator
- **Source document** (2,967 characters; `dsid_1653d1ed11a1403ca8e44bef03a610a9`, row 15,643 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Rollout plan: feature gated behind experiment flag "obs.route_tags_tool_calls" for three
  > days in canary, then staged rollout to 25% traffic, then full rollout assuming no
  > regressions.
- **Why it was missed:** Every content word differs: "tool invocation" for function calls,
  "observability change" for telemetry, "kill switch" for an experiment flag. The worst case:
  13,209th in BM25, 41,930th in gte. Picked instead: other PRs and tickets about feature
  flags and staged rollouts. *Total vocabulary mismatch.*

### qst_0212: Google Drive, observability slice storyboard

- **Query:** How should we set up low-noise oncall dashboards and tiered paging for a streaming
  text generation service so that responders can jump from a latency spike to a trace with
  hashed prompt references and cache hotness context and resolve issues within about 15
  minutes?
- **Expected answer** (shortened): A slice-first design: dashboards and alerts per
  high-value slice (customer tier, route, model variant, KV-cache warmth); a span contract
  with `prompt_sha` (hash only) and `kv_hit` tags; KV warmth heuristics; graded per-slice
  alerts (P2 at p95 > baseline×1.5 for 15m, P1 at ×2 for 10m, P0 on SLA breach) with noise
  controls.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: incidents (BM25 #22; 0.897)
  2. google_drive: operators-triage-timeline-and-dashboard-matrix (gte #19; 0.865)
  3. google_drive: Telemetry weave: operator-lens notebook (BM25 #17; 0.862)
  4. confluence: operational-signal-workflow-and-comm-templates-2026 (BM25 #20; 0.859)
  5. confluence: Tiered On‑Call Orchestration and Customer Status Playbook (BM25 #24; 0.852)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `low-noise oncall dashboards tiered paging streaming text generation latency spike hashed prompt references cache hotness 15 minutes`:
    1. google_drive: Observability slice storyboard and alert finessing notes ✅
    2. slack: 5566001234
    3. google_drive: Spike forensics: WAF signature crafting and exit strategy
- **Source document** (7,440 characters; `dsid_5085c9f4b4484f298dabd02c99d9ce03`, row 144,836 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Span names: ingress.request -> scheduler.queue -> model.exec -> emitter.stream ->
  > exporter.kv_lookup. Required span tags: request_id, tenant_id, route, model_variant,
  > kv_hit (true/false), batch_id (if batched), prompt_sha (hash of prompt prefix),
  > token_count. Trace size policy: prefer compact spans; only include prompt_sha, not raw
  > prompt.
- **Why it was missed:** The notes call it a "slice-first approach" with "trace anchors" and
  "prompt snapshots"; the question says "low-noise dashboards", "hashed prompt references",
  "cache hotness". BM25 ranks it 101st. The agent found it. *Paraphrase; just below the cut.*

### qst_0213: HubSpot, BrambleBox Content Labs

- **Query:** For a content templating company planning tens of millions of very short
  generations during a Q3 ramp, what was the requested plan for a time limited evaluation
  that included a large free usage allowance?
- **Expected answer:** A 14-day trial with hosted API keys and 50 million free tokens for
  benchmarking.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. gmail: Request: short eval usage pack for self-serve signup (BrightAI) (gte #5; 0.843)
  2. gmail: MAP: Sequencing capacity reservations & procurement milestones for Q3 evaluation (gte #23; 0.820)
  3. gmail: Eval materials curation, MAP gating and close timeline (gte #11; 0.820)
  4. gmail: Evaluation plan & usage headroom — clarifying overage / burst terms (BM25 #15; 0.814)
  5. gmail: Re: Evaluation acceleration / short extension & packaging options (gte #12; 0.811)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `content templating tens of millions short generations Q3 evaluation free usage allowance`:
    1. hubspot: ZeptoQuill Marketing Platform
    2. hubspot: Aurabloom Marketing Apps
    3. gmail: Request: short eval usage pack for self-serve signup (BrightAI)
  - keyword `content templating tens of millions Q3`:
    1. hubspot: Silkbridge CodeFoundry
    2. slack: random
    3. google_drive: Burst crediting pricing — raw notes
  - search `"content templating" OR "tens of millions" "Q3" evaluation`: refused, search limit reached
- **Source document** (3,244 characters; `dsid_ad14c3cb06c4494093dfcfff5098d89d`, row 177,040 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > Next steps (explicit):
  > - Provide 14-day trial API keys + 50M free tokens for benchmark.
- **Why it was missed:** The record says "content marketing template generator", "trial API
  keys", "50M free tokens"; the question says "content templating company", "time limited
  evaluation", "large free usage allowance". BM25 ranks it 104th. Picked instead: emails about
  evaluation usage packs. *Paraphrase; just below the cut.*

### qst_0214: Linear, capacity triage and rollback plan

- **Query:** In the mid March 2026 incident where a regional dedicated serving cluster briefly
  ran out of spare capacity and the automatic scaler couldn't add nodes due to provider limit
  type errors, what exact build version did on call revert to in order to restore capacity?
- **Expected answer:** runtime-1.21.4 (rolled back from runtime-1.21.7).
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: 1712104523 (gte #10; 0.896)
  2. slack: eng-oncall (gte #18; 0.895)
  3. slack: incidents (gte #22; 0.894)
  4. slack: eng-oncall (gte #17; 0.884)
  5. slack: incidents (gte #21; 0.882)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `revert build version dedicated serving cluster capacity autoscaler March 2026`:
    1. jira: Dedicated capacity throughput collapse caused by intermittent GPU slot blocking from background chec
    2. linear: Autoscaler prematurely scaled down dedicated capacity after short burst, causing request rejections
    3. linear: Capacity recovery retro and targeted remediations for dedicated pools after scaling regression
  - search `dedicated-eu-2 INC-2026-487 quota_exceeded revert build version`:
    1. slack: eng-oncall
    2. slack: eng-infra
    3. slack: incidents
- **Source document** (6,654 characters; `dsid_303196c831af4d1bb38a1e2546228a09`, row 221,653 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > 2026-03-13T09:27Z - Autoscale controller reported repeated scale-up errors:
  > 'quota_exceeded' … 2026-03-13T09:42Z - Manual rollback: rolled back runtime build
  > runtime-1.21.7 -> runtime-1.21.4 on two affected zones; autoscaler recovered and
  > headroom returned to >30% by 09:50Z.
- **Why it was missed:** The ticket talks about a "capacity shortfall" and "headroom" in
  "Dedicated pools", "quota_exceeded". Dozens of Slack alerts say "autoscaler failed to
  provision nodes", and they rank higher everywhere. *Near-duplicate incident alerts.*

### qst_0218: Linear, actionable incident cards

- **Query:** In the private network hosted cluster console, what are the planned quick actions
  on the compact outage summary tile that on call engineers can use to stop a risky release
  or switch to a safer model option?
- **Expected answer:** **Pause Rollout** and **Trigger Fallback Variant**; the card also has
  **Add to Canary**.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. linear: Oncall Navigator: Rapid Actions & Telemetry for Private Deployments (gte #5; 0.867)
  2. linear: Ops Slate: Quick Network Diagnostics & Canary Risk Surface for Private Console (BM25 #2, gte #8; 0.864)
  3. google_drive: On-Prem First Responder Checklist + Troubleshooting Lab Notes (BM25 #15; 0.863)
  4. linear: Compact pathway annotations for Private Cluster console (gte #11; 0.861)
  5. linear: Connection playback + cost simulator for private console rollout surface (gte #6; 0.855)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"compact outage summary" OR "quick actions" stop risky release switch safer model`:
    1. slack: 2723456789
    2. slack: eng-releases
    3. confluence: Runtime 1.22 Release Notes — KV Cache Compaction + Tier-Aware Eviction
  - keyword `"outage summary"`:
    1. gmail: Request for service credit following 2027-04-16 incident
    2. gmail: Ownership of partial uptime hit during transit outage
    3. gmail: CascadeCrypt SLA credit request and issuance plan
- **Source document** (5,739 characters; `dsid_462ab0251b704d878310f53e13d6f515`, row 194,633 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > 2) Each card exposes primary actions: Add to Canary, Pause Rollout, Trigger Fallback
  > Variant, Open Debug Shell (link), and Mark Investigating.
- **Why it was missed:** The ticket's words are "incident card", "Private Cluster 'Status'
  panel", "Pause Rollout", "Trigger Fallback Variant"; the question says "compact outage
  summary tile", "stop a risky release", "switch to a safer model option". Picked instead: two
  other design tickets for private-cluster consoles ("Oncall Navigator", "Ops Slate"). *UI
  names paraphrased away, plus look-alike tickets.*

### qst_0219: Gmail, third-party IP indemnity redlines

- **Query:** In the negotiation where a healthcare customer pushed for stronger protection
  against outside patent or copyright lawsuits for an AI hosting service, what dollar limit
  did the vendor propose per claim as the compromise?
- **Expected answer:** A per-claim cap of the greater of 12 months of fees or $1,000,000.
- **Given answer:** In the negotiation with Solara Health (SOL-4721)… the greater of
  $5,000,000 or 12 months of fees. *(A different negotiation.)*
- **Pipeline's top 5** (rank in each search, reranker score):
  1. gmail: Re: MSA negotiation - indemnity & defense control (Solara Health / SOL-4721) (BM25 #12; 0.914)
  2. gmail: Re: Brightscale MSA / Order Form - litigation control & escrow alignment (BM25 #17; 0.901)
  3. gmail: Patent backstop: proposed MSA carve-out and escrow/insurance coordination (BM25 #1; 0.898)
  4. gmail: Nexera MSA redlines — indemnity, AI-output & cap negotiations (BM25 #10; 0.885)
  5. gmail: Indemnity negotiation: HelixCloud order form redlines (BM25 #4; 0.882)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `healthcare customer patent copyright indemnity cap per claim compromise`:
    1. gmail: Re: MSA negotiation - indemnity & defense control (Solara Health / SOL-4721)
    2. gmail: Order Form redlines — IP indemnity carve-out for patent claims
    3. gmail: Order Form: Third-party IP indemnity — scope & control ✅
- **Source document** (11,647 characters; `dsid_b395ebb516614698bcd038cdb7933d4b`, row 91,646 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > 3) Cap & insurance: We cannot accept an uncapped $5M demand for order-form customers.
  > Proposed commercial compromise: a per-claim cap equal to the greater of (i) 12 months of
  > fees paid under the Order Form or (ii) $1,000,000.
- **Why it was missed:** The thread is about "third-party IP indemnity"; the question says
  "outside patent or copyright lawsuits". Several other indemnity negotiations with
  healthcare customers rank higher. The agent found it. *Paraphrase plus similar
  negotiations.*

### qst_0220: Slack #eng-runtime, OOM with KV prefetch

- **Query:** In our GPU inference runtime, what configuration change was shown to eliminate
  out of memory crashes caused by combining lookahead cache allocation with aggressive
  request coalescing on long context runs, even though it slightly increased tail latency?
- **Expected answer:** Drop `prefetch_window` from 6 to 2 and `batch_trigger` from 8 to 4
  (with allocator tuning `slab_coalesce=true`, `max_reuse_age=3`): OOMs went to 0, p95 rose
  about 18 ms.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: eng-runtime (gte #6; 0.919)
  2. google_drive: Execution-path sweep for attention kernels: FlashAttention vs paged vs fused ops (BM25 #3; 0.905)
  3. slack: incidents (gte #16; 0.890)
  4. google_drive: Tail chunking and deferred-free patterns — OOM/root-cause notes (BM25 #10, gte #2; 0.882)
  5. google_drive: Eager sync vs deferred all-reduce trials (gte #1; 0.879)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `lookahead cache allocation aggressive request coalescing long context runs OOM tail latency`:
    1. slack: eng-runtime ✅
    2. linear: Darkhorse allocator fallback and KV reshard hotfix for cross-tenant OOM cascade
    3. google_drive: Oversubscription SLO-replay and OOM hypotheses
  - search `"lookahead" "coalescing" OOM tail latency configuration`:
    1. google_drive: Host-offload activation & PCIe-aware shard rebalancing notes
    2. linear: Deadline-aware microburst packer for streaming chat to reduce tail latency and GPU memory churn
    3. jira: Intermittent long-tail latency in structured-output mode when model emits large enumerations
- **Source document** (3,026 characters; `dsid_72b2bd3c2a094fdca6d5f3187a752c24`, row 226,364 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > mike: Short-term mitigation: reduce prefetch_window to 2 and lower batch_trigger to 4. …
  > Example flags: `slab_coalesce=true`, `max_reuse_age=3`
  >
  > anna: Tried that on a canary run. Results after 200 requests: peak_gpu down ~12%,
  > compaction_events down to 4, OOMs -> 0. Latency p50 +3ms, p95 +18ms (acceptable for this
  > workload).
- **Why it was missed:** The thread names the settings: `prefetch_window`, `batch_trigger`,
  "KV prefetch"; the question describes them as "lookahead cache allocation" and "aggressive
  request coalescing". Many other Slack threads are about OOMs. The agent found it.
  *Paraphrase of configuration names.*

### qst_0221: Confluence, Employee Lifecycle Operating Manual

- **Query:** When someone new starts at Redwood Inference, what is the target turnaround time
  for getting them fully set up with accounts and tools in North America versus
  Europe/Asia-Pacific?
- **Expected answer:** 5 business days for the US and 10 business days for EMEA/APAC.
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: eng-infra (BM25 #6; 0.891)
  2. gmail: Re: Welcome — pick a time for your Redwood trial (gte #13; 0.883)
  3. gmail: Global demo: confirming final slot across PT / CEST / IST (calendar attached) (BM25 #23; 0.879)
  4. gmail: Redwood quickstart — choose a time that works for your team (gte #14; 0.879)
  5. gmail: Welcome to Redwood — choose a 20-minute API setup slot (gte #19; 0.873)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `onboarding turnaround SLA accounts tools North America Europe Asia Pacific`:
    1. slack: customer-success
    2. gmail: Preferred invoice intake & PO stamping for North America vendor onboarding
    3. gmail: Re: SLA packaging, onboarding credits & canary coverage plan
  - search `employee onboarding IT setup accounts laptop "turnaround time" "new hire"`:
    1. slack: people-ops
    2. slack: new-hires
    3. slack: new-hires
    5. confluence: Employee Lifecycle Operating Manual ✅
- **Source document** (8,970 characters; `dsid_5323aeb9f89042c9a62505d409c4d94a`, row 3,917 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > 9) Metrics and SLAs for People Ops services
  > - New hire full provisioning SLA: 5 business days (US), 10 business days (EMEA/APAC).
- **Why it was missed:** The answer is one line in a 9,000-character operating manual about
  the whole employee lifecycle, and the question says "turnaround time" and "set up with
  accounts and tools" for "provisioning SLA". The agent found it. *One line in a long
  document.*

### qst_0224: Slack #docs, audit-log query examples

- **Query:** In the public documentation for searching system activity records, what
  identifier should examples use instead of showing a person's email address to avoid
  exposing sensitive personal data?
- **Expected answer:** `tokenized_actor_id` (instead of `user_email`).
- **Given answer:** I could not find this information in the company's documents.
- **Pipeline's top 5** (rank in each search, reranker score):
  1. slack: PR cleans audit-log docs: normalize field names, remove PII from samples, add migration no (gte #4; 0.886)
  2. slack: docs (BM25 #24; 0.871)
  3. confluence: Sensitive Actions Audit Event Taxonomy (Schema v1) (gte #18; 0.851)
  4. confluence: PII Redaction & Sensitive Fields Policy for Audit Events (BM25 #6; 0.843)
  5. slack: docs (BM25 #3; 0.842)
- **Agent's own searches** (run 14), top 3 of each, ✅ marks the expected document:
  - search `"documentation" "audit" search actor identifier email sensitive personal data`:
    1. confluence: PII Redaction & Sensitive Fields Policy for Audit Events
    2. slack: eng-platform
    3. confluence: Audit Event Schema v2 (Canonical) + SIEM Mapping (Splunk/Datadog)
- **Source document** (4,041 characters; `dsid_96088536cbcc449eb676e94bf800486a`, row 291,488 of `data/enterprise_rag_bench/data/documents/test.parquet`):
  > elaine: … re: examples that reference "user_email" we should switch to "actor_id" to
  > avoid showing emails in docs.
  > …
  > sec_ari: … We should call out tokenization: prefer tokenized_actor_id in examples.
- **Why it was missed:** The thread says "audit-log query examples", `user_email`,
  `actor_id`; the question says "searching system activity records" and "a person's email
  address". Picked instead: a PR description about cleaning PII from audit-log docs, which
  is about the same change. *Paraphrase of field names.*

## Why search fails: four patterns

Most cases have more than one cause; these are the main ones.

| Pattern | Cases | What happens |
|---|---:|---|
| **Paraphrase** | almost all | Semantic questions are written to avoid the document's words: product names, ids, field and setting names, jargon. BM25 needs those words; gte only partly bridges the gap. |
| **Look-alike documents** | 0178, 0184, 0186, 0196, 0201, 0204, 0214, 0218, 0219, 0220 | The corpus has many documents on the same topic (deep-dive scheduling emails, OOM threads, CRM records of similar customers). The question doesn't say which one, so better-matching look-alikes win. In 0184 and 0204 only the customer's name would decide, and the question leaves it out. |
| **A detail in a long document** | 0187, 0190, 0221 (and partly 0178, 0207) | The answer is one line in a long spec, release notes or manual. gte embeds the whole document as one vector, which describes its main topic, not that line. gte ranks these 2,000th to 196,000th. |
| **Just below the cut** | 0185, 0191, 0197, 0212, 0213 | The document ranks 37th to 104th in one search: found, but not deep enough. In 0197 it was inside BM25's top 50, but the fusion step dropped it. |

## What I take from it

- **The reranker isn't the problem; the candidates are.** In 12 of the 20 cases the reranker
  scores the expected document above everything the pipeline picked. Had it been among the
  candidates, it would probably have been first, and the answer probably right.
- **The fusion step throws away good candidates.** It merges two lists of 50 (up to 100
  documents) and reranks only the top 50 by fusion score. A document found by one search alone
  can't score above one found by both, so single-search hits ranked 30th to 50th are always
  dropped. Reranking the whole merged list (up to 100 documents) fixes 0197 at no LLM cost,
  for about twice the reranker's GPU time.
- **Going deeper would reach most of the rest.** The better of the two searches ranks the
  expected document within the top 200 in 7 cases, the top 500 in 11 and the top 1,000 in 16.
  But reranking 500 full documents per question with the current reranker would take about 40
  seconds on my laptop. It needs a cheaper middle stage: rerank short excerpts (the first 512
  tokens) of a few hundred candidates, then rerank the best 50 in full.
- **One vector per document hides details.** For release notes, specs and manuals, whole-
  document vectors can't find a single line. Smaller gte chunks (say 2,000 characters, several
  vectors per long document) would give the line its own vector. Re-embedding only the 3% of
  documents over 10,000 characters would take a few hours, not a night.
- **Rewriting the question helps a little.** The agent found 6 of these 20 by rewording, all
  from the paraphrase and just-below-the-cut groups. Writing a hypothetical answer in a
  document's words and searching with it (HyDE) attacks the same problem in one step,
  without an agent loop.
- **Some misses can't be fixed by search alone.** In 0184 and 0204 the question fits several
  documents equally well. Those may be partly a benchmark artefact: a real user would usually
  name the customer.
- **The reranker's scores are bunched together.** Picked documents and expected documents
  mostly score between 0.85 and 0.95, and the decisive gaps are often 0.005 to 0.03. A
  stronger reranker might separate them better; that's worth one comparison.

### Caveats

- 20 of 50 cases, all semantic questions; other question types may fail differently.
- The patterns and reasons are my reading of each case, not measured.
- The reranker score for the expected document uses its first chunk, as the pipeline does;
  for documents over 24,000 characters (none here) it would miss the rest.

## What's next

In order of cost:

1. **Rerank the whole merged list** instead of the fused top 50. Free to measure with the
   retrieval evaluation.
2. **Deeper candidates with a cheap middle stage**: take 200–500 from each search, rerank short
   excerpts, then the best 50 in full. Measure recall and time per question.
3. **HyDE**: search with a short hypothetical answer written by a cheap LLM, alongside the
   question.
4. **Smaller gte chunks for long documents**, to give details their own vectors.
5. **Score distributions on the questions the pipeline got right**, to compare with the
   medians above and see whether any score can tell "the answer isn't in the candidates".
   Free: no LLM calls.
