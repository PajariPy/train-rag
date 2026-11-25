Understand how the corpus could be grouped into buckets by what the documents say, source by
source, so search (or an agent) can narrow down where to look before ranking.

## Why
41 of the 125 semantic benchmark questions still fail with today's best setup (67.2% in the
top 5). Two patterns dominate (brainstorm 006):
- **near-twin families:** many documents from one template on one topic, told apart only by
  the customer, region, date or numbers. A topic bucket would hold the twins together, so
  separating them takes attributes, not topics;
- **side details:** the answer is one line in a document about something else, so a bucket
  made from the document's main topic points the question elsewhere.

So the open question is what a useful bucket is: a topic, a source structure (Slack channel,
Linear ticket description), or attributes the question states (source, date, customer type).

## How
Per source, in the data preprocessing problem (the second problem in
`docs/research/research_map.excalidraw`, linked from step 12):
- **Linear (step 1):** 99.96% of tickets have a description, a short summary written by the
  source itself. Idea: search the descriptions, then open the full ticket. But the
  description holds the answer in only 25 of the 58 Linear questions.
- **Slack (step 2):** 35 channels hold 96.8% of threads. Idea: a channel filter or boost, as
  an agent tool rather than a hard filter (qst_0229's answer sits in #sales).
- **Reading:** the Slack review page (`exploration/document_review/`, 50 threads, short /
  long / random + their 5 most similar), notes in `review_notes.json`; parsed samples of
  every source in notebook 010.
- Next sources: Gmail, Jira, Confluence, the rest.

Related: [[Approach human-style data problems]], the document growth map
(document growth problem in `docs/research/research_map.excalidraw`, the bucketing idea),
`docs/brainstorm/006_semantic_failures_up_close.md`.

## Decision (16 Nov)
Topic modelling and clustering are stopped: search already reaches the right topic, and the
failures are near-twins inside it. Clustering adds little over cosine search on the
fine-tuned gte vectors (DiskANN index), and per-document keywords are what BM25 already does.
The path is mining with a language model: our own schema per source, one SLM pass over the
corpus (~$15–35), plus rules and NER for what the text states. See steps 3 and 4 of the data
preprocessing problem.

## Plan (16 Nov): zero-shot attribute mining, Slack first
**Why Slack first:** Slack is 55.8% of the corpus (285,605 threads) and the least structured
source: the title is only the channel name, and each thread is a single vector. 6 of its 13
semantic questions fail, and we haven't addressed it properly yet. The other sources have
more structure (ticket fields, email headers, CRM records) and are smaller, so a tactic that
works on Slack should carry over. We fight the toughest soldier first and learn how to
defeat the whole army in the process.

**The split:**
- **concrete fields** (customer, region, dates, ids, people) → GLiNER2 or NuExtract,
  zero-shot, from a schema we write;
- **abstracted attributes** (industry, customer size, document purpose) → GLiNER2's
  classification, zero-shot;
- **fine-tune only if that's too weak.** The papers suggest it may be: fine-tuned small
  models beat zero-shot on extraction (Sub-Billion Super-Frontier, ELMTEX). The LoRA
  route (~$15–35) stays as the fallback.
- **Store scores, not just labels,** so a filter can be tightened later without another
  pass.
- Mined fields become **agent tools or soft filters**, not hard routing.

**Steps:**
1. Write the Slack schema from the review notes (`exploration/document_review/`) and the
   failure patterns (brainstorm 006).
2. Label ~200 Slack threads as the reference: DeepSeek (~$0.10, paid: ask first) plus a hand
   check on the review page.
3. Run GLiNER2 and NuExtract on them and score each field. Check GLiNER2's input limit
   (split long threads by message if needed).
4. Dev pilot: extract the right documents and candidates of the dev questions (~40k
   documents, local) and measure the retrieval effect (an agent filter, the fields embedded
   with the text) on the semantic dev questions. Confirm once on the benchmark.
5. If it holds: the full Slack pass, then the same tactic on the other sources.

**GLiNER2 notes** (map step 5a; [TDS article](https://towardsdatascience.com/gliner2-extracting-structured-information-from-text/),
[repo](https://github.com/fastino-ai/GLiNER2)):
- one schema runs entities, classification, relations and structured fields in a single
  pass (`create_schema()...extract()`), which fits "no second pass on 500k";
- `include_confidence=True` gives the scores to store; LoRA adapters (2–10 MB) are the
  cheap fallback before the SLM route;
- it extracts what the text states well, but doesn't infer (`gender` null though Ada is
  "daughter"). So inferred attributes go to classification and get checked hardest;
- `extract_json` records mix people up (Babbage and her husband became Ada's parents; Lord
  Byron got his wife's "reformer"), while the same `parent_of` as a relation was right. So:
  one value per field, links as relations, and validate every record (fields are optional,
  even `name` can be null);
- label names matter (`alias` worked, `same_as` didn't): try 2–3 names per field in step 3;
- input is chunked at 384 tokens: split threads by message and merge the fields per thread.

## Draft Slack schema (16 Nov, for review)
From reading 12 threads across 11 channels (`exploration/document_review/`) and the 79 Slack
benchmark questions. Speaker and channel are already parsed and aren't extracted again.
Spans are at most 8 words (`max_width` 8 in `gliner2-large-v1`).

**Entity types (spans):**

| Label | Description for the model | Examples from the threads |
|---|---|---|
| `topic` | the main subject of the thread, as a short phrase | ampere q4_0 seqlen=1024 regression; 403 org_mismatch after key rotation |
| `person` | a person's name or handle mentioned in the text | @Raj, Matt, Omar |
| `team` | a team, group or department | eng-runtime, SRE, People Ops, EngOps |
| `client` | a customer company, tenant or customer org | acme-inc, zenith, AcmeSec, Arkadia Finance |
| `region` | a cloud region, data centre or location | us-west-2, eu-west, us-east rack |
| `product` | a product, plan or tier the company sells | Redwood Private, Fast tier, Dedicated, Redwood CLI |
| `feature` | a user-facing feature or capability | midflight backend swap, Optimization Suggestions, provisional admit |
| `component` | an internal service, system part or tool | auth-proxy, audit-shipper, autotuner, `/v1/generate` |
| `model` | an ML model name | redwood-7b, gpt-small |
| `version` | a version, release or variant | runtime 1.19, q4_0, `db_test_fixture_v2` |
| `environment` | a deployment environment or stage | prod, staging, canary, perf-lab |
| `bug` | a problem, error or failure, as its short symptom | 403 org_mismatch; SSO failures; flaky DB tests |
| `cause` | the root cause of a problem | token minted under the wrong org; autotuner heuristic |
| `fix` | a fix, mitigation or action taken or planned | revert autotuner rule; issue temp token |
| `config` | a configuration setting or feature flag | `saml.allowArtifactFallback`, `edge.midflight_swap` |
| `metric` | a metric, alert or measured quantity | p95 latency, `edge.swap.fallback`, `bench.reject_rate` |
| `identifier` | a ticket, incident, PR or request id | PERF-1189, INC-2025-042, PR #9482, `req_12ab34cd` |
| `date_time` | a date, time, deadline or time window | 4/17, 10:00–10:30 PT, EOD PST, H1 2025 |
| `document` | a named document, RFC, template or runbook | RFC edge-midflight-swap, PRD template |

**Relations (head → tail):**

| Relation | Head → tail | Example |
|---|---|---|
| `owns` | person → fix | Matt → auth-proxy patch |
| `reported` | person or client → bug | Leo → 403 org_mismatch |
| `works_for` | person → client | Leo → acme-inc |
| `member_of` | person → team | Sam → People Ops |
| `affects` | bug → component, feature, product or client | SSO failures → zenith |
| `occurs_in` | bug → region or environment | regression → canary |
| `caused_by` | bug → cause | 403 → token minted under the wrong org |
| `fixed_by` | bug → fix | regression → revert autotuner rule |
| `changes` | fix → config | artifact fallback → `saml.allowArtifactFallback` |
| `tracked_in` | bug or fix → identifier | regression → PERF-1189 |
| `part_of` | feature or component → product | midflight swap → edge |

GLiNER2 can't restrict head and tail types, so wrong-type pairs are filtered afterwards.

**Left out for now** (sentences, not spans): decisions, thread kind (incident, how-to,
proposal, onboarding, social), plain numbers (BM25 already matches them).

**Risks:** `cause` and `fix` are often longer than 8 words; `identifier`, `config` and
`date_time` may be cheaper with regex; GLiNER2 is sensitive to label names, so try 2–3 names
for weak types.

**Device:** `gliner2-large-v1` (DeBERTa-v3-large, 340M) fits the laptop GPU (RTX 2000 Ada,
8 GB) with room to spare. Rough guess, to measure on the 50 review threads: hours, not days,
for all 285k threads on the GPU; CPU only would take days.

## Evaluation (16 Nov)
**Reference:** DeepSeek V4.1 Flash, off-peak, labels the same schema. Every value must be copied
word for word from the thread; a value not found in the text is flagged. Hand check 30–50
threads on the review page before trusting the rest. Threads from the training side, spread
over channels and short / long / random; none behind the benchmark questions. Cost ≈ $0.0009
a thread: 13-thread trial ≈ $0.01, 200 threads ≈ $0.20 (paid: ask first).

**Models:** `gliner2-large-v1` (DeBERTa-v3-large, 340M, ~512-token window, spans ≤ 8 words,
threads split by message) and `gliner2.5-base-v1` (DeBERTa-v3-base, 194M, 4,096 window,
no span limit, whole threads). 30 of the 50 review threads exceed 384 words; none exceeds
4,096.

**Items:** unique values per thread, not mentions. Entity = (thread, type, text);
relation = (thread, relation, head text, tail text). Text lowercased, spaces and punctuation
trimmed.

**Matching:** exact (same type, same text) and partial (same type, overlapping spans).

**Metrics:**
- precision (how many found items are in the reference), recall (how many reference items
  were found), F1 and support, per field and per relation;
- micro (counts pooled, frequent fields dominate) and macro (mean over fields, each field
  equal), separately for entities and relations;
- a type-confusion table (right span, wrong type), to merge or relabel types;
- a threshold sweep per field (0.3–0.9) from the stored scores, to pick each field's
  threshold;
- time per thread on the GPU.

**Rollout:** small sample first, adjust, then scale:
1. 13-thread trial: DeepSeek labels + both models; fix the prompt, labels and descriptions.
2. ~200 threads: hand check, full scores, pick the model, thresholds and the fields to keep.
3. Dev pilot (~40k documents) for the retrieval effect, then the full Slack pass.

## Trial result (16 Nov): zero-shot GLiNER is too weak
13 threads, DeepSeek labels as reference ($0.0079), code in `exploration/slack_extraction/`
(sample, label, extract, score; report in `data/report.md`).
- gliner2-large (384-word chunks) best: micro F1 0.41 exact / 0.48 partial, 0.83 s a thread
  on the laptop GPU. gliner2.5-base: 0.33 / 0.40 on chunks, 0.28 / 0.32 on whole threads
  (the long window lowers its recall).
- Names work: person 0.85, client 0.83, date_time 0.71, version 0.70, team 0.60 (partial F1).
- Descriptive fields fail: feature 0.17, config 0.16, component 0.14, fix 0.09; topic 0.28.
- Relations ≈ 0 for every type.
- Part of the gap is the reference (long DeepSeek phrases, loose types).

Verdict (user, 16 Nov): not worth pursuing as is; we need something more powerful.

## Decision (16 Nov): a fine-tuned SLM
Options weighed: fine-tune GLiNER2-large (labels ≈ $1, still a short-span picker), a small
generative model with LoRA on DeepSeek labels (≈ $15–35), DeepSeek on all Slack (≈ $175
off-peak at the measured $0.0006 a thread, and again for every source). **Chosen: the SLM.**
It writes phrases, topics and relations, and once trained it runs cheaply on every source.
Reuse the fact-extraction setup (Qwen3.5-2B LoRA, `exploration/fact_extraction/train.py`,
Vast.ai) and score it with `exploration/slack_extraction/score.py` against held-out DeepSeek
labels.

**Order (16 Nov):** first [[Build the Slack semantic dev set]] (measurement is on dev, never
the benchmark), then [[Test mined fields as search filters]] with DeepSeek labels as the
ceiling, and only if they help, [[Train the Slack field extractor SLM]].

Only if GLiNER2's classification is weak: compare decision models for the closed
attributes (Clef-flash self-hosted; Jev or Gemini Decisions paid, ~$35–80 per attribute on
the full corpus: ask first).
