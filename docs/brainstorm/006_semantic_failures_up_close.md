# Semantic failures up close: what gte and BM25 find instead

Status: analysis notes for the brainstorm, 16 November 2026. Follows the 16 November decision in
[article 006](../articles/006_first_lora_and_checking_the_labels.md) and step 12d of the
[research map](../research/research_map.excalidraw).

**Setup:** today's best one, on the benchmark's 125 semantic questions:
- fine-tuned gte searches with the question, top 50;
- BM25 searches with the question plus Gemini's 4 rewrites appended, top 50;
- the reranker scores the union of both lists (about 95 documents).

A question **fails** when none of its right documents reaches the reranked top 5. That leaves
**41 failures**:

- **17 lost by the reranker:** the right document is among the candidates, but at least 5
  others score higher.
- **24 lost by search:** the right document is in neither top-50 list.

Data: `exploration/query_expansion/failure_cases.py`, which writes `data/failure_cases.json`
and a readable `data/failure_cases.md`. It's local and free. Each document is shown with the
passage closest to the gold answer.

How to read the ranks:
- **gte#** and **BM25#** are positions in each top 50; "–" means not in it.
- **rerank#** is the position in the reranked union of about 95 documents.
- **Score** is the reranker's score. The top-5 documents usually score 0.88–0.93.

## What the 41 have in common

### 1. Families of near-twins: same type, same template, same topic

The corpus is synthetic. Many documents are written from one template on one theme, and differ
only in the customer, region, date and numbers. The questions hide the name ("a healthcare
client", "a payments company", "a tiny two person dev shop"), so the topic is all that's left to
match, and every twin matches it.

| Family | Questions | Twins in the candidates |
|---|---|---|
| HubSpot notes on small SaaS chat assistants, latency targets in the POC | 0209, 0254, 0287, 0204, 0251 | QuickReply, Zenport, MicroGlint, FiberFrame, Velvet Echo, Petite Halo, StellarBridge, GoldenMarsh… |
| Jira: a pinned model version bypassed during a rollout | 0256, 0282 | 6+ tickets per question: sessions routed to the canary or quantized variant despite a pin |
| Linear: benchmark store + comparison UI + triage workflow | 0275 | 7 tickets with almost the same title |
| Linear: Private-cluster console tiles with quick actions | 0218 | Oncall Navigator, Ops Slate, Tile Deck, Inline Recovery Toolbar |
| Linear: multi-region routing and load shedding | 0285, 0240 | hedging, failover coordinators, quota limiters |
| Gmail: renewal emails with reranker latency SLAs | 0206 | 5 renewal emails, each with its own p95 for the reranker |
| Gmail: scheduling a technical deep dive with a health client | 0184 | PulseHealth, MedSure Health, Medallion Health, Pioneer Cardiology |
| Jira: 429 storms after synchronized credential refresh | 0181 | SSO, signed-URL, signed-key and cron variants, most with "add jitter" |
| Confluence: incident runbooks with update cadences | 0216, 0197 | 10+ playbooks; cadences differ by a few minutes |
| Fireflies: demo calls showing the Optimize tab | 0260 | Hummingbird, Acorn, LumenDocs |

In about half of the failures (20 of 41, by my reading), the documents that beat the right one
are twins like these.

**How the twins differ from the right document:**
- **Numbers:** 250 ms at 50 sessions, against 400 ms or 200 sessions in a twin.
- **The customer's details:** two people, a Vercel frontend, CORS.
- **One mechanism:** a tokenizer-hash check, signed cookies, fingerprint grouping.
- **Constraints the question states but the search ignores:** a date ("early March 2026",
  "late July 2025"), a region ("US East"), an industry ("healthcare", "payments", "shipping")
  or a source ("email thread", "call").

### 2. The answer is a side detail of a document about something else

In 15 failures the answer is one line in a document whose title and main topic are elsewhere.
gte embeds the whole chunk, BM25 weighs the whole document, and both see the main topic:

| Question asks about | The answer sits in |
|---|---|
| where NCCL fingerprint files live (0246) | a PR about a wavefront collective scheduler |
| KV-aware batching that cut p99 (0243) | a PR titled "dynamic per-channel requant…" |
| stepped batching sizes (0276) | a PR on stepped batching, a quant-safety linter and unit-econ smoke tests |
| the dual-signer workaround key (0229) | Slack **#sales** |
| the regional 40/40/20 GPU commit (0185) | a Slack #eng-platform thread about rehydration |
| auto-actions on safety regressions (0187) | acceptance criterion AC7 of "Acceptance matrix for Optimize quality shocks" |
| the removed field causing 400s (0190) | the breaking-changes list in "Optimize 1.3 release notes" |
| the canary ramp schedule (0234) | step 3 of "Optimize Conductor 4.0 release notes" |
| the sandbox lifetime and warning (0293) | one line of a ticket about provisioning a playground |
| output-stability rule for a shipping customer (0291) | one bullet of the "Tideway Shipping… evaluation kit" |
| mitigation for hallucinated citations (0283) | one bullet in a "reading scrapbook" |
| the security smoke-test date (0242) | one line of a 12.7k-character call transcript |
| the workspace watermark (0267) | point 3 of a Slack #eng-runtime thread on allocators |
| the canary pass thresholds (0297) | "mini-canary-solo-notebook" |
| who makes the demo GIF (0296) | a Slack #customer-success thread; two near-copies of it rank higher |

Slack documents are titled only with the channel name (`eng-infra`, `sales`, `docs`), so their
titles don't help either.

### 3. The question renames the key term

The question avoids the corpus's words, and the rewrites don't always guess them:
- "version lock" stands for *pin*;
- "attention memory" for *KV cache*;
- "GPU communication tuning fingerprints" for *NCCL autotuner*;
- "inference tuning product" for *Optimize*;
- "short lived credentials" for *signed cookies*;
- "traffic gatekeeper… time boundary" for *epoch anchors*;
- "compact outage summary tile" for *incident card*.

Patterns 1 and 2 almost always come with this one.

### 4. Doubtful gold answers (4 questions)

- **qst_0298:** a second document from the same meeting, "Tethys Systems… (Notes v2)", says
  the lead time was *updated to 3–5 weeks (was 2–4)*. The gold answer uses the older 2–4. BM25
  ranks the newer notes #2.
- **qst_0249:** the question says "Frankfurt secondary region". The document is about
  `eu-central-2`, which on AWS is Zurich; Frankfurt is `eu-central-1`. The question's
  paraphrase points at a different region.
- **qst_0282:** the question asks about "a traffic surge in US East for a healthcare customer
  with dedicated capacity". The right document (Jira "Pinned model unexpectedly demoted during
  edge rate-shedding…") is only gte #36 and rerank #10, so the question fails. Two *wrong*
  documents in gte's list, at #1 (a Jira ticket) and #3 (a Confluence incident review), are
  about BrightHealthAI: a healthcare customer in a dedicated pool during a 12 March surge in
  us-east. They match the question's details better than the right document does. The
  difference: there, traffic fell back to a *legacy quantized* variant, not an *experimental*
  canary. So either the question is ambiguous, or it hinges on the single word
  "experimental".
- **qst_0248:** the right document sets a Finance threshold at "midpoint + 15%". The twins give
  other rules: Director for ≤10% above the band maximum, VP People Ops for anything out of band.
  The gold answer, "+10% to +25% → Hiring Manager + HRBP", has to be read from a table further
  down the document.

### 5. When the right document is a candidate, the reranker is often close

- **5 of the 17 reranker losses are at rank 6–8**, just outside the top 5:
  - 0289: rank 6, 0.896 against 0.908 at #1;
  - 0181, 0235, 0293: rank 7;
  - 0275: rank 8.

  0216 and 0282 come next, at ranks 9 and 10.
- **4 of the 24 search misses would rank #1 if they were candidates**, because their reranker
  score is above the top pick's (known from other searches in the check):

  | Question | Right document | Top pick |
  |---|---:|---:|
  | 0197 | 0.896 | 0.881 |
  | 0224 | 0.924 | 0.864 |
  | 0227 | 0.936 | 0.899 |
  | 0276 | 0.921 | 0.905 |

  For these four, the problem is purely search.
- **The reranker also places some right documents far down:**

  | Question | Right document's rank |
  |---|---:|
  | 0248 | 48 |
  | 0249 | 38 |
  | 0298 | 28 |
  | 0254 | 26 |
  | 0209 | 23 |
  | 0234 | 23 |

  These are the template families, where a twin covers the topic more fully than the right
  document.

### What this means for bucketing

- **Pattern 1, the families:** the twins share a source, a topic and a template, so they would
  land in the same bucket as the right document. Bucketing by topic can't separate them.
  Telling them apart takes *attributes*: customer industry, region, date, the numbers, the
  source.
- **Pattern 2, the side details:** a bucket made from a document's main topic labels it by that
  topic. Routing would send the question to the bucket of its own topic, not to the one holding
  the answer.
- **Where routing could help:** questions that name a source or a date. Examples: 0235 says
  "email thread" (the reranker's #1 is a HubSpot note); 0242 and 0298 say "call" (the
  reranker's picks are emails and Slack); 0256, 0260 and 0227 give dates. That's metadata
  filtering on fields we already have, not bucketing.

## Case notes: lost by the reranker (17)

### qst_0181: staggering credential refreshes to stop 429s
- **Right:** Jira "Intermittent 429s from signed-cookie refresh loop in ephemeral worker pool
  during bulk ingest". gte#8, BM25#21, rerank#7 (0.884).
- **gte found:** Jira tickets on 429 storms (orchestrator burst windows, cron slotting, retry
  misalignment). **BM25 found:** a Jira ticket on SSO token-refresh 429s, a Slack #support
  thread on refreshes without jitter, a Jira ticket on signed-URL refresh waves.
- **Similar:** all of them describe synchronized credential refresh causing 429 bursts, and
  most give the same fix (add jitter).
- **Different:** only the right one has signed cookies, ephemeral workers, a *partial worker
  restart* and jitter of 30–120 s. The top pick is a Slack #devex thread about a Kotlin sample.

### qst_0206: the reranking p95 in a 2027 renewal (four bands → two levels)
- **Right:** Gmail "Renewal: Q3 committed GPU ramp + tier consolidation". gte –, BM25#2,
  rerank#12 (0.893).
- **gte found:** Slack #product on burst billing, renewal emails for Peregrine and Sproutline.
  **BM25 found:** a capacity addendum (16k model + reranker), the right one, a 24k-context
  renewal.
- **Similar:** they're all renewal emails with reranker latency SLAs.
- **Different:** each email has its own numbers (p95 < 80 ms, median ≤ 40 ms, ≤ 310 ms). Only
  the right one has the *tier consolidation* (4 → 2) and "candidate-rerank ≤ 310 ms at 32 GPU
  Base".

### qst_0209: POC pass criteria for an in-product live chat helper
- **Right:** HubSpot "Banyan Loop AI". gte –, BM25#2, rerank#23 (0.867).
- **gte found:** HubSpot QuickReply Stack, Atomline Chatflow, Northhelm ChatStudio. **BM25
  found:** HubSpot FiberFrame Chatwise, the right one, Honeygrove Assistantium.
- **Similar:** these are all HubSpot notes on a small SaaS embedding a streaming chat assistant,
  each with a first-token target. Honeygrove even has "median first-byte < 250 ms".
- **Different:** the right one has "< 250 ms at **50** concurrent small sessions" and fetches
  user-event context. The twins have 200–400 sessions or other targets.

### qst_0216: update cadence for the shared incident channel
- **Right:** Confluence "Tribe Liaison & Status-Wire Protocol: Incident Playbook". gte#1,
  BM25#50, rerank#9 (0.822).
- **gte found:** the right one first, then an outage coordination kit and a guide to SLO
  contracts. **BM25 found:** an on-call guide (every 15 min until containment), an
  incident-comms page (Sev1 every 30–60 min), chat-ops templates.
- **Similar:** all of them are incident-comms playbooks with update cadences.
- **Different:** the right one is the *cross-team* (Tribe Liaison) Status-Wire with T+10 /
  T+30 / every 30–60 min and 15 min if customer-impacting. The twins quote close but different
  cadences. gte had it first, and the reranker dropped it to #9.

### qst_0234: the ramp schedule for the new inference cost optimizer
- **Right:** Confluence "Optimize Conductor 4.0 release notes and migration guide". gte#42,
  BM25#3, rerank#23 (0.811).
- **gte found:** Linear tickets on cost-optimizer canaries (1→5→20→100%) and a GitHub PR on a
  prefill optimizer. **BM25 found:** a Linear rolling policy conductor, a Confluence PRD for a
  canary workflow, the right one.
- **Similar:** they're all staged rollouts with gates.
- **Different:** only the right one has the doubling steps (10→20→40→80→100) and 24 h waits,
  in step 3 of release notes. Google Drive rollout runbooks win on topic.

### qst_0235: the four contract markup areas for a payments pilot
- **Right:** Gmail "Re: Next steps & contract sent to procurement — quick recap". gte –,
  BM25#49, rerank#7 (0.892).
- **gte found:** HubSpot Cedar Ridge Payments, a Forkstream payments dossier, a HubSpot note on
  a support AI. **BM25 found:** Gmail threads on a liability carve-out, indemnity allocation and
  Acme Payments indemnity.
- **Similar:** they're all procurement and counsel redlines, several with payments companies.
- **Different:** only the right one lists four areas (US residency + 30-day export, 7-year
  audit logs, a lower cap, IP tweaks). The question says "email thread", but the reranker's #1
  is a HubSpot note.

### qst_0248: who approves an offer 20% above midpoint
- **Right:** Confluence "Evidence-driven offer evaluation and onboarding trigger playbook".
  gte#2, BM25#46, rerank#48 (0.842).
- **gte found:** Confluence offer-negotiation guidance, the right one, a hiring tension matrix.
  **BM25 found:** a Confluence comp-and-bands playbook, a Gmail thread on remote salary floors,
  the offer guidance.
- **Similar:** all of them cover approval rules for offers above band.
- **Different:** each document has its own thresholds, and some contradict the gold answer.
  The reranker prefers concrete approval emails. See the doubtful-answers section.

### qst_0249: readiness checklist for a GPU spot EKS cluster in "Frankfurt"
- **Right:** Slack `…euc2-eks-spot-gpu-provisioning.json`. gte –, BM25#15, rerank#38 (0.888).
- **gte found:** three Slack #eng-infra threads with GPU smoke tests, plus a kubeadm checklist.
  **BM25 found:** Slack #eng-infra bootstrap checklists.
- **Similar:** all are EKS/GPU nodegroup setup with an `nvidia-smi` check, and several score
  above 0.92.
- **Different:** the right one is eu-central-2 with spot nodes and a 9-step list. The twins are
  generic and titled `eng-infra`. The question's "Frankfurt" doesn't match eu-central-2 (see the
  doubtful-answers section).

### qst_0254: account notes for a two-person dev shop building a browser coding assistant
- **Right:** HubSpot "Copper Kite Works". gte –, BM25#43, rerank#26 (0.857).
- **gte found:** HubSpot Sidecar API Labs, Marmalade Rover Studio, Coralwood Interactives.
  **BM25 found:** HubSpot Inkbridge Collective, a Google Drive doc on starter credits, HubSpot
  Slate Needle.
- **Similar:** all are HubSpot notes on indie developers building code helpers on Vercel, who
  are cost-sensitive and want streaming and credits. Velvet Echo and Petite Halo (a solo
  founder) are nearly the same story.
- **Different:** only the right one has founder + 1, CORS and token exposure, $200 promo
  credits and 18k tokens in 48 h.

### qst_0256: mitigation at the routing layer when a version pin is given
- **Right:** Jira "Cohort signal loss during synchronized canary flip caused unpinned routing to
  quantized variant". gte#13, BM25 –, rerank#12 (0.893).
- **gte found:** three Jira tickets on pins bypassed by canary routing, edge retries and
  handshake drift. **BM25 found:** a Linear ghost-policy ticket, Jira pinned-model drift, Jira
  "version lock cleared".
- **Similar:** this is the densest family in the corpus. Every one is a pinned session routed
  to another variant during a rollout.
- **Different:** the right one has the q4 build, tokenizer hashes and a route-resolver guard
  that rejects mismatched fallbacks. "Early March 2026" and "single tenant" are in the question,
  and the twins are also dated March 2026.

### qst_0275: the doc on a benchmark result store, comparison UI and triage
- **Right:** Linear "Compact benchmark result store + comparison canvas and triage playbook".
  gte#35, BM25#9, rerank#8 (0.899).
- **gte found:** Linear "Performance drift inspector…", "Throughput ledger…", "Regression
  Forensics Hub…". **BM25 found:** "Throughput ledger…", "Benchmark lineage store…", "Result
  Aggregator…".
- **Similar:** at least 7 Linear tickets with the same three parts.
- **Different:** only the right one has fingerprint grouping (a 64-bit hash plus an embedding
  signature) and 30 days of detail / 1 year of rollups. The twins keep 90 days / forever or
  90 days / 2 years.

### qst_0282: why pinned conversations hit an experimental fallback in a US East surge
- **Right:** Jira "Pinned model unexpectedly demoted during edge rate-shedding…". gte#36,
  BM25 –, rerank#10 (0.922, against 0.937 at #1).
- **gte found:** Jira and a Confluence incident review about BrightHealthAI's us-east surge.
  **BM25 found:** a Jira API-gateway misroute, a priority-inversion ticket, an edge-to-core
  primer.
- **Similar:** the same pin-bypass family.
- **Different:** the right one is about edge *rate-shedding* with no pin exemption. The
  BrightHealthAI twins match the question's healthcare, us-east and surge better. See the
  doubtful-answers section.

### qst_0283: mitigating citation hallucination after aggressive context compression
- **Right:** Google Drive "Longform reading scrapbook: compression, KV heuristics, and RAG
  probes". gte#34, BM25#6, rerank#17 (0.870).
- **gte found:** Google Drive reading notes and truncation probes. **BM25 found:** a context
  window matrix, a paper-curation memo, conversation-context notes.
- **Similar:** they're all long-context research notes about hallucination.
- **Different:** the answer is one bullet in a scrapbook of conference takeaways: calibrated
  retrieval masking, entropy-drop detectors and a KV reset after retrieval.

### qst_0287: POC latency and peak agents for a support assistant with tool calls
- **Right:** HubSpot "Kestrel Helpdesk Systems". gte –, BM25#2, rerank#14 (0.839).
- **gte found:** HubSpot EmberField, Peregrine Response Grid, Elmbridge SupportWorks. **BM25
  found:** HubSpot Emberbrook AgentWorks (a 2-week POC), the right one, Moonlit HelpMatrix.
- **Similar:** these are HubSpot support-assistant accounts with streaming and tool calls.
  StellarBridge, the #1 pick, also has "first-token < 1.5 s p95".
- **Different:** the right one has a 500 ms median target and **50** concurrent agent sessions.
  StellarBridge has 120.

### qst_0289: cause of truncated streams and overcounted usage on a dedicated east-coast deployment
- **Right:** Jira "MintLance concurrent session stall causing truncated completions and
  temporary invoice/billing delta…". gte –, BM25#8, rerank#6 (0.896, against 0.908 at #1).
- **gte found:** a control-plane shift log, Linear on a retry + KV-replay usage spike, a Jira
  5xx killchain. **BM25 found:** a Jira CardinalPay truncation + billing spike, Jira on a March
  invoice spike, a Slack incident on truncated streams.
- **Similar:** they're all truncation or billing incidents with retries.
- **Different:** only the right one has a KV miss storm, a router safety path resetting streams
  and an idempotency gap in billing acknowledgements. Its title names the customer, not the
  cause. It missed the top 5 by one place.

### qst_0293: auto-shutdown of the cross-team sandbox, its lifetime and warning
- **Right:** Jira "Provision 'integration-playground' Jira project + Confluence space + Slack
  channel…". gte –, BM25#14, rerank#7 (0.812).
- **gte found:** Jira on sandbox permissions, Confluence on an SDK sandbox, a GitHub CI
  sandbox. **BM25 found:** a Confluence entitlement sandbox (72 h maximum), Jira on ephemeral
  experiment sandboxes with Jira, Confluence and Slack plus TTL, Jira on an autosweeper.
- **Similar:** they're all ephemeral workspaces with TTLs. The BM25 #2 is almost the same
  provisioning story.
- **Different:** only the right one has 14 days and a 72 h email warning. The answer is one
  line of a provisioning ticket.

### qst_0298: lead time for an isolated EU environment with customer keys (partner call)
- **Right:** Fireflies "Tethys Systems - SI Channel & Cloud GPU Fit Discovery". gte –, BM25#31,
  rerank#28 (0.817).
- **gte found:** HubSpot NorthBridge, Stratify and EdenBridge, all EU and KMS. **BM25 found:** a
  Gmail thread on SI onboarding, "Tethys Systems… (Notes v2)" (3–5 weeks), a Fireflies Sundial
  call.
- **Similar:** they're all EU Private + KMS deployments for SIs.
- **Different:** the right one is the call transcript (2–4 weeks). Its own later notes say 3–5.
  The question says "call", but the reranker picks Gmail and Slack.

## Case notes: lost by search (24)

Here the right document is in neither top 50. "Score" is given when another search in the
check returned the right document, so the reranker scored it. "No score" means no search in
the check ever found it.

### qst_0184: when the deep dive with the health client is scheduled
- **Right:** Gmail "CytoHealth — schedule private hosting architecture review". No score.
- **gte found:** Fireflies Pioneer Cardiology, Gmail threads with Medallion Health and MedSure
  Health. **BM25 found:** a MedCorps pilot, HubSpot Cedarstone Health, Atlas Labs.
- **Similar:** they're all scheduling 60–90-minute deep dives for Private hosting with health
  customers.
- **Different:** CytoHealth and Tue 24 Oct 2028, 10:00–11:15 PT. The question names neither.

### qst_0185: splitting a year-long commit for ~8 accelerators across NA, EU and SEA
- **Right:** Slack #eng-platform. No score.
- **gte found:** a Google Drive GPU BOM calculator, a Gmail OrchestraAI tri-region commit, a
  Google Drive procurement plan. **BM25 found:** three Google Drive NA/EU/APAC capacity plans.
- **Similar:** they're all regional capacity planning with burst buffers.
- **Different:** the answer (40/40/20 + 10% spot burst) is one message in a Slack thread on
  another subject.

### qst_0187: automated actions when harmful outputs persist for two windows
- **Right:** Linear "Acceptance matrix for Optimize quality shocks and auto-remediation". No
  score.
- **gte found:** Confluence fail-safe patterns, a Google Drive misuse corpus, a Linear canary
  framework. **BM25 found:** an A/B failure-mode matrix, a Linear experimentation runtime, a
  safety fallback contract.
- **Similar:** all are auto-rollback with PagerDuty.
- **Different:** only the right one has 75% traffic back to the last good model, a prefilled
  incident and PagerDuty + Console alerts, in acceptance criterion AC7. "Harmful output
  warnings" stands for *quality shocks / safety regression*.

### qst_0190: what broke older route configs in the February 2026 update
- **Right:** Confluence "Optimize 1.3 release notes — Performance Profiles & Cost Telemetry".
  No score.
- **gte found:** a Linear routing-spec validator, a Slack #devex thread on SDK payloads, a
  GitHub config-diff schema. **BM25 found:** a canonical request schema, a compatibility shim,
  a JSON validation taxonomy.
- **Similar:** they're all about schema validation and 400s.
- **Different:** `optimize.legacy_hint` removed in Optimize 1.3, one line under "breaking
  changes". "Inference tuning product" stands for *Optimize*.

### qst_0197: the first 60–90 minutes of a major outage
- **Right:** Confluence "Ten-Step Restoration Sprint & Shadow Mentoring Handbook". Score 0.896,
  **above the top pick's 0.881**.
- **gte found:** a first-five-minutes cookbook, a ten-minute checklist, a responder manual.
  **BM25 found:** first-responder protocols, on-call pathways, a capability matrix.
- **Similar:** ten or more incident runbooks with timed steps.
- **Different:** the five timed steps (0–5, 5–10, 10–20, 20–40, 30–50 min). The reranker would
  put it first. It's a pure search miss.

### qst_0204: next steps for a mid-market analytics SaaS (shared trial → isolated)
- **Right:** HubSpot "Pillarhaven Insightworks". No score.
- **gte found:** HubSpot Helmshore, WillowBend, Pebble Hemisphere. **BM25 found:** HubSpot
  Oxley Vertex, Lighthouse Canvas, Sequoia Haven.
- **Similar:** they're all mid-market SaaS deals moving from a hosted POC to Dedicated pricing.
- **Different:** only the right one has the dates (3/4, 3/10, the week of 3/8) and the parity
  suite. The question says "private network isolated", but the document says "Dedicated".

### qst_0207: rollout schedule and kill switch for tool-invocation tracking
- **Right:** GitHub "per-route function-call telemetry and JSONL output tagging". Score 0.860
  (the top pick has 0.878).
- **gte found:** GitHub PRs on invocation timelines, streaming counters, rollback dashboards.
  **BM25 found:** a GitHub PR with flags at 5→25→100%, Linear tickets on canary plans.
- **Similar:** they're all PRs and tickets with feature flags and staged rollouts.
- **Different:** `obs.route_tags_tool_calls` with 3 days of canary → 25% → full.

### qst_0214: the build on-call reverted to in mid-March 2026
- **Right:** Linear "Capacity triage, forensic telemetry, and scaling rollback plan for
  Dedicated pools". Score 0.885 (the top pick has 0.907).
- **gte found:** Linear tickets on the autoscaler and reservations after capacity events.
  **BM25 found:** Slack incidents on GPU quota exhaustion, a Terraform spot fallback.
- **Similar:** they're all autoscalers blocked by provider quotas.
- **Different:** runtime-1.21.7 → 1.21.4, one line in a timeline.

### qst_0218: quick actions on the compact outage tile in the Private console
- **Right:** Linear "Actionable incident cards + cost-aware rollout controls for private
  clusters". Score 0.812 (the top pick has 0.867).
- **gte found:** a Linear emergency panel, Cluster Overview panels, a Tile Deck. **BM25 found:**
  a compact network health dashboard, Ops Slate, playbook tiles.
- **Similar:** they're all Linear UI tickets on Private-cluster console tiles with one-click
  actions.
- **Different:** "Pause Rollout" and "Trigger Fallback Variant" on an *incident card*. The
  twins have other action sets.

### qst_0224: which identifier docs examples should use instead of email
- **Right:** Slack #docs. Score 0.924, **above the top pick's 0.864**.
- **gte found:** Google Drive processing-activity records, a security questionnaire sheet, a
  SIEM evidence export. **BM25 found:** two other Slack #docs threads on legal wording, the
  PII redaction policy.
- **Similar:** they're about PII in public docs and audit logs.
- **Different:** the swap of `user_email` → `actor_id`/`tokenized_actor_id`. Pure search miss.

### qst_0227: April 2025 bursty streaming with 15–30 s gaps
- **Right:** Gmail "Urgent: Conversation stream slowdown — customer impact & next steps". Score
  0.936, **above the top pick's 0.899**.
- **gte found:** Jira streaming stalls (packet reordering, silent SSE disconnects), a Google
  Drive streaming advisory. **BM25 found:** a Gmail realtime plateau, Jira SSE byte gaps, Jira
  auth refresh during SSE.
- **Similar:** they're all streaming stalls with client workarounds.
- **Different:** westus-edge-3, KV-size backpressure, a 10 s client timeout + one retry. Pure
  search miss.

### qst_0229: the temporary key accepting both old and new signing
- **Right:** Slack **#sales**. No score.
- **gte found:** Linear KMS signing + compat shim, a Jira gateway JWT rotation (dual-JWKS), a
  GitHub credential linter. **BM25 found:** a token-throttle notebook, Linear credential-epoch
  bleed, Slack on JWKS kids.
- **Similar:** all are key rotation with temporary dual acceptance.
- **Different:** `sk_poc_helio_dual` with `accepted_signers` v1-hmac and v2-ecdsa and a 24 h
  TTL, found in a sales channel.

### qst_0240: why parts of the gatekeeper disagree on the quota time boundary in failover
- **Right:** Linear "Investigate & fix rate-policy jitter that starves tenant capacity during
  regional failovers". Score 0.885 (the top pick has 0.903).
- **gte found:** Linear limiter failover, a limiter heartbeat protocol, a Confluence region
  split. **BM25 found:** a Google Drive API-key window probe, failover load-shedding notes, a
  Linear quota drift.
- **Similar:** they're all multi-region limiter inconsistencies causing 429s.
- **Different:** the race between epoch anchors and the policy-cache refresh. "Time boundary"
  stands for *epoch anchors*.

### qst_0242: the date the security lead set for the image smoke test
- **Right:** Fireflies "HaloTech x Redwood: Marketplace listing + ref-arch integration sync"
  (12.7k characters). No score.
- **gte found:** three Slack #partnerships threads on catalog publishing. **BM25 found:** a
  listing security playbook, a marketplace SKU email, a partner onboarding checklist.
- **Similar:** all are cloud-marketplace listing work with security checks.
- **Different:** Liam, 2025-03-24, the AMI smoke test, in one line of a long call. The
  question says "call".

### qst_0243: the change that cut latency spikes from KV footprint jumps
- **Right:** GitHub "dynamic per-channel requant + KV-aware jitter-resilient
  continuous-batching". Score 0.842 (the top pick has 0.911).
- **gte found:** a Linear stream aggregator with KV-aware batching, a GitHub NCCL/KV coalescer,
  comm sparsity notes. **BM25 found:** Google Drive attention-chunking notes, kernel variance,
  a FlashAttention sweep.
- **Similar:** they're all batching and KV-memory latency work. The Linear stream aggregator is
  very close.
- **Different:** the KV sizing hint + adaptive hold windows. The title leads with requant.

### qst_0246: where the GPU comm-tuning fingerprints are stored, and upgrade handling
- **Right:** GitHub "wavefront-collective-scheduler-compact-kernel-selector". Score 0.842 (the
  top pick has 0.890).
- **gte found:** Confluence GPU node profiles, Google Drive k8s GPU compat notes, a CUDA/NCCL
  lab journal. **BM25 found:** the same compat notes, an on-prem GPU upgrade playbook, control
  plane rotation notes.
- **Similar:** they're all GPU/NCCL upgrades and compatibility.
- **Different:** `/var/lib/redwood/nccl-fingerprints`, rotate or back up, in one line of a PR
  about a scheduler.

### qst_0251: embedding throughput and top-10 latency targets in pre-sales notes
- **Right:** HubSpot "PearlGate Embed Labs". No score.
- **gte found:** HubSpot Ironpine QueryGrid, EmberLynx, Ashfield Embedsuite. **BM25 found:**
  HubSpot Amberfield, Zephyrus, CaseHarbor.
- **Similar:** all are HubSpot embedding and semantic-search accounts with nightly batches and
  latency targets.
- **Different:** about 400 qps peak and < 150 ms for the top 10. RailCove (< 150 ms P95) is a
  near-twin.

### qst_0260: the console tuning tip from a late-July 2025 vendor call
- **Right:** Fireflies "Sandstone Corp — Search prototype: embeddings + rerank walkthrough". No
  score.
- **gte found:** Slack #product on Optimize, a Google Drive optimizer cookbook, a Bayesian
  quant playbook. **BM25 found:** HubSpot Octant, Fireflies Acorn, a Google Drive pilot
  journal.
- **Similar:** the reranker's picks are all Fireflies demo calls showing the Optimize tab.
- **Different:** batch size 8 + quantization profile x4 for the *search route*, on the date
  given.

### qst_0267: the change that best cut tail latency by stopping attention-kernel fallback
- **Right:** Slack #eng-runtime. No score.
- **gte found:** a Google Drive queue-length sweep, Linear paged-KV prefetch, a Linear prefill
  shaper. **BM25 found:** Google Drive notes on slab fragmentation and FlashAttention,
  scratchpad thrashing, MoE chunking.
- **Similar:** all are memory fragmentation causing kernel fallback and tail latency.
- **Different:** the ~8 MB per-pool workspace watermark (p99 ~310 ms), point 3 of a Slack
  thread.

### qst_0276: the proposal for stepped batch sizes under latency budgets
- **Right:** GitHub "Introduce stepped batching policy, quant-safety linter, and unit-econ
  smoke tests". Score 0.921, **above the top pick's 0.905**.
- **gte found:** a Linear econ-config lab, Google Drive cohort autotune presets, a Linear
  resilience lattice. **BM25 found:** Google Drive microburst presets, prefill thresholds,
  sequence-length-aware batching.
- **Similar:** they're all batching policies for mixed short and long requests.
- **Different:** `[1, 4, 16]` with per-step latency bounds. Pure search miss.

### qst_0285: SLO Sentinel, prefetching for failover and graded breakers
- **Right:** Linear "SLO Sentinel: Prefetch Routing and Graded Circuit Breakers for
  Multi-Region Requests". No score.
- **gte found:** Linear telemetry-driven dispatch, a failover coordinator, a geo-fallback shed
  plan. **BM25 found:** Linear SLO attestation, a regional fallback mismatch, locality hedging.
- **Similar:** these are all Linear multi-region routing controllers with shedding, a dense
  family.
- **Different:** an EWMA risk score, KV prefetch warmup, and Green/Yellow/Red breakers with
  their thresholds.

### qst_0291: output-stability rule in the shipping customer's checklist
- **Right:** Google Drive "Tideway Shipping integration - inference evaluation kit". No score.
- **gte found:** Confluence safe config patterns, a Linear export flow, an SLO pilot canvas.
  **BM25 found:** a benchmark FAQ, a Safety Net release checklist, chaos drills.
- **Similar:** release and readiness checklists.
- **Different:** 3 cold-starts and 5 warm-starts with string equivalence. "Shipping" is in the
  title, but in the other documents it means *to ship a release*, so it doesn't help BM25.

### qst_0296: who makes the animated demo for the renewal review
- **Right:** Slack #customer-success. Score 0.879 (the top pick has 0.905).
- **gte found:** Google Drive QBR deck templates. **BM25 found:** Linear animated onboarding
  GIFs, Linear catalog tiles, a renewal checklist.
- **Similar:** the reranker's #1 and #3 are near-copies of the same Slack thread (demo prep,
  slide 3 heatmap), without the GIF line.
- **Different:** Alan commits to a GIF of the onboarding dashboard.

### qst_0297: canary pass/fail thresholds for the first hour of a KV-eviction tweak
- **Right:** Google Drive "mini-canary-solo-notebook". No score.
- **gte found:** Confluence experimentation manuals with SLO envelopes, a Linear rollout
  playbook, a governance checklist. **BM25 found:** a Linear microburst absorber, an experiment
  matrix, Google Drive KV prefetch notes.
- **Similar:** they're all experiment safety thresholds for 5xx and p99.
- **Different:** 5xx > 0.1% relative or p99 > +10 ms in the first 60 minutes, in a notebook
  with a generic title.
