The final evaluation of the pipeline (`field-agent-clarify`): fix how the
evaluation drifted from what was built, add the metrics readers know, add an off-topic set
with the refusal decision, time every stage, then run it. Comparable with run 31
(`hybrid-ft-rerank`, 77.6% correct, strict judge). ^desc

- **Design:** [[docs/superpowers/specs/2026-11-23-evaluation-update-design|evaluation update spec]]
- **What drifted:** [[docs/brainstorm/007_evaluation_drift|brainstorm 007]]
- **Mining done:** [[Mine the corpus with the SLM]] (23 Nov); import and embed are [[import-embed-mined-fields]].
- **Scope (23 Nov, budget):** only `field-agent-clarify`, one run. Agent noise is 6–10 points,
  so the result is reported as a single run with that caveat.
- **Estimated cost:** ≈ $32–36, ≈ $27 with every cut: see Budget below.
- **Left out on purpose:** private data ([[Private and personal data questions]]), a
  pre-search gate, prompt tuning.

## Decision (23 Nov): single, already-composed questions
This evaluation feeds the pipeline one standalone question, as if an LLM had already
composed the search query from the chat. It does **not** test composing the query from the
conversation, nor accuracy inside a conversation. That is a separate task with its own test
set, metrics and budget: [[Rewrite the query from the chat history]]. Also on the research
map: problem "Final evaluation", step 1.

## Semantic retrieval first (decision 23 Nov)
Before paying for the full run (≈ $20), measure retrieval alone on the 125 semantic questions:
`evaluate-search --retriever field-agent --question-type semantic` (≈ $2–3; the agent's
searches and check, no answer, no asking) against the free first search
(`--retriever hybrid-ft-union`). Lenient semantic accuracy ≈ recall@5 × 0.96, so this tells
most of the semantic score. Needs [[import-embed-mined-fields]] first. Research map: problem
"Final evaluation", step 2.

**Result (23 Nov), first 90 semantic questions (budget $2–3, spent $2.69):**

| | R@1 | R@5 | R@10 | R@20 |
|---|--:|--:|--:|--:|
| first search (`hybrid-ft-union`) | 48.9% | 63.3% | 68.9% | 68.9% |
| agent (`field-agent`) | 62.2% | 63.3% | 71.1% | 73.3% |

The agent puts the right case first far more often (+13.3 points at rank 1), but recall@5 is
unchanged, so expected lenient semantic accuracy ≈ 61% (run 31: 54.4%). In the "60–65%" band:
try the free levers (traces of the misses, answer from 8–10 documents) before the full run.
All 125 semantic questions with the first search: R@1 47.2%, R@5 62.4%, R@10 68.8%, R@20 71.2%.

## Budget (23 Nov)
Prices are high; every paid step asks first.

| Task | What it is | Cost | Possible cut |
|---|---|---|---|
| [[import-embed-mined-fields]] | Load the SLM's fields for all 511,958 documents and embed their profiles, so field search covers the whole corpus | free | — |
| [[ambiguity-labels-alternate-gold]] | Gemini rates each question's top 5: label the question consistent / specific / ambiguous and keep the twins that also answer fully (alternate gold) | ≈ $5–6 | only the 125 semantic questions → ≈ $1.5 |
| [[memory-notes-v3-tiers]] | Rewrite the simulated person's memories in tiers (vague, concrete, wrong, none), so asking is tested on realistic memories | ≈ $2 | — |
| [[judge-strict-lenient]] | Judge every answer twice: strict (comparable with run 31) and lenient (a right mentioned case counts); check on run-31 answers | ≈ $0.10 | — |
| [[citation-metrics]] | Is the right document cited as the main case, as a mention, or not at all | free | — |
| [[retrieval-metrics]] | MRR, nDCG, precision, hit rate, recall; strict and with alternate gold | free | — |
| [[agent-effort-metrics]] | Turns, tool calls, searches until the right document, asks, tokens per question, from the traces | free | — |
| [[stage-timing-cost]] | Wall time per stage, latency p50 / p95, cost per question and per 1,000 | free | — |
| [[refusal-counts]] | Correct "not found" replies on off-topic and info-not-found questions; wrong refusals on answerable ones | free | — |
| [[report-splits]] | Every metric by question type, ambiguity, note tier, source, and the 33 questions with a cut gold document | free | — |
| [[ragas-judge-build-eval]] | Our Gemini judge for the four RAGAS metrics on a stratified sample of 100 questions, checked against the `ragas` library on run-31 answers | ≈ $3–6 | skip the check → ≈ $1.5 |
| [[off-topic-set-build]] | One Gemini call writes 50–100 off-topic questions; our local search flags risky ones; the user reviews them | < $0.05 | — |
| [[refusal-verdict]] | A 4th verdict in the agent's check, "no document answers", leading to the "not found" reply | free | — |
| [[smoke-test-clarify]] | Mention-cap check and 20 questions end to end before the full run | ≈ $1.7 | — |
| [[full-evaluation-run]] | `field-agent-clarify` on 500 + off-topic questions, one run, judged both ways; RAGAS on the 100-question sample | ≈ $20 | — |
| [[results-write-up]] | Report, speed and cost options, research map, todo board, article | free | — |
| [[off-topic-set-verified-rounds]] (parked) | The careful off-topic set: wide search and ratings per candidate, rerunnable as the corpus grows | — | ≈ $10–16 if reopened |
| [[fix-cut-long-documents]] (parked) | Full fields for the 3,151 long documents the SLM read cut | — | ≈ $1 version if reopened |
| **Total** | | **≈ $32–36** | **≈ $27 with every cut** |

## Tasks
In build order; each task's note has the why, how, cost and "done when".

- [x] [[import-embed-mined-fields]] (free) ![[import-embed-mined-fields#^desc]]
- [ ] [[ambiguity-labels-alternate-gold]] (paid ≈ $5–6) ![[ambiguity-labels-alternate-gold#^desc]]
- [ ] [[memory-notes-v3-tiers]] (paid ≈ $2) ![[memory-notes-v3-tiers#^desc]]
- [ ] [[judge-strict-lenient]] (paid ≈ $0.10) ![[judge-strict-lenient#^desc]]
- [ ] [[citation-metrics]] ![[citation-metrics#^desc]]
- [ ] [[retrieval-metrics]] ![[retrieval-metrics#^desc]]
- [ ] [[agent-effort-metrics]] ![[agent-effort-metrics#^desc]]
- [ ] [[stage-timing-cost]] ![[stage-timing-cost#^desc]]
- [ ] [[refusal-counts]] ![[refusal-counts#^desc]]
- [ ] [[report-splits]] ![[report-splits#^desc]]
- [ ] [[ragas-judge-build-eval]] (paid ≈ $3–6) ![[ragas-judge-build-eval#^desc]]
- [ ] [[off-topic-set-build]] (paid < $0.05, plus your review) ![[off-topic-set-build#^desc]]
- [ ] [[refusal-verdict]] ![[refusal-verdict#^desc]]
- [ ] [[smoke-test-clarify]] (paid ≈ $1.7) ![[smoke-test-clarify#^desc]]
- [ ] [[full-evaluation-run]] (paid ≈ $20) ![[full-evaluation-run#^desc]]
- [ ] [[results-write-up]] ![[results-write-up#^desc]]

## Parked
Also in the main board's **Backlog** column (tag `#evaluation`).

- [ ] [[fix-cut-long-documents]]: 33 questions touched, marked and reported apart ![[fix-cut-long-documents#^desc]]
- [ ] [[off-topic-set-verified-rounds]]: the careful, rerunnable off-topic set ![[off-topic-set-verified-rounds#^desc]]
