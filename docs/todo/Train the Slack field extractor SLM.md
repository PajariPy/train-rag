Fine-tune a small generative model to mine the field schema from every document of every
source in one pass (decided 20 Nov: all sources now, not Slack first).

## Why
Zero-shot GLiNER was too weak on the 13-thread trial (micro F1 0.41 exact / 0.48 partial;
bugs, causes, fixes, topics and every relation fail). DeepSeek on all Slack would cost about
$175 and again per source. Decision (16 Nov): an SLM ([[Understand how data could be
bucketed by content]]).

## How
Only after [[Test mined fields as search filters]] shows the fields help.
1. Training labels come from the field test ([[Test mined fields as search filters]],
   section "Labels reused as SLM training data"): the candidates of half A of the dev
   questions (≈ 10,000 documents, about half not Slack), ~1,000 random Slack threads and ~1,000 train-split
   questions, all with the frozen prompt. Label the 200 held-out threads from
   `exploration/slack_extraction/` for scoring. Never train on a `dev_slack` source thread
   or question, or on half B's candidates.
2. LoRA on Qwen3.5-2B with the fact-extraction setup (`exploration/fact_extraction/train.py`,
   Vast.ai: paid, ask first).
3. Score on the held-out 200 with `exploration/slack_extraction/score.py` (per field and
   relation, micro and macro).
4. The full Slack pass; store the raw output so new fields are re-parsed, not re-mined.

## Cost estimate (16 Nov)
Measured inputs: DeepSeek trial on 13 threads ($0.0079: 706 output tokens a thread with the
first schema); first fact LoRA (Qwen3.5-2B r16, 2,695 examples × 3 epochs, median 2,020
tokens: 68 min on an RTX 4090 at $0.42/h = $1.01, plus $0.74 of failed starts); Slack threads
median 470 words (~1,000 tokens). The extended schema (26 fields, 16 relations) is assumed to
double the output to ~1,400 tokens.

**Fine-tuning (one time):**
| Part | Basis | Cost |
|---|---|---:|
| DeepSeek labels | paid by the field test (≈ $0.85 extra for random threads and questions) | ≈ $0.85 |
| LoRA runs | ~50 min each on a 4090 ≈ $0.40; 3–5 runs with failures | $1.5–3 |
| **Total** | | **≈ $2.5–4** |

**Serving = the mining pass** (vLLM on a rented RTX 4090, $0.40/h). Output dominates:
285k threads × 700–1,400 tokens = 200–400M output tokens. vLLM speed for Qwen3.5-2B is **not
measured**; plausible 2,000–5,000 output tokens/s with batching:
| Scope | Output tokens | Hours | Cost |
|---|---:|---:|---:|
| Slack, 285k threads | 200–400M | 11–56 h | $5–25 |
| Whole corpus, 512k docs (longer docs) | ~400–800M | 22–110 h | $10–45 |
The laptop GPU (RTX 2000 Ada) is free but several times slower: days to weeks.

At query time the question's fields cost ≈ $0.0002 with DeepSeek, or nothing with the SLM.

**DeepSeek instead of the SLM:** Slack $175 (measured $0.0006 a thread, first schema) to
≈ $285 (extended schema); whole corpus ≈ $350–550; and the full price again after every
schema change, where the SLM only re-runs the GPU pass.

**Measured (16 Nov), vLLM 0.33.0, base Qwen3.5-2B, one RTX 4090** (`exploration/slack_extraction/vllm_bench.sh`):
1,000 real Slack threads with the extraction instructions in front (1,600 input tokens a
thread), output forced to a fixed length:

| Output per thread | Threads/s | Output tokens/s | Slack, 285k | Whole corpus, 512k |
|---:|---:|---:|---|---|
| 1,400 tokens (extended schema) | 6.1 | 8,556 | 13 h ≈ **$4.80** | 23 h ≈ **$8.70** |
| 700 tokens (first schema) | 10.8 | 7,578 | 7.3 h ≈ $2.70 | 13 h ≈ $4.90 |

At $0.37/h; add ~$0.50 of setup per machine and pick a host without download fees (see
`docs/vast_ai.md`). Serve the LoRA merged into the weights, so the trained model runs at the
same speed. Assumes other sources' documents are about as long as Slack threads.

**So the SLM route is ≈ $3–5 (fine-tuning) + ≈ $5 (Slack) or ≈ $9 (everything) ≈ $8–15 in
total**, against $175–550 with DeepSeek. Run cost: $0.10 GPU + $0.44 download fees + $0.01
for a host that never booted.


## Plan (20 Nov): all sources in one run
Full plan: `~/.claude/plans/eventual-jumping-moon.md`; research map, data preprocessing
problem, step 9; article 007, "Next: one extractor for every source".

- **Labels:** 12,930 `train` items in `exploration/slack_extraction/data/labels.jsonl`
  (6,289 Slack, 1,487 Linear, 1,282 Drive, 1,114 Gmail, 1,001 questions, 478 Fireflies,
  430 Confluence, 386 Jira, 293 GitHub, 170 HubSpot) + **~2,000 extra** DeepSeek labels for
  the thin sources (Gmail is 24% of the corpus, 9% of training), from documents in no
  evaluation set.
- **Split stays:** half B's 12,531 candidate documents are never trained on, so the
  field-search test can be rerun on the SLM's own fields. A 70/30 re-split would leave 60
  questions for that test, too few with 6–10 points of run-to-run noise.
- **Cross-validation: would be good, no budget** (5 trainings ≈ $15, 30+ GPU hours). One run,
  scored on every held-out document per source (≈ ±1–2 points); a second run with one
  changed setting if borderline.
- **Run:** Qwen3.5-2B LoRA r16, 2 epochs, `--max-tokens 6144`, RTX 4090; smoke run first.
  **Weights & Biases** project `train-rag-fields`; train.log copied back.
- **Score:** parse rate and per-field F1 per source vs DeepSeek (`score.py`), then the half-B
  field-search test. Mining only after that.

| Fine-tuning | Cost |
|---|---|
| 2,000 extra labels (DeepSeek, off-peak) | ≈ $2.10 |
| Smoke run + machine setup | ≈ $1.00 |
| LoRA, 2 epochs, ~14.9k examples, ≈ 6.5 h | ≈ $2.70 |
| Scoring 12.5k held-out | ≈ $0.20 |
| Download fees, failed boots | ≈ $0.50–1.00 |
| **Total** | **≈ $6.50–7** (+ ≈ $3 for a second run) |

Mining afterwards: 512k documents, ≈ 615M output tokens, ≈ $9–13 on one 4090 (21–30 h).

## Result: training (21 Nov)
Run `fields-r16-e2`; adapter, logs and W&B (rag-train/train-rag-fields) in
`exploration/slm_fields/runs/fields-r16-e2/`. Progress page workflow "SLM fine-tuning, all
sources · 2026-11-20 09:31".

- **Extra labels:** 1,998 of 2,000 (2 failed), DeepSeek off-peak $1.84. Training set:
  14,919 examples (`build_sft.py`); eval 400 (40 per source); held-out 12,711.
- **Training:** Qwen3.5-2B LoRA r16, 2 epochs, 1,866 steps, **9.6 h** on one RTX 4090 in
  Thailand ($0.40/h) with the fast DeltaNet kernels (16 s/step). Planned 6.5 h: the median
  example is 3,729 tokens, longer than estimated.
- **Loss:** 0.46 → 0.14. Eval loss 0.247 (step 200) → 0.171, flat from step ~1,600: no
  overfitting, a third epoch would add little.
- **First look** (20 held-out documents, end of training): JSON valid 100%, mean field word
  F1 vs DeepSeek **0.78**. Strong: person, client, region, model (≥ 0.99), question 0.90,
  product 0.89, topic 0.84, fix 0.84. Weak: decision 0.30, impact 0.40, action_item 0.56,
  document 0.65, date_time 0.67, concept 0.68: the judgement fields, where DeepSeek is least
  consistent too.
- **Failed first attempt:** ssh dropped during setup, the fast-kernel flag read as empty,
  the run started on the slow path (68 s/step) and was destroyed: $0.46. `vast_train.sh` and
  `attach_train.sh` now time out every ssh call and run jobs detached.
- **Cost:** labels $1.84 + Vast $5.23 (incl. the failed attempt) = **$7.07** (plan ≈ $6.50–7).

Next: score all 12,711 held-out documents per source (`vast_score.sh` → `score_heldout.py`).

## Result: held-out score (21 Nov)
All 12,711 held-out documents (`test` + `extraction_test`, never trained on), generated with
vLLM 0.33.0 on the merged model (greedy, documents cut at 5,000 prompt tokens), scored
against DeepSeek's labels with `exploration/slm_fields/score_heldout.py`. Report:
`exploration/slm_fields/runs/fields-r16-e2/score.md` and `score.json`.

- **JSON valid 99.5%**; 0.4% of outputs hit the 3,000-token limit. 80 min of generation on
  one RTX 4090; scoring run $0.74 (incl. ~30 min idle from a script bug, fixed).
- **Every source is close:** mean field word F1 0.733 (Fireflies) to 0.813 (Gmail); Slack
  0.787 on 6,584 documents; all 0.785. Value F1 (exact value, micro) 0.555–0.623.
- **Names and identifiers match DeepSeek almost exactly** (word / exact value F1): person
  0.98 / 0.98, client 0.97 / 0.90, region 0.96 / 0.88, model 0.95 / 0.81, identifier
  0.85 / 0.81, date_time 0.85 / 0.80, team 0.81 / 0.79.
- **Phrases match in words, not in wording:** topic 0.75 / 0.16, bug 0.72 / 0.28, cause
  0.72 / 0.20, fix 0.69 / 0.26. These feed field search through embeddings, so the
  wording matters less than the words.
- **Weakest:** decision 0.55 / 0.06 and impact 0.59 / 0.04, the judgement fields.
- Compare: zero-shot GLiNER reached exact micro F1 0.41 vs DeepSeek on 13 threads; the SLM
  reaches 0.57 on 12,711 documents of every source.
- **Go criteria from the plan:** parse rate ≥ 98% ✓; names at DeepSeek's level ✓.
- **Total fine-tuning cost:** labels $1.84 + Vast $6.00 = **$7.84**.

Next: the half-B field-search test on the SLM's own fields (the held-out outputs cover half
B's candidates), then mining the corpus (≈ $9–13; paid, ask first).

## Result: field search with the SLM's fields, level 1 (22 Nov)
The "Fields as a search" test of article 007, run once per extractor on the same questions
and pool (`exploration/slm_fields/field_search_compare.py`, free and local; report
`runs/fields-r16-e2/field_search.md`): half B's 100 dev questions (55 reached by today's
search, 45 missed), pool = the 12,511 held-out documents. Each side embeds its own fields for
the question and the documents. No agent, no reranker.

| Right thread in the top … (100 questions) | 10 | 50 | 100 | 300 | Median rank |
|---|---:|---:|---:|---:|---:|
| Field search, DeepSeek's fields (all profiles) | 38 | 68 | 80 | 96 | 19 |
| Field search, SLM's fields (all profiles) | 38 | 66 | 80 | 90 | 23 |

- **The SLM's fields search as well as DeepSeek's** at the depths that matter (top 10–100,
  every profile mix within 1–6 points; one question = 1 point, so this is noise level).
- Slightly weaker deep down: missed questions in the top 300, 82% vs 93%.
- Field search still rescues what today's search misses: top 50 for the missed group 44–49%
  (SLM) vs 2% for gte on the raw text.
- 65 of 12,711 SLM outputs didn't parse (empty fields); none is a half-B question or right
  thread.

Next: level 2 (agent + reranker, small Gemini cost), or go straight to mining.
Field search vs today's hybrid search (with or without the reranker) was started and dropped
(22 Nov): the pipeline uses both searches together (`search_documents` and `search_fields`),
so ranking one against the other doesn't inform any decision. The script still prints the
hybrid rows (`hybrid_ranks`) for reference.
