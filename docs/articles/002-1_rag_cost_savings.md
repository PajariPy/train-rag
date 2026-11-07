# How to make RAG cheaper: what others do, and what fits us

Status: notes, November 2026. A companion to
[First retrieval baselines](002_first_retrieval_baselines.md), whose cost section showed that
the LLM answer step is about 90–99% of the bill.

## Where our money goes

At 50 requests per second (about 131 million queries a month), 50 requests per minute (about
2.2 million) and 10 requests per minute (about 438,000), with Gemini Flash reading 10
documents per answer:

| Part | 50 req/s, ~$/month | Share | 50 req/min, ~$/month | Share | 10 req/min, ~$/month | Share |
|---|---:|---:|---:|---:|---:|---:|
| Search (BM25, gte, Postgres, all on CPU) | ~$0.9k | <0.1% | ~$0.3k | ~1% | ~$0.3k | ~5% |
| Reranker, top 30 (GPU) | ~$27k | ~2% | ~$0.6–1.2k | ~3–5% | ~$0.6k | ~10% |
| **LLM answer, 10 documents (~13,000 input tokens)** | **~$1.43M** | **~98%** | **~$24k** | **~95%** | **~$4.8k** | **~84%** |

At 10 requests per minute the servers are mostly idle, but they run around the clock, so the
fixed costs show. One L4 for the reranker is busy about 15% of the time. A GPU that scales to
zero between requests, or a CPU-only setup without the reranker, would matter more here than
at high traffic.

The LLM bill is `(input tokens + output tokens) × price per token`. Every saving below cuts
one of those three things: how many tokens go in, how many come out, or the price per token.

## What companies do

What I found is mostly vendor blogs and engineers' own write-ups, so the percentages are
self-reported and depend heavily on the workload. The techniques themselves come up
consistently across sources.

The commonly recommended order is to measure first, then turn on provider prompt caching,
then add semantic caching, then route between models, and to compress the context throughout.

| Technique | What it does | Claimed saving | Fit for us |
|---|---|---|---|
| **Fewer, better documents** | rerank, then send fewer documents to the LLM | 60–80% of RAG costs combined with caching and routing (vendor guide); reranking is put at 3–7% of spend against 90–95% for the LLM | **Already measured.** The reranked top 5 finds more of the right documents than the unreranked top 10 (76.2% vs 71.4%), about 45% off the LLM bill |
| **Context compression** | send only the relevant passages, or drop filler tokens (LLMLingua-2) | about 50% (project claim) | **Likely our biggest lever.** Documents average about 1,300 tokens. The best chunk plus its neighbours could be about 300, cutting input from about 13k to 2–3k tokens (my estimate) |
| **Cheaper model or routing** | send easy questions to a small model and hard ones to a big one (LiteLLM, RouteLLM) | 50–70% combined with caching | **Big.** Bedrock's Nova Lite or Claude Haiku 5.5 cost 5–13× less than Gemini Flash. Our question types already suggest the split |
| **Semantic caching** | reuse an earlier answer when a new question means the same | up to ~70% on repetitive traffic (Redis, vendor) | Depends on how often real questions repeat. Answers go stale when documents change |
| **Provider prompt caching** | the repeated start of a prompt is billed at about 10% | 59–70% (ProjectDiscovery, measured) | **Small for us.** Only the instructions repeat; the retrieved documents change every question |
| **Distillation / fine-tuning** | train a small model on a big model's answers | 5–30× cheaper per success (TensorZero, vendor); about 50% (one team, an 8B model) | **Later.** Needs real traffic, weeks of work, and retraining when documents change. Small models may stay weak at combining several sources |
| **Infrastructure discounts** | savings plans, spot GPUs, batch for offline jobs | 30–70% off hardware | Only touches search and the reranker, which are under 10% of our bill |

### Notes on each

- **Prompt caching only matches the start of a prompt.** ProjectDiscovery's hit rate went from
  7% to 84% once they moved content that changes out of the start of their prompts. For us,
  the fixed instructions must come first and the retrieved documents after.
- **Semantic caching has a correctness risk.** It compares embeddings of questions and
  returns an old answer when they're close enough. A threshold that's too loose answers a
  different question; one that's too strict never hits. Lab results like a 98.5% hit rate
  come from pre-warmed caches.
- **Routing needs quality measurement per route.** It only saves money if the cheap model is
  good enough for the questions sent to it.
- **Compression has a recall risk.** Cutting a document to its best passage loses answers that
  need two distant parts of it. Our intra_document_reasoning questions test exactly that.

## LLM prices to compare

Prices per million tokens, standard tier, from a third-party summary of Amazon Bedrock pricing
(October 2026), not AWS's own page. "Per 1k answers" assumes 13,000 input and 300 output
tokens; "5 documents" assumes 6,500 input tokens. A month is about 131 million answers at 50
requests per second, about 2.2 million at 50 per minute and about 438,000 at 10 per minute.

| Model | Size | Price-implied active size (rough) | What it is | Input $/M | Output $/M | Per 1k answers, 10 docs | Per 1k answers, 5 docs | 50 req/s per month, 10 docs | 50 req/min per month, 10 docs | 10 req/min per month, 10 docs |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon Nova Micro | not published | ~1–4B | text only, 128K context; Amazon's smallest | 0.035 | 0.14 | $0.50 | $0.27 | ~$65k | ~$1.1k | ~$0.2k |
| Amazon Nova Lite | not published | ~2–6B | text, image, video; 300K context | 0.06 | 0.24 | $0.85 | $0.46 | ~$112k | ~$1.9k | ~$0.4k |
| Claude Haiku 5.5* | not published | ~3–10B | Anthropic's small tier | 0.10 | 0.50 | $1.45 | $0.80 | ~$190k | ~$3.2k | ~$0.6k |
| OpenAI gpt-oss-120b | 117B total, 5.1B active (MoE) | 5.1B (known) | open weights, 128K context | 0.15 | 0.60 | $2.13 | $1.16 | ~$280k | ~$4.7k | ~$0.9k |
| Meta Llama 4 Scout | 109B total, 17B active (MoE) | 17B (known) | open weights | 0.17 | 0.66 | $2.41 | $1.30 | ~$316k | ~$5.3k | ~$1.1k |
| Amazon Nova 2 Lite | not published | ~10–35B | Amazon's newer small tier | 0.30 | 2.50 | $4.65 | $2.70 | ~$611k | ~$10k | ~$2.0k |
| *Gemini 3.8 Flash (now)* | not published | ~25–80B | Google's mid tier | *0.75* | *3.75* | *$10.90* | *$6.00* | *~$1.43M* | *~$24k* | *~$4.8k* |
| Amazon Nova Pro | not published | ~25–80B | Amazon's mid tier; 300K context | 0.80 | 3.20 | $11.40 | $6.20 | ~$1.49M | ~$25k | ~$5.0k |
| Claude Haiku 4.5 | not published | ~35–100B | Anthropic's previous small tier | 1.00 | 5.00 | $14.50 | $7.90 | ~$1.91M | ~$32k | ~$6.4k |
| Claude Sonnet 5.5 | not published | ~70–200B | Anthropic's mid tier | 2.00 | 10.00 | $29.00 | $16.00 | ~$3.8M | ~$64k | ~$12.7k |

**On size:** only the open-weight models publish their parameter counts. Amazon, Anthropic and
Google don't, so the table gives the vendor's tier instead of guessing a number. gpt-oss-120b
and Llama 4 Scout are mixtures of experts: each token passes through only the "active"
parameters (5.1B and 17B), which is why they're cheap to run despite their total size. Size
is also a weak guide to quality: a newer small model often beats an older larger one. Our own
answer evaluation, on our questions, is the real test.

**How the price-implied size is estimated.** The cost of serving a token grows roughly with
the parameters it passes through, so I used the two open models as rulers. Per 1,000 answers,
gpt-oss-120b costs $2.13 for 5.1B active parameters (about $0.42 per billion), and Llama 4
Scout $2.41 for 17B (about $0.14 per billion). Dividing each closed model's cost per 1,000
answers by that range gives the estimate. Treat it as an order of magnitude, nothing more:

- **The two rulers disagree by 3×,** so every estimate is a 3× range.
- **Prices aren't costs.** A vendor's margin, its own chips (Google's TPUs, Amazon's
  Trainium) and how busy its servers are all move the price. A high margin makes a model look
  bigger than it is.
- **It's active size, not total.** A mixture-of-experts model can be several times larger in
  total than its active size.
- **Long-context pricing and caching** aren't modelled.

\*Haiku 5.5's price is for prompts up to 100k tokens, and it's unusually low for a Claude
model. Check it on AWS's pricing page before relying on it.

- **Batch pricing (50% off) doesn't apply**: a live system has to answer right away.
- **Halving the documents saves about 45%, not 50%.** Only the input halves; the output and
  the fixed prompt don't.
- **Gemini's Flash-Lite tier** is likely much cheaper than Flash. I haven't checked its price
  yet.

## What I'd do, in order

1. **Send passages instead of whole documents.** Use the reranked top 5, but only each
   document's matching chunk with its neighbours, using the `previous_id` and `next_id` links
   the ingestion pipeline already stores. It needs no new model or service.
2. **Compare cheaper LLMs** (Nova Lite, Claude Haiku 5.5, gpt-oss-120b, Gemini Flash-Lite,
   Gemini Flash as the reference). Give them the same 100 questions and the same retrieved
   passages, and grade them with our judge. That shows price and quality side by side, for
   cents to a few dollars per model.
3. **Route by difficulty.** Send basic and constrained questions to the cheapest model that
   holds up, and completeness and conflicting-info questions to a stronger one.
4. **Semantic caching, once there is real traffic** to measure repetition on. On the
   benchmark every question is different, so it would only show a fake win.

Steps 1 and 2 multiply. For example, cutting each answer's input from about 13k to 3k tokens
and moving from Gemini Flash to Nova Lite would take the 50-requests-per-second LLM bill from
about $1.43M to roughly $30k a month. That's arithmetic, not a result: it holds only if
answer quality holds, which is what the answer evaluation is for.

## Sources

Cost-saving techniques:

- [Towards Data Science: RAG Is Burning Money](https://towardsdatascience.com/rag-is-burning-money-i-built-a-cost-control-layer-to-fix-it/)
- [Big Data Boutique: LLM Cost Optimization Techniques](https://bigdataboutique.com/blog/llm-cost-optimization-techniques)
- [ProjectDiscovery: How We Cut LLM Costs by 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [PromptUnit: The Hidden Costs in Your RAG Pipeline](https://www.promptunit.ai/blog/rag-pipeline-hidden-costs)
- [pcables: Cut RAG Costs](https://pcables.com/cut-rag-costs-optimizing-embeddings-storage-and-context-budgets)
- [Redis: LLM Token Optimization](https://redis.io/blog/llm-token-optimization-speed-up-apps/)
- [PremAI: 8 Strategies That Cut API Spend by 80%](https://www.premai.io/blog/llm-cost-optimization-8-strategies-that-cut-api-spend-by-80-2026-guide/)
- [TianPan: LLM Routing and Model Cascades](https://tianpan.co/blog/2025-11-03-llm-routing-model-cascades)
- [Kunal Ganglani: Reduce LLM API Costs 60%](https://www.kunalganglani.com/blog/reduce-llm-api-costs-production)
- [SuperCompress: Query-Aware Compression](https://www.supercompress.dev/reduce-llm-costs)
- [awesome-llm-token-optimization](https://github.com/pleasedodisturb/awesome-llm-token-optimization)
- [DEV: 7 techniques to reduce LLM costs](https://dev.to/adamo_software/how-to-reduce-llm-costs-in-production-07-techniques-you-need-to-know-2gde)

Smaller and distilled models:

- [TensorZero: Distillation, 5–30× cheaper inference](https://www.tensorzero.com/blog/distillation-programmatic-data-curation-smarter-llms-5-30x-cheaper-inference/)
- [DEV: fine-tuned Llama 3.1 replacing RAG](https://dev.to/johalputt/we-ditched-rag-20-for-fine-tuned-llama-31-50-better-accuracy-for-internal-docs-aci)
- [Decoding ML: blueprint to cut LLM costs](https://decodingml.substack.com/p/our-blueprint-to-cut-llm-costs-by)
- [Classmethod: Can RAG be replaced by fine-tuning?](https://dev.classmethod.jp/en/articles/rag-vs-finetuning-kb-search/)
- [ITEA GENIUS project: smaller models, comparable results](https://itea4.org/news/smaller-models-comparable-results-reduced-costs-learnings-from-the-genius-project.html)
- [Sustainable AI trilemma: LLM agents and RAG (arXiv 2501.08262)](https://arxiv.org/pdf/2501.08262)

LLM prices:

- [CloudForecast: AWS Bedrock Pricing (2026)](https://www.cloudforecast.io/blog/aws-bedrock-pricing/)
- [costbench: Amazon Bedrock Pricing 2026](https://costbench.com/software/llm-api-providers/amazon-bedrock/)
- [Wring: AWS Bedrock Models guide](https://wring.co/blog/aws-bedrock-llm-models-guide)
- [cloudburn: Amazon Bedrock Pricing](https://cloudburn.io/blog/amazon-bedrock-pricing)
