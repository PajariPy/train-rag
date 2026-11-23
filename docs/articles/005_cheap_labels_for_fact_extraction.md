# Cheap labels for a small fact-extraction model

Status: notes and decision, 7 November 2026. Follows
[Wrong answers after the judge](004_wrong_answers_after_the_judge.md) and the model choice in
[`docs/brainstorm/002_small_models_for_fact_extraction.md`](../brainstorm/002_small_models_for_fact_extraction.md).

## The plan, and the bill that worried me

The search failures in post 004 are near-duplicate documents that differ in one detail: a
fleet of "about eight" GPUs against 48, a 1-year term against 3 years. The plan is to extract
those details from every document with a small model:

1. a bigger LLM labels a training set: each document's key facts, names and values, as JSON;
2. LoRA on **Qwen3.5-0.8B** (Apache 2.0, chosen for cheap training and serving);
3. the small model extracts from all 512k documents on a rented GPU (an RTX 3090 on Vast.ai
   at about $0.17/h, about $2–3 for the whole corpus).

The GPU part is cheap. The labels were not: **about $27** for 10,000 documents with our
Gemini 3.8 Flash ($0.75 per million input tokens, $3.75 per million output tokens), at about
1.6k input and 400 output tokens per document.

## Four levers

| Lever | What changes | Effect |
|---|---|---|
| Fewer examples | Start with 3k documents, not 10k; a similar LoRA extraction study trained on 6k, and small models on narrow format tasks stop gaining early | cost about ÷ 3; add more only if the learning curve still climbs |
| Compact output | Terse JSON of entities and values, about 150 tokens instead of 400 | output cost more than halves; easier for a 0.8B model to learn |
| Batch or off-peak pricing | Gemini batch, Qwen batch, DeepSeek off-peak | about −50% |
| A cheaper labelling model | A Chinese API, or an open model on a rented GPU | **about 5–50× cheaper** |

## Labelling model prices

Per 1M tokens, from aggregator sites (they disagree; the official pricing pages decide).
Costs assume about 1.6k input tokens per document.

| Model | Input / output | 3k docs, compact JSON | 10k docs, full facts | Training on its outputs |
|---|---|---:|---:|---|
| Gemini 3.8 Flash (current) | $0.75 / $3.75 | ≈ $5.30 | ≈ $27 | restricted for "competing models"; unclear for us |
| **DeepSeek V4.1 Flash, off-peak** | $0.15 / $0.60 | **≈ $1.00** | ≈ $4.80 | **explicitly allowed** |
| DeepSeek V4.1 Flash, peak | $0.30 / $1.20 | ≈ $2.00 | ≈ $9.60 | explicitly allowed |
| Qwen3.8-flash, batch | $0.15 / $0.47, −50% | ≈ $0.45 | ≈ $2.10 | not checked |
| qwen3.7-flash (≤ 32k), batch | $0.03 / $0.13, −50% | ≈ $0.10 | ≈ $0.50 | not checked |
| GLM-4.5-Flash (Zhipu) | free, rate-limited | $0 | $0 | not checked |
| MiniMax M2.x | $0.30 / $1.20 | ≈ $2.00 | ≈ $9.60 | not checked |
| Open Qwen teacher on a rented 4090 | GPU time | ≈ $0.50 | ≈ $1–2 | Apache 2.0: no restriction |

DeepSeek's off-peak price applies outside 01:00–04:00 and 06:00–10:00 UTC on weekdays.

## What else counts besides price

- **Terms.** DeepSeek's Open Platform terms (January 2025) let users apply outputs to
  "training other models (such as model distillation)". That is the clearest position among
  the APIs for a model we might later use in a product. A third-party site reports an
  anti-distillation clause in DeepSeek's *open-weights* licence; it is unverified and concerns
  the weights, not the API.
- **Where the data goes.** Requests go to servers in China (Qwen's international endpoint is
  in Singapore). Fine for this synthetic, CC-licensed benchmark; not for real company
  documents.
- **Label quality.** A strong API model should label at least as well as a local 9B teacher.
  A Gemini spot-check of about 200 labels (about $0.50) measures it either way.

## Decision

**Label with DeepSeek V4.1 Flash, off-peak, in compact JSON, starting with 3,000 documents
(about $1).** The 3,000 are sampled by source and include the 2,370 documents of the failure
pools from post 004, so the same labels also run the pilot: do extracted facts separate the
near-duplicates in our 46 failures? Then LoRA on Qwen3.5-0.8B, and more labels only if the
learning curve asks for them.

## Sources

- [DeepSeek API pricing, November 2026 (BenchLM)](https://benchlm.ai/deepseek/api-pricing)
- [DeepSeek pricing breakdown (TechJack Solutions)](https://techjacksolutions.com/ai-tools/deepseek/deepseek-pricing/)
- [DeepSeek API pricing, October 2026 (Morph)](https://www.morphllm.com/deepseek-api)
- [DeepSeek Open Platform Terms of Service](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html)
- [DeepSeek model licence anti-distillation clause (ConductAtlas, third party)](https://conductatlas.com/platform/deepseek/deepseek-model-license/anti-distillation-clause/)
- [Qwen API pricing, November 2026 (BenchLM)](https://benchlm.ai/alibaba/api-pricing)
- [Alibaba Cloud Model Studio pricing capture (usagepricing)](https://www.usagepricing.com/blueprint/alibaba-cloud-model-studio)
- [Z.ai GLM API free plan (CostBench)](https://costbench.com/software/llm-api-providers/glm-api/free-plan/)
- [MiniMax M2.7 API pricing (DEV Community)](https://dev.to/owen_fox/minimax-m27-api-pricing-2026-free-tier-setup-and-how-it-stacks-against-deepseek-and-kimi-34)
- [Chinese LLM pricing deep dive (JadeGate)](https://jadegate-web.pages.dev/blog/chinese-llm-pricing-deep-dive)
- [How Small Can You Go? LoRA fine-tuning 270M–8B models for merchant information extraction](https://arxiv.org/html/2606.08051v1)
