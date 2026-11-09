# Small models (about 1–2B) for fact extraction

Status: research notes, 7 November 2026. Feeds the LoRA plan in article 004, section 12.

## 1. The task we need a model for

- **Input:** one document of the corpus: a Slack thread, email thread, ticket, pull request,
  meeting transcript or wiki page. Median 4,300 characters (about 1.1k tokens); Fireflies
  transcripts run to about 11k characters.
- **Output:** the document's key facts and the named things and values in it (products,
  systems, regions, people, numbers, dates, versions), as JSON.
- **Why:** the 31 clean search failures are near-duplicates that differ in one such detail
  (qst_0185: a fleet of "about eight" GPUs vs 48, 24 + 12 + 4, or 18 nodes).
- **Plan:** Gemini labels about 5–10k documents, LoRA on a small model, the small model
  extracts from all 512k documents on a rented GPU (an RTX 3090 at about $0.17/h).

So what matters, in order: following an output format exactly (JSON), extracting details
faithfully, handling 1–6k-token inputs, gaining from fine-tuning, and a licence that allows
commercial use.

## 2. Which benchmarks are relevant

No public benchmark measures "extract key facts from enterprise chat and email". The closest
proxies, and how much each tells us:

| Benchmark | Measures | Relevance to us |
|---|---|---|
| **Extraction F1 after LoRA** (merchant extraction, 2026) | Fine-tuned models pulling 10 fields from noisy text into JSON | **High:** the same kind of task, after the same kind of fine-tuning |
| **Fine-tuned rank across 8 tasks** (distil labs, 2025) | 12 models fine-tuned on 10k synthetic examples each, incl. document understanding and QA | **High** for picking a base model to fine-tune |
| **IFEval** | Following explicit format and content instructions | **Medium:** a proxy for producing clean JSON before fine-tuning |
| **News summarization study** (19 small models) | Relevance, coherence, factual consistency | Medium: faithfulness to the source, but not extraction |
| **MMLU / MMLU-Pro** | General knowledge | Low: we extract what's in the text, not what the model knows |
| **Our own pilot** (46 failures, 2,370 pool documents) | Whether extracted facts separate the near-duplicates | **The real benchmark.** Nothing above replaces it |

## 3. Comparison

Numbers are from model cards and the papers listed; sources mix settings, so compare
within a column with care. "FT extraction F1" is the merchant-extraction paper (best
checkpoint after LoRA).

| Model | Params | Released | Context | IFEval | MMLU (-Pro) | Fine-tuning evidence | Licence / commercial use |
|---|---:|---|---:|---:|---:|---|---|
| **Qwen3.5-2B** | 2B | Feb–Mar 2026 | 262k | 61.2 (non-thinking), 78.6 (thinking) | Pro 55.3 / 66.5 | **FT extraction F1 95.18** | **Apache 2.0: yes** |
| **Qwen3.5-0.8B** | 0.8B | Mar 2026 | 262k | — | — | **FT extraction F1 94.75**, "matches models 2.5–4× larger" | **Apache 2.0: yes** |
| **Granite 4.0-1B** (IBM) | 1.6B | Oct 2025 | 128k | **77.4** | 59.4 | Best base model for summarization in a small community test; no FT extraction numbers | **Apache 2.0: yes** |
| **Qwen3-1.7B** | 1.7B | Apr 2025 | 32k | 68.2 | 64.4 | Fine-tuned average rank 4.44 of 12 (best under 3B) | Apache 2.0: yes |
| **Gemma 3 1B** | 1B | Mar 2025 | 32k | 80.2 | 38.8 | FT extraction F1 93.93 | Gemma Terms of Use: yes, with Google's prohibited-use policy and redistribution conditions |
| **Llama 3.2 1B** | 1.2B | Sep 2024 | 128k | 59.5 | 49.3 | **Most tunable** (tunability rank 1 of 12) | Llama 3.2 Community Licence: yes, with attribution ("Built with Llama"), Meta's acceptable-use policy, separate licence above 700M monthly users |
| **SmolLM2-1.7B** | 1.7B | Feb 2025 | 8k | 56.7 | (19.3 reported, looks wrong) | Tunability rank 4 of 12 | Apache 2.0: yes |
| **LFM2-1.2B-Extract** (Liquid AI) | 1.2B | 2025 | 32k | — | — | **Built for extraction** (JSON/XML/YAML); only judge-based claims vs Gemma 3 27B | LFM Open Licence: **free only below $10M company revenue** |
| *Upper bound:* Qwen3-4B-Instruct-2507 | 4B | Jul 2025 | 256k | — | — | **Best fine-tuned rank** (2.25 of 12); Qwen3.5-4B FT extraction F1 96.60 | Apache 2.0: yes |

Context lengths for Qwen3-1.7B, Gemma 3 1B, Llama 3.2 1B, SmolLM2 and LFM2 are from memory of
their model cards, not re-checked in this pass.

## 4. How many training iterations

Two published recipes, both close to ours:

| | Merchant extraction (2026) | distil labs (2025) |
|---|---|---|
| Training examples | 6,125 (labelled) | 10,000 (synthetic, from a teacher LLM) |
| Epochs | 6 | 4 |
| Steps | **2,298** at effective batch 16 | not stated (about 2,500 at batch 16) |
| LoRA rank / alpha | 8 / 16 (rank 32 gained ≤ 0.2 F1) | 64 |
| Learning rate | 1e-4, cosine, 10% warmup | 5e-5, linear |
| Checkpoints | every 200 steps, best kept | — |
| Result | 0.8B–2B within about 2 F1 points of an 8B model | Fine-tuned students match or beat the teacher on 7 of 8 tasks |

**For us:** about **5–10k Gemini-labelled documents, 3–6 epochs, LoRA rank 8–16, about
2,000–4,000 steps**, with a checkpoint every 200 steps scored on held-out labelled documents.
Bigger ranks bought almost nothing in the merchant study. "Fine-tuning matters more than base
model choice" is distil labs' main finding.

## 5. Decision (7 November 2026)

**We start with Qwen3.5-0.8B.** It is cheaper to train and to serve over 512k documents than
the 2B, a training iteration takes only a few hours, and in the extraction study it was only
0.4 F1 points behind the 2B. Other models (Qwen3.5-2B, Granite 4.0-1B, a 4B) stay as later
experiments.

**Caveat found later the same day:** the Qwen3.5 small models are vision-language models
("Causal Language Model with Vision Encoder") with a new hybrid attention (Gated DeltaNet),
loaded with `AutoModelForMultimodalLM` and only supported by `transformers` from the main
branch. LoRA on that architecture is new ground. The text-only Qwen3-0.6B / Qwen3-1.7B
(Apache 2.0, mature tooling) are the fallback; see section 6.

**Updated decision: Qwen3.5-2B.** Chosen over the 0.8B for the best published extraction result
at this size (95.2 F1 after LoRA) and a 262k context. Its vision encoder is accepted on purpose:
it could handle visual data (PDFs, screenshots) later. LoRA goes on the text layers only, the
vision part stays frozen, and vLLM serves it text-only (`--language-model-only`). A short
training test on the laptop decides whether the tooling works; if not, Qwen3-1.7B on the same
code.

## 6. How the training data is fed to the model

Fine-tuning teaches the small model to do exactly what DeepSeek did, with the same prompt.
Each training example is one chat with three messages, the same ones DeepSeek received plus
its answer:

| Message | Content | Source |
|---|---|---|
| **system** | the extraction instructions, word for word | `fact_extraction_instructions` in `exploration/fact_extraction/prompts.yaml` |
| **user** | `Document ({source}: {title}):` then the document's normalized text, cut at 30,000 characters | the corpus, normalized as for indexing |
| **assistant** | the JSON label: `about`, `entities`, `facts` | DeepSeek's answer, in `exploration/fact_extraction/data/labels.json` |

A real example (qst_0185's expected document, shortened):

```text
system:    You read one internal company document (a chat thread, email thread, ticket,
           pull request, meeting transcript or wiki page) and list what it says, for a
           search index. ... Return JSON with three fields: about, entities, facts ...

user:      Document (slack: eng-platform):

           Amanda: we need to lock Q3 reserved capacity before ...
           Karthik: proposal — commit 40% us-east, 40% eu-west, 20% ap-southeast,
           plus ~10% global burst on spot ...
           ...

assistant: {"about": "Slack thread in eng-platform discussing Q3 reserved GPU capacity
            allocation across regions, comparing committed pricing, egress, availability
            and SLO tradeoffs.",
            "entities": ["Amanda", "Karthik", "Maria", "Eli", "Julian", "infra-bot",
                         "billing-bot", "us-east", "eu-west", "ap-southeast", "A100", ...],
            "facts": ["Karthik proposes committing 40% capacity in us-east, 40% in eu-west,
                       20% in ap-southeast.",
                      "Karthik proposes a 10% global burst pool on low-cost spot-ish
                       capacity for short surges.",
                      "us-east committed price is 17.0 dollars per GPU-hour vs on-demand
                       24.0, about 29% savings.", ...]}
```

How the trainer uses it:

1. **One token sequence.** The model's chat template turns the three messages into one
   sequence: `<system> instructions <user> document <assistant> JSON <end>`. Median length
   about 1.1k tokens of document plus about 370 of JSON; sequences are capped at about
   4–6k tokens.
2. **Input versus target.** The model reads the whole sequence, but the loss is computed
   **only on the assistant part** (the JSON and the end token). It learns to write the
   label given the instructions and the document; it is not trained to reproduce the
   document or the instructions.
3. **LoRA.** The base model's weights stay frozen; small adapter matrices (rank 16) on its
   linear layers are the only weights that change.
4. **Split.** 2,700 of the 3,000 labelled documents train; 300, stratified by source, are
   held out to check the result (valid JSON rate, and how close the facts and entities are
   to DeepSeek's).

At extraction time the same system and user messages go in, and the model writes the JSON
(greedy decoding, no thinking), which is parsed and stored per document. The 2,370 pilot
documents are never trained on: they test whether the extracted facts separate the
near-duplicate documents of the 46 search failures.

## 7. Recommendation (before the decision)

1. **Qwen3.5-2B as the main candidate.** Apache 2.0, the strongest fine-tuned extraction
   evidence at this size (95.18 F1), and a 262k context, so long transcripts fit. Train and
   run it in **non-thinking mode**: the card warns thinking mode can loop.
2. **Qwen3.5-0.8B as the cheap option.** Only 0.4 F1 points behind the 2B in the extraction
   study, and about 2–3× faster to run over 512k documents.
3. **Granite 4.0-1B as the alternative.** Apache 2.0, the best instruction-following score
   (IFEval 77.4) among the clearly-licensed models, and a 128k context.
4. **Avoid for a product:** LFM2 (revenue cap), and prefer Apache-licensed models over Gemma
   and Llama, whose custom terms allow commercial use but add conditions to track.
5. If the 2B falls short, Qwen3-4B / Qwen3.5-4B fits a rented 24 GB 3090 with plain LoRA.

## 8. Gaps

- No benchmark here is our task. The pilot on the 46 failures decides.
- Several numbers come from secondary sources (blogs, a jailbreak paper's table); the model
  cards are the primary source for IFEval and licences.
- Licences should get a legal check before any production use.

## Sources

- [How Small Can You Go? LoRA fine-tuning 270M–8B models for merchant information extraction (arXiv 2606.08051)](https://arxiv.org/html/2606.08051v1)
- [distil labs: we benchmarked 12 small language models across 8 tasks to find the best base model for fine-tuning](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning)
- [Evaluating small language models for news summarization (arXiv 2502.00641)](https://arxiv.org/abs/2502.00641)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Granite 4.0-1B model card](https://huggingface.co/ibm-granite/granite-4.0-1b)
- [LFM2-1.2B-Extract model card](https://huggingface.co/LiquidAI/LFM2-1.2B-Extract/blob/main/README.md)
- [Qwen3 technical report (arXiv 2505.09388)](https://arxiv.org/pdf/2505.09388)
- [Can small language models reliably resist jailbreak attacks? (arXiv 2503.06519), source of the instruct IFEval/MMLU table](https://arxiv.org/pdf/2503.06519)
- [SmolLM2 paper (arXiv 2502.02737)](https://arxiv.org/pdf/2502.02737)
- [lillm-summarizer: small-model summarization and LoRA experiments](https://github.com/nchapman/lillm-summarizer)
- [Low-resource fine-tuning for multi-task structured information extraction with a 1B model (arXiv 2509.08381)](https://arxiv.org/pdf/2509.08381)
- [Qwen 3.5 small series licence and release notes](https://dev.to/alanwest/qwen-35-small-four-models-zero-api-cost-a-quick-benchmark-50f7)
- [Local LLM commercial licence guide](https://www.promptquorum.com/pt/local-llms/local-llm-commercial-license-guide-business)
- [Gemma licence discussion on Hugging Face](https://hface.pages.dev/google/gemma-7b/discussions/122)
