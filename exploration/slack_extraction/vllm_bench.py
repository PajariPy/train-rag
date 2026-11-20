"""Measure vLLM's speed for Qwen3.5-2B on the Slack extraction job (runs on the rented GPU).

1,000 real Slack threads, each with the extraction instructions in front (label.py), as
chat prompts with thinking off. The output length is forced (ignore_eos) to the sizes we
expect: 1,400 tokens (extended schema) and 700 (the first schema, measured with DeepSeek).
Speed doesn't depend on a LoRA fine-tune, so the base model stands in for the trained one.

Writes bench_result.json next to itself.

Usage (on the machine, from vllm_bench.sh):
    python vllm_bench.py bench_prompts.json
"""

import json
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3.5-2B"
LENGTHS = (1400, 700)


def main() -> None:
    """Load the model, run both output lengths and write the timings."""
    data = json.loads(Path(sys.argv[1]).read_text())
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": data["system"]}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in data["texts"]
    ]
    input_tokens = sum(len(tokenizer.encode(prompt)) for prompt in prompts)
    llm.generate(prompts[:8], SamplingParams(max_tokens=16))  # warm up
    results = []
    for length in LENGTHS:
        params = SamplingParams(max_tokens=length, ignore_eos=True, temperature=0)
        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        seconds = time.perf_counter() - start
        output_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        results.append({
            "output_length": length,
            "documents": len(prompts),
            "seconds": round(seconds, 1),
            "documents_per_second": round(len(prompts) / seconds, 2),
            "output_tokens_per_second": round(output_tokens / seconds),
            "input_tokens": input_tokens,
        })
        print(json.dumps(results[-1]), flush=True)
    Path(__file__).with_name("bench_result.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
