"""Write the fields for every held-out document with the merged SLM in vLLM (runs on the GPU).

Input: prompts.jsonl from vast_score.sh, first line {"system": ...}, then one
{"doc_id", "source", "text"} per document. Greedy decoding, thinking off, the same chat the
training used. A document whose prompt is longer than PROMPT_TOKENS is cut from its end
(training cut documents the same way to fit 6,144 tokens with the answer).

Writes outputs.jsonl ({"doc_id", "source", "text", "finish", "cut"} per document, appended
chunk by chunk so a restart resumes) and logs "progress docs X/Y" for the progress page.
Stops early if the first chunk mostly fails to parse: the merge went wrong, don't pay for
the rest.

Usage (on the machine): python generate_vllm.py merged prompts.jsonl outputs.jsonl
"""

import json
import sys
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT_TOKENS = 5_000
MAX_NEW_TOKENS = 3_000
CHUNK = 512


def main() -> None:
    """Generate in chunks and append the outputs."""
    model, prompts_path, out_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    lines = prompts_path.read_text().split("\n")
    system = json.loads(lines[0])["system"]
    docs = [json.loads(line) for line in lines[1:] if line.strip()]
    done = set()
    if out_path.exists():
        done = {json.loads(line)["doc_id"] for line in out_path.read_text().split("\n") if line}
    todo = [d for d in docs if d["doc_id"] not in done]

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=PROMPT_TOKENS + MAX_NEW_TOKENS + 64,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tokenizer = llm.get_tokenizer()

    def prompt(text: str) -> tuple[str, bool]:
        cut = False
        for _ in range(8):
            chat = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            over = len(tokenizer.encode(chat)) - PROMPT_TOKENS
            if over <= 0:
                return chat, cut
            text, cut = text[: max(0, len(text) - over * 4 - 100)], True
        return chat, cut

    params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    start = time.time()
    print(f"progress docs {len(done)}/{len(docs)}", flush=True)
    for first in range(0, len(todo), CHUNK):
        chunk = todo[first : first + CHUNK]
        built = [prompt(d["text"]) for d in chunk]
        outputs = llm.generate([chat for chat, _ in built], params, use_tqdm=False)
        rows = [
            {
                "doc_id": d["doc_id"], "source": d["source"], "text": o.outputs[0].text,
                "finish": o.outputs[0].finish_reason, "cut": cut,
            }
            for d, (_, cut), o in zip(chunk, built, outputs, strict=True)
        ]
        with out_path.open("a") as out:
            out.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        if first == 0 and not done:
            valid = 0
            for r in rows:
                try:
                    json.loads(r["text"])
                    valid += 1
                except json.JSONDecodeError:
                    pass
            print(f"first chunk: {valid}/{len(rows)} parse", flush=True)
            if valid < 0.5 * len(rows):
                print("most outputs don't parse: stopping", flush=True)
                sys.exit(2)
        finished = len(done) + first + len(chunk)
        print(f"progress docs {finished}/{len(docs)} ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
