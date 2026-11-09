r"""Write facts with the fine-tuned model: Qwen3.5-2B plus a LoRA adapter from train.py.

Reads the chat examples of a JSONL file from build_dataset.py (by default the held-out
documents, which the model never saw in training), lets the model write the JSON for each,
and saves the outputs as {doc_id: {about, entities, facts}}, the shape judge_labels.py reads
with --labels. Each output is saved as soon as it is written, so a rerun continues where the
last one stopped. Output that doesn't parse is kept under "raw" with empty lists.

Runs without the database. Usage:
    uv run --no-project --with "transformers @ git+https://github.com/huggingface/transformers" \
        --with peft --with accelerate --with torch --with torchvision --with pillow \
        python generate.py --adapter runs/qwen35-2b-r16-e3/adapter \
        --data data/sft_eval.jsonl --out data/outputs_qwen35-2b-r16-e3.json [--limit N]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"


def main() -> None:
    """Generate and save the facts of every document in the file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1_000)
    parser.add_argument("--max-prompt-tokens", type=int, default=3_500)
    args = parser.parse_args()

    examples = [
        json.loads(line) for line in args.data.read_text().split("\n") if line.strip()
    ][: args.limit]
    outputs = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [e for e in examples if e["doc_id"] not in outputs]
    print(f"{len(examples)} documents, {len(outputs)} already done, {len(todo)} to go", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()

    start = time.time()
    for done, example in enumerate(todo, 1):
        messages = example["messages"][:2]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ids.shape[1] > args.max_prompt_tokens:
            # Shorten only the document, as in training, so the instructions stay whole.
            over = ids.shape[1] - args.max_prompt_tokens
            user = messages[1]["content"]
            messages = [messages[0], {**messages[1], "content": user[: len(user) - over * 4 - 100]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            output = model.generate(
                ids.to(model.device), max_new_tokens=args.max_new_tokens, do_sample=False
            )
        text = tokenizer.decode(output[0][ids.shape[1] :], skip_special_tokens=True).strip()
        try:
            parsed = json.loads(text)
            label = {
                key: parsed.get(key, [] if key != "about" else "")
                for key in ("about", "entities", "facts")
            }
        except (json.JSONDecodeError, AttributeError):
            label = {"about": "", "entities": [], "facts": [], "raw": text}
        outputs[example["doc_id"]] = {**label, "source": example["source"]}
        args.out.write_text(json.dumps(outputs, indent=1, ensure_ascii=False))
        per_doc = (time.time() - start) / done
        print(
            f"{done}/{len(todo)} {example['doc_id'][:14]} {ids.shape[1]} tokens in, "
            f"{per_doc:.1f} s/doc, about {per_doc * (len(todo) - done) / 60:.0f} min left",
            flush=True,
        )


main()
