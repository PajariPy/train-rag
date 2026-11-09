r"""LoRA fine-tuning of Qwen3.5-2B to write a document's facts as JSON, like DeepSeek did.

Reads the chat examples from build_dataset.py (no database needed, so it runs on a rented
GPU). Each example becomes one token sequence through the model's own chat template:
instructions and document as the prompt (with the empty thinking block of non-thinking mode,
exactly what the model sees when it generates), then the JSON and the end-of-turn token as the
target. The loss is computed on the target only. A document too long for MAX_TOKENS is cut so
the JSON always fits. LoRA goes on the text model's linear layers; the vision encoder and the
output layer stay frozen.

After training, the model writes the JSON for EVAL_GENERATE held-out documents, and the run
logs how many parse, and how close their facts and entities are to DeepSeek's. Everything is
logged to Weights & Biases (project train-rag-facts) and to train.log.

Usage:
    python train.py --data DIR --out DIR [--limit N] [--max-steps N] [--max-tokens N]
        [--batch N] [--accum N] [--generate N] [--name NAME]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL = "Qwen/Qwen3.5-2B"
PROJECT = "train-rag-facts"
IGNORE = -100
WORD = re.compile(r"\w+")
log = logging.getLogger("train")


def read_jsonl(path: Path, limit: int | None) -> list[dict]:
    r"""The examples in a JSONL file, the first `limit` of them if given.

    Split on "\n" only: `splitlines()` also splits on Unicode separators (U+2028 and others)
    that can occur inside a document's text, which cuts a record in two.
    """
    lines = path.read_text().split("\n")

    return [json.loads(line) for line in lines[:limit] if line.strip()]


def prompt_text(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    """The system and user messages as the model sees them before it answers."""
    return tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode(
    tokenizer: AutoTokenizer, example: dict, max_tokens: int
) -> dict[str, list[int]] | None:
    """One example as input ids and labels: the prompt is masked, the JSON is learned.

    If prompt and answer don't fit in `max_tokens`, only the document is shortened (from its
    end) and the chat rebuilt, so the instructions, the assistant header and the JSON stay
    whole. Returns None if even an empty document wouldn't fit.
    """
    messages = example["messages"]
    target_ids = tokenizer(messages[2]["content"] + "<|im_end|>\n", add_special_tokens=False)[
        "input_ids"
    ]
    user = messages[1]["content"]
    for _attempt in range(6):
        chat = [messages[0], {**messages[1], "content": user}]
        prompt_ids = tokenizer(prompt_text(tokenizer, chat), add_special_tokens=False)["input_ids"]
        over = len(prompt_ids) + len(target_ids) - max_tokens
        if over <= 0:
            return {
                "input_ids": prompt_ids + target_ids,
                "labels": [IGNORE] * len(prompt_ids) + target_ids,
            }
        if not user:
            return None
        # About 4 characters per token, plus a margin; repeat until it fits.
        user = user[: max(0, len(user) - over * 4 - 100)]

    return None


class Collator:
    """Pads a batch to its longest example; padding is masked in the loss."""

    def __init__(self, pad_id: int) -> None:
        """Remember the padding token."""
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Stack one batch."""
        width = max(len(item["input_ids"]) for item in batch)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], IGNORE) for b in batch]),
            "attention_mask": torch.tensor(
                [pad([1] * len(b["input_ids"]), 0) for b in batch]
            ),
        }


def overlap(predicted: list[str], expected: list[str]) -> float:
    """Word-level F1 between two lists of strings, joined; 1.0 when both are empty."""
    got = set(WORD.findall(" ".join(predicted).lower()))
    want = set(WORD.findall(" ".join(expected).lower()))
    if not got and not want:
        return 1.0
    if not got or not want:
        return 0.0
    precision, recall = len(got & want) / len(got), len(got & want) / len(want)

    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@torch.no_grad()
def check_outputs(model, tokenizer, examples: list[dict], max_new_tokens: int = 1_000) -> dict:  # noqa: ANN001
    """Let the model write the JSON for held-out documents and compare it with DeepSeek's."""
    model.eval()
    parsed, fact_f1, entity_f1, samples = 0, [], [], []
    for example in examples:
        prompt = tokenizer(prompt_text(tokenizer, example["messages"]), return_tensors="pt")
        output = model.generate(
            **prompt.to(model.device), max_new_tokens=max_new_tokens, do_sample=False
        )
        text = tokenizer.decode(output[0][prompt["input_ids"].shape[1] :], skip_special_tokens=True)
        expected = json.loads(example["messages"][2]["content"])
        try:
            got = json.loads(text)
            parsed += 1
            fact_f1.append(overlap(got.get("facts", []), expected["facts"]))
            entity_f1.append(overlap(got.get("entities", []), expected["entities"]))
        except (json.JSONDecodeError, AttributeError):
            fact_f1.append(0.0)
            entity_f1.append(0.0)
        if len(samples) < 5:
            samples.append([example["doc_id"], example["source"], text[:2000]])

    return {
        "check/json_valid": parsed / len(examples),
        "check/fact_word_f1": sum(fact_f1) / len(examples),
        "check/entity_word_f1": sum(entity_f1) / len(examples),
        "samples": samples,
    }


def main() -> None:
    """Train, save the adapter, and check it on held-out documents."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None, help="examples per split")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-tokens", type=int, default=4_096)
    # Batch 1: the output scores alone (sequence x 248k vocabulary x 4 bytes) are about 4 GB
    # per 4k-token example, so a batch of 4 needs about 16 GB just for them.
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--generate", type=int, default=50, help="held-out docs to generate")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(args.out / "train.log"), logging.StreamHandler()],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train = read_jsonl(args.data / "sft_train.jsonl", args.limit)
    held_out = read_jsonl(args.data / "sft_eval.jsonl", args.limit)
    encoded_train = [encode(tokenizer, e, args.max_tokens) for e in train]
    encoded_eval = [encode(tokenizer, e, args.max_tokens) for e in held_out]
    train_set = [e for e in encoded_train if e is not None]
    eval_set = [e for e in encoded_eval if e is not None]
    skipped = len(encoded_train) - len(train_set) + len(encoded_eval) - len(eval_set)
    lengths = sorted(len(e["input_ids"]) for e in train_set)
    log.info(
        "examples: train %d, eval %d, skipped (too long even without document) %d; "
        "tokens median %d, max %d",
        len(train_set), len(eval_set), skipped, lengths[len(lengths) // 2], lengths[-1],
    )
    steps_per_epoch = -(-len(train_set) // (args.batch * args.accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)

    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False
    names = {
        n.split(".")[-1]
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "visual" not in n and not n.endswith("lm_head")
    }
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        # Text layers only: any module path without "visual", ending in one of the names.
        target_modules=r"^(?!.*visual).*\.(" + "|".join(sorted(names)) + r")$",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training = TrainingArguments(
        output_dir=str(args.out),
        run_name=args.name,
        report_to="wandb",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * total_steps)),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        train_sampling_strategy="group_by_length",  # less padding per batch
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=Collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    model.enable_input_require_grads()
    start = time.time()
    trainer.train()
    log.info("trained in %.1f min", (time.time() - start) / 60)
    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")

    model.config.use_cache = True
    result = check_outputs(model, tokenizer, held_out[: args.generate])
    samples = result.pop("samples")
    log.info("held-out check: %s", result)
    for doc_id, source, text in samples:
        log.info("sample %s (%s): %s", doc_id, source, text)
    import wandb  # noqa: PLC0415 - only once the trainer has started the run

    if wandb.run is not None:
        wandb.log(result)
        wandb.log({"check/samples": wandb.Table(["doc_id", "source", "output"], samples)})
        wandb.finish()


main()
