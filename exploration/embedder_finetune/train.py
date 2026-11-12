r"""Fine-tune an embedder (gte-modernbert-base by default) on our questions with hard negatives.

Design: docs/brainstorm/005_synthetic_training_questions.md (data), 004 (the negatives).

Data (from export.py, no database needed, so it also runs on a rented GPU):
- training_set.jsonl: question, positive (its source document) and 5 negatives picked
  below the reranker's biggest score drop;
- dev_set.jsonl: the same for 422 dev questions, plus every candidate the search found;
- documents.jsonl: the text of each document, as production embedded it.

Training: sentence-transformers with MultipleNegativesRankingLoss in its cached form
(GradCache), so a large batch fits a small GPU. Each question is pulled towards its
document and pushed away from its 5 negatives and every other document in the batch. The
batch sampler keeps duplicate texts out of a batch (a duplicate would be a false negative).

Dev check, before and after: for each dev question, rank its right document among its own
search candidates (the fused top 50 plus the right document) by cosine. Reports hit@1, @5,
@10 and MRR, overall and per question type. No re-embedding of the corpus needed. The
benchmark is not touched here: it is for the final test only.

Everything is logged to Weights & Biases (project train-rag-embedder) and train.log; when
the train_rag package and its database are reachable, also to the progress page.

Usage:
    python train.py --data DIR --out DIR [--limit N] [--max-steps N] [--epochs N]
        [--batch N] [--mini-batch N] [--max-tokens N] [--encode-batch N] [--lr F]
        [--eval-questions N]
        [--dev-limit N] [--model NAME] [--qwen-prompt] [--grad-checkpointing]
        [--name NAME] [--no-wandb]

    # smoke test on the laptop:
    uv run python exploration/embedder_finetune/train.py \\
        --data exploration/embedder_finetune/data --out exploration/embedder_finetune/runs/smoke \\
        --limit 64 --max-steps 4 --max-tokens 1024 --eval-questions 10 --dev-limit 20 \\
        --no-wandb
"""

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

if TYPE_CHECKING:
    from train_rag.progress.recorder import RunProgress

MODEL = "Alibaba-NLP/gte-modernbert-base"
# Qwen3-Embedding reads questions with an instruction in front (its documents get none);
# the same text is used in training and in the dev checks.
QWEN_QUERY_PROMPT = (
    "Instruct: Given a question about a company's internal documents (chats, emails, "
    "tickets, pages, meetings), retrieve the document that answers it\nQuery:"
)
PROJECT = "train-rag-embedder"
PRODUCTION_TOKENS = 8192  # what the index uses; the before/after check always uses it
NEGATIVES = 5
SEED = 7
log = logging.getLogger("train")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    r"""The rows of a JSONL file.

    Split on "\n" only: `splitlines()` also splits on Unicode separators that can occur
    inside a document's text.

    Args:
        path: The file.

    Returns:
        One dict per line.
    """
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def training_dataset(rows: list[dict[str, Any]], texts: dict[str, str]) -> Dataset:
    """The (question, positive, negative_1..5) table the loss reads.

    Args:
        rows: Training rows.
        texts: doc_id -> document text.

    Returns:
        One row per question whose documents all have a text.
    """
    columns: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        doc_ids = [row["positive"], *row["negatives"][:NEGATIVES]]
        if len(doc_ids) != NEGATIVES + 1 or any(doc_id not in texts for doc_id in doc_ids):
            continue

        columns["anchor"].append(row["question"])
        columns["positive"].append(texts[row["positive"]])
        for index, doc_id in enumerate(doc_ids[1:], 1):
            columns[f"negative_{index}"].append(texts[doc_id])

    log.info("training examples: %d of %d rows", len(columns["anchor"]), len(rows))

    return Dataset.from_dict(dict(columns))


class CandidateRankingEvaluator(SentenceEvaluator):
    """Rank each dev question's right document among its own search candidates."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        texts: dict[str, str],
        name: str,
        max_tokens: int | None = None,
        batch_size: int = 4,
        query_prompt: str | None = None,
    ) -> None:
        """Keep the questions and the candidate texts.

        Args:
            rows: Dev rows, each with its candidates in `scores`.
            texts: doc_id -> document text.
            name: Prefix of the metric names.
            max_tokens: Read documents up to this many tokens (None: the model's setting).
            batch_size: Texts embedded at once.
            query_prompt: Text put before each question (None: the question alone).
        """
        super().__init__()
        self.rows = [row for row in rows if row["positive"] in texts]
        self.texts = texts
        self.name = name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.query_prompt = query_prompt
        self.primary_metric = "mrr"

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        """Embed questions and candidates, rank, and score.

        Args:
            model: The model being trained.
            output_path: Unused.
            epoch: Unused.
            steps: Unused.

        Returns:
            hit@1, hit@5, hit@10 and MRR overall, and hit@10 per question type.
        """
        # 1. Embed every candidate once (at production length for the full check) and the questions.
        doc_ids = sorted(
            {doc_id for row in self.rows for doc_id in row["scores"] if doc_id in self.texts}
        )
        trained_length = model.max_seq_length

        if self.max_tokens is not None:
            model.max_seq_length = self.max_tokens

        try:
            documents = model.encode(
                [self.texts[doc_id] for doc_id in doc_ids],
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            questions = model.encode(
                [row["question"] for row in self.rows],
                prompt=self.query_prompt,
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        finally:
            model.max_seq_length = trained_length

        # 2. Rank each right document among its own question's candidates by cosine.
        position = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        ranks: dict[str, list[int]] = defaultdict(list)

        for row, question in zip(self.rows, questions, strict=True):
            candidates = [position[doc_id] for doc_id in row["scores"] if doc_id in position]
            similarity = documents[candidates] @ question
            right = similarity[candidates.index(position[row["positive"]])]
            rank = int((similarity > right).sum()) + 1

            ranks["all"].append(rank)
            ranks[row["question_type"]].append(rank)

        # 3. Turn ranks into hit@k and MRR, overall and per question type.
        metrics: dict[str, float] = {}

        for group, values in ranks.items():
            if group == "all":
                for cut in (1, 5, 10):
                    metrics[f"hit@{cut}"] = sum(rank <= cut for rank in values) / len(values)
                metrics["mrr"] = sum(1 / rank for rank in values) / len(values)
            else:
                metrics[f"{group}_hit@10"] = sum(rank <= 10 for rank in values) / len(values)

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)

        return metrics


class ProgressCallback(TrainerCallback):
    """Report each optimizer step to the progress page."""

    def __init__(self, progress: "RunProgress") -> None:
        """Keep the open progress record.

        Args:
            progress: An entered `RunProgress`.
        """
        self.progress = progress

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Advance by one step.

        Args:
            args: Training arguments.
            state: Trainer state (holds the latest loss).
            control: Trainer control.
            **kwargs: Unused.
        """
        losses = [entry["loss"] for entry in state.log_history if "loss" in entry]

        note = f"loss {losses[-1]:.3f}" if losses else None
        self.progress.advance("train", 1, note=note)


def open_progress(name: str, total: int) -> "RunProgress | None":
    """Open a progress-page record if the train_rag database is reachable.

    Args:
        name: Run name shown on the page.
        total: Total optimizer steps.

    Returns:
        An entered `RunProgress`, or None on a machine without the database.
    """
    try:
        from train_rag.database import get_engine  # noqa: PLC0415 - optional on a rented GPU
        from train_rag.progress.recorder import RunProgress  # noqa: PLC0415

        return RunProgress(get_engine(), name, total, unit="steps").__enter__()
    except Exception as error:  # noqa: BLE001 - no database is fine, just no page
        log.info("no progress page: %s", error)

        return None


def main() -> None:
    """Check before, train, check after, save."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only this many training rows")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32, help="questions per optimizer step")
    parser.add_argument("--mini-batch", type=int, default=2, help="texts per forward pass")
    parser.add_argument("--max-tokens", type=int, default=PRODUCTION_TOKENS)
    parser.add_argument("--encode-batch", type=int, default=4, help="texts at once in dev checks")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval-questions", type=int, default=100, help="dev subset in training")
    parser.add_argument("--dev-limit", type=int, default=None, help="smoke test: fewer dev rows")
    parser.add_argument("--model", default=MODEL, help="base model to fine-tune")
    parser.add_argument("--qwen-prompt", action="store_true", help="Qwen3's query instruction")
    parser.add_argument("--grad-checkpointing", action="store_true", help="less memory, slower")
    parser.add_argument("--name", default="gte-mnrl-hn")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # 1. Logging: console, train.log, Weights & Biases.
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "train.log")],
    )
    os.environ.setdefault("WANDB_PROJECT", PROJECT)

    # 2. Data: document texts, training examples, dev questions (full and a quick subset).
    texts = {row["doc_id"]: row["text"] for row in read_jsonl(args.data / "documents.jsonl")}
    train_rows = read_jsonl(args.data / "training_set.jsonl")[: args.limit]
    dev_rows = read_jsonl(args.data / "dev_set.jsonl")[: args.dev_limit]

    train_set = training_dataset(train_rows, texts)
    dev_subset = random.Random(SEED).sample(dev_rows, min(args.eval_questions, len(dev_rows)))
    query_prompt = QWEN_QUERY_PROMPT if args.qwen_prompt else None
    full_check = CandidateRankingEvaluator(
        dev_rows,
        texts,
        "dev",
        max_tokens=PRODUCTION_TOKENS,
        batch_size=args.encode_batch,
        query_prompt=query_prompt,
    )

    # 3. Model and loss.
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_tokens
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)

    # 4. Dev check before training: the baseline.
    start = time.time()
    before = full_check(model)
    log.info("before (%.1f min): %s", (time.time() - start) / 60, before)

    # 5. Training setup: steps, a quick dev check 5 times, a checkpoint at each.
    steps_per_epoch = math.ceil(len(train_set) / args.batch)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    eval_steps = max(1, total_steps // 5)

    training = SentenceTransformerTrainingArguments(
        output_dir=str(args.out),
        run_name=args.name,
        report_to="none" if args.no_wandb else "wandb",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=1,
        seed=SEED,
        prompts={"anchor": query_prompt} if query_prompt else None,
        gradient_checkpointing=args.grad_checkpointing,
    )

    progress = open_progress(f"embedder fine-tune {args.name} (local)", total_steps)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training,
        train_dataset=train_set,
        loss=loss,
        evaluator=CandidateRankingEvaluator(
            dev_subset,
            texts,
            "dev_subset",
            batch_size=args.encode_batch,
            query_prompt=query_prompt,
        ),
        callbacks=[ProgressCallback(progress)] if progress is not None else [],
    )

    # 6. Train.
    try:
        start = time.time()
        trainer.train()
        log.info("trained in %.1f min", (time.time() - start) / 60)
    finally:
        if progress is not None:
            progress.__exit__(None, None, None)

    model.save(str(args.out / "final"))

    # 7. Dev check after training, saved next to the model.
    after = full_check(model)
    log.info("after: %s", after)

    result = {
        "before": before,
        "after": after,
        "args": {**vars(args), "data": str(args.data), "out": str(args.out)},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=1))

    for metric in sorted(before):
        log.info("%-40s %.3f -> %.3f", metric, before[metric], after[metric])

    if not args.no_wandb:
        import wandb  # noqa: PLC0415 - only once the trainer has started the run

        if wandb.run is not None:
            wandb.log({f"before/{key}": value for key, value in before.items()})
            wandb.log({f"after/{key}": value for key, value in after.items()})
            wandb.finish()


if __name__ == "__main__":
    main()
