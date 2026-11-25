r"""Score the dev reranking set with a ColBERT (late-interaction) model, and time it.

For every question, every candidate is scored with MaxSim: each question token takes its best
match among the document's token vectors, and those best matches are summed. Uses
sentence-transformers' MultiVectorEncoder (no extra library).

- Questions: normalized as the pipeline does, encoded once.
- Documents: every unique candidate (`title + "\n" + text`, as the cross-encoder reads it),
  encoded once, shortest first, then scored against every question that has it as a candidate.
  Vectors are discarded after scoring, so nothing large is kept.
- Batches: sized by a token budget (`--batch-tokens`), so whole documents (8,192 tokens) fit
  the GPU; the scores are checkpointed and a stopped run resumes where it left off.
- Timing: document encoding (what indexing would cost) and, for a sample of questions, the
  question-time cost with stored document vectors (question encode + MaxSim over its candidates).
- Candidates: today's ~97 per question (candidates.jsonl), or with `--deep` the gte and BM25
  top 200 each (~380 per question, deep_found.json from deep_candidates.py).

Output in `data/`:
    scores_<name>.json / deep_scores_<name>.json   {qid: {doc_id: score}}
    colbert_<name>_stats.json / deep_colbert_<name>_stats.json   settings, tokens, timings

Run from the main checkout:
    uv run python <workspace>/exploration/reranker_dev/colbert.py
        --model lightonai/LateOn --name lateon [--deep] [--doc-length 8192] [--limit 10]
        [--batch-tokens 8192] [--workflow KEY --step NAME]
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import NamedTuple, cast

import pandas as pd
import torch
from orderings import DATA, FoundCandidate, ScoreTable, load_questions
from sentence_transformers import MultiVectorEncoder

from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_text
from train_rag.progress.recorder import RunProgress

MAX_BATCH = 64
CHARS_PER_TOKEN = 4.0
CHECKPOINT_DOCUMENTS = 2000
SAMPLE_QUESTIONS = 20


class PoolItem(NamedTuple):
    """One question and the ids of its candidates."""

    qid: str
    question: str
    doc_ids: list[str]


def maxsim(query: torch.Tensor, document: torch.Tensor) -> float:
    """Sum over question tokens of the best dot product with any document token.

    Args:
        query: Question token vectors (question tokens × dim).
        document: Document token vectors (document tokens × dim).

    Returns:
        The MaxSim score.
    """
    return float((query @ document.T).max(dim=1).values.sum())


def load_pool(deep: bool) -> tuple[list[PoolItem], dict[str, str]]:
    """Load the questions with their candidates, and every candidate's reranker input text.

    Args:
        deep: Use the deep candidates (top 200 per search) instead of today's ~97.

    Returns:
        The pool, and the title and text (joined by a newline) per document id.
    """
    questions = load_questions()
    frames = [pd.read_parquet(DATA / "documents.parquet")]

    if deep:
        found = cast(
            dict[str, list[FoundCandidate]], json.loads((DATA / "deep_found.json").read_text())
        )
        frames.append(pd.read_parquet(DATA / "deep_documents.parquet"))
        pool = [
            PoolItem(
                question["qid"],
                question["question"],
                [cand["doc_id"] for cand in found[question["qid"]]],
            )
            for question in questions
        ]
    else:
        pool = [
            PoolItem(
                question["qid"],
                question["question"],
                [cand["doc_id"] for cand in question["candidates"]],
            )
            for question in questions
        ]

    docs = pd.concat(frames).drop_duplicates("doc_id").set_index("doc_id")
    needed = {doc_id for item in pool for doc_id in item.doc_ids}
    texts = {doc_id: f"{docs.at[doc_id, 'title']}\n{docs.at[doc_id, 'text']}" for doc_id in needed}

    return pool, texts


def candidate_users(pool: list[PoolItem]) -> dict[str, list[int]]:
    """Map each candidate document to the questions that have it as a candidate.

    Args:
        pool: Questions with their candidate ids.

    Returns:
        Question positions per document id.
    """
    users: dict[str, list[int]] = {}
    for position, item in enumerate(pool):
        for doc_id in item.doc_ids:
            users.setdefault(doc_id, []).append(position)

    return users


def token_batches(
    order: list[str], texts: dict[str, str], args: argparse.Namespace
) -> list[list[str]]:
    """Split documents (shortest first) into batches that stay within a token budget.

    Args:
        order: Document ids sorted by text length.
        texts: Reranker input text per document id.
        args: Command-line arguments (doc_length, batch_tokens).

    Returns:
        Batches of document ids; each batch's size times its longest document fits the budget.
    """
    batches: list[list[str]] = []
    batch: list[str] = []
    for doc_id in order:
        longest = min(args.doc_length, len(texts[doc_id]) / CHARS_PER_TOKEN)
        if batch and ((len(batch) + 1) * longest > args.batch_tokens or len(batch) >= MAX_BATCH):
            batches.append(batch)
            batch = []
        batch.append(doc_id)

    if batch:
        batches.append(batch)

    return batches


def load_checkpoint(path: Path) -> tuple[ScoreTable, dict[str, int], float]:
    """Load a stopped run's scores, token counts and encoding seconds, if any.

    Args:
        path: The checkpoint file.

    Returns:
        Scores per question id, token count per finished document id, encoding seconds.
    """
    if not path.exists():
        return {}, {}, 0.0

    saved = json.loads(path.read_text())

    return saved["scores"], saved["tokens"], saved["encode_seconds"]


def save_checkpoint(path: Path, scores: ScoreTable, tokens: dict[str, int], seconds: float) -> None:
    """Write the run's progress so a stopped run can resume.

    Args:
        path: The checkpoint file.
        scores: Scores per question id and document id so far.
        tokens: Token count per finished document id.
        seconds: Encoding seconds so far.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"scores": scores, "tokens": tokens, "encode_seconds": seconds})
    )
    temporary.replace(path)


def score_documents(
    model: MultiVectorEncoder,
    pool: list[PoolItem],
    query_vectors: list[torch.Tensor],
    texts: dict[str, str],
    args: argparse.Namespace,
) -> tuple[ScoreTable, list[int], float]:
    """Encode every unique candidate once and score it against the questions that have it.

    Args:
        model: The ColBERT model.
        pool: Questions with their candidate ids.
        query_vectors: Each question's token vectors, in pool order.
        texts: Reranker input text per document id.
        args: Command-line arguments (name, deep, doc_length, batch_tokens, checkpoint, progress).

    Returns:
        Scores per question id and document id, token count per document, encoding seconds.
    """
    users = candidate_users(pool)
    saved, tokens, encode_seconds = load_checkpoint(args.checkpoint)
    scores: ScoreTable = {item.qid: saved.get(item.qid, {}) for item in pool}
    order = sorted(
        (doc_id for doc_id in users if doc_id not in tokens), key=lambda doc_id: len(texts[doc_id])
    )
    batches = token_batches(order, texts, args)
    since_checkpoint = 0

    with RunProgress(
        get_engine(),
        f"ColBERT {args.name}{' deep' if args.deep else ''}: encode and score candidates",
        len(users),
        unit="documents",
        workflow=args.workflow,
        step=args.step,
    ) as progress:
        if tokens:
            progress.advance("resumed", len(tokens), note=f"{len(tokens)} from checkpoint")

        for batch in batches:
            torch.cuda.synchronize()
            started = time.perf_counter()
            vectors = model.encode_document(
                [texts[doc_id] for doc_id in batch], batch_size=len(batch)
            )
            torch.cuda.synchronize()
            encode_seconds += time.perf_counter() - started

            for doc_id, document in zip(batch, vectors, strict=True):
                tokens[doc_id] = int(document.shape[0])
                for position in users[doc_id]:
                    scores[pool[position].qid][doc_id] = maxsim(query_vectors[position], document)

            since_checkpoint += len(batch)
            if since_checkpoint >= CHECKPOINT_DOCUMENTS:
                save_checkpoint(args.checkpoint, scores, tokens, encode_seconds)
                since_checkpoint = 0

            progress.advance(
                "encoding", len(batch), note=f"{len(tokens) / encode_seconds:.1f} docs/s"
            )

    return scores, list(tokens.values()), encode_seconds


def question_time_ms(
    model: MultiVectorEncoder,
    pool: list[PoolItem],
    cleaned: list[str],
    texts: dict[str, str],
    args: argparse.Namespace,
) -> float:
    """Time a question with stored document vectors: encode it, MaxSim over its candidates.

    Args:
        model: The ColBERT model.
        pool: Questions with their candidate ids.
        cleaned: Normalized question texts, in pool order.
        texts: Reranker input text per document id.
        args: Command-line arguments (doc_length, batch_tokens), for the encoding batches.

    Returns:
        Median milliseconds per question over a fixed random sample.
    """
    sample = random.Random(7).sample(range(len(pool)), min(SAMPLE_QUESTIONS, len(pool)))
    timings: list[float] = []
    for position in sample:
        doc_ids = sorted(pool[position].doc_ids, key=lambda doc_id: len(texts[doc_id]))
        stored = [
            document
            for batch in token_batches(doc_ids, texts, args)
            for document in model.encode_document(
                [texts[doc_id] for doc_id in batch], batch_size=len(batch)
            )
        ]
        torch.cuda.synchronize()
        started = time.perf_counter()
        query = model.encode_query([cleaned[position]])[0]
        _ = [maxsim(query, document) for document in stored]
        torch.cuda.synchronize()
        timings.append(1000 * (time.perf_counter() - started))

    return statistics.median(timings)


def main() -> None:
    """Load the model, encode and score, time, save."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--doc-length", type=int, default=1024)
    parser.add_argument(
        "--deep", action="store_true", help="score the deep candidates (top 200 each)"
    )
    parser.add_argument("--limit", type=int, help="smoke test: first N questions")
    parser.add_argument(
        "--batch-tokens", type=int, default=8192, help="token budget per encoding batch"
    )
    parser.add_argument("--workflow")
    parser.add_argument("--step")
    args = parser.parse_args()

    pool, texts = load_pool(args.deep)
    if args.limit:
        pool = pool[: args.limit]

    model = MultiVectorEncoder(
        args.model, device="cuda", model_kwargs={"torch_dtype": torch.float16}
    )
    module = model[0]
    defaults = {"query_length": module.query_length, "document_length": module.document_length}
    module.document_length = args.doc_length

    cleaned = [normalize_text(item.question) for item in pool]
    started = time.perf_counter()
    query_vectors = cast(list[torch.Tensor], model.encode_query(cleaned, batch_size=64))
    query_encode_ms = 1000 * (time.perf_counter() - started) / len(cleaned)

    prefix = "deep_" if args.deep else ""
    suffix = "-smoke" if args.limit else ""
    scores_file = DATA / f"{prefix}scores_{args.name}{suffix}.json"
    args.checkpoint = DATA / f"{prefix}scores_{args.name}{suffix}.partial.json"

    scores, tokens, encode_seconds = score_documents(model, pool, query_vectors, texts, args)
    scores_file.write_text(json.dumps(scores))
    args.checkpoint.unlink(missing_ok=True)
    print(f"saved {scores_file.name}; timing a question with stored vectors")

    per_question_ms = question_time_ms(model, pool, cleaned, texts, args)

    at_cap = sum(count >= args.doc_length for count in tokens)
    stats = {
        "model": args.model,
        "name": args.name,
        "deep": args.deep,
        "model_defaults": defaults,
        "document_length": args.doc_length,
        "batch_tokens": args.batch_tokens,
        "questions": len(pool),
        "unique_documents": len(tokens),
        "doc_tokens_mean": round(statistics.mean(tokens), 1),
        "doc_tokens_median": statistics.median(tokens),
        "doc_tokens_max": max(tokens),
        "share_at_cap": round(at_cap / len(tokens), 3),
        "documents_per_second_encode": round(len(tokens) / encode_seconds, 2),
        "query_encode_ms": round(query_encode_ms, 2),
        "question_time_ms_median": round(per_question_ms, 2),
        "gpu": torch.cuda.get_device_name(0),
    }
    (DATA / f"{prefix}colbert_{args.name}{suffix}_stats.json").write_text(
        json.dumps(stats, indent=1)
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
