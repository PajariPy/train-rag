"""Retrieval-only evaluation: how many expected documents a retriever finds, no LLM."""

from collections import defaultdict
from collections.abc import Callable, Sequence
from statistics import fmean

from train_rag.evaluation.metrics import document_recall
from train_rag.models import Question
from train_rag.models.evaluation import RetrievalMetrics
from train_rag.retrieval import Retriever

DEFAULT_KS = (1, 5, 10, 20)


def retrieval_metrics(
    hits: Sequence[tuple[Question, list[str]]], indexed: set[str], ks: Sequence[int]
) -> RetrievalMetrics:
    """Average recall at each k over questions that expect documents.

    Args:
        hits: Each question with its retrieved doc ids, best first.
        indexed: Doc ids present in the index.
        ks: Cut-offs to measure.

    Returns:
        The averaged metrics; recall is 0 when no question expects documents.
    """
    scored = [(question, doc_ids) for question, doc_ids in hits if question.expected_doc_ids]

    return RetrievalMetrics(
        questions=len(scored),
        recall_at={
            k: fmean(document_recall(q.expected_doc_ids, ids[:k]) or 0.0 for q, ids in scored)
            if scored
            else 0.0
            for k in ks
        },
        reachable=fmean(
            document_recall(q.expected_doc_ids, sorted(indexed & set(q.expected_doc_ids))) or 0.0
            for q, _ids in scored
        )
        if scored
        else 0.0,
    )


def evaluate_retrieval(
    retriever: Retriever,
    questions: Sequence[Question],
    indexed: set[str],
    ks: Sequence[int] = DEFAULT_KS,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, RetrievalMetrics]:
    """Retrieve for every question and measure recall at each k, overall and per type.

    Args:
        retriever: The evaluated retriever.
        questions: Benchmark questions.
        indexed: Doc ids present in the index.
        ks: Cut-offs to measure.
        progress: Called with (done, total) after each question.

    Returns:
        Metrics keyed by question type, plus `all`.
    """
    hits: list[tuple[Question, list[str]]] = []
    for question in questions:
        hits.append((question, [hit.doc_id for hit in retriever.retrieve(question.text, max(ks))]))
        if progress is not None:
            progress(len(hits), len(questions))
    by_type: dict[str, list[tuple[Question, list[str]]]] = defaultdict(list)
    for question, doc_ids in hits:
        by_type[str(question.question_type)].append((question, doc_ids))

    return {
        "all": retrieval_metrics(hits, indexed, ks),
        **{name: retrieval_metrics(group, indexed, ks) for name, group in sorted(by_type.items())},
    }


def format_retrieval_report(title: str, metrics: dict[str, RetrievalMetrics]) -> str:
    """Format retrieval metrics as a markdown table.

    Args:
        title: Report heading.
        metrics: Metrics keyed by question type.

    Returns:
        The report text.
    """
    ks = sorted(next(iter(metrics.values())).recall_at)
    lines = [
        f"## {title}",
        "",
        "| Question type | Questions | " + " | ".join(f"Recall@{k}" for k in ks) + " | In index |",
        "|---|---:|" + "---:|" * (len(ks) + 1),
    ]
    for name, item in metrics.items():
        recalls = " | ".join(f"{item.recall_at[k]:.1%}" for k in ks)
        lines.append(f"| {name} | {item.questions} | {recalls} | {item.reachable:.1%} |")

    return "\n".join(lines)
