"""Format a run summary as a table with the benchmark's leaderboard metrics."""

from train_rag.models.evaluation import MetricSummary, RunSummary

HEADER = (
    f"{'question type':<26}{'questions':>10}{'score':>8}{'correct':>9}{'complete':>10}"
    f"{'doc recall':>12}{'extra docs':>12}{'errors':>8}"
)


def format_row(name: str, metrics: MetricSummary) -> str:
    """Format one line of the table.

    Args:
        name: Row label.
        metrics: Metrics of the row.

    Returns:
        The formatted line.
    """
    recall = f"{metrics.document_recall:.1%}" if metrics.document_recall is not None else "—"

    return (
        f"{name:<26}{metrics.questions:>10}{metrics.overall_score:>8.1%}"
        f"{metrics.correctness:>9.1%}{metrics.completeness:>10.1%}{recall:>12}"
        f"{metrics.invalid_extra_docs:>12.2f}{metrics.errors:>8}"
    )


def format_report(title: str, summary: RunSummary) -> str:
    """Format a run summary: one row per question type, then the overall row.

    Args:
        title: Heading, e.g. the run id and system name.
        summary: The run summary.

    Returns:
        The report text.
    """
    rows = [format_row(name, metrics) for name, metrics in summary.by_type.items()]
    rule = "-" * len(HEADER)

    return "\n".join([title, HEADER, rule, *rows, rule, format_row("overall", summary.overall)])
