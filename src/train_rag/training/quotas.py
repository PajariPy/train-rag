"""How many training questions to make per question type and per source.

The mix copies the benchmark's one-document questions: its type counts (175 basic,
125 semantic, 40 intra-document reasoning, 20 miscellaneous) and, per type, how its right
documents spread over the sources (EnterpriseRAG-Bench paper, Table 10).
"""

from collections.abc import Mapping

TYPE_WEIGHTS: dict[str, int] = {
    "basic": 175,
    "semantic": 125,
    "intra_document_reasoning": 40,
    "miscellaneous": 20,
}
SOURCE_ORDER = (
    "confluence",
    "fireflies",
    "github",
    "gmail",
    "google_drive",
    "hubspot",
    "jira",
    "linear",
    "slack",
)
SOURCE_WEIGHTS: dict[str, dict[str, int]] = {
    "basic": dict(zip(SOURCE_ORDER, (19, 11, 22, 15, 24, 18, 21, 24, 21), strict=True)),
    "semantic": dict(zip(SOURCE_ORDER, (15, 4, 14, 18, 14, 14, 23, 10, 13), strict=True)),
    "intra_document_reasoning": dict(zip(SOURCE_ORDER, (2, 3, 2, 9, 0, 1, 2, 9, 12), strict=True)),
}
MIN_CHARS = 200
MIN_INTRA_DOCUMENT_CHARS = 3000  # the benchmark's --min-doc-length


def split_by_weight(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """Split a total in proportion to weights, rounding by largest remainder.

    Args:
        total: The number to split.
        weights: Weight per key; a zero weight gets nothing.

    Returns:
        A count per key; the counts sum to `total`.
    """
    weight_sum = sum(weights.values())
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    by_remainder = sorted(exact, key=lambda key: exact[key] - counts[key], reverse=True)
    for key in by_remainder[: total - sum(counts.values())]:
        counts[key] += 1

    return counts


def is_usable_source(char_count: int, question_type: str) -> bool:
    """Tell whether a document is long enough to write a question of this type from.

    Args:
        char_count: The document's length in characters.
        question_type: The question type.

    Returns:
        True if the document can be a source.
    """
    if question_type == "intra_document_reasoning":
        return char_count >= MIN_INTRA_DOCUMENT_CHARS

    return char_count >= MIN_CHARS
