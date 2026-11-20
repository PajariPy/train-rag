"""Score both GLiNER models against DeepSeek's labels, per field and per relation.

An item is a distinct value per thread: an entity is (thread, type, text), a relation is
(thread, relation, head, tail). Text is compared lowercased, with spaces collapsed and
surrounding punctuation trimmed. Two match levels:

- exact: same type and the same text;
- partial: same type and overlapping text (one contains the other, or they share at least
  half their words).

Matching is one to one within a thread and type, exact pairs first. Reference values that
DeepSeek didn't copy from the text can't be found as spans and are left out (counted).

Reports precision, recall, F1 and support per field and relation, micro (counts pooled) and
macro (mean over fields that have reference items), a type-confusion count (a found value
that is in the reference under another type) and a threshold sweep per field.

Writes data/report.md and prints it.

Usage:
    uv run python exploration/slack_extraction/score.py [--threshold 0.5]
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from schema import ENTITY_TYPES, RELATIONS

HERE = Path(__file__).parent
LABELS = HERE / "data" / "labels.json"
EXTRACTIONS = HERE / "data" / "extractions"
REPORT = HERE / "data" / "report.md"
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def norm(text: str) -> str:
    """Lowercase, collapse spaces, trim surrounding punctuation."""
    return re.sub(r"\s+", " ", text.lower()).strip(" \t\n.,;:!?\"'`()[]{}")


def overlaps(a: str, b: str) -> bool:
    """Whether two normalised values overlap: one contains the other or half the words match."""
    if a in b or b in a:
        return True
    words_a, words_b = set(a.split()), set(b.split())

    return len(words_a & words_b) >= 0.5 * max(len(words_a), len(words_b))


def match(reference: list[str], found: list[str], partial: bool) -> int:
    """How many found values match a reference value, one to one, exact pairs first."""
    left = list(reference)
    hits = 0
    rest = []
    for value in found:
        if value in left:
            left.remove(value)
            hits += 1
        else:
            rest.append(value)
    if partial:
        for value in rest:
            other = next((ref for ref in left if overlaps(value, ref)), None)
            if other is not None:
                left.remove(other)
                hits += 1

    return hits


def match_relations(
    reference: list[tuple[str, str]], found: list[tuple[str, str]], partial: bool
) -> int:
    """Relations match when both ends match (exactly, or overlapping when partial)."""
    left = list(reference)
    hits = 0
    for head, tail in sorted(found, key=lambda pair: pair not in reference):
        for ref in left:
            same = ref == (head, tail)
            close = partial and overlaps(head, ref[0]) and overlaps(tail, ref[1])
            if same or close:
                left.remove(ref)
                hits += 1
                break

    return hits


@dataclass
class Counts:
    """True positives, found and reference counts for one field."""

    hits: int = 0
    found: int = 0
    reference: int = 0

    def scores(self) -> tuple[float, float, float]:
        """Precision, recall and F1."""
        precision = self.hits / self.found if self.found else 0.0
        recall = self.hits / self.reference if self.reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        return precision, recall, f1


def reference_items(
    labels: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[tuple[str, str]]]], int]:
    """DeepSeek's values per thread and type, and its relations per thread and relation.

    Returns:
        Entities, relations, and how many values were dropped as not in the text.
    """
    entities: dict[str, dict[str, list[str]]] = {}
    relations: dict[str, dict[str, list[tuple[str, str]]]] = {}
    dropped = 0
    for label in labels.values():
        doc_id = str(label["doc_id"])
        missing = {norm(value) for value in label["not_in_text"]}  # type: ignore[union-attr]
        entities[doc_id] = {}
        for name, values in label["entities"].items():  # type: ignore[union-attr]
            if name not in ENTITY_TYPES:
                continue
            kept = sorted({norm(value) for value in values} - missing - {""})
            dropped += len({norm(value) for value in values} & missing)
            entities[doc_id][name] = kept
        relations[doc_id] = {}
        for row in label["relations"]:  # type: ignore[union-attr]
            head, tail = norm(row["head"]), norm(row["tail"])
            if row["relation"] in RELATIONS and head not in missing and tail not in missing:
                relations[doc_id].setdefault(row["relation"], []).append((head, tail))

    return entities, relations, dropped


def count(
    entities: dict[str, dict[str, list[str]]],
    relations: dict[str, dict[str, list[tuple[str, str]]]],
    extraction: dict[str, dict[str, object]],
    threshold: float,
    partial: bool,
) -> dict[str, Counts]:
    """Counts per field and per relation for one model at one threshold."""
    counts = {name: Counts() for name in [*ENTITY_TYPES, *RELATIONS]}
    for doc_id, reference in entities.items():
        found = extraction.get(doc_id, {"entities": {}, "relations": []})
        for name in ENTITY_TYPES:
            ref = reference.get(name, [])
            values = found["entities"].get(name, {})  # type: ignore[union-attr]
            kept = sorted({norm(v) for v, score in values.items() if score >= threshold} - {""})
            counts[name].hits += match(ref, kept, partial)
            counts[name].found += len(kept)
            counts[name].reference += len(ref)
        for name in RELATIONS:
            ref_pairs = relations[doc_id].get(name, [])
            pairs = sorted({
                (norm(row["head"]), norm(row["tail"]))
                for row in found["relations"]  # type: ignore[union-attr]
                if row["relation"] == name and row["score"] >= threshold
            })
            counts[name].hits += match_relations(ref_pairs, pairs, partial)
            counts[name].found += len(pairs)
            counts[name].reference += len(ref_pairs)

    return counts


def summary(counts: dict[str, Counts], names: list[str]) -> tuple[tuple[float, ...], ...]:
    """Micro (pooled) and macro (mean over fields with reference items) P, R, F1."""
    pooled = Counts(
        sum(counts[n].hits for n in names),
        sum(counts[n].found for n in names),
        sum(counts[n].reference for n in names),
    )
    present = [counts[n].scores() for n in names if counts[n].reference]
    macro = tuple(sum(s[i] for s in present) / len(present) if present else 0.0 for i in range(3))

    return pooled.scores(), macro


def confusion(
    entities: dict[str, dict[str, list[str]]],
    extraction: dict[str, dict[str, object]],
    threshold: float,
) -> Counter[tuple[str, str]]:
    """(found type, reference type) for found values that sit in the reference under another type."""
    pairs: Counter[tuple[str, str]] = Counter()
    for doc_id, reference in entities.items():
        found = extraction.get(doc_id, {"entities": {}})["entities"]
        for name, values in found.items():  # type: ignore[union-attr]
            for value, score in values.items():
                value = norm(value)
                if score < threshold or value in reference.get(name, []):
                    continue
                for other, ref in reference.items():
                    if other != name and value in ref:
                        pairs[(name, other)] += 1

    return pairs


def fmt(value: float) -> str:
    """A score as two decimals."""
    return f"{value:.2f}"


def main() -> None:
    """Score every model in data/extractions/ and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    labels = json.loads(LABELS.read_text())
    entities, relations, dropped = reference_items(labels)
    runs = {
        path.stem: json.loads(path.read_text()) for path in sorted(EXTRACTIONS.glob("*.json"))
    }
    labelled = set(entities)
    extractions = {
        name: {k: v for k, v in run["threads"].items() if k in labelled}
        for name, run in runs.items()
    }
    models = list(extractions)
    lines = [
        f"# Slack extraction scores ({len(entities)} threads, threshold {args.threshold})",
        "",
        f"Reference: DeepSeek labels; {dropped} values not copied from the text left out.",
        "Speed: " + ", ".join(
            f"{name} {runs[name]['seconds_per_thread']:.2f} s/thread on {runs[name]['device']}"
            for name in models
        ),
        "",
    ]
    tables = {
        (name, partial): count(entities, relations, extractions[name], args.threshold, partial)
        for name in models
        for partial in (False, True)
    }
    for title, names in (("Entities", list(ENTITY_TYPES)), ("Relations", list(RELATIONS))):
        header = "| field | support | " + " | ".join(
            f"{m} P | {m} R | {m} F1 | {m} F1 partial" for m in models
        ) + " |"
        lines += [f"## {title}", "", header, "|" + "---|" * (2 + 4 * len(models))]
        for field in names:
            row = [field, str(tables[(models[0], False)][field].reference)]
            for m in models:
                p, r, f = tables[(m, False)][field].scores()
                row += [fmt(p), fmt(r), fmt(f), fmt(tables[(m, True)][field].scores()[2])]
            lines.append("| " + " | ".join(row) + " |")
        for label, index in (("**micro**", 0), ("**macro**", 1)):
            row = [label, ""]
            for m in models:
                exact = summary(tables[(m, False)], names)[index]
                part = summary(tables[(m, True)], names)[index]
                row += [fmt(exact[0]), fmt(exact[1]), fmt(exact[2]), fmt(part[2])]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## Type confusion (found as → reference type, exact text)", ""]
    for m in models:
        top = confusion(entities, extractions[m], args.threshold).most_common(10)
        lines.append(f"- **{m}:** " + ("; ".join(f"{a} → {b} ×{n}" for (a, b), n in top) or "none"))
    lines += ["", "## Threshold sweep (partial F1 per field)", ""]
    header = "| field | " + " | ".join(f"{m} @{t}" for m in models for t in SWEEP) + " |"
    lines += [header, "|" + "---|" * (1 + len(models) * len(SWEEP))]
    sweeps = {
        (m, t): count(entities, relations, extractions[m], t, True) for m in models for t in SWEEP
    }
    for field in [*ENTITY_TYPES, *RELATIONS]:
        cells = [fmt(sweeps[(m, t)][field].scores()[2]) for m in models for t in SWEEP]
        lines.append(f"| {field} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"
    REPORT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
