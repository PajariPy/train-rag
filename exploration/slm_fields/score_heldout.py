"""Score the SLM's fields on the held-out documents against DeepSeek's labels, per source.

Reads runs/<name>/outputs.jsonl (from vast_score.sh) and data/sft_heldout.jsonl (DeepSeek's
answers, with values not in the text removed, as in training). Two measures per field:

- word F1: the field's values joined, word sets compared, averaged over documents (the
  training's check, 1.0 when both are empty);
- value F1: values compared one by one (lowercased, spaces collapsed), counts pooled over
  documents (micro), counting only documents where DeepSeek or the SLM has a value.

Plus the share of outputs that parse as JSON and the share cut at the token limit.
Writes runs/<name>/score.md and score.json and prints the report. Free and local.

Usage: uv run python exploration/slm_fields/score_heldout.py --name fields-r16-e2
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
WORD = re.compile(r"\w+")


def read(path: Path) -> list[dict]:
    """JSONL rows, split on newlines only (texts hold U+2028)."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def norm(value: object) -> str:
    """Lowercase with spaces collapsed."""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def word_f1(predicted: list[str], expected: list[str]) -> float:
    """Word-set F1 of the joined values; 1.0 when both are empty."""
    got = set(WORD.findall(" ".join(predicted).lower()))
    want = set(WORD.findall(" ".join(expected).lower()))
    if not got and not want:
        return 1.0
    if not got or not want:
        return 0.0
    hits = len(got & want)

    return 2 * hits / (len(got) + len(want))


def f1(hits: int, found: int, reference: int) -> float:
    """F1 from pooled counts."""
    return 2 * hits / (found + reference) if found + reference else 0.0


def main() -> None:
    """Score and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run = HERE / "runs" / args.name
    reference = {
        r["doc_id"]: json.loads(r["messages"][2]["content"])
        for r in read(HERE / "data" / "sft_heldout.jsonl")
    }
    outputs = read(run / "outputs.jsonl")

    words: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    values: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    docs: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # docs, parsed, cut-off, long
    for row in outputs:
        expected = reference[row["doc_id"]]["fields"]
        try:
            got = json.loads(row["text"])
            fields = got.get("fields", {}) if isinstance(got, dict) else {}
            ok = True
        except json.JSONDecodeError:
            fields, ok = {}, False
        for source in (row["source"], "all"):
            stats = docs[source]
            stats[0] += 1
            stats[1] += ok
            stats[2] += row["finish"] == "length"
            stats[3] += bool(row.get("cut"))
        for name, want in expected.items():
            mine = fields.get(name, []) if isinstance(fields, dict) else []
            mine = mine if isinstance(mine, list) else [mine]
            score = word_f1(list(map(str, mine)), want)
            got_set, want_set = {norm(v) for v in mine} - {""}, {norm(v) for v in want} - {""}
            for source in (row["source"], "all"):
                words[source][name].append(score)
                counts = values[source][name]
                counts[0] += len(got_set & want_set)
                counts[1] += len(got_set)
                counts[2] += len(want_set)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    sources = ["all", *sorted(s for s in docs if s != "all")]
    fields = list(words["all"])
    result = {
        source: {
            "documents": docs[source][0],
            "json_valid": docs[source][1] / docs[source][0],
            "cut_at_limit": docs[source][2] / docs[source][0],
            "document_shortened": docs[source][3] / docs[source][0],
            "word_f1_mean": mean([mean(words[source][f]) for f in fields]),
            "value_f1_micro": f1(*(sum(values[source][f][i] for f in fields) for i in range(3))),
            "word_f1": {f: mean(words[source][f]) for f in fields},
            "value_f1": {f: f1(*values[source][f]) for f in fields},
        }
        for source in sources
    }
    (run / "score.json").write_text(json.dumps(result, indent=1))

    lines = [
        f"# Held-out score: {args.name} vs DeepSeek ({len(outputs):,} documents)", "",
        "| Source | Docs | JSON valid | Cut at limit | Word F1 (mean of fields) | Value F1 (micro) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in sources:
        r = result[source]
        lines.append(
            f"| {source} | {r['documents']:,} | {r['json_valid']:.1%} | {r['cut_at_limit']:.1%} "
            f"| {r['word_f1_mean']:.3f} | {r['value_f1_micro']:.3f} |"
        )
    lines += ["", "| Field | Word F1 | Value F1 |", "|---|---:|---:|"]
    for name in sorted(fields, key=lambda f: -result["all"]["word_f1"][f]):
        lines.append(
            f"| {name} | {result['all']['word_f1'][name]:.3f} "
            f"| {result['all']['value_f1'][name]:.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (run / "score.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
