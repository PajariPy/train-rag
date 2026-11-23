"""Turn the SLM's mining outputs into one labels file that `train-rag import-fields` reads.

Reads every runs/mine-r16-e2/out/part*/shard_*.jsonl and the held-out scoring outputs
(runs/fields-r16-e2/outputs.jsonl, documents only, not questions). Writes
runs/mine-r16-e2/labels.jsonl in the labels-file shape (`id`, `key` = `<id>:<prompt hash>`,
`fields`, `labels`, plus `relations`). The model's text is kept as `output` with `finish` and
`cut`, so a schema change re-parses it and never re-mines. An output that isn't valid JSON is
kept too, with empty fields and `parse_error`. Prints valid JSON per source.

Free and local. Usage:
    uv run python exploration/slm_fields/to_labels.py
    uv run train-rag import-fields --file exploration/slm_fields/runs/mine-r16-e2/labels.jsonl \
        --extractor slm-fields-r16-e2
"""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
RUN = HERE / "runs" / "mine-r16-e2"
SCORED = HERE / "runs" / "fields-r16-e2" / "outputs.jsonl"
OUT = RUN / "labels.jsonl"
PROMPT = "fc8ae8878988"  # the frozen prompt the SLM was trained on (build_sft.VERSION)


def label_line(row: dict) -> dict:
    """Build one labels-file line from one output row.

    Args:
        row: {doc_id, source, text, finish, cut} from generate_vllm.py.

    Returns:
        The line, with empty fields when the output doesn't parse.
    """
    line = {
        "id": row["doc_id"],
        "key": f"{row['doc_id']}:{PROMPT}",
        "source": row["source"],
        "output": row["text"],
        "finish": row.get("finish"),
        "cut": row.get("cut"),
    }
    try:
        parsed = json.loads(row["text"])
        if not isinstance(parsed, dict):
            raise ValueError("not an object")
    except ValueError:
        return {**line, "fields": {}, "labels": {}, "parse_error": True}

    return {
        **line,
        "fields": parsed.get("fields") or {},
        "labels": parsed.get("labels") or {},
        "relations": parsed.get("relations") or [],
    }


def main() -> None:
    """Merge the outputs, write the labels file and print valid JSON per source."""
    files = sorted(RUN.glob("out/part*/shard_*.jsonl")) + [SCORED]
    seen: set[str] = set()
    total: Counter[str] = Counter()
    valid: Counter[str] = Counter()
    with OUT.open("w") as out:
        for path in files:
            with path.open() as file:
                for text in file:
                    if not text.strip():
                        continue
                    row = json.loads(text)
                    if row["doc_id"].startswith("question:") or row["doc_id"] in seen:
                        continue
                    seen.add(row["doc_id"])
                    line = label_line(row)
                    total[row["source"]] += 1
                    valid[row["source"]] += not line.get("parse_error")
                    out.write(json.dumps(line, ensure_ascii=False) + "\n")

    for source in sorted(total):
        share = valid[source] / total[source]
        print(f"{source:13} {total[source]:>8,} documents, valid JSON {share:.1%}")
    print(f"{'all':13} {sum(total.values()):>8,} documents, "
          f"valid JSON {sum(valid.values()) / max(1, sum(total.values())):.1%} -> {OUT}")


if __name__ == "__main__":
    main()
