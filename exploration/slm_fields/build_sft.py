r"""Turn DeepSeek's field labels into chat examples for fine-tuning the SLM, every source.

Each example is the chat DeepSeek saw plus its answer: system = the extraction instructions
(label.instructions()), user = the document text from inputs.jsonl, assistant = the label as
compact JSON {"fields", "labels", "relations"}. Only labels from the frozen prompt
(VERSION) are used.

Name values that don't appear in the text (`not_in_text`, 23% of the rows: mostly
normalised spellings such as "p95 latency" or a full name the text never writes) are removed
from the target, with the relations that use them, instead of dropping the whole row: the
model learns to copy names as the text writes them.

Splits (never mixed):
- sft_train.jsonl: `train` items (half A's candidates, random Slack threads, train
  questions, and the extra batch of 20 Nov for thin sources);
- sft_eval.jsonl: EVAL_PER_SOURCE `test` items per source, for the loss during training;
- sft_heldout.jsonl: every `test` and `extraction_test` item, for scoring after training.

Free and local. Usage:
    uv run python exploration/slm_fields/build_sft.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EXTRACTION = HERE.parent / "slack_extraction"
sys.path.insert(0, str(EXTRACTION))
from label import instructions  # noqa: E402

INPUTS = EXTRACTION / "data" / "inputs.jsonl"
LABELS = EXTRACTION / "data" / "labels.jsonl"
OUT = HERE / "data"
VERSION = "fc8ae8878988"  # the frozen prompt of the field test
EVAL_PER_SOURCE = 40
SEED = 11


def target(row: dict) -> str:
    """The label as the model should write it, without values missing from the text."""
    missing = set(row.get("not_in_text") or [])
    fields = {
        name: [v for v in values if v not in missing] for name, values in row["fields"].items()
    }
    relations = [
        r for r in row["relations"] if r["head"] not in missing and r["tail"] not in missing
    ]
    answer = {"fields": fields, "labels": row["labels"], "relations": relations}

    return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    """Build the three files and print their sizes per source."""
    system = instructions()
    with INPUTS.open() as lines:  # not splitlines(): texts hold U+2028 separators
        items = {item["id"]: item for item in map(json.loads, lines)}
    rows = {}
    with LABELS.open() as lines:
        for row in map(json.loads, lines):
            if row["key"].endswith(VERSION) and row["id"] in items:
                rows[row["id"]] = row

    def example(row: dict) -> dict:
        item = items[row["id"]]
        return {
            "doc_id": row["id"], "source": item["source"], "use": item["use"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": item["text"]},
                {"role": "assistant", "content": target(row)},
            ],
        }

    train = [example(r) for r in rows.values() if items[r["id"]]["use"] == "train"]
    held = [
        example(r) for r in rows.values() if items[r["id"]]["use"] in ("test", "extraction_test")
    ]
    rng = random.Random(SEED)
    by_source: dict[str, list[dict]] = {}
    for e in held:
        if e["use"] == "test":
            by_source.setdefault(e["source"], []).append(e)
    evaluation = [
        e for source in sorted(by_source)
        for e in rng.sample(by_source[source], min(EVAL_PER_SOURCE, len(by_source[source])))
    ]
    rng.shuffle(train)

    OUT.mkdir(exist_ok=True)
    for name, examples in (("train", train), ("eval", evaluation), ("heldout", held)):
        path = OUT / f"sft_{name}.jsonl"
        path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
        print(f"{path.name}: {len(examples):,}", dict(Counter(e["source"] for e in examples)))
    stripped = sum(bool(rows[e["doc_id"]].get("not_in_text")) for e in train)
    print(f"train rows with values removed (not in text): {stripped:,}")


if __name__ == "__main__":
    main()
