r"""Turn the labelled training documents into chat examples for fine-tuning.

Each example is the chat DeepSeek saw, plus its answer: system = the extraction instructions,
user = "Document ({source}: {title}):" and the normalized text (cut at MAX_CHARS, as when
labelling), assistant = the JSON label. EVAL_SHARE of the documents of every source (spread
over its size bands) are held out for evaluation. The pilot documents are not in here.

Writes data/sft_train.jsonl and data/sft_eval.jsonl: one {"doc_id", "source", "band",
"messages"} per line, ready to copy to a training machine without the database.

Usage (from the main checkout, which has the data):
    uv run --project ~/projects/rag-implementation/train-rag python \
        exploration/fact_extraction/build_dataset.py
"""

import json
import random
from pathlib import Path

import yaml
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_documents
from train_rag.repository.document import ids_of_doc_ids, load_documents

HERE = Path(__file__).parent
DATA = HERE / "data"
MAX_CHARS = 30_000  # the same cut as when labelling
EVAL_SHARE = 0.1
SEED = 7


def main() -> None:
    """Build the chat examples and split them into training and evaluation sets."""
    prompts = yaml.safe_load((HERE / "prompts.yaml").read_text())
    rows = json.loads((DATA / "sample.json").read_text())["train"]
    labels = {v["doc_id"]: v for v in json.loads((DATA / "labels.json").read_text()).values()}
    rows = [row for row in rows if row["doc_id"] in labels]
    with Session(get_engine()) as session:
        ids = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, [r["doc_id"] for r in rows])
        records = load_documents(session, sorted(ids))
    sources = {record.doc_id: record.source_type for record in records}
    documents = {doc.doc_id: doc for doc in normalize_documents(records)}

    examples = []
    for row in rows:
        doc, label = documents[row["doc_id"]], labels[row["doc_id"]]
        title = doc.title.splitlines()[0] if doc.title else ""
        user = prompts["fact_extraction_prompt"].format(
            source=sources[row["doc_id"]], title=title, text=doc.text[:MAX_CHARS]
        )
        answer = {key: label[key] for key in ("about", "entities", "facts")}
        examples.append(
            {
                **row,
                "messages": [
                    {"role": "system", "content": prompts["fact_extraction_instructions"]},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
                ],
            }
        )

    # Hold out every tenth document of each source and band, chosen at random.
    generator = random.Random(SEED)
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for example in examples:
        groups.setdefault((str(example["source"]), int(example["band"])), []).append(example)
    train, held_out = [], []
    for group in groups.values():
        generator.shuffle(group)
        cut = max(1, round(len(group) * EVAL_SHARE))
        held_out += group[:cut]
        train += group[cut:]
    generator.shuffle(train)

    for name, items in (("sft_train.jsonl", train), ("sft_eval.jsonl", held_out)):
        with (DATA / name).open("w") as out:
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"train {len(train)}, eval {len(held_out)} -> {DATA}")


main()
