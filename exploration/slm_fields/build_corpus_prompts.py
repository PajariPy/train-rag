"""Write the prompts for mining the whole corpus with the SLM, in 3 parts of gzipped shards.

Every document of the corpus, as the SLM saw documents in training
(`build_inputs.document_text`), minus the documents the held-out scoring already mined
(`runs/fields-r16-e2/outputs.jsonl`, reused at import). The system prompt is the training's
(`label.instructions()`), checked against `data/sft_heldout.jsonl`.

The parts go to 3 GPUs, so they are balanced by estimated work, not by document count: the
prompt tokens (characters / 4, cut at the 5,000 the generator allows) plus the mean output.
Each part is shuffled (long documents spread over its shards) and cut into shards of at most
SHARD_DOCUMENTS, in generate_vllm.py's format: a {"system"} line, then {doc_id, source, text}.

Output: runs/mine-r16-e2/part{1,2,3}/shard_NN.jsonl.gz and runs/mine-r16-e2/manifest.json.

Free and local (database only). Usage:
    uv run python exploration/slm_fields/build_corpus_prompts.py
"""

import gzip
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from sqlmodel import Session, col, select

HERE = Path(__file__).parent
EXTRACTION = HERE.parent / "slack_extraction"
sys.path.insert(0, str(EXTRACTION))
from build_inputs import DATASET, document_text  # noqa: E402
from label import instructions  # noqa: E402

from train_rag.database import get_engine  # noqa: E402
from train_rag.parsers import default_registry  # noqa: E402
from train_rag.repository.document import load_documents  # noqa: E402
from train_rag.repository.inferred_messages import load_inferred_messages  # noqa: E402
from train_rag.tables import DocumentRecord  # noqa: E402

SCORED = HERE / "runs" / "fields-r16-e2" / "outputs.jsonl"
HELDOUT = HERE / "data" / "sft_heldout.jsonl"
OUT = HERE / "runs" / "mine-r16-e2"
PARTS = 3
SHARD_DOCUMENTS = 50_000
LOAD_BATCH = 5_000
PROMPT_TOKENS = 5_000  # generate_vllm.PROMPT_TOKENS
MEAN_OUTPUT_TOKENS = 1_200  # ≈ 615M output tokens / 512k documents (card estimate)
SEED = 23


def work(text: str) -> int:
    """Estimate one document's tokens: its prompt, cut like the generator, plus an answer.

    Args:
        text: The document text.

    Returns:
        Estimated tokens.
    """
    return min(len(text) // 4, PROMPT_TOKENS) + MEAN_OUTPUT_TOKENS


def main() -> None:
    """Load every document, leave out the mined ones, balance the parts, write the shards."""
    system = instructions()
    trained = json.loads(HELDOUT.read_text().split("\n", 1)[0])["messages"][0]["content"]
    if system != trained:
        raise SystemExit("label.instructions() differs from the training's system prompt")

    with SCORED.open() as file:
        mined = {json.loads(line)["doc_id"] for line in file if line.strip()}
    print(f"already mined by the scoring run: {len(mined):,} (questions included)", flush=True)

    rows: dict[str, dict[str, Any]] = {}
    start = time.time()
    with Session(get_engine()) as session:
        registry = default_registry(inferred_messages=load_inferred_messages(session))
        ids = list(
            session.exec(
                select(DocumentRecord.id)
                .where(DocumentRecord.dataset == DATASET)
                .order_by(col(DocumentRecord.id))
            )
        )
        for first in range(0, len(ids), LOAD_BATCH):
            batch = [i for i in ids[first : first + LOAD_BATCH] if i is not None]
            for document in load_documents(session, batch):
                if document.doc_id in mined or document.doc_id in rows:
                    continue  # mined already, or the doc_id stored twice
                text = document_text(document, registry)
                rows[document.doc_id] = {
                    "doc_id": document.doc_id,
                    "source": str(document.source_type),
                    "text": text,
                }
            done = min(first + LOAD_BATCH, len(ids))
            print(f"loaded {done:,}/{len(ids):,} ({time.time() - start:.0f} s)", flush=True)

    documents = sorted(rows.values(), key=lambda row: (-work(row["text"]), row["doc_id"]))
    parts: list[list[dict[str, Any]]] = [[] for _ in range(PARTS)]
    totals = [0] * PARTS
    for row in documents:  # longest first, each to the lightest part
        lightest = totals.index(min(totals))
        parts[lightest].append(row)
        totals[lightest] += work(row["text"])

    rng = random.Random(SEED)
    manifest: dict[str, Any] = {
        "documents": len(documents),
        "skipped_mined": len(mined),
        "parts": {},
    }
    header = json.dumps({"system": system}) + "\n"
    for number, part in enumerate(parts, start=1):
        rng.shuffle(part)
        folder = OUT / f"part{number}"
        folder.mkdir(parents=True, exist_ok=True)
        shards = {}
        for index, first in enumerate(range(0, len(part), SHARD_DOCUMENTS), start=1):
            shard = part[first : first + SHARD_DOCUMENTS]
            name = f"shard_{index:02d}.jsonl.gz"
            with gzip.open(folder / name, "wt", encoding="utf-8") as out:
                out.write(header)
                out.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in shard)
            shards[name] = {
                "documents": len(shard),
                "tokens": sum(work(row["text"]) for row in shard),
            }
        manifest["parts"][f"part{number}"] = {
            "documents": len(part),
            "tokens": totals[number - 1],
            "shards": shards,
        }
        print(f"part{number}: {len(part):,} documents, ≈{totals[number - 1] / 1e6:.0f}M tokens, "
              f"{len(shards)} shards", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(documents):,} documents to mine; manifest in {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
