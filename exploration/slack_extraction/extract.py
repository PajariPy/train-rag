"""Run both GLiNER models over the sampled Slack threads with the schema, keeping scores.

- gliner2.5-base-v1 (boundary, DeBERTa-v3-base, 194M) reads each thread whole (4,096 window);
- gliner2-large-v1 (span, DeBERTa-v3-large, 340M) reads it in 384-word chunks, 64 overlap,
  because of its shorter window;
- base-chunked: gliner2.5-base-v1 on the same chunks, to tell the model apart from the window.

Everything scoring at least 0.3 is kept with its confidence, so score.py can sweep the
threshold per field without another run. A relation's score is the lower of its two ends.
Local, on the GPU when there is one.

Writes data/extractions/<model>.json: doc id -> entities and relations, plus seconds per
thread.

Usage:
    uv run python exploration/slack_extraction/extract.py [--limit 13] [--model base|base-chunked|large]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from gliner2 import AutoExtractor

from schema import ENTITY_TYPES, RELATIONS

HERE = Path(__file__).parent
THREADS = HERE / "data" / "threads.json"
OUTPUT = HERE / "data" / "extractions"
MODELS = {
    "base": "fastino/gliner2.5-base-v1",
    "base-chunked": "fastino/gliner2.5-base-v1",
    "large": "fastino/gliner2-large-v1",
}
FLOOR = 0.3
BATCH = 4


def best_scores(items: list[Any]) -> dict[str, float]:
    """Distinct values of one type with their highest confidence."""
    best: dict[str, float] = {}
    for item in items:
        if isinstance(item, dict) and "text" in item:
            text = str(item["text"])
            best[text] = max(best.get(text, 0.0), float(item.get("confidence", 1.0)))

    return best


def relation_rows(name: str, items: list[Any]) -> dict[tuple[str, str, str], float]:
    """(relation, head, tail) -> score, the lower of the two ends' confidences."""
    rows: dict[tuple[str, str, str], float] = {}
    for item in items:
        if not isinstance(item, dict) or "head" not in item or "tail" not in item:
            continue
        head, tail = item["head"], item["tail"]
        if not (isinstance(head, dict) and isinstance(tail, dict)):
            continue
        key = (name, str(head.get("text")), str(tail.get("text")))
        score = min(float(head.get("confidence", 1.0)), float(tail.get("confidence", 1.0)))
        rows[key] = max(rows.get(key, 0.0), score)

    return rows


def tidy(output: dict[str, Any]) -> dict[str, Any]:
    """The model's raw answer as {entities: {type: {value: score}}, relations: [...]}.

    The span model sometimes puts relations at the top level, keyed "name: description";
    those are folded in with the rest.
    """
    entities = {
        name: best_scores(items)
        for name, items in (output.get("entities") or {}).items()
        if name in ENTITY_TYPES
    }
    relations: dict[tuple[str, str, str], float] = {}
    groups = dict(output.get("relation_extraction") or {})
    groups.update({key: value for key, value in output.items() if ": " in key})
    for key, items in groups.items():
        name = key.split(": ")[0]
        if name in RELATIONS and isinstance(items, list):
            for row, score in relation_rows(name, items).items():
                relations[row] = max(relations.get(row, 0.0), score)

    return {
        "entities": entities,
        "relations": [
            {"relation": name, "head": head, "tail": tail, "score": round(score, 4)}
            for (name, head, tail), score in sorted(relations.items())
        ],
    }


def main() -> None:
    """Extract the first --limit threads with one or both models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", choices=[*MODELS, "both"], default="both")
    args = parser.parse_args()

    threads = json.loads(THREADS.read_text())[: args.limit]
    texts = [thread["text"] for thread in threads]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for short in MODELS if args.model == "both" else [args.model]:
        model = AutoExtractor.from_pretrained(MODELS[short]).to(device)
        schema = (
            model.create_schema()
            .entities(dict(ENTITY_TYPES))
            .relations({name: description for name, (description, _, _) in RELATIONS.items()})
        )
        model.extract(texts[0][:300], schema)  # warm up before timing
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if short == "base":
            raw = model.batch_extract(
                texts, schema, batch_size=BATCH, threshold=FLOOR, include_confidence=True
            )
        else:
            raw = model.batch_extract_long(
                texts, schema, batch_size=BATCH, threshold=FLOOR, include_confidence=True,
                chunk_size=384, chunk_overlap=64,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        result = {
            "model": MODELS[short],
            "device": device,
            "seconds_per_thread": round(seconds / len(texts), 4),
            "threads": {thread["doc_id"]: tidy(out) for thread, out in zip(threads, raw)},
        }
        path = OUTPUT / f"{short}.json"
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{short}: {len(texts)} threads, {seconds / len(texts):.3f} s each on {device}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
