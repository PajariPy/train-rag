r"""Label documents with their key facts using DeepSeek, for the fact-extraction training set.

Reads the sets written by sample.py, sends each document with the instructions in
prompts.yaml to DeepSeek (`deepseek-flash`, JSON mode), checks the answer has the expected
shape, and stores it in data/labels.json keyed by document id and a hash of the prompt, so a
rerun never pays twice and a prompt change starts fresh labels.

DeepSeek charges half outside its peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays);
the script refuses to start in peak hours unless --peak is given.

Usage (from the main checkout, which has the data and the .env):
    uv run --project ~/projects/rag-implementation/train-rag python \\
        exploration/fact_extraction/label.py {train|pilot} [--limit N] [--peak] [--show N]
"""

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.processing.normalize import normalize_documents
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents

HERE = Path(__file__).parent
SAMPLE = HERE / "data" / "sample.json"
LABELS = HERE / "data" / "labels.json"
ENV = Path.home() / "projects" / "rag-implementation" / "train-rag" / ".env"
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_CHARS = 30_000  # longer documents (a few transcripts and wiki pages) are cut
WORKERS = 8
RETRIES = 3
# Dollars per million tokens: (cache-miss input, cache-hit input, output), off-peak and peak.
PRICES = {"off-peak": (0.15, 0.003, 0.60), "peak": (0.30, 0.006, 1.20)}
PEAK_HOURS = {1, 2, 3, 6, 7, 8, 9}  # UTC, weekdays


class FactLabel(BaseModel):
    """What a document says, in the shape the small model will learn to write."""

    about: str
    entities: list[str]
    facts: list[str]


def peak_now() -> bool:
    """Whether DeepSeek's peak prices apply right now."""
    now = datetime.now(UTC)

    return now.weekday() < 5 and now.hour in PEAK_HOURS


def prompt_hash(prompts: dict[str, str]) -> str:
    """A short hash of the model and prompts, so labels made with another prompt aren't reused."""
    blob = json.dumps(
        [MODEL, prompts["fact_extraction_instructions"], prompts["fact_extraction_prompt"]]
    )

    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class Labeller:
    """Calls DeepSeek, keeps the labels and counts the tokens; safe across worker threads."""

    def __init__(self, prompts: dict[str, str], key: str) -> None:
        """Load the saved labels and set up the client.

        Args:
            prompts: The instructions and prompt template.
            key: The DeepSeek API key.
        """
        self.prompts = prompts
        self.version = prompt_hash(prompts)
        self.client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)
        self.lock = threading.Lock()
        self.labels: dict[str, dict[str, object]] = (
            json.loads(LABELS.read_text()) if LABELS.exists() else {}
        )
        self.tokens = {"miss": 0, "hit": 0, "output": 0}
        self.failed: list[str] = []

    def key(self, doc_id: str) -> str:
        """The label's key: the document and the prompt version."""
        return f"{doc_id}:{self.version}"

    def label(self, doc_id: str, source: str, title: str, body: str) -> None:
        """Label one document unless a label for it and this prompt exists.

        Args:
            doc_id: The document's id.
            source: Its source type.
            title: Its title.
            body: Its normalized text.
        """
        if self.key(doc_id) in self.labels:
            return
        user = self.prompts["fact_extraction_prompt"].format(
            source=source, title=title.splitlines()[0] if title else "", text=body[:MAX_CHARS]
        )
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": self.prompts["fact_extraction_instructions"]},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            # On by default: it spends the whole token budget reasoning and returns no answer.
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 1_500,
        }
        for _attempt in range(RETRIES):
            try:
                response = self.client.post(URL, json=payload)
                response.raise_for_status()
                data = response.json()
                label = FactLabel.model_validate_json(data["choices"][0]["message"]["content"])
            except (httpx.HTTPError, ValidationError, KeyError, ValueError):
                continue
            usage = data.get("usage", {})
            with self.lock:
                self.tokens["hit"] += usage.get("prompt_cache_hit_tokens", 0)
                self.tokens["miss"] += usage.get(
                    "prompt_cache_miss_tokens", usage.get("prompt_tokens", 0)
                )
                self.tokens["output"] += usage.get("completion_tokens", 0)
                self.labels[self.key(doc_id)] = {"doc_id": doc_id, **label.model_dump()}

            return
        with self.lock:
            self.failed.append(doc_id)

    def save(self) -> None:
        """Write the labels through a temporary file, so a crash never leaves it half written."""
        with self.lock:
            partial = LABELS.with_suffix(".json.tmp")
            partial.write_text(json.dumps(self.labels, ensure_ascii=False, indent=1))
            partial.replace(LABELS)

    def cost(self) -> float:
        """Dollars for the tokens counted, at the price that applies now."""
        miss, hit, output = PRICES["peak" if peak_now() else "off-peak"]

        return (
            self.tokens["miss"] * miss + self.tokens["hit"] * hit + self.tokens["output"] * output
        ) / 1e6


def round_robin(rows: list[dict[str, object]]) -> list[str]:
    """Training documents ordered source by source and band by band, in turns.

    So `--limit N` takes a spread of sources and sizes, not the first N of one source.
    """
    groups: dict[tuple[object, object], list[str]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["band"]), []).append(str(row["doc_id"]))
    queues = [groups[key] for key in sorted(groups, key=lambda key: (key[1], str(key[0])))]
    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))

    return ordered


def main() -> None:
    """Label one set of documents, or show labels already made."""
    parser = argparse.ArgumentParser()
    parser.add_argument("set", choices=["train", "pilot"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--peak", action="store_true", help="run even at peak prices")
    parser.add_argument("--show", type=int, default=0, help="print this many labels and stop")
    args = parser.parse_args()

    prompts = yaml.safe_load((HERE / "prompts.yaml").read_text())
    sample = json.loads(SAMPLE.read_text())
    doc_ids = round_robin(sample["train"]) if args.set == "train" else sample["pilot"]
    doc_ids = doc_ids[: args.limit] if args.limit else doc_ids
    labeller = Labeller(prompts, dotenv_values(ENV)["DEEPSEEK_API_KEY"] or "")

    if args.show:
        for doc_id in doc_ids[: args.show]:
            label = labeller.labels.get(labeller.key(doc_id))
            print(json.dumps(label, indent=1, ensure_ascii=False))
        return
    if peak_now() and not args.peak:
        raise SystemExit("DeepSeek peak hours: prices are doubled. Rerun later or pass --peak.")

    with Session(get_engine()) as session:
        internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, doc_ids)
        records = load_documents(session, sorted(internal))
    sources = {record.doc_id: record.source_type for record in records}
    by_id = {doc.doc_id: doc for doc in normalize_documents(records)}
    todo = [doc_id for doc_id in doc_ids if doc_id in by_id]
    print(f"{args.set}: {len(todo)} documents, prompt {labeller.version}", flush=True)

    def work(doc_id: str) -> None:
        doc = by_id[doc_id]
        labeller.label(doc_id, str(sources[doc_id]), doc.title, doc.text)

    progress = RunProgress(get_engine(), f"label facts with DeepSeek ({args.set})", len(todo))
    try:
        with progress, ThreadPoolExecutor(WORKERS) as pool:
            for done, _ in enumerate(pool.map(work, todo), 1):
                if done % 10 == 0 or done == len(todo):
                    progress.advance(
                        "labelling", 10 if done % 10 == 0 else done % 10,
                        note=f"DeepSeek: ${labeller.cost():.3f}",
                    )
                if done % 100 == 0:
                    labeller.save()
                    print(f"{done}/{len(todo)} · ${labeller.cost():.3f}", flush=True)
    finally:
        labeller.save()
    print(
        f"done: {len(todo) - len(labeller.failed)} labelled, {len(labeller.failed)} failed, "
        f"{labeller.tokens} tokens, about ${labeller.cost():.3f}",
        flush=True,
    )


main()
