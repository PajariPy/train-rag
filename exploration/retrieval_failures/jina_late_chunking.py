r"""Does late chunking with jina-embeddings-v3 lift the expected document in its pool?

Reads the pools written by pools.py: for each problem case, gte's own top 50 plus the
expected document. Every pool document is scored three ways with jina-embeddings-v3, by
its best chunk's cosine similarity to the query:

- naive: 1,000-character chunks, each embedded on its own;
- late: the document embedded in windows of WINDOW_CHARS (most documents fit in one),
  then the token vectors inside each chunk's span averaged (late chunking, Günther et al. 2024);
- whole: one vector for the document (mean of its first WHOLE_TOKENS tokens).

The expected document's rank among itself and gte's top 50 is recorded for each.

Runs outside the project environment, because jina's model code needs transformers 4:

    uv run --no-project --python 3.13 --with "transformers<5" --with einops \\
        --with "torch==2.15.0" --with numpy \\
        python exploration/retrieval_failures/jina_late_chunking.py POOLS.json OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "jinaai/jina-embeddings-v3"
CHUNK_CHARS = 1_000
# About 3,000 tokens: without flash-attn, attention memory grows with the square of the length.
WINDOW_CHARS = 12_000
WHOLE_TOKENS = 4_096  # the whole-document vector reads at most this many tokens
PASSAGE, QUERY = "retrieval.passage", "retrieval.query"


def spans(text: str, size: int = CHUNK_CHARS) -> list[tuple[int, int]]:
    """Cut a text into pieces of at most `size` characters, at line ends or spaces.

    Args:
        text: The document.
        size: Longest piece.

    Returns:
        Character spans (start, end) covering the text, in order.
    """
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        if text[start:end].strip():
            pieces.append((start, end))
        start = end

    return pieces


class Jina:
    """jina-embeddings-v3 on the GPU, with direct access to token vectors."""

    def __init__(self) -> None:
        """Load the model in half precision."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float16)
        self.model = self.model.cuda().eval()
        self.instructions = self.model._task_instructions
        self.task_ids = self.model._adaptation_map

    def encode(
        self, texts: list[str], task: str, max_length: int = 1024, batch_size: int = 16
    ) -> np.ndarray:
        """Embed texts the normal way (mean pooling, instruction added), as unit vectors."""
        vectors = self.model.encode(texts, task=task, max_length=max_length, batch_size=batch_size)

        return np.asarray(vectors, dtype=np.float32)

    @torch.no_grad()
    def late_chunks(self, text: str, pieces: list[tuple[int, int]]) -> np.ndarray:
        """Late chunking: embed the text in windows, then mean-pool each piece's tokens.

        Args:
            text: The document.
            pieces: Character spans of its chunks.

        Returns:
            One unit vector per piece, in order.
        """
        prefix = self.instructions[PASSAGE]
        vectors = []
        window_start = 0
        window: tuple[torch.Tensor, np.ndarray] | None = None
        for start, end in pieces:
            if window is None or end > window_start + WINDOW_CHARS:
                window_start = start
                window = self.window(prefix, text[window_start : window_start + WINDOW_CHARS])
            states, offsets = window
            low, high = start - window_start + len(prefix), end - window_start + len(prefix)
            starts, ends = offsets[:, 0], offsets[:, 1]
            inside = (starts >= low) & (ends <= high) & (ends > starts)
            chosen = states[torch.from_numpy(inside).cuda()]
            vector = chosen.mean(0) if len(chosen) else states.mean(0)
            vectors.append(torch.nn.functional.normalize(vector.float(), dim=0).cpu().numpy())

        return np.stack(vectors)

    def window(self, prefix: str, text: str) -> tuple[torch.Tensor, np.ndarray]:
        """Token vectors and character offsets of one window, instruction included."""
        encoded = self.tokenizer(
            prefix + text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=8192,
        )
        offsets = encoded.pop("offset_mapping")[0].numpy()
        encoded = {key: value.cuda() for key, value in encoded.items()}
        mask = torch.full((1,), self.task_ids[PASSAGE], dtype=torch.int32, device="cuda")
        states = self.model(**encoded, adapter_mask=mask)[0][0]

        return states, offsets


def rank(target: float, others: list[float]) -> int:
    """Rank of the target score among itself and the others (1 = top)."""
    return 1 + sum(other > target for other in others)


def main() -> None:
    """Score every pool three ways and write the expected document's ranks."""
    data = json.loads(Path(sys.argv[1]).read_text())
    pools, documents = data["pools"], data["documents"]
    jina = Jina()
    pieces = {doc_id: spans(text) for doc_id, text in documents.items()}

    scores: dict[str, dict[str, np.ndarray]] = {"naive": {}, "late": {}, "whole": {}}
    for number, (doc_id, text) in enumerate(documents.items(), 1):
        chunk_texts = [text[start:end] for start, end in pieces[doc_id]] or [text]
        scores["naive"][doc_id] = jina.encode(chunk_texts, PASSAGE)
        scores["late"][doc_id] = jina.late_chunks(text, pieces[doc_id] or [(0, len(text))])
        if number % 200 == 0:
            print(f"embedded {number}/{len(documents)} documents", flush=True)
            torch.cuda.empty_cache()
    whole_ids = list(documents)
    whole = jina.encode([documents[d] for d in whole_ids], PASSAGE, WHOLE_TOKENS, batch_size=1)
    scores["whole"] = {d: whole[i : i + 1] for i, d in enumerate(whole_ids)}

    results = []
    for qid, pool in pools.items():
        query = jina.encode([pool["question"]], QUERY)[0]
        target = pool["expected"]
        others = [d for d in pool["gte_top50"] if d != target and d in documents]
        row: dict[str, object] = {"question_id": qid, "pool": len(others) + 1}
        for method, vectors in scores.items():
            best = {d: float(np.max(vectors[d] @ query)) for d in [target, *others]}
            row[method] = {
                "rank": rank(best[target], [best[d] for d in others]),
                "target": best[target],
                "top": max(best[d] for d in others),
            }
        results.append(row)
    Path(sys.argv[2]).write_text(json.dumps(results, indent=1))
    print(f"wrote {sys.argv[2]}", flush=True)


main()
