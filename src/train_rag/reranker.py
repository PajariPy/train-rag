"""Rerankers: score how well each document answers a query.

Two kinds: a cross-encoder (gte-reranker-modernbert) reads the query and a document together;
a late-interaction model (LateOn) compares their token vectors with MaxSim.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from sentence_transformers import CrossEncoder, MultiVectorEncoder
from sentence_transformers.base.modules.transformer import Transformer

GTE_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"
LATEON = "lightonai/LateOn"
# The model's whole context, and the cut the accepted pipeline reranks on.
FULL_TOKENS = 8192
SHORT_TOKENS = 1024
# Token budget of one LateOn encoding batch (documents x longest document); fits 8 GB.
BATCH_TOKENS = 8192
CHARS_PER_TOKEN = 4.0


class Reranker(ABC):
    """Scores query and document pairs; implementations are interchangeable."""

    name: ClassVar[str]

    @abstractmethod
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each text against the query.

        Args:
            query: The question or search text.
            texts: Document texts.

        Returns:
            One relevance score per text, in order; higher is more relevant.
        """


class GteModernBertReranker(Reranker):
    """gte-reranker-modernbert-base: a cross-encoder from the same family as the embedder.

    It reads the query and a document together (up to 8,192 tokens), so it can judge
    relevance far more precisely than comparing two separate vectors, but it has to run
    once per document. Runs in half precision. Cut to the first 1,024 tokens, it ranks as
    well as on the full text at about 2.3x the speed (article 006).
    """

    name: ClassVar[str] = "gte-reranker-modernbert-base"

    def __init__(self, batch_size: int = 4, max_tokens: int = FULL_TOKENS) -> None:
        """Load the model in half precision, on the GPU when there is one.

        Args:
            batch_size: Pairs scored at once.
            max_tokens: Tokens of each query and document pair read; the rest is cut.
        """
        self.max_tokens = max_tokens
        self.model = CrossEncoder(
            GTE_RERANKER,
            max_length=max_tokens,
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each text against the query.

        Texts are scored shortest first, so each batch holds texts of similar length and
        little padding is computed.

        Args:
            query: The question or search text.
            texts: Document texts.

        Returns:
            One relevance score per text, in order; higher is more relevant.
        """
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        scores = self.model.predict(
            [(query, texts[index]) for index in order], batch_size=self.batch_size
        )
        by_index = dict(zip(order, scores.tolist(), strict=True))

        return [float(by_index[index]) for index in range(len(texts))]


def maxsim(query: torch.Tensor, document: torch.Tensor) -> float:
    """Sum over query tokens of the best dot product with any document token.

    Args:
        query: Query token vectors (query tokens x dim).
        document: Document token vectors (document tokens x dim).

    Returns:
        The MaxSim score.
    """
    return float((query @ document.T).max(dim=1).values.sum())


class LateOnReranker(Reranker):
    """LateOn: a late-interaction (ColBERT) model, the accepted reranker since 25 Nov.

    Each document becomes one vector per token and the query one per token; a document's
    score is the sum, over query tokens, of the best match among its tokens (MaxSim). The
    document vectors don't depend on the query, so in production they are computed at
    ingestion and a query costs only its own encoding plus MaxSim (~14 ms for ~97
    candidates). Here they are computed on the fly, which is slower but needs no stored
    vectors. Reads the first 1,024 tokens: on the dev set whole documents ranked worse.
    """

    name: ClassVar[str] = "lateon"

    def __init__(self, max_tokens: int = SHORT_TOKENS, batch_tokens: int = BATCH_TOKENS) -> None:
        """Load the model in half precision, on the GPU when there is one.

        Args:
            max_tokens: Tokens of each document read; the rest is cut.
            batch_tokens: Token budget of one encoding batch.

        Raises:
            TypeError: If the model's first module isn't a Transformer.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MultiVectorEncoder(
            LATEON, device=device, model_kwargs={"torch_dtype": torch.float16}
        )
        module = self.model[0]
        if not isinstance(module, Transformer):
            raise TypeError(f"{LATEON} should start with a Transformer module")

        module.document_length = max_tokens
        self.max_tokens = max_tokens
        self.batch_tokens = batch_tokens

    def batches(self, texts: Sequence[str]) -> list[list[int]]:
        """Group text positions, shortest first, into batches within the token budget.

        Args:
            texts: Document texts.

        Returns:
            Batches of positions into `texts`.
        """
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        batches: list[list[int]] = []
        batch: list[int] = []
        for index in order:
            longest = min(self.max_tokens, len(texts[index]) / CHARS_PER_TOKEN)
            if batch and (len(batch) + 1) * longest > self.batch_tokens:
                batches.append(batch)
                batch = []
            batch.append(index)

        if batch:
            batches.append(batch)

        return batches

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each text against the query with MaxSim.

        Args:
            query: The question or search text.
            texts: Document texts.

        Returns:
            One relevance score per text, in order; higher is more relevant.
        """
        query_vectors = self.model.encode_query([query])[0]
        scores = [0.0] * len(texts)
        for batch in self.batches(texts):
            documents = self.model.encode_document(
                [texts[index] for index in batch], batch_size=len(batch)
            )
            for index, document in zip(batch, documents, strict=True):
                scores[index] = maxsim(query_vectors, document)

        return scores
