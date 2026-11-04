"""Rerankers: score how well each document answers a query, reading both together."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from sentence_transformers import CrossEncoder

GTE_RERANKER = "Alibaba-NLP/gte-reranker-modernbert-base"


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
    once per document. Runs in half precision.
    """

    name: ClassVar[str] = "gte-reranker-modernbert-base"
    max_tokens: ClassVar[int] = 8192

    def __init__(self, batch_size: int = 4) -> None:
        """Load the model in half precision, on the GPU when there is one.

        Args:
            batch_size: Pairs scored at once.
        """
        self.model = CrossEncoder(
            GTE_RERANKER,
            max_length=self.max_tokens,
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
