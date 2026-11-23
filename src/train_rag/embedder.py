"""Text embedders: turn texts into vectors for the embedding tables."""

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch
from sentence_transformers import SentenceTransformer

MINI_LM = "sentence-transformers/all-MiniLM-L6-v2"
GTE_MODERNBERT = "Alibaba-NLP/gte-modernbert-base"
# Where the chosen fine-tuned gte lives (a training run's `final` folder, copied here);
# FINETUNED_EMBEDDER in the environment overrides it.
FINETUNED_GTE = "models/gte-modernbert-ft"


def finetuned_model_path() -> str:
    """The folder of the fine-tuned gte weights.

    Returns:
        FINETUNED_EMBEDDER from the environment, or `FINETUNED_GTE`.
    """
    return os.environ.get("FINETUNED_EMBEDDER", FINETUNED_GTE)


class Embedder(ABC):
    """Embeds texts as vectors of a fixed size; implementations are interchangeable."""

    name: ClassVar[str]
    dimensions: ClassVar[int]
    max_tokens: ClassVar[int]
    finetuned: ClassVar[bool] = False  # fine-tuned models keep their vectors in own tables

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts.

        Args:
            texts: Texts to embed.

        Returns:
            One normalized vector per text, in order.
        """


class MiniLmEmbedder(Embedder):
    """all-MiniLM-L6-v2: small and fast, 384 dimensions.

    It reads at most 256 word pieces per text (about 1,000 characters); the rest of a
    longer text is cut off.
    """

    name: ClassVar[str] = "all-MiniLM-L6-v2"
    dimensions: ClassVar[int] = 384
    max_tokens: ClassVar[int] = 256

    def __init__(self, batch_size: int = 128) -> None:
        """Load the model, on the GPU when there is one.

        Args:
            batch_size: Texts embedded at once.
        """
        self.model = SentenceTransformer(MINI_LM)
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts as unit vectors, so cosine distance ranks them.

        Args:
            texts: Texts to embed.

        Returns:
            One 384-dimensional vector per text, in order.
        """
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )

        return [vector.tolist() for vector in vectors]


class GteModernBertEmbedder(Embedder):
    """gte-modernbert-base: reads up to 8,192 tokens (about 30,000 characters), 768 dimensions.

    Runs in half precision; a long text costs far more than a short one, so batches are
    small.
    """

    name: ClassVar[str] = "gte-modernbert-base"
    dimensions: ClassVar[int] = 768
    max_tokens: ClassVar[int] = 8192

    def __init__(self, batch_size: int = 4, model_path: str = GTE_MODERNBERT) -> None:
        """Load the model in half precision, on the GPU when there is one.

        Args:
            batch_size: Texts embedded at once.
            model_path: Hugging Face name or local folder of the weights.
        """
        self.model = SentenceTransformer(model_path, model_kwargs={"torch_dtype": torch.float16})
        self.model.max_seq_length = self.max_tokens
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts as unit vectors, so cosine distance ranks them.

        Args:
            texts: Texts to embed.

        Returns:
            One 768-dimensional vector per text, in order.
        """
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True
        )

        return [vector.astype("float32").tolist() for vector in vectors]


class FineTunedGteEmbedder(GteModernBertEmbedder):
    """gte-modernbert-base fine-tuned on our own questions with hard negatives.

    Same size and settings as the base model; its vectors live in the `_768_ft` tables.
    Design and runs: docs/brainstorm/005, exploration/embedder_finetune/.
    """

    name: ClassVar[str] = "gte-modernbert-ft"
    finetuned: ClassVar[bool] = True

    def __init__(self, model_path: str | None = None, batch_size: int = 4) -> None:
        """Load the fine-tuned weights.

        Args:
            model_path: Folder of the saved model; `finetuned_model_path()` by default.
            batch_size: Texts embedded at once.
        """
        super().__init__(batch_size=batch_size, model_path=model_path or finetuned_model_path())


EMBEDDERS: dict[str, type[Embedder]] = {
    "minilm": MiniLmEmbedder,
    "gte": GteModernBertEmbedder,
}
