"""Build a BM25 keyword index over whole documents and save it to disk."""

import json
import time
from array import array
from pathlib import Path

import bm25s
import Stemmer
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import get_engine
from train_rag.models import SourceType
from train_rag.models.retrieval import IndexedDocument
from train_rag.processing.normalize import normalize_content, normalize_text
from train_rag.repository.document import document_ids, load_documents
from train_rag.sampling import sample_documents

BM25_DIR = Path("data/bm25")
DOCUMENTS_FILE = "documents.json"
DOCUMENT_BATCH = 5_000


def bm25_index_dir(sample: int | None) -> Path:
    """Name the folder of a BM25 index.

    Args:
        sample: Number of distractors of the test sample, or None for every document.

    Returns:
        `data/bm25/all`, or `data/bm25/sample_<n>` for a sample.
    """
    return BM25_DIR / ("all" if sample is None else f"sample_{sample}")


def tokenize(texts: list[str], stemmer: Stemmer.Stemmer) -> list[list[str]]:
    """Split texts into lowercased, stemmed words without English stopwords.

    Args:
        texts: Texts to split.
        stemmer: English stemmer.

    Returns:
        The words of each text.
    """
    tokens = bm25s.tokenize(
        texts, stopwords="en", stemmer=stemmer.stemWords, return_ids=False, show_progress=False
    )

    return [list(words) for words in tokens]


def index_bm25(sample: int | None = None) -> Path:
    """Index the normalized title and text of every document (or of the test sample) with BM25.

    Args:
        sample: Number of random distractors of the test sample, or None for every
            document.

    Returns:
        The folder the index was saved to.
    """
    started = time.perf_counter()
    stemmer = Stemmer.Stemmer("english")
    dataset = EnterpriseRagBenchLoader.name
    documents: list[IndexedDocument] = []
    # Words are kept as ids in compact arrays: as Python strings the whole corpus (about
    # 300M words) would not fit in memory.
    vocabulary: dict[str, int] = {}
    corpus: list[array[int]] = []

    with Session(get_engine()) as session:
        selected = sample_documents(session, dataset, sample) if sample is not None else None
        for source_type in SourceType:
            ids = [
                document_id
                for document_id in document_ids(session, dataset, source_type)
                if selected is None or document_id in selected
            ]
            for start in range(0, len(ids), DOCUMENT_BATCH):
                batch = load_documents(session, ids[start : start + DOCUMENT_BATCH])
                texts = [
                    normalize_text(document.title) + "\n" + normalize_content(document.content)
                    for document in batch
                ]
                corpus.extend(
                    array("i", (vocabulary.setdefault(word, len(vocabulary)) for word in words))
                    for words in tokenize(texts, stemmer)
                )
                documents.extend(
                    IndexedDocument(
                        document_id=document.id or 0,
                        doc_id=document.doc_id,
                        source_type=source_type,
                    )
                    for document in batch
                )
            print(f"{source_type}: tokenized {len(ids):,} documents", flush=True)

    model = bm25s.BM25()
    model.index((corpus, vocabulary), show_progress=False)
    index_dir = bm25_index_dir(sample)
    model.save(index_dir, show_progress=False)
    (index_dir / DOCUMENTS_FILE).write_text(
        json.dumps([document.model_dump(mode="json") for document in documents])
    )
    print(
        f"Indexed {len(documents):,} documents into {index_dir} "
        f"in {time.perf_counter() - started:.0f}s"
    )

    return index_dir
