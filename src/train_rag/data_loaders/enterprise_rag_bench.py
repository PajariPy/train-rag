"""Data loader for EnterpriseRAG-Bench stored as local parquet files."""

from collections.abc import Sequence
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from train_rag.data_loaders.base import DataLoader
from train_rag.models import Artifact, Corpus, Question, SourceType

DOCUMENTS_FILE = Path("data/documents/test.parquet")
QUESTIONS_FILE = Path("data/questions/test.parquet")


class EnterpriseRagBenchLoader(DataLoader):
    """Load EnterpriseRAG-Bench from a directory laid out like the Hugging Face dataset.

    Expected layout: `data/documents/test.parquet` and `data/questions/test.parquet`.
    """

    name = "enterprise_rag_bench"

    def __init__(self, data_dir: Path, source_types: Sequence[SourceType] | None = None) -> None:
        """Configure the loader.

        Args:
            data_dir: Root of the downloaded dataset.
            source_types: Sources whose documents to load; all sources when None.
                Questions are always loaded in full.
        """
        self.data_dir = data_dir
        self.source_types = source_types

    def load(self) -> Corpus:
        """Load artifacts and questions.

        Returns:
            The normalized corpus.

        Raises:
            FileNotFoundError: If the dataset files are missing.
        """
        return Corpus(
            source=self.name,
            artifacts=self.load_artifacts(),
            questions=self.load_questions(),
        )

    def load_artifacts(self) -> list[Artifact]:
        """Load documents, filtered to the configured source types.

        Returns:
            The normalized artifacts.
        """
        filters = None
        if self.source_types is not None:
            filters = pc.field("source_type").isin([str(source) for source in self.source_types])
        table = pq.read_table(self.resolve(DOCUMENTS_FILE), filters=filters)

        return [
            Artifact(
                id=row["doc_id"],
                source_type=row["source_type"],
                title=row["title"],
                content=row["content"],
            )
            for row in table.to_pylist()
        ]

    def load_questions(self) -> list[Question]:
        """Load the benchmark questions.

        Returns:
            The normalized questions.
        """
        table = pq.read_table(self.resolve(QUESTIONS_FILE))

        return [
            Question(
                id=row["question_id"],
                question_type=row["question_type"],
                text=row["question"],
                gold_answer=row["gold_answer"],
                answer_facts=row["answer_facts"],
                source_types=row["source_types"],
                expected_doc_ids=row["expected_doc_ids"],
            )
            for row in table.to_pylist()
        ]

    def resolve(self, relative: Path) -> Path:
        """Locate a dataset file.

        Args:
            relative: Path of the file inside the dataset root.

        Returns:
            The absolute path of the file.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        path = self.data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"No dataset file at {path}; run `uv run train-rag-download`.")

        return path
