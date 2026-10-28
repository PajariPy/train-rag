"""Shared test fixtures."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from train_rag.models import Artifact

DOCUMENTS_DIR = Path(__file__).parent / "fixtures" / "documents"


@pytest.fixture
def real_document() -> Callable[[str], Artifact]:
    """Load real EnterpriseRAG-Bench documents saved under `fixtures/documents`.

    Returns:
        A function that loads a fixture by file stem as an artifact.
    """

    def load(name: str) -> Artifact:
        """Load one saved document.

        Args:
            name: File stem of the fixture, e.g. `slack_channel`.

        Returns:
            The document as an artifact.
        """
        raw = json.loads((DOCUMENTS_DIR / f"{name}.json").read_text())

        return Artifact(
            id=raw["doc_id"],
            source_type=raw["source_type"],
            title=raw["title"],
            content=raw["content"],
        )

    return load
