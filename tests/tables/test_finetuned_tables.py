"""Tests for the separate tables of the fine-tuned embedder."""

import pytest

from train_rag.models import SourceType
from train_rag.tables import EMBEDDING_TABLES, FINETUNED_TABLES, tables_for


def test_finetuned_tables_are_separate() -> None:
    """Every source gets its own fine-tuned table, apart from the base model's."""
    assert FINETUNED_TABLES[SourceType.SLACK].__tablename__ == "slack_embeddings_768_ft"
    assert {table.__tablename__ for table in FINETUNED_TABLES.values()}.isdisjoint(
        table.__tablename__ for table in EMBEDDING_TABLES[768].values()
    )


def test_tables_for_picks_by_embedder_kind() -> None:
    """A fine-tuned embedder searches its own tables; the others keep theirs."""
    assert tables_for(768, finetuned=True) is FINETUNED_TABLES
    assert tables_for(768, finetuned=False) is EMBEDDING_TABLES[768]
    assert tables_for(384, finetuned=False) is EMBEDDING_TABLES[384]


def test_finetuned_embedder_uses_its_own_tables() -> None:
    """The fine-tuned gte stores under its own name, in the fine-tuned tables."""
    from train_rag.embedder import FineTunedGteEmbedder, GteModernBertEmbedder

    assert FineTunedGteEmbedder.name != GteModernBertEmbedder.name
    assert FineTunedGteEmbedder.dimensions == 768
    assert tables_for(FineTunedGteEmbedder.dimensions, FineTunedGteEmbedder.finetuned) is (
        FINETUNED_TABLES
    )
    assert not GteModernBertEmbedder.finetuned


def test_finetuned_model_path_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The weights folder can be set per run; otherwise the default folder is used."""
    from train_rag.embedder import FINETUNED_GTE, finetuned_model_path

    monkeypatch.delenv("FINETUNED_EMBEDDER", raising=False)
    assert finetuned_model_path() == FINETUNED_GTE

    monkeypatch.setenv("FINETUNED_EMBEDDER", "runs/x/final")
    assert finetuned_model_path() == "runs/x/final"


def test_finetuned_retrievers_are_registered() -> None:
    """The evaluation can search with the fine-tuned embedder, alone and in the hybrid."""
    from train_rag.evaluation.registry import RETRIEVERS

    assert {"gte-ft-chunk", "hybrid-ft", "hybrid-ft-rerank"} <= RETRIEVERS.keys()
