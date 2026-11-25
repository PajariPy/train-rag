"""Tests for LateOn's MaxSim score and its token-budget batches."""

import torch

from train_rag.reranker import LateOnReranker, maxsim


def test_maxsim_sums_each_query_tokens_best_match() -> None:
    """Each query token takes its best document token; the best matches are summed."""
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    document = torch.tensor([[0.9, 0.1], [0.2, 0.5], [0.0, 0.3]])

    assert abs(maxsim(query, document) - (0.9 + 0.5)) < 1e-6


def test_batches_stay_within_the_token_budget() -> None:
    """Short texts share a batch; texts at the token cap go one per batch."""
    reranker = LateOnReranker.__new__(LateOnReranker)
    reranker.max_tokens = 1024
    reranker.batch_tokens = 1500
    texts = ["x" * 400, "x" * 100_000, "x" * 400, "x" * 100_000]

    batches = reranker.batches(texts)

    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert [0, 2] in batches
    assert all(len(batch) == 1 for batch in batches if 1 in batch or 3 in batch)
