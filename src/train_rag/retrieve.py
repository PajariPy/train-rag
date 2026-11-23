"""Answer one search query from the command line with the retrieval pipeline."""

from train_rag.models.retrieval import RetrievedDocument
from train_rag.pipelines.retrieval import RetrievalPipeline


async def run_retrieval(
    query: str, limit: int = 10, rerank: bool = True
) -> list[RetrievedDocument]:
    """Load the pipeline, answer one query and close it.

    Args:
        query: The question or search text.
        limit: Number of documents.
        rerank: Rerank the fused documents.

    Returns:
        The documents, best first.
    """
    pipeline = RetrievalPipeline.load(rerank=rerank)
    try:
        return await pipeline.run(query, limit)
    finally:
        await pipeline.close()
