"""The test sample: a small corpus to compare retrieval methods on before indexing all."""

from sqlmodel import Session

from train_rag.repository.document import ids_of_doc_ids, sample_document_ids
from train_rag.repository.question import load_questions

SAMPLE_SEED = "chunk-sample"


def sample_documents(session: Session, dataset: str, distractors: int) -> set[int]:
    """Pick every document a benchmark question cites, plus random other documents.

    The same seed always picks the same documents, so every retrieval method indexed on
    the sample searches the same corpus and their recall can be compared.

    Args:
        session: Active database session.
        dataset: Dataset to sample from.
        distractors: Number of random documents added to the cited ones.

    Returns:
        The row ids of the sample.
    """
    cited = [
        doc_id
        for question in load_questions(session, dataset)
        for doc_id in question.expected_doc_ids
    ]

    return ids_of_doc_ids(session, dataset, cited) | sample_document_ids(
        session, dataset, distractors, SAMPLE_SEED
    )
