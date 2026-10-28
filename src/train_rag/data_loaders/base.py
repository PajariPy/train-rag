"""Common interface for all data loaders."""

from abc import ABC, abstractmethod

from train_rag.models import Corpus


class DataLoader(ABC):
    """Load data from some place and normalize it into a Corpus.

    Subclasses own the details of where data lives and how it is shaped;
    the rest of the pipeline only ever sees `Corpus`.
    """

    name: str

    @abstractmethod
    def load(self) -> Corpus:
        """Load the full corpus from the source.

        Returns:
            The normalized corpus.
        """
