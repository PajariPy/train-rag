"""What a person remembers about the document behind their question."""

from pydantic import BaseModel


class MemoryNote(BaseModel):
    """A person's memory of the document they are asking about: big picture, few details.

    Some notes hold one detail that is slightly wrong, as human memory often is;
    `wrong_detail` says which, so those questions can be scored apart.
    """

    question_id: str
    note: str
    wrong_detail: str | None = None

    @property
    def clean(self) -> bool:
        """Tell whether every detail in the note is true.

        Returns:
            True when the note has no misremembered detail.
        """
        return self.wrong_detail is None
