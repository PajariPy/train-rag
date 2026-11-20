"""What the field-search agent does for one question: its tool calls, check and result."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from train_rag.models.evaluation import TraceStep
from train_rag.models.retrieval import SearchHit

CaseVerdict = Literal["same_answer", "one_case_fits", "several_cases_fit"]


class ToolCall(BaseModel):
    """One tool call of the model: its id (when the API gives one), name and arguments."""

    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    """The agent's check of its final documents: does the answer depend on which is picked?

    `fits` are the documents' ids (the agent's labels turned into dataset ids) that fit
    the question as asked.
    """

    reasoning: str = ""
    verdict: CaseVerdict
    fits: list[str] = Field(default_factory=list)


class AgentAsk(BaseModel):
    """One question the agent put to the person, and the reply."""

    ask: str
    reply: str


class AgentRun(BaseModel):
    """The agent's work on one question.

    `documents` are the final top documents, in full, fitting ones first; `verdict` is
    the last check (None when the agent never finished); `side_by_side` says that several
    cases still fit when it stopped, so the answer shows them side by side.
    """

    documents: list[SearchHit] = Field(default_factory=list)
    picks: list[str] = Field(default_factory=list)
    candidates: int = 0
    searches: list[dict[str, Any]] = Field(default_factory=list)
    asks: list[AgentAsk] = Field(default_factory=list)
    checks: list[Verdict] = Field(default_factory=list)
    side_by_side: bool = False
    trace: list[TraceStep] = Field(default_factory=list)

    @property
    def verdict(self) -> Verdict | None:
        """The last check.

        Returns:
            The verdict, or None.
        """
        return self.checks[-1] if self.checks else None
