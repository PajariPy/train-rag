"""The agent's loop for one question: search, finish, rerank, check, maybe ask, and stop.

1. The agent searches with `search_documents` and `search_fields` (at most `MAX_SEARCHES`
   in total) and may ask the person (`ask_user`, at most `max_asks`; each reply adds a
   search). It calls `finish` with the documents that answer.
2. Its picks and every search's results, fused, are reranked against the question; the
   top `FINAL_DOCUMENTS` are its final documents.
3. The check, one more turn on the same conversation: the final documents in full (cut at
   `MAX_DOCUMENT_CHARS`) with the baseline disagreement prompt; the agent calls
   `judge_cases`. If several cases fit and it can still ask, it is told to ask and finish
   again (at most `MAX_CHECKS` checks); otherwise the answer shows the cases side by side.
"""

from train_rag.evaluation.field_agent.chat import GeminiChat
from train_rag.evaluation.field_agent.searches import MAX_SEARCHES, AgentSearches
from train_rag.evaluation.field_agent.tools import ASK_USER, FINISH, JUDGE_CASES, SEARCH_TOOLS
from train_rag.evaluation.person import Person
from train_rag.evaluation.pipeline_answer import MAX_DOCUMENT_CHARS
from train_rag.mined_fields.profiles import source_line
from train_rag.models.field_agent import AgentAsk, AgentRun, ToolCall, Verdict
from train_rag.models.retrieval import SearchHit
from train_rag.prompts import prompt

FINAL_DOCUMENTS = 5
SEARCHES_PER_REPLY = 1
MAX_CHECKS = 2
# Turns allowed beyond the tool budgets, for refused or malformed calls.
SPARE_TURNS = 3
NO_ONE_TO_ASK = "There is no one to ask, or no questions left."
SEVERAL = "several_cases_fit"


class AgentLoop:
    """One question's conversation between the agent and its tools."""

    def __init__(
        self,
        chat: GeminiChat,
        searches: AgentSearches,
        question: str,
        person: Person | None,
        max_asks: int,
    ) -> None:
        """Set up the loop.

        Args:
            chat: The conversation, started with the question.
            searches: The searches and their budget.
            question: The question, which the final rerank reads.
            person: Who to ask, or None.
            max_asks: Questions the agent may ask.
        """
        self.chat = chat
        self.searches = searches
        self.question = question
        self.person = person
        self.max_asks = max_asks if person is not None else 0
        self.run = AgentRun()
        self.scores: dict[str, float] = {}  # reranker scores against the question
        self.checking = False
        self.done = False

    def allowed(self) -> list[str]:
        """The tools the agent may call in the next turn.

        Returns:
            Tool names.
        """
        if self.checking:
            return [JUDGE_CASES]
        tools = [*self.searches.allowed(), FINISH]
        if len(self.run.asks) < self.max_asks:
            tools.append(ASK_USER)

        return tools

    def execute(self) -> AgentRun:
        """Run the conversation to its end.

        Returns:
            What the agent did and its final documents.
        """
        asking = self.max_asks * (1 + SEARCHES_PER_REPLY)
        turns = MAX_SEARCHES + asking + 2 * MAX_CHECKS + SPARE_TURNS
        for _turn in range(turns):
            calls = self.chat.step(self.allowed())
            if not calls:
                break
            self.handle(calls)
            if self.done:
                break
        if not self.run.documents:  # never finished: rerank what the searches found
            self.final_documents([])
        self.run.searches = self.searches.records
        self.run.trace = self.chat.trace

        return self.run

    def handle(self, calls: list[ToolCall]) -> None:
        """Answer every call of one turn; `finish` and `judge_cases` end the turn's work.

        Args:
            calls: The model's calls.
        """
        final = next((call for call in calls if call.name in (FINISH, JUDGE_CASES)), None)
        for call in calls:
            if final is not None and call is not final:
                self.chat.reply(call, f"Not run: you called {final.name}.")
            elif call.name in SEARCH_TOOLS and not self.checking:
                output, shown = self.searches.run(call)
                self.chat.reply(call, output, shown)
            elif call.name == ASK_USER and not self.checking:
                self.chat.reply(call, self.ask(str(call.args.get("question") or "")))
            elif call.name == FINISH and not self.checking:
                self.finish(call)
            elif call.name == JUDGE_CASES and self.checking:
                self.judge(call)
            else:
                self.chat.reply(call, f"{call.name} can't be called now.")

    def ask(self, question: str) -> str:
        """Put the agent's question to the person, within the budget.

        Args:
            question: The agent's question.

        Returns:
            The reply, or why there is none.
        """
        if self.person is None or len(self.run.asks) >= self.max_asks or not question:
            return NO_ONE_TO_ASK
        reply = self.person.reply(question)
        self.run.asks.append(AgentAsk(ask=question, reply=reply))
        self.searches.extra += SEARCHES_PER_REPLY

        return reply

    def finish(self, call: ToolCall) -> None:
        """Rerank the picks and the searches' results, and show the final ones for the check.

        Args:
            call: The `finish` call.
        """
        ids = call.args.get("ids")
        self.run.picks = self.searches.resolve(ids if isinstance(ids, list) else [])
        documents = self.final_documents(self.run.picks)
        listing = "\n\n".join(
            f"[{self.searches.label(hit.doc_id)}] {source_line(hit.source_type.value, None)}:\n"
            f"{hit.text[:MAX_DOCUMENT_CHARS]}"
            for hit in documents
        )
        text = f"{prompt('case_check_turn')}\n\nQuestion: {self.question}\n\n{listing}"
        self.chat.reply(call, text, [hit.doc_id for hit in documents])
        self.checking = True

    def final_documents(self, picks: list[str]) -> list[SearchHit]:
        """Rerank the candidates against the question and load the top ones in full.

        Args:
            picks: The agent's picks, dataset ids.

        Returns:
            The final documents, best first.
        """
        candidates = self.searches.candidates(picks)
        self.run.candidates = len(candidates)
        reranked = self.searches.retriever.rerank(self.question, candidates, self.scores)
        top = [document.doc_id for document in reranked[:FINAL_DOCUMENTS]]
        self.run.documents = self.searches.retriever.read_documents(top)

        return self.run.documents

    def judge(self, call: ToolCall) -> None:
        """Record the check; ask on several cases while possible, else stop.

        Args:
            call: The `judge_cases` call.
        """
        verdict = str(call.args.get("verdict") or "same_answer")
        fits = call.args.get("fits")
        check = Verdict(
            reasoning=str(call.args.get("reasoning") or ""),
            verdict=verdict if verdict in ("one_case_fits", SEVERAL) else "same_answer",
            fits=self.searches.resolve(fits if isinstance(fits, list) else []),
        )
        self.run.checks.append(check)
        fitting = [hit for hit in self.run.documents if hit.doc_id in check.fits]
        self.run.documents = fitting + [
            hit for hit in self.run.documents if hit.doc_id not in check.fits
        ]
        can_ask = len(self.run.asks) < self.max_asks and len(self.run.checks) < MAX_CHECKS
        if check.verdict == SEVERAL and can_ask:
            self.chat.reply(call, prompt("case_check_ask", reasoning=check.reasoning))
            self.checking = False
            return

        self.run.side_by_side = check.verdict == SEVERAL
        self.done = True
