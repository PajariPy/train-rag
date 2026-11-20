"""The agent's two searches, their budget, the result cards and the candidate list.

`search_documents` is the accepted first search over the whole corpus (fine-tuned gte on the
query, BM25 on the query with Gemini's rewrites appended, the union reranked on the first
1,024 tokens). `search_fields` searches the mined fields (`FieldIndex`). Every document the
agent is shown gets a short id (`d1`, `d2`, ...) that stays the same for the conversation.
"""

from typing import Any, Protocol

from train_rag.evaluation.field_agent.tools import SEARCH_DOCUMENTS, SHOWN, SNIPPET_CHARS
from train_rag.evaluation.hybrid import to_hits
from train_rag.fusion import reciprocal_rank_fusion
from train_rag.mined_fields.index import PROFILE_COLUMNS, FieldIndex
from train_rag.mined_fields.profiles import field_card, plain_card, source_line
from train_rag.mined_fields.schema import AREAS, SOURCE_KINDS
from train_rag.models import SourceType
from train_rag.models.field_agent import ToolCall
from train_rag.models.retrieval import RetrievedDocument, SearchHit
from train_rag.query_rewrite import QueryRewriter

# Searches per question across both tools, and how many of them may be general searches
# (each costs a Gemini rewrite and about 6 s of reranking; a field search is a lookup).
MAX_SEARCHES = 4
MAX_GENERAL_SEARCHES = 2
# Documents of a general search kept for the candidate list (the whole reranked union).
GENERAL_KEPT = 100
# Candidates the final rerank reads: the agent's picks, then its searches fused.
AGENT_CANDIDATES = 100


class DocumentSearch(Protocol):
    """The retrieval pipeline as the agent uses it (`PipelineRetriever`)."""

    def search(
        self, query: str, limit: int, keyword_query: str | None = None
    ) -> list[RetrievedDocument]:
        """Search the whole corpus and rerank."""
        ...

    def rerank(
        self, query: str, documents: list[RetrievedDocument], scores: dict[str, float]
    ) -> list[RetrievedDocument]:
        """Order documents from any search by the reranker."""
        ...

    def read_documents(self, doc_ids: list[str]) -> list[SearchHit]:
        """Load documents' full text, title on top."""
        ...


def snippet(text: str) -> str:
    """The start of a text, whitespace collapsed, as a result card shows it.

    Args:
        text: The text.

    Returns:
        Its first `SNIPPET_CHARS` characters.
    """
    return " ".join(text.split())[:SNIPPET_CHARS]


class AgentSearches:
    """Runs the agent's searches for one question and remembers what they found."""

    def __init__(
        self, retriever: DocumentSearch, index: FieldIndex, rewriter: QueryRewriter
    ) -> None:
        """Start with no searches.

        Args:
            retriever: The general search and the reranker.
            index: The mined fields.
            rewriter: Writes the BM25 query of a general search.
        """
        self.retriever = retriever
        self.index = index
        self.rewriter = rewriter
        self.label_of: dict[str, str] = {}
        self.doc_of: dict[str, str] = {}
        self.rankings: dict[str, list[SearchHit]] = {}
        self.records: list[dict[str, Any]] = []
        self.extra = 0  # searches added by the person's replies

    def label(self, doc_id: str) -> str:
        """The short id the agent sees for a document, made on first sight.

        Args:
            doc_id: The dataset id.

        Returns:
            E.g. "d3".
        """
        if doc_id not in self.label_of:
            label = f"d{len(self.label_of) + 1}"
            self.label_of[doc_id] = label
            self.doc_of[label] = doc_id

        return self.label_of[doc_id]

    def resolve(self, labels: list[Any]) -> list[str]:
        """Turn the agent's ids into dataset ids, dropping unknown ones and repeats.

        Args:
            labels: Ids as the agent wrote them.

        Returns:
            The dataset ids, in order.
        """
        return list(
            dict.fromkeys(
                self.doc_of[str(label).strip("[] ")]
                for label in labels
                if str(label).strip("[] ") in self.doc_of
            )
        )

    def allowed(self) -> list[str]:
        """The search tools the agent may still call.

        Returns:
            Tool names; empty when the budget is used up.
        """
        if len(self.records) >= MAX_SEARCHES + self.extra:
            return []
        general = sum(record["tool"] == SEARCH_DOCUMENTS for record in self.records)
        tools = ["search_fields"]
        if general < MAX_GENERAL_SEARCHES:
            tools.insert(0, SEARCH_DOCUMENTS)

        return tools

    def run(self, call: ToolCall) -> tuple[str, list[str]]:
        """Run one search call within the budget.

        Args:
            call: A `search_documents` or `search_fields` call.

        Returns:
            What the agent receives and the shown documents' ids.
        """
        if call.name not in self.allowed():
            return f"No {call.name} left: use another tool or call finish.", []
        query = str(call.args.get("query") or "").strip()
        if not query:
            return "Bad call: give a query.", []
        if call.name == SEARCH_DOCUMENTS:
            return self.search_documents(query)

        return self.search_fields(query, call.args)

    def search_documents(self, query: str) -> tuple[str, list[str]]:
        """Run the general search and show its top results.

        Args:
            query: The agent's query.

        Returns:
            The result cards and their doc ids.
        """
        keyword_query = self.rewriter.keyword_query(query)
        documents = self.retriever.search(query, GENERAL_KEPT, keyword_query)
        name = f"{SEARCH_DOCUMENTS} {len(self.records) + 1}"
        self.rankings[name] = to_hits(documents)
        self.records.append({"tool": SEARCH_DOCUMENTS, "query": query})
        shown = documents[:SHOWN]
        cards = [self.card(document.doc_id, document.title, document.text) for document in shown]

        return self.listing(cards), [document.doc_id for document in shown]

    def search_fields(self, query: str, args: dict[str, Any]) -> tuple[str, list[str]]:
        """Run a field search and show its top results.

        Args:
            query: The agent's query.
            args: The call's other arguments: profile and filters.

        Returns:
            The result cards and their doc ids.
        """
        profile = str(args.get("profile") or "all")
        profile = profile if profile in PROFILE_COLUMNS else "all"
        areas = [area for area in args.get("areas") or [] if area in AREAS]
        sources = [source for source in args.get("sources") or [] if source in SOURCE_KINDS]
        known = set(self.index.channels)
        channels = [
            str(name).lstrip("#")
            for name in args.get("channels") or []
            if str(name).lstrip("#") in known
        ]
        found = self.index.search(query, profile, areas, sources, channels)
        name = f"search_fields {len(self.records) + 1}"
        self.rankings[name] = [
            SearchHit(
                doc_id=self.index.doc_ids[position],
                document_id=self.index.document_ids[position],
                source_type=SourceType(self.index.sources[position]),
                field="fields",
                text="",
                distance=float(rank),
            )
            for rank, position in enumerate(found)
        ]
        self.records.append(
            {
                "tool": "search_fields",
                "query": query,
                "profile": profile,
                "areas": areas,
                "sources": sources,
                "channels": channels,
            }
        )
        shown = [self.index.doc_ids[position] for position in found[:SHOWN]]
        texts = {hit.doc_id: hit.text for hit in self.retriever.read_documents(shown)}
        cards = [self.card(doc_id, "", texts.get(doc_id, "")) for doc_id in shown]

        return self.listing(cards), shown

    def card(self, doc_id: str, title: str, text: str) -> str:
        """One result as the agent sees it, with its short id.

        Args:
            doc_id: The document.
            title: Its title, for a document without mined fields.
            text: Its text (or best chunk).

        Returns:
            The card.
        """
        position = self.index.position.get(doc_id)
        if position is None:
            body = plain_card(title or text.split("\n", 1)[0], snippet(text))
            where = ""
        else:
            body = field_card(self.index.mined[position], snippet(text))
            where = source_line(self.index.sources[position], self.index.channels[position])

        return f"[{self.label(doc_id)}] " + (f"({where}) " if where else "") + body

    @staticmethod
    def listing(cards: list[str]) -> str:
        """Join result cards for the agent.

        Args:
            cards: The cards, best first.

        Returns:
            The tool output.
        """
        return "\n".join(cards) if cards else "No documents found: try other words or filters."

    def candidates(self, picks: list[str]) -> list[RetrievedDocument]:
        """The documents the final rerank reads: the picks, then every search fused.

        Args:
            picks: The dataset ids the agent picked, best first.

        Returns:
            Up to `AGENT_CANDIDATES` documents, without text.
        """
        everything = sum(len(hits) for hits in self.rankings.values())
        fused = reciprocal_rank_fusion(self.rankings, everything)
        by_id = {document.doc_id: document for document in fused}
        picked = [by_id[doc_id] for doc_id in picks if doc_id in by_id]
        chosen = set(picks)
        rest = [document for document in fused if document.doc_id not in chosen]

        return (picked + rest)[:AGENT_CANDIDATES]
