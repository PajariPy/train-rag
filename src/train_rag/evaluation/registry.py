"""Systems and judges selectable by name from the command line."""

from collections.abc import Callable, Sequence

from google import genai

from train_rag.database import get_engine
from train_rag.embedder import GteModernBertEmbedder, MiniLmEmbedder
from train_rag.evaluation.agent import AgentSystem
from train_rag.evaluation.agent_keywords import KeywordAgentSystem
from train_rag.evaluation.agent_snippets import SnippetAgentSystem
from train_rag.evaluation.hybrid import PipelineRetriever
from train_rag.evaluation.judge import AnswerJudge, GeminiJudge
from train_rag.evaluation.pipeline_answer import PipelineAnswerSystem
from train_rag.evaluation.summary_search import SummarySearchSystem
from train_rag.evaluation.system import NoAnswerSystem, OracleSystem, RagSystem
from train_rag.evaluation.usage import UsageMeter
from train_rag.index_bm25 import bm25_index_dir
from train_rag.models import Question
from train_rag.pipelines.retrieval import RetrievalPipeline
from train_rag.retrieval import Bm25Retriever, EmbeddingRetriever, Retriever

SystemFactory = Callable[[Sequence[Question]], RagSystem]
JudgeFactory = Callable[[], AnswerJudge]
RetrieverFactory = Callable[[int | None], Retriever]

# Counts the Gemini tokens of the answers and verdicts of a run.
USAGE = UsageMeter()

SYSTEMS: dict[str, SystemFactory] = {
    OracleSystem.name: OracleSystem,
    NoAnswerSystem.name: lambda _questions: NoAnswerSystem(),
    SummarySearchSystem.name: lambda _questions: SummarySearchSystem(
        EmbeddingRetriever(get_engine(), MiniLmEmbedder(), "summary"), genai.Client()
    ),
    PipelineAnswerSystem.name: lambda _questions: PipelineAnswerSystem(
        PipelineRetriever(RetrievalPipeline.load()), genai.Client(), USAGE
    ),
    AgentSystem.name: lambda _questions: AgentSystem(
        PipelineRetriever(RetrievalPipeline.load()), genai.Client(), USAGE
    ),
    SnippetAgentSystem.name: lambda _questions: SnippetAgentSystem(
        PipelineRetriever(RetrievalPipeline.load()), genai.Client(), USAGE
    ),
    KeywordAgentSystem.name: lambda _questions: KeywordAgentSystem(
        PipelineRetriever(RetrievalPipeline.load()), genai.Client(), USAGE
    ),
}
JUDGES: dict[str, JudgeFactory] = {
    GeminiJudge.name: lambda: GeminiJudge(genai.Client(), usage=USAGE),
}

# Each factory takes the sample size the index was built for (None: the whole corpus);
# only the BM25 index, kept on disk, needs it to find its folder.
RETRIEVERS: dict[str, RetrieverFactory] = {
    "minilm-summary": lambda _sample: EmbeddingRetriever(get_engine(), MiniLmEmbedder(), "summary"),
    "minilm-chunk": lambda _sample: EmbeddingRetriever(get_engine(), MiniLmEmbedder(), "chunk"),
    "gte-chunk": lambda _sample: EmbeddingRetriever(get_engine(), GteModernBertEmbedder(), "chunk"),
    "bm25": lambda sample: Bm25Retriever(bm25_index_dir(sample)),
    "hybrid": lambda sample: PipelineRetriever(
        RetrievalPipeline.load(bm25_index_dir(sample), rerank=False)
    ),
    "hybrid-rerank": lambda sample: PipelineRetriever(
        RetrievalPipeline.load(bm25_index_dir(sample), rerank=True)
    ),
}
