"""Evaluate a RAG system on the benchmark questions and print the leaderboard metrics."""

from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.evaluation import format_report, run_evaluation
from train_rag.evaluation.agent_clarify import ClarifyingSnippetAgent
from train_rag.evaluation.clarify_report import missing_memories
from train_rag.evaluation.person import Person, SimulatedPerson
from train_rag.evaluation.pipeline import PersonFactory
from train_rag.evaluation.registry import JUDGES, PERSON_USAGE, RETRIEVERS, SYSTEMS, USAGE
from train_rag.evaluation.retrieval import evaluate_retrieval, format_retrieval_report
from train_rag.models import Question, QuestionType
from train_rag.progress.recorder import RunProgress
from train_rag.repository.evaluation import wrong_question_ids
from train_rag.repository.question import load_questions
from train_rag.repository.question_memory import load_memories

ENV_FILE = Path(".env")


def evaluate(
    system_name: str,
    judge_name: str,
    question_type: QuestionType | None = None,
    limit: int | None = None,
    workers: int = 8,
    wrong_in: int | None = None,
    question_ids: list[str] | None = None,
) -> int:
    """Run one evaluation and print its report.

    Args:
        system_name: Name of a system in `SYSTEMS`.
        judge_name: Name of a judge in `JUDGES`.
        question_type: Only questions of this type, or None for all.
        limit: Maximum number of questions, or None for all.
        workers: Questions evaluated at the same time.
        wrong_in: Only the questions this earlier run got wrong, or None for all.
        question_ids: Only these questions, or None for all.

    Returns:
        The id of the stored run.

    Raises:
        ValueError: If an id isn't a known question, or, for the clarifying agent, a
            question has no memory note. Checked before any paid call.
    """
    load_dotenv(ENV_FILE)
    engine = get_engine()
    create_tables(engine)
    dataset = EnterpriseRagBenchLoader.name

    with Session(engine) as session:
        questions = load_questions(session, dataset, question_type, limit)
        if wrong_in is not None:
            wrong = wrong_question_ids(session, wrong_in)
            questions = [question for question in questions if question.id in wrong]
        if question_ids is not None:
            known = {question.id for question in questions}
            unknown = [question_id for question_id in question_ids if question_id not in known]
            if unknown:
                raise ValueError(f"Unknown or filtered-out question ids: {unknown}")
            chosen = set(question_ids)
            questions = [question for question in questions if question.id in chosen]
        person_factory: PersonFactory | None = None
        if system_name == ClarifyingSnippetAgent.name:
            ids = [question.id for question in questions]
            memories = load_memories(session, ids)
            missing = missing_memories(ids, memories)
            if missing:
                raise ValueError(f"No memory note for: {missing}; run write-memories first")
            client = genai.Client()

            def make_person(question: Question) -> Person:
                """The simulated person behind one question."""
                return SimulatedPerson(client, PERSON_USAGE, question.text, memories[question.id])

            person_factory = make_person
        system = SYSTEMS[system_name](questions)
        judge = JUDGES[judge_name]()
        print(f"Evaluating {system.name} on {len(questions)} questions, judged by {judge.name}")

        scope = f" wrong in run {wrong_in}" if wrong_in is not None else ""
        name = f"evaluate {system.name} ({question_type or 'all'} questions{scope})"
        with RunProgress(engine, name, len(questions), unit="questions") as progress:

            def advance(done: int, total: int) -> None:
                """Count an evaluated question, on the page and in the terminal."""
                progress.advance("answering and judging", 1, note=f"Gemini: ${USAGE.cost():.2f}")
                print(f"\r{done}/{total}", end="", flush=True)

            run_id, summary = run_evaluation(
                session,
                dataset,
                system,
                judge,
                questions,
                workers,
                progress=advance,
                person_factory=person_factory,
            )

    print()
    print(format_report(f"Run {run_id}: {system.name}, judged by {judge.name}", summary))
    print(USAGE.report())
    if person_factory is not None:
        print(f"Simulated person: {PERSON_USAGE.report()}")

    return run_id


def evaluate_search(
    retriever_name: str,
    question_type: QuestionType | None = None,
    limit: int | None = None,
    sample: int | None = None,
) -> None:
    """Measure how many expected documents a retriever finds, without an LLM.

    Args:
        retriever_name: Name of a retriever in `RETRIEVERS`.
        question_type: Only questions of this type, or None for all.
        limit: Maximum number of questions, or None for all.
        sample: Sample size the BM25 index was built for, or None for the whole corpus.
    """
    retriever = RETRIEVERS[retriever_name](sample)
    with Session(get_engine()) as session:
        questions = load_questions(session, EnterpriseRagBenchLoader.name, question_type, limit)
    indexed = retriever.searchable_doc_ids()

    name = f"evaluate-search {retriever_name} ({question_type or 'all'} questions)"
    with RunProgress(get_engine(), name, len(questions), unit="questions") as progress:
        metrics = evaluate_retrieval(
            retriever,
            questions,
            indexed,
            progress=lambda _done, _total: progress.advance("retrieving", 1),
        )
    print(format_retrieval_report(f"{retriever_name} over {len(indexed):,} documents", metrics))
