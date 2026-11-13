"""Run a RAG system over benchmark questions, grade the answers and store the results."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from sqlmodel import Session

from train_rag.evaluation.judge import AnswerJudge
from train_rag.evaluation.metrics import document_recall, invalid_extra_docs, summarize_run
from train_rag.evaluation.person import Person
from train_rag.evaluation.system import ClarifyingSystem, RagSystem
from train_rag.models import Question
from train_rag.models.evaluation import QuestionResult, RunSummary
from train_rag.repository.evaluation import add_results, create_run, finish_run

BATCH_SIZE = 50

Progress = Callable[[int, int], None]
# Makes the person behind one question, for systems that can ask them.
PersonFactory = Callable[[Question], Person]


def evaluate_question(
    system: RagSystem,
    judge: AnswerJudge,
    question: Question,
    person_factory: PersonFactory | None = None,
) -> QuestionResult:
    """Answer one question, measure retrieval and grade the answer.

    A failure in the system or the judge is recorded on the result instead of stopping
    the run. A system that can ask gets a fresh person for the question when a factory is
    given; every other system sees only the question text.

    Args:
        system: The evaluated RAG system.
        judge: The answer judge.
        question: The benchmark question.
        person_factory: Makes the person behind the question, or None for no one to ask.

    Returns:
        The question's metrics, grade, or error.
    """
    result = QuestionResult(question_id=question.id, question_type=question.question_type)
    try:
        if isinstance(system, ClarifyingSystem) and person_factory is not None:
            answer = system.answer_with(question.text, person_factory(question))
        else:
            answer = system.answer(question.text)
    except Exception as error:  # a system bug must not stop the whole run
        result.error = f"system: {error!r}"

        return result

    result.answer = answer.answer
    result.retrieved_doc_ids = answer.retrieved_doc_ids
    result.cited_doc_ids = answer.cited_doc_ids
    result.trace = answer.trace
    result.document_recall = document_recall(question.expected_doc_ids, answer.retrieved_doc_ids)
    result.extra_docs = invalid_extra_docs(question.expected_doc_ids, answer.retrieved_doc_ids)
    try:
        grade = judge.grade(question, answer.answer)
    except Exception as error:  # a judge failure leaves this question ungraded
        result.error = f"judge: {error!r}"

        return result

    result.correct = grade.correct
    result.completeness = grade.completeness
    result.facts_covered = grade.facts_covered
    result.judge_reasoning = grade.reasoning

    return result


def evaluate_questions(
    system: RagSystem,
    judge: AnswerJudge,
    questions: Sequence[Question],
    workers: int = 8,
    progress: Progress | None = None,
    person_factory: PersonFactory | None = None,
) -> list[QuestionResult]:
    """Evaluate many questions in parallel.

    Args:
        system: The evaluated RAG system.
        judge: The answer judge.
        questions: Benchmark questions.
        workers: Questions evaluated at the same time.
        progress: Called with (done, total) after each question.
        person_factory: Makes the person behind each question, for systems that can ask.

    Returns:
        One result per question, in the questions' order.
    """

    def evaluate(question: Question) -> QuestionResult:
        """Evaluate one question with this run's system and judge."""
        return evaluate_question(system, judge, question, person_factory)

    results: list[QuestionResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(evaluate, questions):
            results.append(result)
            if progress is not None:
                progress(len(results), len(questions))

    return results


def run_evaluation(
    session: Session,
    dataset: str,
    system: RagSystem,
    judge: AnswerJudge,
    questions: Sequence[Question],
    workers: int = 8,
    progress: Progress | None = None,
    person_factory: PersonFactory | None = None,
) -> tuple[int, RunSummary]:
    """Evaluate a system on the questions and store the run with its results.

    Args:
        session: Active database session.
        dataset: Dataset the questions come from.
        system: The evaluated RAG system.
        judge: The answer judge.
        questions: Benchmark questions.
        workers: Questions evaluated at the same time.
        progress: Called with (done, total) after each question.
        person_factory: Makes the person behind each question, for systems that can ask.

    Returns:
        The run id and its summary.
    """
    run = create_run(session, dataset, system.name, judge.name, len(questions))
    if run.id is None:
        raise RuntimeError("The run was stored without an id")

    results = evaluate_questions(system, judge, questions, workers, progress, person_factory)
    for start in range(0, len(results), BATCH_SIZE):
        add_results(session, run.id, results[start : start + BATCH_SIZE])
    summary = summarize_run(results)
    finish_run(session, run, summary)

    return run.id, summary
