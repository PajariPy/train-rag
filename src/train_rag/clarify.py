"""Commands of the evaluation with clarifying questions: write memory notes, report a run."""

from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sqlmodel import Session

from train_rag.data_loaders import EnterpriseRagBenchLoader
from train_rag.database import create_tables, get_engine
from train_rag.evaluation.clarify_report import clarify_report, count_asks
from train_rag.evaluation.memory_notes import (
    NOTE_PROMPT_VERSION,
    has_number,
    leaks_title,
    pick_wrong,
    write_each,
    write_note,
)
from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models import Question
from train_rag.models.memory import MemoryNote
from train_rag.processing.normalize import normalize_documents
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document import ids_of_doc_ids, load_documents
from train_rag.repository.evaluation import results_of_run
from train_rag.repository.question import load_questions
from train_rag.repository.question_memory import load_memories, upsert_memories
from train_rag.tables import QuestionMemoryRecord

ENV_FILE = Path(".env")


def right_documents(session: Session, doc_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
    """Load the title and normalized full text of documents.

    Args:
        session: Active database session.
        doc_ids: Dataset ids of the documents.

    Returns:
        (title, title + text) keyed by dataset id.
    """
    internal = ids_of_doc_ids(session, EnterpriseRagBenchLoader.name, doc_ids)
    clean = normalize_documents(load_documents(session, sorted(internal)))

    return {doc.doc_id: (doc.title, f"{doc.title}\n\n{doc.text}") for doc in clean}


def write_memories(question_ids: list[str], redo: list[str] | None = None) -> None:
    """Write and store the memory note of every question, then print them for review.

    Which notes get a wrong detail is picked from the whole id list, so rewriting some
    notes with `redo` keeps each one's clean-or-wrong choice.

    Args:
        question_ids: All questions of the evaluation.
        redo: Only rewrite these (a subset of `question_ids`), or None for all.

    Raises:
        ValueError: If an id is unknown or a right document is missing.
    """
    load_dotenv(ENV_FILE)
    engine = get_engine()
    create_tables(engine)
    wrong = pick_wrong(question_ids)
    targets = redo if redo is not None else question_ids
    client, usage = genai.Client(), UsageMeter()

    with Session(engine) as session:
        by_id = {
            question.id: question
            for question in load_questions(session, EnterpriseRagBenchLoader.name)
        }
        unknown = [question_id for question_id in targets if question_id not in by_id]
        if unknown or not set(targets) <= set(question_ids):
            raise ValueError(f"Unknown question ids or not in the list: {unknown or targets}")
        questions = [by_id[question_id] for question_id in targets]
        documents = right_documents(session, [q.expected_doc_ids[0] for q in questions])

        written: list[QuestionMemoryRecord] = []

        def write(question: Question) -> MemoryNote:
            """Write one question's note."""
            text = documents[question.expected_doc_ids[0]][1]
            return write_note(client, usage, question, text, wrong=question.id in wrong)

        with RunProgress(engine, "write memory notes", len(questions), unit="notes") as progress:

            def save(note: MemoryNote) -> None:
                """Store one note at once, so a later failure can't lose it."""
                record = QuestionMemoryRecord(
                    question_id=note.question_id,
                    note=note.note,
                    wrong_detail=note.wrong_detail,
                    model=GEMINI_MODEL,
                    prompt_version=NOTE_PROMPT_VERSION,
                )
                upsert_memories(session, [record])
                session.commit()
                written.append(record)
                progress.advance("writing", 1, note=f"{note.question_id} · ${usage.cost():.3f}")

            failed = write_each(questions, write, save)

    for record in written:
        question = by_id[record.question_id]
        title = documents[question.expected_doc_ids[0]][0]
        markers = ("  LEAK?" if leaks_title(record.note, title) else "") + (
            "  NUMBER?" if has_number(record.note) else ""
        )
        print(f"\n## {question.id}{markers}\nQ: {question.text}")
        print(f"Gold answer: {question.gold_answer}\nDocument: {title[:100]}")
        print(record.note)
        if record.wrong_detail:
            print(f"WRONG DETAIL: {record.wrong_detail}")
    wrong_count = sum(record.wrong_detail is not None for record in written)
    print(f"\n{len(written)} notes written, {wrong_count} with a wrong detail")
    if failed:
        print(f"{len(failed)} failed; rewrite them with --redo {' '.join(q.id for q in failed)}")
        for question, error in failed.items():
            print(f"  {question.id}: {error}")
    print(usage.report())


def report_clarification(run_id: int, baseline_id: int, question_ids: list[str]) -> None:
    """Print the clarification report of a run against a zero-shot run.

    Args:
        run_id: The run with clarifying questions.
        baseline_id: The zero-shot run on the same questions.
        question_ids: The questions to compare.
    """
    chosen = set(question_ids)
    with Session(get_engine()) as session:
        run = [r for r in results_of_run(session, run_id) if r.question_id in chosen]
        baseline = {
            r.question_id: r.correct
            for r in results_of_run(session, baseline_id)
            if r.question_id in chosen
        }
        memories = load_memories(session, question_ids)
    print(
        clarify_report(
            baseline,
            {r.question_id: r.correct for r in run},
            {r.question_id: count_asks(r.trace) for r in run},
            memories,
        )
    )
