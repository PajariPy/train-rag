"""Memory notes: which questions get a misremembered detail, and a check for leaked titles."""

import random
import re
from collections.abc import Callable, Sequence

from google import genai
from google.genai import types
from pydantic import BaseModel

from train_rag.evaluation.pipeline_answer import GEMINI_MODEL
from train_rag.evaluation.usage import UsageMeter
from train_rag.models import Question
from train_rag.models.memory import MemoryNote
from train_rag.prompts import prompt

# Share of notes that hold one slightly wrong detail, and the seed that picks them.
WRONG_SHARE = 0.25
SEED = 7
# Bump when the note prompts change, so stored notes say which prompt wrote them.
NOTE_PROMPT_VERSION = "v2"
# Title words shorter than this are too common to count as a leak.
LEAK_WORD_CHARS = 4

WORD = re.compile(r"[a-z0-9]+")
DIGIT = re.compile(r"\d")


def pick_wrong(
    question_ids: Sequence[str], share: float = WRONG_SHARE, seed: int = SEED
) -> set[str]:
    """Choose the questions whose note gets a misremembered detail.

    The choice depends only on the ids, the share and the seed, not on their order.

    Args:
        question_ids: All questions that get a note.
        share: Share of them to pick.
        seed: Random seed.

    Returns:
        The picked question ids.
    """
    ordered = sorted(question_ids)

    return set(random.Random(seed).sample(ordered, round(len(ordered) * share)))


def leaks_title(note: str, title: str) -> bool:
    """Tell whether a note copies the document's title: two long title words in a row.

    Args:
        note: The memory note.
        title: The right document's title.

    Returns:
        True when two consecutive title words of `LEAK_WORD_CHARS`+ characters appear
        consecutively in the note.
    """
    title_words = WORD.findall(title.lower())
    note_words = WORD.findall(note.lower())
    pairs = {
        (first, second)
        for first, second in zip(title_words, title_words[1:], strict=False)
        if len(first) >= LEAK_WORD_CHARS and len(second) >= LEAK_WORD_CHARS
    }

    return any(pair in pairs for pair in zip(note_words, note_words[1:], strict=False))


def has_number(note: str) -> bool:
    """Tell whether a note holds a digit: numbers, ids and dates should not be in it.

    Args:
        note: The memory note.

    Returns:
        True when the note contains any digit, for the reviewer to check.
    """
    return DIGIT.search(note) is not None


def write_each[T](
    items: Sequence[T],
    write: Callable[[T], MemoryNote],
    save: Callable[[MemoryNote], None],
) -> dict[T, str]:
    """Write and save notes one at a time, so one failure doesn't lose the others.

    Args:
        items: What to write a note for, e.g. questions.
        write: Writes one note.
        save: Stores one note as soon as it is written.

    Returns:
        The items that failed, with their error.
    """
    failed: dict[T, str] = {}
    for item in items:
        try:
            note = write(item)
        except Exception as error:  # one failed note must not stop or lose the rest
            failed[item] = repr(error)
            continue
        save(note)

    return failed


class NoteDraft(BaseModel):
    """The note writer's reply."""

    note: str
    wrong_detail: str | None = None


def write_note(
    client: genai.Client, usage: UsageMeter, question: Question, document: str, wrong: bool
) -> MemoryNote:
    """Write what the person behind a question remembers about its document.

    Args:
        client: Gemini client.
        usage: Counts the tokens.
        question: The question, with its gold answer (shown so the note can avoid it).
        document: The right document's text.
        wrong: Whether the note must hold one slightly wrong detail.

    Returns:
        The note; `wrong_detail` is set only when `wrong` is True.

    Raises:
        ValueError: If a note meant to hold a wrong detail doesn't say which one.
    """
    instruction = prompt("memory_note_wrong" if wrong else "memory_note_clean")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt(
            "memory_note_turn",
            question=question.text,
            gold_answer=question.gold_answer,
            instruction=instruction,
            document=document,
        ),
        config=types.GenerateContentConfig(
            system_instruction=prompt("memory_note_instructions"),
            response_mime_type="application/json",
            response_schema=NoteDraft,
        ),
    )
    usage.add(response)
    draft = NoteDraft.model_validate_json(response.text or "")
    if wrong and not draft.wrong_detail:
        raise ValueError(f"The note for {question.id} has no wrong detail")

    return MemoryNote(
        question_id=question.id,
        note=draft.note.strip(),
        wrong_detail=draft.wrong_detail if wrong else None,
    )
