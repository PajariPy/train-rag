"""List who spoke in Fireflies meetings that have no attendees, with an LLM (Gemini).

Some meetings have no attendee list, and their speaker lines are in forms the parser
does not recognize (`Speaker 1 (Jasmine):`, `[S00:00] Ari (Redwood AE):`). The LLM lists
the people who speak, with their role when the text gives one. The lists are stored in
Postgres, where the Fireflies parser attaches them as `attendees_inferred`.

Reads GEMINI_API_KEY from the repository's `.env`.

Run: uv run python exploration/attendee_extractor/extract.py
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.models import Artifact, SourceType
from train_rag.models.parsed import FirefliesMeeting, Person
from train_rag.parsers import FirefliesParser
from train_rag.repository.document import load_artifacts
from train_rag.repository.inferred_attendees import replace_inferred_attendees

DATASET = "enterprise_rag_bench"
ENV_FILE = Path(__file__).parents[2] / ".env"
MODEL = "gemini-3.8-flash"
METHOD = f"{MODEL} speaker extractor"
WORKERS = 8
SYSTEM_PROMPT = """You read meeting transcripts exported from Fireflies. List every person \
who speaks in the meeting.

Use each person's name as written; when a line labels someone both ways, as in \
"Speaker 2 (Nadia)", use the name, not the placeholder. Add the role or company when the \
text gives one, as in "Ari (Redwood AE)". List each person once, in the order they first \
speak. Leave out placeholders without a name ("Speaker 3", "Unknown"), bots, and people \
who are only mentioned but never speak."""


class Speakers(BaseModel):
    """The people who speak in one meeting."""

    attendees: list[Person]


def meetings_without_attendees(session: Session) -> list[Artifact]:
    """Find the meetings for which the parser finds no attendees.

    Args:
        session: Database session.

    Returns:
        The raw meetings.
    """
    parser = FirefliesParser()
    artifacts = load_artifacts(session, DATASET, SourceType.FIREFLIES)

    return [
        artifact
        for artifact in artifacts
        if isinstance(meeting := parser.parse(artifact), FirefliesMeeting) and not meeting.attendees
    ]


def speakers(client: genai.Client, artifact: Artifact) -> list[Person]:
    """Ask the model who speaks in one meeting.

    Args:
        client: Gemini client.
        artifact: Raw meeting.

    Returns:
        The speakers, or an empty list when the model gave no valid answer.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Title: {artifact.title}\n\n<meeting>\n{artifact.content}\n</meeting>",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Speakers,
        ),
    )
    parsed = response.parsed

    return parsed.attendees if isinstance(parsed, Speakers) else []


def main() -> None:
    """List the speakers of every meeting without attendees and store them."""
    load_dotenv(ENV_FILE)
    client = genai.Client()
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        artifacts = meetings_without_attendees(session)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            lists = list(pool.map(lambda artifact: speakers(client, artifact), artifacts))

        found = [
            (artifact.document_id, people)
            for artifact, people in zip(artifacts, lists, strict=True)
            if artifact.document_id is not None and people
        ]
        replace_inferred_attendees(session, METHOD, found)

    for artifact, people in zip(artifacts, lists, strict=True):
        names = ", ".join(f"{p.name}" + (f" ({p.role})" if p.role else "") for p in people)
        print(f"#{artifact.document_id} {artifact.id}: {names or '(none found)'}")
    print(f"Stored speakers for {len(found)} of {len(artifacts)} meetings as {METHOD!r}.")


if __name__ == "__main__":
    main()
