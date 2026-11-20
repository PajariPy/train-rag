"""A web page to read sampled Slack threads one at a time and keep a note on each.

The same layout as the search-failure cases page (exploration/retrieval_failures/cases.py):
a side list of the 50 threads, marked short, long or random; the thread as parsed
messages; its 5 most similar Slack documents; and a note that saves as you type.

Reads data/samples.json (written by build.py). Notes go to review_notes.json next to
this file, outside the ignored data/ folder.

Usage:
    uv run python exploration/document_review/server.py [PORT]
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
SAMPLES = HERE / "data" / "samples.json"
PAGE = HERE / "page.html"
NOTES = HERE / "review_notes.json"


class Note(BaseModel):
    """A note on one document."""

    text: str


def load_samples() -> dict[str, dict[str, Any]]:
    """The sampled threads, keyed by doc id, in their drawn order."""
    return {sample["doc_id"]: sample for sample in json.loads(SAMPLES.read_text())}


def load_notes() -> dict[str, dict[str, str]]:
    """Every saved note, keyed by doc id."""
    return json.loads(NOTES.read_text()) if NOTES.exists() else {}


SAMPLES_BY_ID = load_samples()
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """The page."""
    return PAGE.read_text()


@app.get("/api/samples")
def list_samples() -> list[dict[str, Any]]:
    """Every thread's summary, for the side list."""
    return [
        {
            "doc_id": sample["doc_id"],
            "number": sample["number"],
            "group": sample["group"],
            "title": sample["title"],
            "messages": len(sample["messages"]),
            "chars": sample["chars"],
        }
        for sample in SAMPLES_BY_ID.values()
    ]


@app.get("/api/sample/{doc_id}")
def sample(doc_id: str) -> dict[str, Any]:
    """Everything about one thread, with its similar documents."""
    if doc_id not in SAMPLES_BY_ID:
        raise HTTPException(404, f"no sampled thread {doc_id}")

    return SAMPLES_BY_ID[doc_id]


@app.get("/api/notes")
def notes() -> dict[str, dict[str, str]]:
    """Every saved note."""
    return load_notes()


@app.put("/api/note/{doc_id}")
def save_note(doc_id: str, note: Note) -> dict[str, str]:
    """Save one thread's note; an empty note removes it.

    The file is rewritten through a temporary file, so a crash never leaves it half written.
    """
    if doc_id not in SAMPLES_BY_ID:
        raise HTTPException(404, f"no sampled thread {doc_id}")
    saved = load_notes()
    text = note.text.strip()
    if text:
        saved[doc_id] = {
            "text": text,
            "group": SAMPLES_BY_ID[doc_id]["group"],
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    else:
        saved.pop(doc_id, None)
    partial = NOTES.with_suffix(".json.tmp")
    partial.write_text(json.dumps(dict(sorted(saved.items())), indent=1, ensure_ascii=False))
    partial.replace(NOTES)

    return saved.get(doc_id, {"text": "", "updated": ""})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8768
    print(f"Document review: http://localhost:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
