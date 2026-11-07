"""A small web page that shows the progress of the latest pipeline run."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.models.pipeline_run import RunState
from train_rag.repository.pipeline_run import latest_run_state

PAGE = Path(__file__).with_name("page.html")


def create_app() -> FastAPI:
    """Build the app: the page at `/`, the latest run's state at `/api/progress`.

    Returns:
        The FastAPI app.
    """
    engine = get_engine()
    create_tables(engine)
    app = FastAPI(title="Pipeline progress")

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        """Serve the progress page, which polls `/api/progress`."""
        return PAGE.read_text()

    @app.get("/api/progress")
    def progress() -> RunState | None:
        """Describe the latest run, or None when no run was recorded."""
        with Session(engine) as session:
            return latest_run_state(session)

    return app


def serve_progress_page(port: int) -> None:
    """Serve the progress page on this machine until stopped.

    Args:
        port: Port to listen on.
    """
    print(f"Progress page: http://localhost:{port}")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
