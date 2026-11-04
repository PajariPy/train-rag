"""Progress of a long run (a pipeline or an evaluation), recorded for the progress page."""

import time
from types import TracebackType

from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.repository.pipeline_run import start_run, update_run

LOG_EVERY_SECONDS = 60


class RunProgress:
    """Records how far a run got after every batch, in the `pipeline_runs` table.

    The progress page (`train-rag progress-page`) reads it from there, so it works for a
    run started anywhere. Any loop can report through it: count items (documents,
    questions, ...) and optionally their characters, which give a better time estimate
    when items differ in size. The run is marked `finished` when the block ends normally
    and `stopped` when it ends with an error or Ctrl+C. A plain line is also printed once
    a minute, for logs. Use it from one thread.
    """

    def __init__(
        self,
        engine: Engine,
        pipeline: str,
        total_items: int,
        total_chars: int = 0,
        unit: str = "documents",
    ) -> None:
        """Set up the record of one run.

        Args:
            engine: Database engine; the progress uses its own session.
            pipeline: Name of the run and its settings, shown on the page.
            total_items: Items this run will process.
            total_chars: Their characters, or 0 to estimate time by items alone.
            unit: What the items are, e.g. `documents` or `questions`.
        """
        self.session = Session(engine)
        self.run = start_run(self.session, pipeline, total_items, total_chars, unit)
        self.logged = time.monotonic()

    def __enter__(self) -> "RunProgress":
        """Start recording.

        Returns:
            This progress record.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Mark the run finished, or stopped when it ended with an error or Ctrl+C."""
        status = "finished" if exc_type is None else "stopped"
        update_run(self.session, self.run, self.run.stage, 0, 0, status)
        self.session.close()

    def advance(self, stage: str, items: int, chars: int = 0, note: str | None = None) -> None:
        """Count a finished batch.

        Args:
            stage: What the run is working on now, e.g. the source name.
            items: Items in the batch.
            chars: Their characters, when the run counts them.
            note: Extra line for the page, e.g. the money spent so far; None keeps the
                current one.
        """
        update_run(self.session, self.run, stage, items, chars, note=note)
        if time.monotonic() - self.logged >= LOG_EVERY_SECONDS:
            self.logged = time.monotonic()
            left = self.run.total_chars - self.run.done_chars
            print(
                f"{stage}: {self.run.done_items:,}/{self.run.total_items:,} {self.run.unit}"
                + (f", {left / 1e6:,.0f}M chars left" if self.run.total_chars else "")
                + (f" · {self.run.note}" if self.run.note else ""),
                flush=True,
            )
