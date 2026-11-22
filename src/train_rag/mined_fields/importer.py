"""Import mined fields from a labels file into the `document_fields` table.

The DeepSeek labels (`exploration/slack_extraction/data/labels.jsonl`) and the SLM's output
share one line shape: `id` (the dataset id), `key` (`<id>:<prompt hash>`), `fields` and
`labels` (`thread_kind`, `status`, `area`). Lines for questions (`question:<id>`) and for
other prompts are skipped. Importing again replaces the same extractor's rows.
"""

import json
from collections.abc import Iterator
from itertools import batched
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session

from train_rag.models.mined_fields import LabelLine, MinedFields
from train_rag.progress.recorder import RunProgress
from train_rag.repository.document_fields import document_keys, upsert_document_fields
from train_rag.repository.parsed_document import slack_channels
from train_rag.tables import DocumentFieldsRecord

BATCH_SIZE = 500
QUESTION_PREFIX = "question:"


def read_labels(path: Path, prompt_key: str | None) -> Iterator[LabelLine]:
    """Read the document lines of a labels file.

    Args:
        path: The JSONL file.
        prompt_key: Keep only lines whose key ends with this prompt hash, or None for all.

    Yields:
        One line per labelled document, as written; the last line wins for a repeated id.
    """
    rows: dict[str, LabelLine] = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue

            row = LabelLine.model_validate(json.loads(line))
            if row.id.startswith(QUESTION_PREFIX):
                continue
            if prompt_key is not None and not row.key.endswith(prompt_key):
                continue

            rows[row.id] = row

    yield from rows.values()


def field_records(
    rows: list[LabelLine],
    keys: dict[str, list[tuple[int, str]]],
    channels: dict[int, str],
    extractor: str,
) -> list[DocumentFieldsRecord]:
    """Build the table rows of labelled documents.

    Args:
        rows: Lines of the labels file.
        keys: (row id, source) of the documents with each dataset id.
        channels: Slack channel by document row id.
        extractor: Name of the extractor that wrote the lines.

    Returns:
        One record per document row; ids not in the corpus are left out.
    """
    records = []
    for row in rows:
        mined = MinedFields.from_label(row)
        for document_id, source_type in keys.get(row.id, []):
            records.append(
                DocumentFieldsRecord(
                    document_id=document_id,
                    doc_id=row.id,
                    source_type=source_type,
                    extractor=extractor,
                    raw=row.model_dump(),
                    fields=mined.fields,
                    thread_kind=mined.thread_kind,
                    status=mined.status,
                    areas=mined.areas,
                    channel=channels.get(document_id),
                )
            )

    return records


def import_fields(
    engine: Engine, path: Path, extractor: str, prompt_key: str | None, dataset: str
) -> int:
    """Import a labels file.

    Args:
        engine: Database engine.
        path: The JSONL file.
        extractor: Name stored with the rows, e.g. `deepseek-fc8ae8878988`.
        prompt_key: Keep only lines of this prompt hash, or None for all.
        dataset: Dataset of the documents.

    Returns:
        The number of rows stored.
    """
    rows = list(read_labels(path, prompt_key))
    stored = 0
    title = f"import mined fields: {extractor} from {path.name}"

    with RunProgress(engine, title, len(rows)) as progress, Session(engine) as session:
        for batch in batched(rows, BATCH_SIZE, strict=False):
            keys = document_keys(session, dataset, [row.id for row in batch])
            ids = [document_id for found in keys.values() for document_id, _ in found]
            records = field_records(list(batch), keys, slack_channels(session, ids), extractor)

            upsert_document_fields(session, records)
            stored += len(records)
            progress.advance("importing", len(batch))

    return stored
