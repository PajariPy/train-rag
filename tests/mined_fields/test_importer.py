"""Tests for reading a labels file and building the mined-fields rows."""

import json
from pathlib import Path

from train_rag.mined_fields.importer import field_records, read_labels


def write_labels(path: Path) -> None:
    """Write a labels file with a question, two prompts and a repeated id.

    Args:
        path: Where to write it.
    """
    rows = [
        {"id": "question:d1", "key": "question:d1:new", "fields": {}},
        {"id": "d1", "key": "d1:old", "fields": {"topic": ["old prompt"]}},
        {"id": "d1", "key": "d1:new", "fields": {"topic": ["first"]}},
        {"id": "d1", "key": "d1:new", "fields": {"topic": ["second"]}},
        {"id": "d2", "key": "d2:new", "fields": {"topic": ["other"]}, "labels": {"area": []}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n")


def test_read_labels_keeps_documents_of_one_prompt(tmp_path: Path) -> None:
    """Questions and other prompts are skipped; the last line of a repeated id wins."""
    path = tmp_path / "labels.jsonl"
    write_labels(path)
    rows = list(read_labels(path, "new"))
    assert [row["id"] for row in rows] == ["d1", "d2"]
    assert rows[0]["fields"]["topic"] == ["second"]
    assert len(list(read_labels(path, None))) == 2


def test_field_records_cover_every_row_of_an_id(tmp_path: Path) -> None:
    """A reused dataset id gets a record per document row; unknown ids are left out."""
    path = tmp_path / "labels.jsonl"
    write_labels(path)
    rows = list(read_labels(path, "new"))
    keys = {"d1": [(10, "slack"), (11, "gmail")]}
    records = field_records(rows, keys, {10: "eng-releases"}, "deepseek-test")
    assert [(record.document_id, record.channel) for record in records] == [
        (10, "eng-releases"),
        (11, None),
    ]
    assert records[0].fields["topic"] == ["second"]
    assert records[0].raw["key"] == "d1:new"
    assert records[0].extractor == "deepseek-test"
