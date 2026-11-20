"""Tests for a workflow's overall status from its steps."""

import pytest

from train_rag.repository.workflow import overall_status


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["success", "running", "queued"], "running"),
        (["success", "queued"], "running"),
        (["success", "failed", "queued"], "failed"),
        (["success", "skipped", "success"], "success"),
    ],
)
def test_overall_status(statuses: list[str], expected: str) -> None:
    """A failed step fails the workflow; open steps keep it running; else it succeeded."""
    assert overall_status(statuses) == expected
