"""The search schema must agree with the extraction schema in exploration."""

import importlib.util
from pathlib import Path
from types import ModuleType

from train_rag.mined_fields.schema import AREAS, FIELD_FORMS, STATUSES, THREAD_KINDS

EXTRACTION_SCHEMA = Path(__file__).parents[2] / "exploration/slack_extraction/schema.py"


def extraction_schema() -> ModuleType:
    """Load exploration/slack_extraction/schema.py, which the extraction prompt is built from.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("extraction_schema", EXTRACTION_SCHEMA)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_fields_and_forms_match_the_extraction_schema() -> None:
    """Same fields, same order, same forms."""
    schema = extraction_schema()
    assert FIELD_FORMS == {name: form for name, (form, _) in schema.FIELDS.items()}


def test_labels_match_the_extraction_schema() -> None:
    """Same thread kinds, statuses and areas."""
    schema = extraction_schema()
    assert THREAD_KINDS == schema.CLASSES["thread_kind"][1]
    assert STATUSES == schema.CLASSES["status"][1]
    assert AREAS == schema.MULTI_CLASSES["area"][1]
