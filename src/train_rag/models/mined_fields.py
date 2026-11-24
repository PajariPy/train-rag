"""Mined fields of one document, as an extractor (DeepSeek or the SLM) returned them."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from train_rag.mined_fields.schema import AREAS, FIELD_FORMS, STATUSES, THREAD_KINDS


class LabelLine(BaseModel):
    """One line of a labels file, as DeepSeek's labelling or the SLM wrote it.

    Other keys (relations, token counts, ...) are kept, so the line can be stored as is.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    key: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)


class MinedFields(BaseModel):
    """The parsed fields and labels of one document.

    Unknown fields and label values are dropped and missing fields become empty lists,
    so the output of any extractor that follows the schema loads the same way.
    """

    fields: dict[str, list[str]] = Field(default_factory=dict, validate_default=True)
    thread_kind: str | None = None
    status: str | None = None
    areas: list[str] = Field(default_factory=list)

    @field_validator("fields", mode="before")
    @classmethod
    def known_fields(cls, value: object) -> dict[str, list[str]]:
        """Keep the schema's fields, in its order, each as a list of non-empty strings.

        Args:
            value: The extractor's fields: a mapping of field name to values.

        Returns:
            Every schema field with its values; missing fields are empty.
        """
        given = value if isinstance(value, dict) else {}

        return {
            name: [str(item).strip() for item in given.get(name) or [] if str(item).strip()]
            for name in FIELD_FORMS
        }

    @field_validator("thread_kind")
    @classmethod
    def known_kind(cls, value: str | None) -> str | None:
        """Drop a thread kind outside the schema's list.

        Args:
            value: The extractor's label.

        Returns:
            The label, or None.
        """
        return value if value in THREAD_KINDS else None

    @field_validator("status")
    @classmethod
    def known_status(cls, value: str | None) -> str | None:
        """Drop a status outside the schema's list.

        Args:
            value: The extractor's label.

        Returns:
            The label, or None.
        """
        return value if value in STATUSES else None

    @field_validator("areas", mode="before")
    @classmethod
    def known_areas(cls, value: object) -> list[str]:
        """Keep at most three known areas, most important first, without repeats.

        Args:
            value: The extractor's areas.

        Returns:
            The known areas.
        """
        given = value if isinstance(value, list) else []

        return list(dict.fromkeys(area for area in given if area in AREAS))[:3]

    @classmethod
    def from_label(cls, row: LabelLine) -> "MinedFields":
        """Read one line of the labels file (DeepSeek's or the SLM's, same shape).

        Args:
            row: The line: `fields`, and `labels` with `thread_kind`, `status`, `area`.

        Returns:
            The parsed fields.
        """
        labels = row.labels

        return cls(
            fields=row.fields,
            thread_kind=labels.get("thread_kind"),
            status=labels.get("status"),
            areas=labels.get("area") or [],
        )


class FieldProfiles(BaseModel):
    """The three texts field search embeds per document; empty when nothing was mined."""

    subject: str
    problem: str
    names: str


class FieldDocument(BaseModel):
    """A document field search can find: its ids and the filters it matches."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    document_id: int
    source_type: str
    channel: str = ""
