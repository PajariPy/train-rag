import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def imports():
    from collections import Counter

    import marimo as mo
    from sqlmodel import Session

    from train_rag.database import get_engine
    from train_rag.models import SourceType
    from train_rag.parsers import default_registry
    from train_rag.repository.document import get_document, load_artifacts

    return (
        Counter,
        Session,
        SourceType,
        default_registry,
        get_document,
        get_engine,
        load_artifacts,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    # Flagged documents

    Every document a parser flagged because an expected field is missing, plus the ones
    marked not useful (too little content to index), across all parsed types, for review.
    Google Drive and Confluence have no parser, so nothing there is flagged. Parsing the
    whole database takes about two minutes.

    Filter by type and reason, pick a row, and compare the raw document with what the
    parser extracted. `formats` shows which format method found each field.
    """)
    return


@app.cell
def scan(
    Counter,
    Session,
    SourceType,
    default_registry,
    get_engine,
    load_artifacts,
    mo,
):
    DATASET = "enterprise_rag_bench"
    NOT_USEFUL = "not useful"

    def review_reasons(parsed):
        """Missing fields, plus `not useful` for documents with too little content."""
        return [*parsed.missing_fields, *([] if parsed.useful else [NOT_USEFUL])]

    engine = get_engine()
    registry = default_registry()
    parsed_types = [source for source in SourceType if registry.get(source) is not None]
    analysed_counts = Counter()
    flagged = []

    with Session(engine) as _session:
        for _source in mo.status.progress_bar(parsed_types, title="Parsing all documents"):
            for _artifact in load_artifacts(_session, DATASET, _source):
                analysed_counts[_source] += 1
                _parsed = registry.parse(_artifact)
                if _parsed is not None and review_reasons(_parsed):
                    flagged.append((_artifact, _parsed))
    return NOT_USEFUL, analysed_counts, engine, flagged, parsed_types, review_reasons


@app.cell
def summary(Counter, NOT_USEFUL, analysed_counts, flagged, mo, parsed_types, review_reasons):
    flagged_by_type = Counter(parsed.source_type for _artifact, parsed in flagged)
    missing_by_type = Counter(
        (parsed.source_type, field)
        for _artifact, parsed in flagged
        for field in review_reasons(parsed)
    )
    not_useful_count = sum(
        count for (_source, field), count in missing_by_type.items() if field == NOT_USEFUL
    )
    type_rows = [
        {
            "type": str(source),
            "documents": analysed_counts[source],
            "to review": flagged_by_type[source],
            "to review %": f"{flagged_by_type[source] / analysed_counts[source]:.2%}",
        }
        for source in parsed_types
    ]
    field_rows = [
        {
            "type": str(source),
            "reason": field,
            "documents": count,
            "% of type": f"{count / analysed_counts[source]:.2%}",
        }
        for (source, field), count in sorted(missing_by_type.items(), key=lambda item: -item[1])
    ]

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{sum(analysed_counts.values()):,}", label="Parsed documents"),
                    mo.stat(f"{len(flagged):,}", label="To review"),
                    mo.stat(f"{not_useful_count:,}", label="Of which not useful"),
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    mo.ui.table(type_rows, selection=None, label="Per type"),
                    mo.ui.table(field_rows, selection=None, label="Per reason"),
                ],
                widths="equal",
            ),
        ]
    )
    return (missing_by_type,)


@app.cell
def _(flagged, mo):
    type_filter = mo.ui.dropdown(
        options=["all", *sorted({str(parsed.source_type) for _artifact, parsed in flagged})],
        value="all",
        label="Type",
    )
    title_filter = mo.ui.text(placeholder="words in the title", label="Title contains")
    mo.hstack([type_filter, title_filter], justify="start")
    return title_filter, type_filter


@app.cell
def _(missing_by_type, mo, type_filter):
    field_filter = mo.ui.dropdown(
        options=[
            "all",
            *sorted(
                {
                    field
                    for (source, field) in missing_by_type
                    if type_filter.value == "all" or str(source) == type_filter.value
                }
            ),
        ],
        value="all",
        label="Reason",
    )
    field_filter
    return (field_filter,)


@app.cell
def flagged_table(field_filter, flagged, mo, review_reasons, title_filter, type_filter):
    rows = [
        {
            "document_id": artifact.document_id,
            "type": str(parsed.source_type),
            "reason": ", ".join(review_reasons(parsed)),
            "title": artifact.title[:120],
            "chars": len(artifact.content),
            "formats": ", ".join(f"{field}: {method}" for field, method in parsed.formats.items()),
        }
        for artifact, parsed in flagged
        if (type_filter.value == "all" or str(parsed.source_type) == type_filter.value)
        and (field_filter.value == "all" or field_filter.value in review_reasons(parsed))
        and title_filter.value.lower() in artifact.title.lower()
    ]
    table = mo.ui.table(rows, selection="single", page_size=15, label=f"{len(rows):,} documents")
    table
    return (table,)


@app.cell
def detail(Session, engine, flagged, get_document, mo, review_reasons, table):
    import html

    mo.stop(not table.value, mo.md("_Pick a row to see the raw document and the parsed result._"))

    selected_id = table.value[0]["document_id"]
    selected_parsed = next(
        parsed for artifact, parsed in flagged if artifact.document_id == selected_id
    )
    with Session(engine) as _session:
        _document = get_document(_session, selected_id)

    mo.vstack(
        [
            mo.md(
                f"### {_document.title}\n\n"
                f"`#{_document.id}` · `{_document.doc_id}` · {_document.source_type} · "
                f"{_document.char_count:,} chars · "
                f"reason: **{', '.join(review_reasons(selected_parsed))}**\n\n"
                "**Raw document**"
            ),
            mo.Html(
                '<pre style="white-space: pre-wrap; word-break: break-word; margin: 0">'
                f"{html.escape(_document.content)}</pre>"
            ),
        ]
    )
    return (selected_parsed,)


@app.cell(hide_code=True)
def parsed_detail(mo, selected_parsed):
    mo.vstack([mo.md("**Parsed**"), mo.json(selected_parsed.model_dump(mode="json"))])
    return


if __name__ == "__main__":
    app.run()
