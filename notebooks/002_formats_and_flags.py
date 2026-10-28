import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import re
    from collections import Counter, defaultdict

    import altair as alt
    import marimo as mo
    from sqlmodel import Session

    from train_rag.database import get_engine
    from train_rag.models import Artifact, SourceType
    from train_rag.parsers import default_registry
    from train_rag.parsers.text import normalize_newlines, split_sections
    from train_rag.repository.document import (
        count_documents_by_source,
        get_document,
        load_artifacts,
        sample_artifacts,
    )

    return (
        Artifact,
        Counter,
        Session,
        SourceType,
        alt,
        count_documents_by_source,
        default_registry,
        defaultdict,
        get_document,
        get_engine,
        load_artifacts,
        mo,
        normalize_newlines,
        re,
        sample_artifacts,
        split_sections,
    )


@app.cell
def _(mo):
    mo.md("""
    # Document formats and parsing flags

    How many formats each document type comes in, and which documents the parsers could
    not fully parse. Data comes from Postgres (`docker compose up -d` and
    `uv run train-rag import-db` first).

    Pick how many documents to analyse per type. A sample is quick; all documents gives
    exact counts but takes a few minutes.
    """)
    return


@app.cell
def settings(get_engine, mo):
    DATASET = "enterprise_rag_bench"
    EXAMPLES_PER_GROUP = 3
    PREVIEW_CHARS = 5000

    engine = get_engine()
    sample_size = mo.ui.dropdown(
        options={"2,000 per type": 2000, "10,000 per type": 10000, "All documents": 0},
        value="10,000 per type",
        label="Documents to analyse",
    )
    sample_size
    return DATASET, EXAMPLES_PER_GROUP, PREVIEW_CHARS, engine, sample_size


@app.cell
def free_text_formats(Artifact, normalize_newlines, re, split_sections):
    MARKDOWN_HEADING = re.compile(r"(?m)^#{1,6} ")

    def markdown_headings(content: str) -> bool:
        """Format 1: the document has markdown headings."""
        return bool(MARKDOWN_HEADING.search(content))

    def key_sections(content: str) -> bool:
        """Format 2: the document has `key:` sections like tickets do."""
        return bool(split_sections(content))

    def free_text_format(artifact: Artifact) -> str:
        """Rough structure of a document no parser handles (Drive, Confluence)."""
        content = normalize_newlines(artifact.content)
        for matches in (markdown_headings, key_sections):
            if matches(content):
                return matches.__name__

        return "plain_text"

    return (free_text_format,)


@app.cell
def analyse(
    Counter,
    DATASET,
    EXAMPLES_PER_GROUP,
    Session,
    SourceType,
    count_documents_by_source,
    default_registry,
    defaultdict,
    engine,
    free_text_format,
    load_artifacts,
    mo,
    sample_artifacts,
    sample_size,
):
    NO_PARSER = "no parser"
    SINGLE_FORMAT = "document: single format"

    registry = default_registry()
    analysed_counts = Counter()
    flagged_counts = Counter()
    format_counts = defaultdict(Counter)
    missing_counts = defaultdict(Counter)
    format_examples = defaultdict(list)
    missing_examples = defaultdict(list)

    with Session(engine) as _session:
        corpus_counts = count_documents_by_source(_session, DATASET)
        for _source in mo.status.progress_bar(list(SourceType), title="Analysing documents"):
            if sample_size.value:
                _artifacts = sample_artifacts(_session, DATASET, _source, sample_size.value)
            else:
                _artifacts = load_artifacts(_session, DATASET, _source)

            for _artifact in _artifacts:
                analysed_counts[_source] += 1
                _parsed = registry.parse(_artifact)
                if _parsed is None:
                    _formats = [f"structure: {free_text_format(_artifact)}"]
                    _missing = [NO_PARSER]
                else:
                    _formats = [f"{field}: {method}" for field, method in _parsed.formats.items()]
                    _formats += [f"{field}: not found" for field in _parsed.missing_fields]
                    _formats = _formats or [SINGLE_FORMAT]
                    _missing = _parsed.missing_fields

                for _format in _formats:
                    format_counts[_source][_format] += 1
                    if len(format_examples[_source, _format]) < EXAMPLES_PER_GROUP:
                        format_examples[_source, _format].append(_artifact.document_id)

                if _missing:
                    flagged_counts[_source] += 1
                for _field in _missing:
                    missing_counts[_source][_field] += 1
                    if len(missing_examples[_source, _field]) < EXAMPLES_PER_GROUP:
                        missing_examples[_source, _field].append(_artifact.document_id)
    return (
        NO_PARSER,
        analysed_counts,
        corpus_counts,
        flagged_counts,
        format_counts,
        format_examples,
        missing_counts,
        missing_examples,
    )


@app.cell
def show_example(PREVIEW_CHARS, Session, engine, get_document, mo):
    def show_examples(document_ids: list[int]) -> object:
        """Show the raw stored documents in an accordion, longest content cut short."""
        items = {}
        with Session(engine) as session:
            for document_id in document_ids:
                document = get_document(session, document_id)
                if document is None:
                    continue

                content = document.content
                if len(content) > PREVIEW_CHARS:
                    content = f"{content[:PREVIEW_CHARS]}\n\n… ({len(document.content):,} chars)"
                label = f"#{document.id} · {document.doc_id} · {document.title[:80]}"
                items[label] = mo.plain_text(content)

        return mo.accordion(items) if items else mo.md("_No examples._")

    return (show_examples,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Formats per document type

    Formats come from the parsers themselves. When a field is written in several ways, the
    parser has one method per format and tries them in order; each bar shows which method
    found the field, e.g. `review_comments: review_conversation_section`, or `not found`.
    Google Drive and Confluence have no parser; their formats are only a rough structure.
    """)
    return


@app.cell
def formats_overview(SourceType, alt, analysed_counts, format_counts, mo):
    format_rows = [
        {
            "type": str(source),
            "field": document_format.split(": ")[0],
            "format": document_format,
            "documents": count,
            "share": count / analysed_counts[source],
        }
        for source in SourceType
        for document_format, count in format_counts[source].items()
    ]
    overview_rows = [
        {
            "type": str(source),
            "analysed": analysed_counts[source],
            "formats found": sum(
                1
                for document_format in format_counts[source]
                if not document_format.endswith("not found")
            ),
            "most common format": format_counts[source].most_common(1)[0][0],
        }
        for source in SourceType
        if analysed_counts[source]
    ]

    formats_chart = (
        alt.Chart(alt.Data(values=format_rows))
        .transform_calculate(group="datum.type + ' · ' + datum.field")
        .mark_bar()
        .encode(
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title="Share of analysed documents"),
            y=alt.Y("group:N", title=None),
            color=alt.Color("format:N", legend=None),
            order=alt.Order("documents:Q", sort="descending"),
            tooltip=["type:N", "format:N", "documents:Q", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=640, title="Formats per type and field (hover a segment)")
    )
    mo.vstack([mo.ui.table(overview_rows, selection=None), formats_chart])
    return (format_rows,)


@app.cell
def _(SourceType, mo):
    format_source = mo.ui.dropdown(
        options={str(source): source for source in SourceType},
        value="github",
        label="Type",
    )
    format_source
    return (format_source,)


@app.cell
def format_detail(alt, format_rows, format_source, mo):
    source_rows = [row for row in format_rows if row["type"] == str(format_source.value)]
    source_chart = (
        alt.Chart(alt.Data(values=source_rows))
        .mark_bar()
        .encode(
            x=alt.X("documents:Q", title="Documents"),
            y=alt.Y("format:N", sort="-x", title=None),
            color=alt.Color("field:N", title="Field"),
            tooltip=["format:N", "documents:Q", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(width=560, title=f"{format_source.value} formats")
    )
    format_choice = mo.ui.dropdown(
        options=[row["format"] for row in sorted(source_rows, key=lambda row: -row["documents"])],
        label="Show examples of format",
    )
    mo.vstack([source_chart, format_choice])
    return (format_choice,)


@app.cell
def _(format_choice, format_examples, format_source, mo, show_examples):
    mo.stop(format_choice.value is None, mo.md("_Pick a format to see examples._"))
    show_examples(format_examples[format_source.value, format_choice.value])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Unparsed documents

    A document is unparsed when its type has no parser (Google Drive, Confluence) or its
    parser could not find an expected field (flagged). The estimate scales the sample's
    rate to every document of that type in the database.
    """)
    return


@app.cell
def flags_overview(
    NO_PARSER,
    SourceType,
    alt,
    analysed_counts,
    corpus_counts,
    flagged_counts,
    missing_counts,
    mo,
):
    flag_rows = []
    for _source in SourceType:
        _analysed = analysed_counts[_source]
        if not _analysed:
            continue

        _rate = flagged_counts[_source] / _analysed
        flag_rows.append(
            {
                "type": str(_source),
                "in database": corpus_counts.get(str(_source), 0),
                "analysed": _analysed,
                "unparsed": flagged_counts[_source],
                "unparsed %": f"{_rate:.1%}",
                "estimated unparsed in database": round(_rate * corpus_counts.get(str(_source), 0)),
                "reason": "no parser" if missing_counts[_source][NO_PARSER] else "flagged",
            }
        )
    flagged_total = sum(
        row["estimated unparsed in database"] for row in flag_rows if row["reason"] == "flagged"
    )
    no_parser_total = sum(
        row["estimated unparsed in database"] for row in flag_rows if row["reason"] == "no parser"
    )
    database_total = sum(row["in database"] for row in flag_rows)

    missing_rows = [
        {"type": str(source), "missing field": field, "documents": count}
        for source in SourceType
        for field, count in missing_counts[source].items()
        if field != NO_PARSER
    ]
    missing_chart = (
        alt.Chart(alt.Data(values=missing_rows))
        .mark_bar()
        .encode(
            x=alt.X("documents:Q", title="Analysed documents missing the field"),
            y=alt.Y("missing field:N", sort="-x", title=None),
            color=alt.Color("type:N", title="Type"),
            tooltip=["type:N", "missing field:N", "documents:Q"],
        )
        .properties(width=560, title="Missing fields in flagged documents")
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{database_total:,}", label="Documents in database"),
                    mo.stat(f"~{flagged_total:,}", label="Flagged by a parser"),
                    mo.stat(f"{no_parser_total:,}", label="No parser (Drive, Confluence)"),
                ],
                justify="start",
            ),
            mo.ui.table(flag_rows, selection=None),
            missing_chart,
        ]
    )
    return


@app.cell
def _(SourceType, missing_counts, mo):
    flag_source = mo.ui.dropdown(
        options={str(source): source for source in SourceType if missing_counts[source]},
        value="fireflies",
        label="Type",
    )
    flag_source
    return (flag_source,)


@app.cell
def _(flag_source, missing_counts, mo):
    flag_field = mo.ui.dropdown(
        options=[field for field, _count in missing_counts[flag_source.value].most_common()],
        label="Missing field",
    )
    flag_field
    return (flag_field,)


@app.cell
def _(flag_field, flag_source, missing_examples, mo, show_examples):
    mo.stop(flag_field.value is None, mo.md("_Pick a missing field to see unparsed examples._"))
    show_examples(missing_examples[flag_source.value, flag_field.value])
    return


if __name__ == "__main__":
    app.run()
