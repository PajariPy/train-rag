import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import html
    import math
    from collections import defaultdict
    from datetime import date, datetime

    import altair as alt
    import marimo as mo
    from pydantic import BaseModel
    from sqlmodel import Session

    from train_rag.database import get_engine
    from train_rag.models import SourceType
    from train_rag.models.parsed import ParsedDocument
    from train_rag.parsers import default_registry
    from train_rag.repository.document import get_document, load_artifacts
    from train_rag.repository.inferred_attendees import load_inferred_attendees
    from train_rag.repository.inferred_date import load_inferred_dates

    return (
        BaseModel,
        ParsedDocument,
        Session,
        SourceType,
        alt,
        date,
        datetime,
        default_registry,
        defaultdict,
        get_document,
        get_engine,
        html,
        load_artifacts,
        load_inferred_attendees,
        load_inferred_dates,
        math,
        mo,
    )


@app.cell
def _(mo):
    mo.md("""
    # Field sizes

    The size in characters of every parsed field, per type, to find large outliers before
    chunking and embedding. Every document is parsed again, which takes about two minutes.

    How a field is measured:

    - **text** (`description`, `summary`, open text `text`): its length.
    - **list of strings** (`participants`, `topics`, `attachments`): all items joined by `, `.
    - **list of messages, comments or people** (`messages`, `comments`, `attendees`): the sum
      of their text, as `messages`, and the largest single item, as `messages (largest item)`.
    - **nested object** (`thread_starter`, `held_on_inferred`): each text part separately,
      as `thread_starter.text`.
    - `content` is the raw document, for all nine types, including Google Drive and
      Confluence, which have no parser.

    Empty fields and labels (`chat_type`, `precision`) are left out. An **outlier** is a value above Q3 + 3 × IQR (Tukey's far
    out fence), computed per type and field.
    """)
    return


@app.cell
def measure(BaseModel, ParsedDocument, date, datetime):
    from typing import Literal, get_origin

    BASE_FIELDS = set(ParsedDocument.model_fields)

    def text_fields(model):
        """Field names of a model, without labels such as `chat_type` (a Literal)."""
        return [
            name
            for name, field in type(model).model_fields.items()
            if get_origin(field.annotation) is not Literal
        ]

    def text_size(value):
        """Characters of the text parts of one value; None when it holds no text."""
        if isinstance(value, str):
            return len(value)
        if isinstance(value, BaseModel):
            sizes = [text_size(getattr(value, name)) for name in text_fields(value)]
            sizes = [size for size in sizes if size is not None]
            return sum(sizes) if sizes else None
        return None

    def field_sizes(parsed):
        """Yield (field, characters) for every non-empty text field of a parsed document."""
        for name in text_fields(parsed):
            if name in BASE_FIELDS:
                continue
            value = getattr(parsed, name)
            if value is None or isinstance(value, bool | date | datetime):
                continue
            if isinstance(value, str):
                if value:
                    yield name, len(value)
            elif isinstance(value, list):
                if not value:
                    continue
                if all(isinstance(item, str) for item in value):
                    yield name, len(", ".join(value))
                else:
                    sizes = [text_size(item) or 0 for item in value]
                    yield name, sum(sizes)
                    yield f"{name} (largest item)", max(sizes)
            elif isinstance(value, BaseModel):
                for sub in text_fields(value):
                    size = text_size(getattr(value, sub))
                    if size:
                        yield f"{name}.{sub}", size

    return (field_sizes,)


@app.cell
def scan(
    Session,
    SourceType,
    default_registry,
    defaultdict,
    field_sizes,
    get_engine,
    load_artifacts,
    load_inferred_attendees,
    load_inferred_dates,
    mo,
):
    DATASET = "enterprise_rag_bench"

    engine = get_engine()
    sizes = defaultdict(list)
    kinds = defaultdict(int)

    with Session(engine) as _session:
        registry = default_registry(
            load_inferred_dates(_session), load_inferred_attendees(_session)
        )
        for _source in mo.status.progress_bar(list(SourceType), title="Parsing all documents"):
            _parser = registry.get(_source)
            for _artifact in load_artifacts(_session, DATASET, _source):
                sizes[(str(_source), "content")].append((len(_artifact.content), _artifact.document_id))
                if _parser is None:
                    kinds[(str(_source), "not parsed")] += 1
                    continue
                _parsed = _parser.parse(_artifact)
                kinds[(str(_source), type(_parsed).__name__)] += 1
                for _field, _size in field_sizes(_parsed):
                    sizes[(str(_source), _field)].append((_size, _artifact.document_id))
    return engine, kinds, registry, sizes


@app.cell
def stats(math, sizes):
    def percentile(ordered, share):
        """Nearest-rank percentile of an ascending list."""
        return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]

    def fence(ordered):
        """Tukey's far out fence, Q3 + 3 * IQR."""
        q1, q3 = percentile(ordered, 0.25), percentile(ordered, 0.75)

        return q3 + 3 * (q3 - q1)

    stat_rows = []
    for (_source, _field), _values in sizes.items():
        _ordered = sorted(size for size, _document_id in _values)
        _fence = fence(_ordered)
        _p99 = percentile(_ordered, 0.99)
        stat_rows.append(
            {
                "type": _source,
                "field": _field,
                "documents": len(_ordered),
                "median": percentile(_ordered, 0.5),
                "p90": percentile(_ordered, 0.9),
                "p99": _p99,
                "max": _ordered[-1],
                "max / p99": round(_ordered[-1] / _p99, 1) if _p99 else None,
                "fence": round(_fence),
                "outliers": sum(1 for size in _ordered if size > _fence),
            }
        )
    stat_rows.sort(key=lambda row: (row["type"], -row["documents"]))
    return fence, stat_rows


@app.cell
def _(kinds, mo, stat_rows):
    _kind_rows = [
        {"type": source, "parsed as": kind, "documents": count}
        for (source, kind), count in sorted(kinds.items())
    ]
    mo.vstack(
        [
            mo.ui.table(_kind_rows, selection=None, label="What every document was parsed into"),
            mo.ui.table(
                stat_rows,
                selection=None,
                page_size=60,
                label="Characters per field (non-empty values only)",
            ),
        ]
    )
    return


@app.cell
def _(alt, mo, stat_rows):
    _biggest = sorted(
        (row for row in stat_rows if row["field"] != "content" and row["max / p99"]),
        key=lambda row: -row["max / p99"],
    )[:20]
    mo.vstack(
        [
            mo.md(
                "## Where the outliers are\n\nThe 20 fields whose largest value is furthest "
                "above their 99th percentile. A ratio near 1 means no single document stands "
                "out; a ratio of 10 means the largest one is ten times the p99."
            ),
            alt.Chart(alt.Data(values=_biggest))
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("max / p99:Q", title="Largest value ÷ 99th percentile"),
                y=alt.Y("label:N", sort="-x", title=None),
                tooltip=["type:N", "field:N", "p99:Q", "max:Q", "outliers:Q"],
            )
            .transform_calculate(label="datum.type + ' · ' + datum.field")
            .properties(height=alt.Step(18)),
        ]
    )
    return


@app.cell
def _(mo, sizes):
    type_picker = mo.ui.dropdown(
        options=sorted({source for source, _field in sizes}), value="slack", label="Type"
    )
    type_picker
    return (type_picker,)


@app.cell
def _(mo, sizes, type_picker):
    field_picker = mo.ui.dropdown(
        options=sorted(field for source, field in sizes if source == type_picker.value),
        value="content",
        label="Field",
    )
    field_picker
    return (field_picker,)


@app.cell
def _(alt, fence, field_picker, math, mo, sizes, type_picker):
    selected_values = sorted(sizes[(type_picker.value, field_picker.value)], reverse=True)
    _ordered = sorted(size for size, _document_id in selected_values)
    selected_fence = fence(_ordered)

    _bins = {}
    for _size in _ordered:
        _bin = math.floor(math.log10(max(_size, 1)) * 10) / 10
        _bins[_bin] = _bins.get(_bin, 0) + 1
    _histogram = [
        {"from": round(10**low), "to": round(10 ** (low + 0.1)), "low": low, "documents": count}
        for low, count in sorted(_bins.items())
    ]
    _rule = [{"low": math.log10(max(selected_fence, 1))}]

    _bars = (
        alt.Chart(alt.Data(values=_histogram))
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "low:Q",
                bin=alt.Bin(step=0.1),
                title="Characters (log scale: 1 = 10, 2 = 100, 3 = 1,000…)",
            ),
            y=alt.Y("documents:Q", title="Documents"),
            tooltip=["from:Q", "to:Q", "documents:Q"],
        )
    )
    _fence_line = (
        alt.Chart(alt.Data(values=_rule))
        .mark_rule(strokeDash=[4, 4], strokeWidth=2)
        .encode(x="low:Q")
    )
    mo.vstack(
        [
            mo.md(
                f"**{type_picker.value} · {field_picker.value}**: {len(_ordered):,} documents, "
                f"outlier fence {selected_fence:,.0f} characters (dashed line)."
            ),
            (_bars + _fence_line).properties(height=220, width="container"),
        ]
    )
    return selected_fence, selected_values


@app.cell
def _(mo, selected_fence, selected_values):
    largest_table = mo.ui.table(
        [
            {"document_id": document_id, "chars": size, "outlier": size > selected_fence}
            for size, document_id in selected_values[:50]
        ],
        selection="single",
        page_size=10,
        label="The 50 largest values; pick one to read it",
    )
    largest_table
    return (largest_table,)


@app.cell
def _(Session, engine, get_document, html, largest_table, mo):
    mo.stop(not largest_table.value, mo.md("_Pick a row to see the document._"))

    with Session(engine) as _session:
        _document = get_document(_session, largest_table.value[0]["document_id"])
    mo.vstack(
        [
            mo.md(
                f"### {_document.title}\n\n`#{_document.id}` · `{_document.doc_id}` · "
                f"{_document.source_type} · {_document.char_count:,} chars"
            ),
            mo.Html(
                '<pre style="white-space: pre-wrap; word-break: break-word; margin: 0; '
                'max-height: 600px; overflow: auto">'
                f"{html.escape(_document.content)}</pre>"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
