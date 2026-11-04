import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import html
    import math

    import altair as alt
    import marimo as mo
    from sqlmodel import Session

    from train_rag.data_loaders import EnterpriseRagBenchLoader
    from train_rag.database import get_engine
    from train_rag.models import SourceType
    from train_rag.repository.document import document_sizes, get_document

    return (
        EnterpriseRagBenchLoader,
        Session,
        SourceType,
        alt,
        document_sizes,
        get_document,
        get_engine,
        html,
        math,
        mo,
    )


@app.cell
def page_format():
    # A Letter or A4 page with 2.5 cm margins, 14 pt font, single line spacing.
    CHARS_PER_LINE = 65
    LINES_PER_PAGE = 40
    return CHARS_PER_LINE, LINES_PER_PAGE


@app.cell
def _(CHARS_PER_LINE, LINES_PER_PAGE, mo):
    mo.md(f"""
    # Document sizes

    How long the raw documents of every source are, in characters and in printed pages.

    **Pages** are an estimate for a Letter or A4 page with normal margins, a 14 pt font
    and single line spacing: about {CHARS_PER_LINE} characters per line and
    {LINES_PER_PAGE} lines per page, so a page full of text holds about
    {CHARS_PER_LINE * LINES_PER_PAGE:,} characters. Every line break starts a new line,
    so a chat thread of short messages fills more pages than its character count
    suggests.
    """)
    return


@app.cell
def load_sizes(
    CHARS_PER_LINE,
    EnterpriseRagBenchLoader,
    LINES_PER_PAGE,
    Session,
    document_sizes,
    get_engine,
    mo,
):
    engine = get_engine()

    @mo.persistent_cache
    def measure(chars_per_line, lines_per_page):
        """Every document's size as plain dicts, which the cache can store on disk."""
        with Session(get_engine()) as session:
            sizes = document_sizes(session, EnterpriseRagBenchLoader.name, chars_per_line)
        return [
            {
                "document_id": size.document_id,
                "source": size.source_type.value,
                "chars": size.chars,
                "pages": size.lines / lines_per_page,
            }
            for size in sizes
        ]

    with mo.status.spinner("Measuring every document (about a minute the first time)…"):
        rows = measure(CHARS_PER_LINE, LINES_PER_PAGE)
    return engine, rows


@app.cell
def helpers(math):
    def percentile(ordered, share):
        """Value below which `share` of the sorted values fall (nearest rank)."""
        return ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))]

    return (percentile,)


@app.cell
def summary(SourceType, mo, percentile, rows):
    def describe(name, group):
        chars = sorted(row["chars"] for row in group)
        pages = sorted(row["pages"] for row in group)
        total_chars = sum(chars)
        total_pages = sum(pages)
        return {
            "source": name,
            "documents": len(group),
            "total chars (M)": round(total_chars / 1e6, 1),
            "total pages": round(total_pages),
            "median chars": percentile(chars, 0.5),
            "p90 chars": percentile(chars, 0.9),
            "max chars": chars[-1],
            "median pages": round(percentile(pages, 0.5), 1),
            "p90 pages": round(percentile(pages, 0.9), 1),
            "max pages": round(pages[-1], 1),
            "chars per page": round(total_chars / total_pages),
        }

    by_source = {
        source_type.value: [row for row in rows if row["source"] == source_type.value]
        for source_type in SourceType
    }
    summary_rows = [describe(name, group) for name, group in by_source.items()]
    summary_rows.sort(key=lambda row: -row["median pages"])
    summary_rows.append(describe("all", rows))

    _total = summary_rows[-1]
    mo.vstack(
        [
            mo.md(
                f"## Overview\n{_total['documents']:,} documents, "
                f"{_total['total chars (M)']:,} million characters, about "
                f"**{_total['total pages']:,} pages**: a median document is "
                f"{_total['median chars']:,} characters, about "
                f"{_total['median pages']} pages. *Chars per page* below 2,600 means "
                "short lines: the source breaks lines often."
            ),
            mo.ui.table(summary_rows, selection=None, pagination=False, label="Sizes per source"),
        ]
    )
    return (by_source,)


@app.cell
def _(alt, by_source, mo, percentile):
    _ranges = []
    for _name, _group in by_source.items():
        _pages = sorted(row["pages"] for row in _group)
        _ranges.append(
            {
                "source": _name,
                "p10": round(percentile(_pages, 0.1), 2),
                "median": round(percentile(_pages, 0.5), 2),
                "p90": round(percentile(_pages, 0.9), 2),
            }
        )
    _order = [row["source"] for row in sorted(_ranges, key=lambda row: -row["median"])]
    _base = alt.Chart(alt.Data(values=_ranges)).encode(
        y=alt.Y("source:N", sort=_order, title=None),
        tooltip=["source:N", "p10:Q", "median:Q", "p90:Q"],
    )
    _spread = _base.mark_bar(height=14, cornerRadius=4, opacity=0.35).encode(
        x=alt.X("p10:Q", title="Pages per document"), x2="p90:Q"
    )
    _median = _base.mark_tick(thickness=3, size=22).encode(x="median:Q")
    mo.vstack(
        [
            mo.md(
                "## Pages per document\nThe bar runs from the 10th to the 90th percentile; "
                "the tick is the median."
            ),
            (_spread + _median).properties(height=alt.Step(30), width="container"),
        ]
    )
    return


@app.cell
def _(by_source, mo):
    source_picker = mo.ui.dropdown(options=["all", *sorted(by_source)], value="all", label="Source")
    source_picker
    return (source_picker,)


@app.cell
def _(alt, by_source, math, mo, rows, source_picker):
    selected_rows = rows if source_picker.value == "all" else by_source[source_picker.value]

    _bins = {}
    for _row in selected_rows:
        _bin = math.floor(math.log10(max(_row["pages"], 0.01)) * 10) / 10
        _bins[_bin] = _bins.get(_bin, 0) + 1
    _histogram = [
        {
            "low": low,
            "from pages": round(10**low, 2),
            "to pages": round(10 ** (low + 0.1), 2),
            "documents": count,
        }
        for low, count in sorted(_bins.items())
    ]
    mo.vstack(
        [
            mo.md(f"**{source_picker.value}**: {len(selected_rows):,} documents"),
            alt.Chart(alt.Data(values=_histogram))
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(
                    "low:Q",
                    bin=alt.Bin(step=0.1),
                    title="Pages (log scale: -1 = 0.1 page, 0 = 1 page, 1 = 10 pages)",
                ),
                y=alt.Y("documents:Q", title="Documents"),
                tooltip=["from pages:Q", "to pages:Q", "documents:Q"],
            )
            .properties(height=220, width="container"),
        ]
    )
    return (selected_rows,)


@app.cell
def _(mo, selected_rows):
    largest_table = mo.ui.table(
        [
            {
                "document_id": row["document_id"],
                "source": row["source"],
                "chars": row["chars"],
                "pages": round(row["pages"], 1),
            }
            for row in sorted(selected_rows, key=lambda row: -row["pages"])[:50]
        ],
        selection="single",
        page_size=10,
        label="The 50 longest documents in pages; pick one to read it",
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
                f"{_document.source_type} · {_document.char_count:,} chars · "
                f"{largest_table.value[0]['pages']} pages"
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
