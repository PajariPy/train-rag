import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def imports():
    import html
    import json
    import random
    from datetime import UTC, datetime

    import marimo as mo
    from sqlmodel import Session

    from train_rag.data_loaders import EnterpriseRagBenchLoader
    from train_rag.database import get_engine
    from train_rag.processing.normalize import normalize_content
    from train_rag.repository.document import ids_of_doc_ids, load_documents
    from train_rag.repository.question import load_questions

    return (
        EnterpriseRagBenchLoader,
        Session,
        UTC,
        datetime,
        get_engine,
        html,
        ids_of_doc_ids,
        json,
        load_documents,
        load_questions,
        mo,
        normalize_content,
        random,
    )


@app.cell
def _(mo):
    mo.md("""
    # Label questions for ambiguity

    **Ambiguous** means the question, as written, could fairly be answered from a different
    document than the expected one: it doesn't say enough to pick out which document it
    means.

    30 questions from `sample.json`: 10 whose top 2 gte documents both match very closely,
    10 where they match poorly, and 10 in between. They are shown in a fixed shuffled order,
    without their group, so the score doesn't steer the label. Each label is saved to
    `labels.json` when you press Save, and the next unlabelled question opens.
    """)
    return


@app.cell
def load(
    EnterpriseRagBenchLoader,
    Session,
    get_engine,
    ids_of_doc_ids,
    json,
    load_documents,
    load_questions,
    mo,
    random,
):
    HERE = mo.notebook_dir()
    LABELS = HERE / "labels.json"
    sample = json.loads((HERE / "sample.json").read_text())
    random.Random(7).shuffle(sample)
    order = [_row["question_id"] for _row in sample]
    rows = {_row["question_id"]: _row for _row in sample}

    _engine = get_engine()
    with Session(_engine) as _session:
        questions = {
            _q.id: _q
            for _q in load_questions(_session, EnterpriseRagBenchLoader.name)
            if _q.id in rows
        }
        _doc_ids = {
            _doc_id
            for _id in order
            for _doc_id in [
                *questions[_id].expected_doc_ids,
                *(_hit["doc_id"] for _hit in rows[_id]["top2"]),
            ]
        }
        _ids = ids_of_doc_ids(_session, EnterpriseRagBenchLoader.name, sorted(_doc_ids))
        documents = {_d.doc_id: _d for _d in load_documents(_session, sorted(_ids))}
    return LABELS, documents, order, questions, rows


@app.cell
def labels_state(LABELS, json, mo):
    saved_labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    get_labels, set_labels = mo.state(saved_labels)
    return get_labels, saved_labels, set_labels


@app.cell
def current_state(mo, order, saved_labels):
    _unlabelled = [_id for _id in order if _id not in saved_labels]
    get_current, set_current = mo.state(_unlabelled[0] if _unlabelled else order[0])
    return get_current, set_current


@app.cell
def picker(get_current, get_labels, mo, order, set_current):
    _labels = get_labels()
    _options = {
        f"{_i}. {_id} {'✓' if _id in _labels else '·'}": _id for _i, _id in enumerate(order, 1)
    }
    question_picker = mo.ui.dropdown(
        options=_options,
        value=next(_key for _key, _id in _options.items() if _id == get_current()),
        label="Question",
        on_change=set_current,
    )
    return (question_picker,)


@app.cell
def form_cell(get_current, get_labels, mo):
    _saved = get_labels().get(get_current(), {})
    label_form = (
        mo.md("{ambiguous}\n\n{note}")
        .batch(
            ambiguous=mo.ui.radio(
                options=["ambiguous", "not ambiguous"],
                value=_saved.get("label"),
                label="**Label**",
            ),
            note=mo.ui.text_area(
                value=_saved.get("note", ""), label="**Note** (optional)", full_width=True
            ),
        )
        .form(submit_button_label="Save", clear_on_submit=False)
    )
    return (label_form,)


@app.cell
def save(
    LABELS,
    UTC,
    datetime,
    get_current,
    get_labels,
    json,
    label_form,
    order,
    set_current,
    set_labels,
):
    _value = label_form.value
    if _value is not None and _value["ambiguous"] is not None:
        _labels = {
            **get_labels(),
            get_current(): {
                "label": _value["ambiguous"],
                "note": _value["note"].strip(),
                "labelled_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        }
        LABELS.write_text(json.dumps(_labels, indent=1, sort_keys=True))
        _unlabelled = [_id for _id in order if _id not in _labels]
        set_labels(_labels)
        if _unlabelled:
            set_current(_unlabelled[0])
    return


@app.cell
def _(html, mo, normalize_content):
    def document_column(document, heading):
        """One column: who the document is, then its full text, wrapped and scrollable."""
        return mo.vstack(
            [
                mo.md(
                    f"**{heading}**\n\n[{document.source_type}] "
                    f"{document.title.splitlines()[0]}\n\n`{document.doc_id}` · "
                    f"{len(document.content):,} characters"
                ),
                mo.Html(
                    "<pre style='white-space: pre-wrap; height: 70vh; overflow-y: auto; "
                    "font-size: 0.8rem; margin: 0'>"
                    f"{html.escape(normalize_content(document.content))}</pre>"
                ),
            ],
            gap=0.5,
        )

    return (document_column,)


@app.cell
def _(
    document_column,
    documents,
    get_current,
    get_labels,
    label_form,
    mo,
    order,
    question_picker,
    questions,
    rows,
):
    _qid = get_current()
    _question = questions[_qid]
    _expected = [_id for _id in _question.expected_doc_ids if _id in documents]
    _top = rows[_qid]["top2"]
    _top_ids = {_hit["doc_id"] for _hit in _top}
    _facts = "\n".join(f"- {_fact}" for _fact in _question.answer_facts)
    _done = len(get_labels())

    # Columns: the first expected document gte didn't put in its top 2, then gte #1 and #2.
    _columns = []
    _missing = [_id for _id in _expected if _id not in _top_ids]
    if _missing:
        _columns.append(document_column(documents[_missing[0]], "Expected (not in gte's top 2)"))
    for _i, _hit in enumerate(_top, 1):
        _mark = " ✅ expected" if _hit["doc_id"] in _expected else ""
        _columns.append(
            document_column(
                documents[_hit["doc_id"]],
                f"gte #{_i} · similarity {_hit['gte_similarity']:.3f}{_mark}",
            )
        )
    _more = {
        f"Expected: [{documents[_id].source_type}] {documents[_id].title.splitlines()[0]}": (
            document_column(documents[_id], "Expected")
        )
        for _id in _missing[1:]
    }
    _expected_note = (
        f"{len(_expected)} expected document(s)" if _expected else "No expected document"
    )

    mo.vstack(
        [
            mo.hstack(
                [question_picker, mo.md(f"**{_done} / {len(order)} labelled**")],
                justify="start",
                gap=2,
            ),
            mo.md(
                f"### {_qid} ({_question.question_type.value})\n\n"
                f"**Question:** {_question.text}\n\n"
                f"**Gold answer:** {_question.gold_answer}\n\n"
                + (f"**Gold facts:**\n\n{_facts}\n\n" if _facts else "")
                + f"*{_expected_note}.*"
            ),
            label_form,
            mo.hstack(_columns, widths="equal", align="start", gap=1),
            mo.accordion(_more, multiple=True) if _more else mo.md(""),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
